"""Shared preprocessing for the PPG->ECG model.

The input path deliberately mirrors src/dsp.js (resample to 30 Hz, the same
cascaded-biquad zero-phase band-pass, z-scoring) so the model sees at
training time what the browser app will feed it.
"""
import glob
import os

import numpy as np
from scipy import signal

FS_IN = 30          # model input rate (app's analysis rate)
FS_OUT = 120        # model output rate (ECG)
WIN_S = 8           # window length in seconds
N_IN = FS_IN * WIN_S
N_OUT = FS_OUT * WIN_S
FS_BIDMC = 125


# ---- exact port of src/dsp.js bandpass ------------------------------------
def _biquad(kind, f0, fs, q=np.sqrt(0.5)):
    w0 = 2 * np.pi * f0 / fs
    cos, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    if kind == 'lowpass':
        b = [(1 - cos) / 2, 1 - cos, (1 - cos) / 2]
    else:
        b = [(1 + cos) / 2, -(1 + cos), (1 + cos) / 2]
    a0 = 1 + alpha
    return np.array(b) / a0, np.array([1, -2 * cos / a0, (1 - alpha) / a0])


def _apply(x, ba):
    b, a = ba
    dc = b.sum() / a.sum()
    # Same initial state as dsp.js: steady state for constant input x[0].
    zi = signal.lfilter_zi(b, a) * x[0] if np.isfinite(dc) else None
    return signal.lfilter(b, a, x, zi=zi)[0] if zi is not None else signal.lfilter(b, a, x)


def js_bandpass(x, fs=FS_IN, lo=0.7, hi=4.0):
    stages = [_biquad('highpass', lo, fs)] * 2 + [_biquad('lowpass', hi, fs)] * 2
    y = np.asarray(x, float) - np.mean(x)
    for s in stages:
        y = _apply(y, s)
    y = y[::-1]
    for s in stages:
        y = _apply(y, s)
    return y[::-1].copy()


def zscore(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-8 else 1.0)


# ---- data -----------------------------------------------------------------
def load_bidmc(data_dir):
    """Return list of (record_id, ppg@30Hz pre-filter, ecg@120Hz filtered)."""
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, 'bidmc_*_Signals.csv'))):
        rid = os.path.basename(f).split('_')[1]
        d = np.genfromtxt(f, delimiter=',', skip_header=1)
        ppg, ecg = d[:, 2], d[:, 5]  # PLETH, lead II
        t125 = np.arange(len(ppg)) / FS_BIDMC
        t30 = np.arange(0, t125[-1], 1 / FS_IN)
        ppg30 = np.interp(t30, t125, ppg)  # linear, like dsp.resample
        sos = signal.butter(2, [0.5, 40], btype='band', fs=FS_BIDMC, output='sos')
        ecg_f = signal.sosfiltfilt(sos, ecg)
        ecg120 = signal.resample_poly(ecg_f, 24, 25)
        n = min(len(ppg30) // FS_IN, len(ecg120) // FS_OUT)
        out.append((rid, ppg30[: n * FS_IN], ecg120[: n * FS_OUT]))
    return out


def model_input(ppg30_raw):
    """Full-record input as the app produces it: band-passed (not yet z-scored)."""
    return js_bandpass(ppg30_raw)


def windows(x30, y120, hop_s):
    """Yield aligned (input, target) windows; both z-scored per window."""
    hop_in, hop_out = int(hop_s * FS_IN), int(hop_s * FS_OUT)
    k = 0
    while k * hop_in + N_IN <= len(x30) and k * hop_out + N_OUT <= len(y120):
        yield (x30[k * hop_in: k * hop_in + N_IN], y120[k * hop_out: k * hop_out + N_OUT])
        k += 1


def split_subjects(ids, seed=0, n_test=11, n_val=6):
    rng = np.random.default_rng(seed)
    ids = list(rng.permutation(sorted(ids)))
    return ids[n_test + n_val:], ids[n_test: n_test + n_val], ids[:n_test]


# ---- beat detectors and per-record alignment -------------------------------
def dominant_hr_hz(x, fs):
    f, p = signal.periodogram(x - x.mean(), fs, nfft=max(8192, len(x)))
    band = (f >= 0.7) & (f <= 3.5)
    return f[band][np.argmax(p[band])]


def ppg_peaks(x30, fs=FS_IN):
    """Systolic peaks; refractory period tied to the dominant HR to skip dicrotic notches."""
    hr = dominant_hr_hz(x30, fs)
    p, _ = signal.find_peaks(x30, distance=max(2, int(0.6 * fs / hr)), prominence=0.3 * x30.std())
    return p / fs


def r_peaks(y, fs=FS_OUT):
    """Pan-Tompkins-style: 5-20 Hz band, squared slope, then snap to |ECG| maximum."""
    sos = signal.butter(2, [5, 20], btype='band', fs=fs, output='sos')
    e = np.gradient(signal.sosfiltfilt(sos, y)) ** 2
    e = np.convolve(e, np.ones(int(0.12 * fs)) / int(0.12 * fs), mode='same')
    cand, _ = signal.find_peaks(e, distance=int(0.25 * fs), height=0.3 * np.percentile(e, 98))
    yc = y - np.median(y)
    w = int(0.06 * fs)
    idx = [c - w + np.argmax(np.abs(yc[max(0, c - w): c + w])) for c in cand if c >= w]
    if not idx:
        return np.array([])
    pol = np.sign(np.median(yc[idx])) or 1.0
    idx = [c - w + np.argmax(pol * yc[max(0, c - w): c + w]) for c in cand if c >= w]
    return np.unique(idx) / fs


def ecg_polarity(y):
    r = r_peaks(y)
    if not len(r):
        return 1.0
    yc = y - np.median(y)
    return float(np.sign(np.median(yc[(r * FS_OUT).astype(int)])) or 1.0)


TARGET_DELAY_S = 0.25   # convention: PPG systolic peak follows the R-peak by 250 ms


def record_offset(x30, y120):
    """Circular-median of (PPG peak - preceding R) modulo RR, i.e. the record's
    channel offset as seen by the peaks. Only defined modulo one beat."""
    r, p = r_peaks(y120), ppg_peaks(x30)
    if len(r) < 10 or len(p) < 10:
        return None, None
    rr = np.median(np.diff(r))
    d = []
    for tp in p:
        prior = r[r < tp]
        if len(prior):
            d.append(((tp - prior[-1]) % rr) / rr * 2 * np.pi)
    ang = np.angle(np.mean(np.exp(1j * np.array(d))))
    spread = 1 - np.abs(np.mean(np.exp(1j * np.array(d))))  # 0 = perfectly consistent
    return (ang % (2 * np.pi)) / (2 * np.pi) * rr, spread


def align_record(x30, y120):
    """Shift the ECG so the R->PPG-peak delay equals TARGET_DELAY_S.
    Returns (x30, y120_aligned, consistency) or None if beats are unreliable."""
    off, spread = record_offset(x30, y120)
    if off is None:
        return None
    shift = int(round((off - TARGET_DELAY_S) * FS_OUT))   # >0: ECG must move later
    y = np.roll(y120, shift)
    trim = abs(shift) + FS_OUT
    n_s = (len(y) - 2 * trim) // FS_OUT
    y = y[trim: trim + n_s * FS_OUT]
    x = x30[trim * FS_IN // FS_OUT: trim * FS_IN // FS_OUT + n_s * FS_IN]
    return x, y, spread
