// Signal processing for remote photoplethysmography (rPPG).
// Pure functions, no DOM access, so they can be unit-tested in Node.

export const HR_MIN_HZ = 0.7; // 42 bpm
export const HR_MAX_HZ = 4.0; // 240 bpm
const BEAT_REFRACTORY_S = 0.3;  // fixed refractory for the rhythm-independent path (> ~0.23 s dicrotic delay)
const HARMONIC_RATIO = 0.4;    // f/2 peak this strong relative to f -> treat f as a harmonic (sim: harmonics 0.47-0.73, noise <= 0.28)

const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
const std = (a) => {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length || 1));
};
const median = (a) => {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const k = s.length >> 1;
  return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2;
};

/**
 * Linearly resample irregular samples onto a uniform grid.
 * @param {number[]} t   timestamps in seconds (strictly increasing)
 * @param {number[][]} channels  arrays aligned with t
 * @param {number} fs    target sample rate (Hz)
 */
export function resample(t, channels, fs) {
  if (t.length < 2) return { t: [], channels: channels.map(() => []) };
  const t0 = t[0];
  const n = Math.floor((t[t.length - 1] - t0) * fs) + 1;
  const out = channels.map(() => new Array(n));
  const tu = new Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const ti = t0 + i / fs;
    tu[i] = ti;
    while (j < t.length - 2 && t[j + 1] < ti) j++;
    const span = t[j + 1] - t[j];
    const w = span > 0 ? (ti - t[j]) / span : 0;
    for (let c = 0; c < channels.length; c++) {
      out[c][i] = channels[c][j] * (1 - w) + channels[c][j + 1] * w;
    }
  }
  return { t: tu, channels: out };
}

/**
 * POS (Plane-Orthogonal-to-Skin) pulse extraction.
 * Wang, den Brinker, Stuijk, de Haan, IEEE TBME 64(7), 2017.
 *
 * Polarity: more blood absorbs more light, so skin gets darker in systole
 * (pulsatile signature ≈ -[0.33, 0.77, 0.53] in normalised RGB). The raw POS
 * projection therefore falls in systole; it is negated here so the output
 * follows the conventional PPG polarity (up = more blood volume), matching
 * contact pulse oximeters.
 */
export function pos(r, g, b, fs, windowSec = 1.6) {
  const n = r.length;
  const l = Math.max(2, Math.round(windowSec * fs));
  const h = new Array(n).fill(0);
  if (n < l) return h;
  for (let start = 0; start + l <= n; start++) {
    const R = r.slice(start, start + l);
    const G = g.slice(start, start + l);
    const B = b.slice(start, start + l);
    const mr = mean(R), mg = mean(G), mb = mean(B);
    if (!mr || !mg || !mb) continue;
    const s1 = new Array(l), s2 = new Array(l);
    for (let i = 0; i < l; i++) {
      const rn = R[i] / mr, gn = G[i] / mg, bn = B[i] / mb;
      s1[i] = gn - bn;
      s2[i] = gn + bn - 2 * rn;
    }
    const sd2 = std(s2);
    const alpha = sd2 > 1e-12 ? std(s1) / sd2 : 0;
    const p = s1.map((v, i) => v + alpha * s2[i]);
    const mp = mean(p);
    for (let i = 0; i < l; i++) h[start + i] -= p[i] - mp;
  }
  return h;
}

// RBJ audio-EQ-cookbook biquads.
function biquad(type, f0, fs, q = Math.SQRT1_2) {
  const w0 = (2 * Math.PI * f0) / fs;
  const cos = Math.cos(w0), alpha = Math.sin(w0) / (2 * q);
  let b0, b1, b2;
  if (type === 'lowpass') {
    b0 = (1 - cos) / 2; b1 = 1 - cos; b2 = (1 - cos) / 2;
  } else {
    b0 = (1 + cos) / 2; b1 = -(1 + cos); b2 = (1 + cos) / 2;
  }
  const a0 = 1 + alpha, a1 = -2 * cos, a2 = 1 - alpha;
  return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
}

function applyBiquad(x, [b0, b1, b2, a1, a2]) {
  const y = new Array(x.length);
  let x1 = x[0], x2 = x[0], y1 = 0, y2 = 0;
  // Initialise state to steady state for a constant input equal to x[0].
  const dc = (b0 + b1 + b2) / (1 + a1 + a2);
  y1 = y2 = Number.isFinite(dc) ? dc * x[0] : 0;
  for (let i = 0; i < x.length; i++) {
    const v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = v;
    y[i] = v;
  }
  return y;
}

/** Zero-phase band-pass (4th-order Butterworth-style, run forward and backward). */
export function bandpass(x, fs, lo = HR_MIN_HZ, hi = HR_MAX_HZ) {
  if (x.length < 4) return [...x];
  const stages = [
    biquad('highpass', lo, fs), biquad('highpass', lo, fs),
    biquad('lowpass', hi, fs), biquad('lowpass', hi, fs),
  ];
  const m = mean(x);
  let y = x.map((v) => v - m);
  for (const s of stages) y = applyBiquad(y, s);
  y.reverse();
  for (const s of stages) y = applyBiquad(y, s);
  return y.reverse();
}

/** Hann-windowed periodogram evaluated on a fine frequency grid. */
export function spectrum(x, fs, fLo = 0.5, fHi = 5, df = 0.01) {
  const n = x.length;
  const m = mean(x);
  const w = x.map((v, i) => (v - m) * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1))));
  const freqs = [], power = [];
  for (let f = fLo; f <= fHi + 1e-9; f += df) {
    let re = 0, im = 0;
    const k = (2 * Math.PI * f) / fs;
    for (let i = 0; i < n; i++) {
      re += w[i] * Math.cos(k * i);
      im -= w[i] * Math.sin(k * i);
    }
    freqs.push(f);
    power.push(re * re + im * im);
  }
  return { freqs, power };
}

/**
 * Heart rate from the spectral peak, plus an SNR (dB) in the style of
 * de Haan & Jeanne 2013: power near f0 and 2·f0 vs. the rest of the band.
 */
export function spectralHR(x, fs) {
  const { freqs, power } = spectrum(x, fs, 0.5, 2 * HR_MAX_HZ + 0.3);
  let best = -1, bestP = -Infinity;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] < HR_MIN_HZ || freqs[i] > HR_MAX_HZ) continue;
    if (power[i] > bestP) { bestP = power[i]; best = i; }
  }
  if (best < 0) return { bpm: NaN, snrDb: -Infinity, freqs, power };
  // A sharp systolic upstroke puts a lot of power in the 2nd harmonic, which can
  // beat the fundamental. If there is a clear peak near f/2, prefer it.
  const half = freqs[best] / 2;
  if (half >= HR_MIN_HZ) {
    let hi = -1, hp = -Infinity;
    for (let i = 0; i < freqs.length; i++) {
      if (Math.abs(freqs[i] - half) <= 0.08 && power[i] > hp) { hp = power[i]; hi = i; }
    }
    const isLocalMax = hi > 0 && hi < freqs.length - 1 && power[hi] >= power[hi - 1] && power[hi] >= power[hi + 1];
    if (isLocalMax && hp >= HARMONIC_RATIO * bestP) best = hi;
  }
  const f0 = freqs[best];
  let sig = 0, noise = 0;
  for (let i = 0; i < freqs.length; i++) {
    const f = freqs[i];
    if (f < HR_MIN_HZ || f > 2 * HR_MAX_HZ) continue;
    if (Math.abs(f - f0) <= 0.1 || Math.abs(f - 2 * f0) <= 0.2) sig += power[i];
    else noise += power[i];
  }
  const snrDb = noise > 0 ? 10 * Math.log10(sig / noise) : Infinity;
  return { bpm: f0 * 60, snrDb, freqs, power };
}

/**
 * Systolic peak detection on a band-passed pulse signal.
 * Returns peak positions in (fractional) samples, refined by parabolic interpolation.
 */
export function findPeaks(x, fs, hrHz = NaN) {
  // Refractory period: 60% of the beat period when the heart rate is known, so
  // the dicrotic wave and noise bumps between beats are not counted as beats;
  // otherwise a fixed 180 bpm limit.
  const minDist = Math.max(2, Math.round((Number.isFinite(hrHz) ? 0.6 / hrHz : 60 / (HR_MAX_HZ * 60 * 0.75)) * fs));
  const win = Math.max(3, Math.round(1.5 * fs));
  const candidates = [];
  for (let i = 1; i < x.length - 1; i++) {
    if (!(x[i] > x[i - 1] && x[i] >= x[i + 1] && x[i] > 0)) continue;
    // Local amplitude threshold: a fraction of the local positive envelope.
    const a = Math.max(0, i - win), b = Math.min(x.length, i + win);
    let localMax = 0;
    for (let k = a; k < b; k++) if (x[k] > localMax) localMax = x[k];
    if (x[i] < 0.3 * localMax) continue;
    candidates.push(i);
  }
  // Enforce refractory period, keeping the larger peak.
  const kept = [];
  for (const i of candidates) {
    const last = kept[kept.length - 1];
    if (last !== undefined && i - last < minDist) {
      if (x[i] > x[last]) kept[kept.length - 1] = i;
    } else kept.push(i);
  }
  return kept.map((i) => {
    const y0 = x[i - 1], y1 = x[i], y2 = x[i + 1];
    const d = y0 - 2 * y1 + y2;
    return d !== 0 ? i + (0.5 * (y0 - y2)) / d : i;
  });
}

/**
 * Inter-beat intervals and time-domain HRV.
 * Intervals that are physiologically implausible or deviate >25% from the
 * median are rejected (they are almost always detection errors in rPPG).
 */
export function hrv(peakSamples, fs) {
  const ibi = [];
  for (let i = 1; i < peakSamples.length; i++) ibi.push((peakSamples[i] - peakSamples[i - 1]) / fs);
  const med = median(ibi);
  const valid = ibi.map((v) => v >= 60 / (HR_MAX_HZ * 60) && v <= 60 / (HR_MIN_HZ * 60) && Math.abs(v - med) <= 0.25 * med);
  const good = ibi.filter((_, i) => valid[i]);
  const succ = [];
  for (let i = 1; i < ibi.length; i++) if (valid[i] && valid[i - 1]) succ.push((ibi[i] - ibi[i - 1]) * 1000);
  return {
    ibi,
    valid,
    bpm: good.length ? 60 / mean(good) : NaN,
    sdnnMs: good.length > 1 ? std(good) * 1000 : NaN,
    rmssdMs: succ.length ? Math.sqrt(mean(succ.map((d) => d * d))) : NaN,
    acceptedFraction: ibi.length ? good.length / ibi.length : 0,
  };
}

/**
 * Full pipeline: irregular RGB samples → uniform pulse waveform + metrics.
 * @param {{t:number[], r:number[], g:number[], b:number[]}} s  t in seconds
 */
export function analyze(s, fs = 30) {
  const { t, channels: [r, g, b] } = resample(s.t, [s.r, s.g, s.b], fs);
  if (t.length < fs * 4) return null;
  const raw = pos(r, g, b, fs);
  const bvp = bandpass(raw, fs);
  const sd = std(bvp) || 1;
  const norm = bvp.map((v) => v / sd);
  const spec = spectralHR(norm, fs);
  // Two peak sets: one with a refractory period tied to the spectral HR (best
  // for steady rhythms), one with a fixed 0.3 s refractory that does not rely
  // on the spectral HR (which is unreliable when the rhythm is irregular).
  const peaksSpectral = findPeaks(norm, fs, spec.bpm / 60);
  const peaksFixed = findPeaks(norm, fs, 0.6 / BEAT_REFRACTORY_S);
  const quality = assessQuality(raw, norm, peaksFixed, spec, fs);
  // An irregular rhythm has no single spectral peak; when only the beat-based
  // check passes, report the mean beat rate instead of the spectral peak.
  const peaks = quality.source === 'beats' ? peaksFixed : peaksSpectral;
  const bpm = quality.source === 'beats' ? meanBeatRate(peaks, fs) : spec.bpm;
  return { fs, t, rgb: { r, g, b }, bvp: norm, peaks, spectral: spec, hrv: hrv(peaks, fs), quality, bpm };
}

/** 60 / mean of all physiologically plausible inter-beat intervals (no regularity filter). */
export function meanBeatRate(peaks, fs) {
  const ibi = [];
  for (let i = 1; i < peaks.length; i++) {
    const v = (peaks[i] - peaks[i - 1]) / fs;
    if (v >= 1 / HR_MAX_HZ && v <= 1 / HR_MIN_HZ) ibi.push(v);
  }
  return ibi.length ? 60 / mean(ibi) : NaN;
}

/**
 * Beat-shape consistency (template-matching SQI, after Li & Clifford 2012):
 * mean correlation of each beat segment with the leave-one-out average beat.
 * Band-passed noise aligned on its own maxima also scores ~0.8, so this is
 * only used together with bandPowerRatioDb.
 */
export function beatSQI(y, peaks) {
  const ibi = [];
  for (let i = 1; i < peaks.length; i++) ibi.push(peaks[i] - peaks[i - 1]);
  if (ibi.length < 3) return 0;
  const med = median(ibi);
  const pre = Math.round(0.3 * med), post = Math.round(0.5 * med);
  const segs = peaks.map(Math.round).filter((p) => p - pre >= 0 && p + post < y.length).map((p) => y.slice(p - pre, p + post + 1));
  const n = segs.length;
  if (n < 4) return 0;
  const L = segs[0].length;
  const sum = new Array(L).fill(0);
  for (const sg of segs) for (let i = 0; i < L; i++) sum[i] += sg[i];
  const corr = (a, c) => {
    const ma = mean(a), mc = mean(c);
    let num = 0, da = 0, dc = 0;
    for (let i = 0; i < L; i++) { num += (a[i] - ma) * (c[i] - mc); da += (a[i] - ma) ** 2; dc += (c[i] - mc) ** 2; }
    return da && dc ? num / Math.sqrt(da * dc) : 0;
  };
  return segs.reduce((acc, sg) => acc + corr(sg, sum.map((v, i) => (v - sg[i]) / (n - 1))), 0) / n;
}

/**
 * Power density in the cardiac band (0.7–4 Hz) relative to 4 Hz–Nyquist, in dB,
 * on the un-filtered POS signal. A pulse concentrates power below 4 Hz whatever
 * the rhythm; sensor noise spreads across the whole band. Motion is also
 * in-band, which is why beatSQI is required as well.
 */
export function bandPowerRatioDb(raw, fs) {
  const top = fs / 2 - 0.5;
  const { freqs, power } = spectrum(raw, fs, HR_MIN_HZ, top, 0.05);
  let inb = 0, out = 0;
  freqs.forEach((f, i) => { if (f <= HR_MAX_HZ) inb += power[i]; else out += power[i]; });
  if (!out) return Infinity;
  return 10 * Math.log10((inb / (HR_MAX_HZ - HR_MIN_HZ)) / (out / (top - HR_MAX_HZ)));
}

/**
 * Signal quality from two independent checks; the better one wins:
 *  - spectral: SNR of the dominant frequency (rewards a steady rhythm)
 *  - beats: band-power ratio AND beat-shape consistency (rhythm-independent,
 *    so irregular rhythms are not marked poor just for being irregular)
 * Thresholds were chosen on simulated data and checked on held-out seeds (see
 * README): the beat path adds ~0.6% false accepts on pulse-free input, on top of
 * the spectral path's ~2.9%. Not clinically calibrated.
 */
export function assessQuality(raw, bvp, peaks, spec, fs) {
  const spectral = qualityLabel(spec.snrDb);
  const bandDb = bandPowerRatioDb(raw, fs);
  const sqi = beatSQI(bvp, peaks);
  const beats = bandDb >= 5 && sqi >= 0.88 ? 'good' : bandDb >= 3.5 && sqi >= 0.88 ? 'fair' : 'poor';
  const rank = { none: -1, poor: 0, fair: 1, good: 2 };
  const useBeats = rank[beats] > rank[spectral];
  return { label: useBeats ? beats : spectral, source: useBeats ? 'beats' : 'spectral', spectralSnrDb: spec.snrDb, bandRatioDb: bandDb, beatSqi: sqi };
}

/** Coarse signal quality label from SNR. Thresholds are heuristic. */
export function qualityLabel(snrDb) {
  if (!Number.isFinite(snrDb)) return 'none';
  if (snrDb >= 3) return 'good';
  if (snrDb >= 0) return 'fair';
  return 'poor';
}
