"""Train and evaluate the PPG->ECG reconstruction model on BIDMC.

    python3 train.py --data /path/to/bidmc_csv --epochs 25

BIDMC's ECG and PPG channels carry an arbitrary per-record offset (an
artefact of the monitor, not physiology), so each record is first re-aligned
to a fixed R->PPG-peak delay (common.TARGET_DELAY_S). Records whose beat
timing is too inconsistent to align (circular spread > MAX_SPREAD) are
excluded up front.

Evaluation is 5-fold cross-validation by subject (each subject is tested by
a model that never saw them; fixed epoch count, no early stopping on test
data) and compares against a template baseline: a population-average ECG beat placed
at each detected PPG beat. If the network does not beat the template, it has
learned beat timing and nothing else.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy import signal

from common import (FS_IN, FS_OUT, N_IN, N_OUT, align_record, js_bandpass, load_bidmc,
                    model_input, ppg_peaks, r_peaks, windows, zscore)
from model import PPG2ECG

HERE = os.path.dirname(os.path.abspath(__file__))
NOISE_LEVELS = [0.0, 0.5, 1.0]   # band-limited noise std relative to signal std


def bandlimited_noise(n, rng):
    return zscore(js_bandpass(rng.standard_normal(n + 2 * FS_IN * 5))[FS_IN * 5: FS_IN * 5 + n])


# ---- datasets -------------------------------------------------------------
class WindowSet(torch.utils.data.Dataset):
    def __init__(self, recs, hop_s, augment, seed=0):
        self.items = []
        for _, x, y in recs:
            for xi, yi in windows(x, y, hop_s):
                self.items.append((xi.astype(np.float32), zscore(yi).astype(np.float32)))
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.noise = bandlimited_noise(FS_IN * 3600, self.rng) if augment else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y = self.items[i]
        x = zscore(x)
        if self.augment and self.rng.random() < 0.8:
            k = self.rng.uniform(0, 1.0)
            o = self.rng.integers(0, len(self.noise) - N_IN)
            x = zscore(x + k * self.noise[o: o + N_IN])
        return torch.from_numpy(x.astype(np.float32))[None], torch.from_numpy(y)[None]


# ---- full-record inference (same procedure as the browser app) -------------
def stitch_predict(model, x30, hop_s=4):
    """Overlapping windows, triangular cross-fade. Returns ECG at FS_OUT."""
    n_out = len(x30) // FS_IN * FS_OUT
    hop_in, hop_out = hop_s * FS_IN, hop_s * FS_OUT
    starts = list(range(0, max(1, len(x30) - N_IN + 1), hop_in))
    if starts[-1] + N_IN < len(x30):
        starts.append(len(x30) - N_IN)
    w = np.bartlett(N_OUT + 2)[1:-1]
    acc, wsum = np.zeros(n_out), np.zeros(n_out)
    with torch.no_grad():
        batch = np.stack([zscore(x30[s: s + N_IN]) for s in starts]).astype(np.float32)
        ys = model(torch.from_numpy(batch)[:, None]).numpy()[:, 0]
    for s, y in zip(starts, ys):
        o = s // FS_IN * FS_OUT
        m = min(N_OUT, n_out - o)
        ww = w.copy()
        if s == 0:
            ww[: N_OUT // 2] = 1
        if s == starts[-1]:
            ww[N_OUT // 2:] = 1
        acc[o: o + m] += ww[:m] * y[:m]
        wsum[o: o + m] += ww[:m]
    return acc / np.maximum(wsum, 1e-6)


# ---- template baseline -----------------------------------------------------
TPL_PRE, TPL_POST = 0.25, 0.45


def fit_template(recs):
    beats, delays = [], []
    pre, post = int(TPL_PRE * FS_OUT), int(TPL_POST * FS_OUT)
    for _, x, y in recs:
        yz = zscore(y)
        r = r_peaks(y)
        for t in r:
            i = int(round(t * FS_OUT))
            if i - pre >= 0 and i + post <= len(yz):
                beats.append(yz[i - pre: i + post])
        for tp in ppg_peaks(x):
            prior = r[(r < tp) & (r > tp - 0.6)]
            if len(prior):
                delays.append(tp - prior[-1])
    return np.mean(beats, 0), float(np.median(delays))


def template_predict(x30, template, delay):
    n_out = len(x30) // FS_IN * FS_OUT
    rt = ppg_peaks(x30) - delay
    t = np.arange(n_out) / FS_OUT
    out = np.full(n_out, template[0])
    idx = np.searchsorted(rt, t + TPL_PRE) - 1     # latest beat starting before t
    ok = idx >= 0
    off = np.where(ok, t - rt[np.clip(idx, 0, None)] + TPL_PRE, -1)
    valid = ok & (off >= 0) & (off < TPL_PRE + TPL_POST)
    out[valid] = template[np.minimum((off[valid] * FS_OUT).astype(int), len(template) - 1)]
    return out


# ---- metrics ---------------------------------------------------------------
def segment_metrics(pred, true, tol=0.075):
    """Per 8 s segment: Pearson r, R-peak sensitivity/PPV counts, timing error."""
    rs, tp, fn, fp, errs = [], 0, 0, 0, []
    for k in range(len(true) // N_OUT):
        p, y = pred[k * N_OUT:(k + 1) * N_OUT], true[k * N_OUT:(k + 1) * N_OUT]
        if p.std() < 1e-6:
            rs.append(0.0)
        else:
            rs.append(float(np.corrcoef(p, y)[0, 1]))
        rt = r_peaks(y)
        rp = r_peaks(p)
        used = set()
        for t in rt:
            j = np.argmin(np.abs(rp - t)) if len(rp) else None
            if j is not None and abs(rp[j] - t) <= tol and j not in used:
                used.add(j); tp += 1; errs.append(abs(rp[j] - t))
            else:
                fn += 1
        fp += len(rp) - len(used)
    return rs, tp, fn, fp, errs


def evaluate(recs, predictor, seed=123):
    rng = np.random.default_rng(seed)
    res = {}
    for k in NOISE_LEVELS:
        per_subj_r, TP, FN, FP, E = [], 0, 0, 0, []
        for _, x, y in recs:
            xi = x if k == 0 else zscore(x) + k * bandlimited_noise(len(x), rng)
            rs, tp, fn, fp, errs = segment_metrics(predictor(xi), y)
            per_subj_r.append(float(np.median(rs)))
            TP += tp; FN += fn; FP += fp; E += errs
        res[f'noise_{k}'] = {
            'pearson_r_median_over_subjects': round(float(np.median(per_subj_r)), 3),
            'pearson_r_iqr': [round(float(v), 3) for v in np.percentile(per_subj_r, [25, 75])],
            'pearson_r_per_subject': [round(v, 3) for v in per_subj_r],
            'r_peak_sensitivity': round(TP / max(TP + FN, 1), 3),
            'r_peak_ppv': round(TP / max(TP + FP, 1), 3),
            'r_peak_timing_err_ms_median': round(1000 * float(np.median(E)), 1) if E else None,
            '_tp': TP, '_fn': FN, '_fp': FP, '_err': E,
        }
    return res


# ---- main ------------------------------------------------------------------
MAX_SPREAD = 0.3


def train_model(recs, epochs, seed, log=''):
    torch.manual_seed(seed)
    ds = WindowSet(recs, hop_s=1, augment=True, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    model = PPG2ECG()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(dl))
    for ep in range(epochs):
        t0, tl = time.time(), 0
        model.train()
        for xb, yb in dl:
            loss = torch.nn.functional.l1_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(xb)
        print(f'{log} epoch {ep + 1:2d}  train L1 {tl / len(ds):.4f}  {time.time() - t0:.0f}s', flush=True)
    return model.eval()


def merge(folds):
    """Combine per-fold evaluate() outputs into subject-level summaries."""
    out = {}
    for k in folds[0]:
        r = sum((f[k]['pearson_r_per_subject'] for f in folds), [])
        tp = sum(f[k]['_tp'] for f in folds); fn = sum(f[k]['_fn'] for f in folds); fp = sum(f[k]['_fp'] for f in folds)
        e = sum((f[k]['_err'] for f in folds), [])
        out[k] = {
            'pearson_r_median_over_subjects': round(float(np.median(r)), 3),
            'pearson_r_iqr': [round(float(v), 3) for v in np.percentile(r, [25, 75])],
            'pearson_r_per_subject': r,
            'r_peak_sensitivity': round(tp / max(tp + fn, 1), 3),
            'r_peak_ppv': round(tp / max(tp + fp, 1), 3),
            'r_peak_timing_err_ms_median': round(1000 * float(np.median(e)), 1) if e else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(HERE, 'checkpoints'))
    ap.add_argument('--skip-cv', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    recs, excluded = [], []
    for rid, p, e in load_bidmc(args.data):
        a = align_record(model_input(p), e)
        if a is None or a[2] > MAX_SPREAD:
            excluded.append(rid)
        else:
            recs.append((rid, a[0], a[1]))
    print(f'usable records {len(recs)}, excluded {len(excluded)}: {excluded}', flush=True)

    results = {'excluded_records': excluded, 'n_records': len(recs), 'epochs': args.epochs}
    if not args.skip_cv:
        ids = list(np.random.default_rng(args.seed).permutation([r[0] for r in recs]))
        folds_m, folds_t, delays = [], [], []
        for f in range(args.folds):
            te_ids = ids[f::args.folds]
            train = [r for r in recs if r[0] not in te_ids]
            test = [r for r in recs if r[0] in te_ids]
            model = train_model(train, args.epochs, args.seed + f, log=f'fold {f + 1}/{args.folds}')
            template, delay = fit_template(train)
            delays.append(delay)
            folds_m.append(evaluate(test, lambda x: stitch_predict(model, x)))
            folds_t.append(evaluate(test, lambda x: template_predict(x, template, delay)))
        results['cv_model'] = merge(folds_m)
        results['cv_template_baseline'] = merge(folds_t)
        results['template_delay_ms_per_fold'] = [round(d * 1000, 1) for d in delays]
        for name in ('cv_model', 'cv_template_baseline'):
            for k, v in results[name].items():
                print(name, k, {kk: vv for kk, vv in v.items() if kk != 'pearson_r_per_subject'}, flush=True)

    # Final model on all usable records, for deployment.
    model = train_model(recs, args.epochs, args.seed, log='final')
    torch.save(model.state_dict(), os.path.join(args.out, 'final.pt'))
    template, delay = fit_template(recs)
    np.save(os.path.join(args.out, 'template.npy'), template)
    with open(os.path.join(HERE, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=1)


if __name__ == '__main__':
    main()
