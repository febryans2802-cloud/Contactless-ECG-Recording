"""Ablations explaining the CV result (see README):
  webcam      : app-matched input (0.7-4 Hz, 30 Hz), subject-independent split
  wideband    : 0.5-12 Hz input (keeps PPG morphology), subject-independent
  same_subject: app-matched input, train on first 70% of every record, test on last 30%
                (the split many published papers use; inflates results)
Single fold, 12 epochs each, so numbers are comparable to each other, not to the CV table.
"""
import json, os, sys
import numpy as np
from common import align_record, js_bandpass, load_bidmc, FS_IN, FS_OUT
from train import MAX_SPREAD, evaluate, fit_template, stitch_predict, template_predict, train_model

data = sys.argv[1]
raw = load_bidmc(data)
def build(lo, hi):
    out = []
    for rid, p, e in raw:
        a = align_record(js_bandpass(p, FS_IN, lo, hi), e)
        if a is not None and a[2] <= MAX_SPREAD:
            out.append((rid, a[0], a[1]))
    return out

def run(name, train, test):
    m = train_model(train, 12, 0, log=name)
    tpl, d = fit_template(train)
    rm = evaluate(test, lambda x: stitch_predict(m, x))['noise_0.0']
    rt = evaluate(test, lambda x: template_predict(x, tpl, d))['noise_0.0']
    keep = lambda r: {k: r[k] for k in ('pearson_r_median_over_subjects', 'r_peak_sensitivity', 'r_peak_ppv')}
    print(name, 'model', keep(rm), 'template', keep(rt), flush=True)
    return {'model': keep(rm), 'template': keep(rt)}

res = {}
for name, (lo, hi) in {'webcam': (0.7, 4.0), 'wideband': (0.5, 12.0)}.items():
    recs = build(lo, hi)
    ids = list(np.random.default_rng(0).permutation([r[0] for r in recs]))
    te = ids[0::5]
    res[name] = run(name, [r for r in recs if r[0] not in te], [r for r in recs if r[0] in te])
recs = build(0.7, 4.0)
cut = lambda r, a, b: (r[0], r[1][int(a * len(r[1]) // FS_IN) * FS_IN: int(b * len(r[1]) // FS_IN) * FS_IN],
                       r[2][int(a * len(r[1]) // FS_IN) * FS_OUT: int(b * len(r[1]) // FS_IN) * FS_OUT])
res['same_subject'] = run('same_subject', [cut(r, 0, 0.7) for r in recs], [cut(r, 0.7, 1.0) for r in recs])
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ablation_results.json'), 'w'), indent=1)
