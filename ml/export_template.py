"""Export the population-average beat for the app's rhythm view.

    python3 export_template.py --data data

Writes model/ecg_template.json and test/template_vector.json (a Python
reference output, so the JS port can be checked against train.template_predict).
"""
import argparse
import json
import os

import numpy as np

from common import FS_IN, FS_OUT, align_record, ecg_polarity, js_bandpass, load_bidmc, model_input
from train import MAX_SPREAD, TPL_POST, TPL_PRE, fit_template, template_predict

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
args = ap.parse_args()

recs = []
for rid, p, e in load_bidmc(args.data):
    a = align_record(model_input(p), e)
    if a is not None and a[2] <= MAX_SPREAD:
        recs.append((rid, a[0], a[1]))
# Flip records with inverted lead II (common in BIDMC) so the average has the
# conventional upright R wave; then set the isoelectric baseline to zero.
tpl, delay = fit_template([(rid, x, ecg_polarity(y) * y) for rid, x, y in recs])
tpl = tpl - np.median(np.r_[tpl[:8], tpl[-8:]])
out = {
    'description': 'Population-average ECG lead II beat, polarity-normalised (R up), baseline 0, z-scored units. NOT the user\'s ECG.',
    'source': f'BIDMC PPG and Respiration Dataset (PhysioNet, ODC-By 1.0), {len(recs)} records',
    'fs': FS_OUT, 'pre_s': TPL_PRE, 'post_s': TPL_POST,
    'delay_s': round(delay, 4),
    'beat': [round(float(v), 5) for v in tpl],
}
with open(os.path.join(HERE, '..', 'model', 'ecg_template.json'), 'w') as f:
    json.dump(out, f)

# Reference vectors for JS parity tests.
rng = np.random.default_rng(0)
t = np.arange(20 * FS_IN) / FS_IN
x = np.sin(2 * np.pi * 1.25 * t) + 0.1 * rng.standard_normal(len(t))
from train import ppg_peaks
peaks = ppg_peaks(x)
bp_in = rng.standard_normal(600) + 5 * np.sin(np.arange(600) / 40)
with open(os.path.join(HERE, '..', 'test', 'template_vector.json'), 'w') as f:
    json.dump({'peaks_s': [float(v) for v in peaks], 'n_in': len(x),
               'expected': [round(float(v), 6) for v in template_predict(x, np.array(out['beat']), out['delay_s'])],
               'bandpass_in': [float(v) for v in bp_in], 'bandpass_out': [float(v) for v in js_bandpass(bp_in)]}, f)
print(f'template: {len(tpl)} samples, delay {delay * 1000:.0f} ms, {len(recs)} records')
