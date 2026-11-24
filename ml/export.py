"""Export trained weights for the browser: model/ppg2ecg.{json,bin} + a test vector.

    python3 export.py --ckpt checkpoints/final.pt
"""
import argparse
import json
import os

import numpy as np
import torch

from common import FS_IN, FS_OUT, N_IN, N_OUT, TARGET_DELAY_S, js_bandpass
from model import PPG2ECG

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'model')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    model = PPG2ECG()
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()

    tensors, blobs, offset = [], [], 0
    for name, p in model.state_dict().items():
        a = p.detach().numpy().astype('<f4').ravel()
        tensors.append({'name': name, 'shape': list(p.shape), 'offset': offset, 'length': a.size})
        blobs.append(a)
        offset += a.size
    np.concatenate(blobs).tofile(os.path.join(OUT, 'ppg2ecg.bin'))

    results = {}
    rp = os.path.join(HERE, 'results.json')
    if os.path.exists(rp):
        with open(rp) as f:
            r = json.load(f)
        results = {k: r[k] for k in ('n_records', 'excluded_records') if k in r}
    manifest = {
        'format': 'ppg2ecg-unet1d-v1',
        'fs_in': FS_IN, 'fs_out': FS_OUT, 'n_in': N_IN, 'n_out': N_OUT,
        'target_delay_s': TARGET_DELAY_S,
        'trained_on': 'BIDMC PPG and Respiration Dataset (PhysioNet), finger PPG -> ECG lead II',
        'training': results,
        'tensors': tensors,
    }
    with open(os.path.join(OUT, 'ppg2ecg.json'), 'w') as f:
        json.dump(manifest, f, indent=1)

    # Reference input/output so the JS port can be checked against PyTorch.
    rng = np.random.default_rng(0)
    t = np.arange(N_IN) / FS_IN
    x = np.sin(2 * np.pi * 1.3 * t) + 0.3 * np.sin(2 * np.pi * 2.6 * t + 1) + 0.1 * rng.standard_normal(N_IN)
    x = ((x - x.mean()) / x.std()).astype(np.float32)
    with torch.no_grad():
        y = model(torch.from_numpy(x)[None, None]).numpy().ravel()
    with open(os.path.join(HERE, '..', 'test', 'ppg2ecg_vector.json'), 'w') as f:
        bp_in = rng.standard_normal(600) + 5 * np.sin(np.arange(600) / 40)
        json.dump({'input': [round(float(v), 6) for v in x], 'output': [round(float(v), 6) for v in y],
                   'bandpass_in': [float(v) for v in bp_in], 'bandpass_out': [float(v) for v in js_bandpass(bp_in)]}, f)
    print(f'exported {offset} params ({offset * 4 / 1e6:.2f} MB) to {os.path.abspath(OUT)}')


if __name__ == '__main__':
    main()
