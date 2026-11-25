import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { synthesizeECG } from '../src/ecgTemplate.js';
import { bandpass } from '../src/dsp.js';

const root = new URL('../', import.meta.url);
const raw = JSON.parse(readFileSync(new URL('model/ecg_template.json', root)));
const tpl = { ...raw, beat: Float32Array.from(raw.beat) };
const vec = JSON.parse(readFileSync(new URL('test/template_vector.json', root)));

test('synthesizeECG matches Python template_predict', () => {
  const y = synthesizeECG(tpl, vec.peaks_s, vec.n_in / 30);
  assert.equal(y.length, vec.expected.length);
  let maxErr = 0;
  for (let i = 0; i < y.length; i++) maxErr = Math.max(maxErr, Math.abs(y[i] - vec.expected[i]));
  assert.ok(maxErr < 1e-4, `max abs error ${maxErr}`);
});

test('one R-peak per pulse, placed delay_s before it', () => {
  const peaks = [1, 2, 3, 4];
  const y = synthesizeECG(tpl, peaks, 5);
  const iR = Math.round(tpl.pre_s * tpl.fs); // R-peak index inside the beat
  const sign = Math.sign(tpl.beat[iR]);
  for (const p of peaks) {
    const i = Math.round((p - tpl.delay_s) * tpl.fs);
    let best = i;
    for (let j = i - 6; j <= i + 6; j++) if (sign * y[j] > sign * y[best]) best = j;
    assert.ok(Math.abs(best - i) <= 2, `R-peak at ${best}, expected ~${i}`);
  }
});

test('Python training-time band-pass (ml/common.py) matches dsp.js', () => {
  const y = bandpass(vec.bandpass_in, 30);
  let maxErr = 0;
  for (let i = 0; i < y.length; i++) maxErr = Math.max(maxErr, Math.abs(y[i] - vec.bandpass_out[i]));
  assert.ok(maxErr < 1e-6, `max abs error ${maxErr}`);
});
