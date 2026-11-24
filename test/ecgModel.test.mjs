import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { forward, loadModelFromBuffers, reconstructECG } from '../src/ecgModel.js';
import { bandpass } from '../src/dsp.js';

const root = new URL('../', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('model/ppg2ecg.json', root)));
const bin = readFileSync(new URL('model/ppg2ecg.bin', root));
const model = loadModelFromBuffers(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const vec = JSON.parse(readFileSync(new URL('test/ppg2ecg_vector.json', root)));

test('JS forward pass matches PyTorch reference output', () => {
  const y = forward(model, Float32Array.from(vec.input));
  assert.equal(y.length, vec.output.length);
  let maxErr = 0;
  for (let i = 0; i < y.length; i++) maxErr = Math.max(maxErr, Math.abs(y[i] - vec.output[i]));
  assert.ok(maxErr < 1e-3, `max abs error ${maxErr}`);
});

test('reconstructECG covers a full recording at the output rate', () => {
  const fs = manifest.fs_in, secs = 23;
  const pulse = Array.from({ length: fs * secs }, (_, i) => Math.sin((2 * Math.PI * 1.2 * i) / fs));
  const ecg = reconstructECG(model, pulse);
  assert.equal(ecg.length, secs * manifest.fs_out);
  assert.ok(ecg.every(Number.isFinite));
  assert.equal(reconstructECG(model, pulse.slice(0, fs * 5)), null, 'too short -> null');
});

test('Python training-time band-pass (ml/common.py) matches dsp.js', () => {
  const y = bandpass(vec.bandpass_in, 30);
  let maxErr = 0;
  for (let i = 0; i < y.length; i++) maxErr = Math.max(maxErr, Math.abs(y[i] - vec.bandpass_out[i]));
  assert.ok(maxErr < 1e-6, `max abs error ${maxErr}`);
});
