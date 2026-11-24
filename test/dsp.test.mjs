import { test } from 'node:test';
import assert from 'node:assert/strict';
import { analyze, bandpass, findPeaks, hrv, resample, spectralHR } from '../src/dsp.js';

// Deterministic PRNG so tests are reproducible.
function rng(seed = 1) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) - 0.5;
}

/**
 * Simulated skin RGB trace: a pulse that modulates mostly green (as real
 * haemoglobin absorption does), plus a shared intensity drift that POS
 * should cancel, plus sensor noise and jittery frame timing.
 */
function synth({ bpm = 72, seconds = 20, fps = 30, noise = 0.3, jitter = 0.004, motion = 4, seed = 7 } = {}) {
  const rand = rng(seed);
  const t = [], r = [], g = [], b = [];
  let ti = 0;
  const f = bpm / 60;
  while (ti < seconds) {
    const pulse = Math.sin(2 * Math.PI * f * ti) + 0.3 * Math.sin(4 * Math.PI * f * ti - 0.8);
    const illum = 1 + 0.02 * Math.sin(2 * Math.PI * 0.2 * ti) + 0.01 * motion * Math.sin(2 * Math.PI * 0.07 * ti);
    t.push(ti);
    r.push(illum * (150 + 0.15 * pulse) + noise * rand());
    g.push(illum * (100 + 0.35 * pulse) + noise * rand());
    b.push(illum * (80 + 0.10 * pulse) + noise * rand());
    ti += 1 / fps + jitter * rand();
  }
  return { t, r, g, b };
}

test('resample produces uniform grid and interpolates linearly', () => {
  const { t, channels: [y] } = resample([0, 1, 3], [[0, 10, 30]], 2);
  assert.deepEqual(t, [0, 0.5, 1, 1.5, 2, 2.5, 3]);
  assert.deepEqual(y, [0, 5, 10, 15, 20, 25, 30]);
});

test('bandpass removes DC and slow drift, keeps cardiac band', () => {
  const fs = 30, n = 600;
  const x = Array.from({ length: n }, (_, i) => 50 + 5 * Math.sin(2 * Math.PI * 0.05 * i / fs) + Math.sin(2 * Math.PI * 1.2 * i / fs));
  const y = bandpass(x, fs);
  const mid = y.slice(100, 500);
  const amp = Math.max(...mid) - Math.min(...mid);
  assert.ok(amp > 1.5 && amp < 2.3, `cardiac amplitude preserved, got ${amp}`);
  assert.ok(Math.abs(mid.reduce((s, v) => s + v, 0) / mid.length) < 0.1, 'DC removed');
});

for (const bpm of [50, 72, 95, 130]) {
  test(`analyze recovers ${bpm} bpm from synthetic RGB`, () => {
    const res = analyze(synth({ bpm }));
    assert.ok(Math.abs(res.spectral.bpm - bpm) < 2, `spectral ${res.spectral.bpm}`);
    assert.ok(Math.abs(res.hrv.bpm - bpm) < 2, `peak-based ${res.hrv.bpm}`);
    assert.ok(res.spectral.snrDb > 3, `snr ${res.spectral.snrDb}`);
  });
}

test('pure noise yields low SNR', () => {
  const rand = rng(3);
  const fs = 30, n = 600;
  const x = Array.from({ length: n }, () => rand());
  const { snrDb } = spectralHR(bandpass(x, fs), fs);
  assert.ok(snrDb < 0, `snr ${snrDb}`);
});

test('findPeaks + hrv on known intervals', () => {
  const fs = 100;
  // Beats at known times with alternating 0.8/0.9 s intervals.
  const beats = [];
  for (let tt = 0.5, k = 0; tt < 20; tt += k++ % 2 ? 0.9 : 0.8) beats.push(tt);
  const n = 20 * fs;
  const x = new Array(n).fill(0).map((_, i) => {
    const ti = i / fs;
    return beats.reduce((s, bt) => s + Math.exp(-((ti - bt) ** 2) / (2 * 0.05 ** 2)), 0);
  });
  const y = bandpass(x, fs);
  const p = findPeaks(y, fs);
  const h = hrv(p, fs);
  assert.ok(Math.abs(p.length - beats.length) <= 1, `${p.length} vs ${beats.length}`);
  assert.ok(Math.abs(h.bpm - 60 / 0.85) < 1.5, `bpm ${h.bpm}`);
  assert.ok(Math.abs(h.rmssdMs - 100) < 15, `rmssd ${h.rmssdMs}`);
});

test('hrv rejects outlier intervals', () => {
  const fs = 10;
  const peaks = [0, 10, 20, 25, 40, 50, 60]; // one spurious beat at 25
  const h = hrv(peaks, fs);
  assert.ok(Math.abs(h.bpm - 60) < 1e-9);
  assert.ok(h.acceptedFraction < 1);
});
