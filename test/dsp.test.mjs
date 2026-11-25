import { test } from 'node:test';
import assert from 'node:assert/strict';
import { analyze, bandpass, findPeaks, hrv, resample, spectralHR } from '../src/dsp.js';

// Deterministic PRNG so tests are reproducible.
function rng(seed = 1) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) - 0.5;
}

/**
 * Simulated skin RGB trace: blood volume darkens the skin, mostly in green
 * (relative pulsatile amplitudes ≈ 0.33 : 0.77 : 0.53 for R : G : B), plus a shared intensity drift that POS
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
    r.push(illum * 150 * (1 - 0.0010 * pulse) + noise * rand());
    g.push(illum * 100 * (1 - 0.0035 * pulse) + noise * rand());
    b.push(illum * 80 * (1 - 0.0024 * pulse) + noise * rand());
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

test('output polarity follows blood volume (up = systole)', () => {
  // Pulse is a narrow systolic bump, so the conventional-polarity waveform
  // must be skewed upward (sharp peaks, broad troughs).
  const fs = 30, t = [], r = [], g = [], b = [];
  for (let i = 0; i < 20 * fs; i++) {
    const ph = ((i / fs) * 1.2) % 1;
    const v = Math.exp(-((ph - 0.2) ** 2) / (2 * 0.06 ** 2)); // blood volume
    t.push(i / fs); r.push(150 * (1 - 0.001 * v)); g.push(100 * (1 - 0.0035 * v)); b.push(80 * (1 - 0.0024 * v));
  }
  const y = analyze({ t, r, g, b }).bvp.slice(60, -60);
  const m = y.reduce((s, v) => s + v, 0) / y.length;
  const skew = y.reduce((s, v) => s + (v - m) ** 3, 0) / y.length;
  assert.ok(skew > 0, `skew ${skew}`);
});

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

// Realistic pulse morphology: sharp systolic peak, dicrotic wave, slow decay.
function realistic({ bpm, dicrotic = 0.38, noise = 0.15, seconds = 60, seed = 3 }) {
  const rand = rng(seed);
  const gs = () => { let s = 0; for (let k = 0; k < 6; k++) s += rand(); return s * Math.SQRT2; }; // ~N(0,1)
  const g = (x, m, sd) => Math.exp(-((x - m) ** 2) / (2 * sd * sd));
  const beats = [];
  for (let x = 0.3; x < seconds + 2; x += (60 / bpm) * (1 + 0.06 * Math.sin(2 * Math.PI * 0.25 * x))) beats.push(x);
  const t = [], r = [], gg = [], b = [];
  for (let i = 0; i < seconds * 30; i++) {
    const tt = i / 30;
    let v = 0;
    for (const bt of beats) { const u = tt - bt; if (u >= 0 && u < 1.1) v += g(u, 0.13, 0.055) + dicrotic * g(u, 0.36, 0.07) + 0.12 * g(u, 0.55, 0.12); }
    t.push(tt); r.push(182 * (1 - 0.001 * v) + noise * gs()); gg.push(132 * (1 - 0.0035 * v) + noise * gs()); b.push(112 * (1 - 0.0024 * v) + noise * gs());
  }
  return { s: { t, r, g: gg, b }, nBeats: beats.filter((x) => x < seconds).length };
}

test('dicrotic wave is not counted as an extra beat', () => {
  for (const bpm of [50, 68, 90]) {
    const { s, nBeats } = realistic({ bpm });
    const a = analyze(s);
    // No extra beats from the dicrotic wave; some misses are expected at this SNR (~0 dB).
    assert.ok(a.peaks.length <= 1.05 * nBeats, `${bpm} bpm: ${a.peaks.length} peaks vs ${nBeats} beats (extra beats)`);
    assert.ok(a.peaks.length >= 0.85 * nBeats, `${bpm} bpm: ${a.peaks.length} peaks vs ${nBeats} beats (missed beats)`);
    assert.ok(Math.abs(a.hrv.bpm - bpm) < 5, `${bpm} bpm: peak-based ${a.hrv.bpm}`);
  }
});

test('sharp pulse without dicrotic wave: spectral HR picks the fundamental, not the 2nd harmonic', () => {
  const { s } = realistic({ bpm: 68, dicrotic: 0, noise: 0 });
  assert.ok(Math.abs(analyze(s).spectral.bpm - 68) < 2);
});

test('harmonic check does not halve genuine heart rates', () => {
  for (const bpm of [90, 110, 140]) for (let seed = 1; seed <= 8; seed++) {
    const r = analyze(synth({ bpm, noise: 0.5, seed, seconds: 15 }));
    assert.ok(Math.abs(r.spectral.bpm - bpm) < 3, `${bpm} bpm seed ${seed}: ${r.spectral.bpm}`);
    const { s } = realistic({ bpm, seed, seconds: 15 });
    const r2 = analyze(s);
    assert.ok(Math.abs(r2.spectral.bpm - bpm) < 5, `realistic ${bpm} bpm seed ${seed}: ${r2.spectral.bpm}`);
  }
});
