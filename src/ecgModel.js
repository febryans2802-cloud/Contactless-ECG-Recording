// Plain-JavaScript inference for the PPG->ECG U-Net trained in ml/.
// Mirrors ml/model.py exactly: Conv1d('same', k=7) + LeakyReLU(0.1),
// MaxPool(2), nearest Upsample(2), channel concat. No dependencies.
//
// The output is an ESTIMATE of what a lead-II ECG might look like given the
// pulse timing. It is generated, not measured. See README.

const SLOPE = 0.1;

/** Parse a manifest + raw float32 weights into a model object. */
export function loadModelFromBuffers(manifest, buffer) {
  const all = new Float32Array(buffer);
  const p = {};
  for (const t of manifest.tensors) p[t.name] = { shape: t.shape, data: all.subarray(t.offset, t.offset + t.length) };
  return { manifest, p };
}

/** Browser loader: fetches model/ppg2ecg.json and .bin relative to baseUrl. */
export async function loadModel(baseUrl = 'model/') {
  const manifest = await (await fetch(baseUrl + 'ppg2ecg.json')).json();
  const buf = await (await fetch(baseUrl + 'ppg2ecg.bin')).arrayBuffer();
  return loadModelFromBuffers(manifest, buf);
}

// Tensors are { c, l, d: Float32Array(c*l) } in channel-major order.
function conv(x, w, b, act) {
  const [cout, cin, k] = w.shape, L = x.l, pad = k >> 1;
  const W = w.data, X = x.d, y = new Float32Array(cout * L);
  for (let o = 0; o < cout; o++) {
    const yo = o * L;
    y.fill(b.data[o], yo, yo + L);
    for (let i = 0; i < cin; i++) {
      const xi = i * L, wb = (o * cin + i) * k;
      for (let j = 0; j < k; j++) {
        const wv = W[wb + j], s = j - pad;
        const t0 = Math.max(0, -s), t1 = Math.min(L, L - s);
        for (let t = t0; t < t1; t++) y[yo + t] += wv * X[xi + t + s];
      }
    }
  }
  if (act) for (let n = 0; n < y.length; n++) if (y[n] < 0) y[n] *= SLOPE;
  return { c: cout, l: L, d: y };
}

function pool(x) {
  const l = x.l >> 1, d = new Float32Array(x.c * l);
  for (let c = 0; c < x.c; c++) for (let t = 0; t < l; t++) d[c * l + t] = Math.max(x.d[c * x.l + 2 * t], x.d[c * x.l + 2 * t + 1]);
  return { c: x.c, l, d };
}

function up(x) {
  const l = x.l * 2, d = new Float32Array(x.c * l);
  for (let c = 0; c < x.c; c++) for (let t = 0; t < l; t++) d[c * l + t] = x.d[c * x.l + (t >> 1)];
  return { c: x.c, l, d };
}

function cat(a, b) {
  const d = new Float32Array(a.d.length + b.d.length);
  d.set(a.d); d.set(b.d, a.d.length);
  return { c: a.c + b.c, l: a.l, d };
}

function block(m, name, x) {
  const P = m.p;
  x = conv(x, P[`${name}.0.weight`], P[`${name}.0.bias`], true);
  return conv(x, P[`${name}.2.weight`], P[`${name}.2.bias`], true);
}

/** One window: Float32Array(n_in) z-scored PPG -> Float32Array(n_out) ECG. */
export function forward(m, input) {
  const x = { c: 1, l: input.length, d: Float32Array.from(input) };
  const a = block(m, 'e1', x);
  const b = block(m, 'e2', pool(a));
  const c = block(m, 'e3', pool(b));
  const mid = block(m, 'mid', pool(c));
  let y = block(m, 'd3', cat(up(mid), c));
  y = block(m, 'd2', cat(up(y), b));
  y = block(m, 'd1', cat(up(y), a));
  y = block(m, 'u1', up(y));
  y = block(m, 'u2', up(y));
  return conv(y, m.p['head.weight'], m.p['head.bias'], false).d;
}

function zscore(a) {
  let s = 0; for (const v of a) s += v;
  const mu = s / a.length;
  let q = 0; for (const v of a) q += (v - mu) ** 2;
  const sd = Math.sqrt(q / a.length) || 1;
  return Float32Array.from(a, (v) => (v - mu) / sd);
}

/**
 * Whole recording: band-passed pulse at fs_in -> estimated ECG at fs_out.
 * Overlapping windows with a triangular cross-fade; same as
 * ml/train.py:stitch_predict, which produced the reported metrics.
 */
export function reconstructECG(m, pulse, hopS = 4) {
  const { fs_in: fi, fs_out: fo, n_in: nIn, n_out: nOut } = m.manifest;
  if (pulse.length < nIn) return null;
  const nOutTot = Math.floor(pulse.length / fi) * fo;
  const hopIn = hopS * fi;
  const starts = [];
  for (let s = 0; s + nIn <= pulse.length; s += hopIn) starts.push(s);
  if (starts[starts.length - 1] + nIn < pulse.length) starts.push(pulse.length - nIn);
  const acc = new Float64Array(nOutTot), wsum = new Float64Array(nOutTot);
  starts.forEach((s, k) => {
    const y = forward(m, zscore(pulse.slice(s, s + nIn)));
    const o = Math.round((s / fi) * fo);
    for (let t = 0; t < nOut && o + t < nOutTot; t++) {
      // numpy.bartlett(nOut + 2)[1:-1]
      let w = 1 - Math.abs((t + 1) - (nOut + 1) / 2) / ((nOut + 1) / 2);
      if (k === 0 && t < nOut / 2) w = 1;
      if (k === starts.length - 1 && t >= nOut / 2) w = 1;
      acc[o + t] += w * y[t];
      wsum[o + t] += w;
    }
  });
  return Float32Array.from(acc, (v, i) => v / Math.max(wsum[i], 1e-6));
}
