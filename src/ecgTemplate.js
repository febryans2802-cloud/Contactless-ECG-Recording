// Rhythm view: places a population-average ECG beat at each detected pulse.
//
// This shows WHEN beats happen, drawn as a generic ECG shape. The shape is
// an average over other people's lead-II ECGs, not the user's. The R-peak is
// placed a fixed delay before each pulse peak by convention, not measured.
// Port of ml/train.py:template_predict (parity-tested).

export async function loadTemplate(url = 'model/ecg_template.json') {
  const t = await (await fetch(url)).json();
  return { ...t, beat: Float32Array.from(t.beat) };
}

/**
 * @param tpl       template from loadTemplate()
 * @param peaksS    pulse-peak times in seconds (ascending)
 * @param durationS length of the output in seconds
 * @returns Float32Array at tpl.fs
 */
export function synthesizeECG(tpl, peaksS, durationS) {
  const { fs, pre_s: pre, post_s: post, delay_s: delay, beat } = tpl;
  const n = Math.floor(durationS) * fs;
  const out = new Float32Array(n).fill(beat[0]);
  const r = peaksS.map((p) => p - delay);
  let k = -1; // index of the latest beat whose window has started
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    while (k + 1 < r.length && r[k + 1] <= t + pre) k++;
    if (k < 0) continue;
    const off = t - r[k] + pre;
    if (off >= 0 && off < pre + post) out[i] = beat[Math.min(Math.floor(off * fs), beat.length - 1)];
  }
  return out;
}
