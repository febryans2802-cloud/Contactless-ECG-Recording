import { analyze, qualityLabel } from './dsp.js';
import { loadFaceDetector, meanSkinRGB, RoiTracker } from './roi.js';

const FS = 30; // analysis sample rate (Hz)
const LIVE_WINDOW_S = 10;
const MIN_SKIN_FRACTION = 0.15;

const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const chart = $('chart');
const ui = {
  status: $('status'), hr: $('hr'), quality: $('quality'), snr: $('snr'),
  rmssd: $('rmssd'), sdnn: $('sdnn'), recTime: $('recTime'),
  btnCamera: $('btnCamera'), btnRecord: $('btnRecord'), btnCsv: $('btnCsv'), btnJson: $('btnJson'),
};

const sampler = document.createElement('canvas');
const sctx = sampler.getContext('2d', { willReadFrequently: true });
const octx = overlay.getContext('2d');

let stream = null;
let tracker = null;
let running = false;
let live = { t: [], r: [], g: [], b: [] };
let rec = null; // { startMs, frames: [{t,r,g,b,skin,face}] }
let lastResult = null; // analysis of the finished recording
let hrHistory = [];
let lastAnalysis = 0;

ui.btnCamera.addEventListener('click', () => (running ? stopCamera() : startCamera()));
ui.btnRecord.addEventListener('click', () => (rec ? stopRecording() : startRecording()));
ui.btnCsv.addEventListener('click', downloadCsv);
ui.btnJson.addEventListener('click', downloadJson);

async function startCamera() {
  ui.btnCamera.disabled = true;
  setStatus('Requesting camera…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false,
    });
  } catch (err) {
    setStatus(`Camera error: ${err.name}`);
    ui.btnCamera.disabled = false;
    return;
  }
  lockCameraSettings(stream.getVideoTracks()[0]);
  video.srcObject = stream;
  await video.play();
  sampler.width = overlay.width = video.videoWidth;
  sampler.height = overlay.height = video.videoHeight;

  setStatus('Loading face detector…');
  tracker = new RoiTracker(await loadFaceDetector());
  running = true;
  live = { t: [], r: [], g: [], b: [] };
  hrHistory = [];
  ui.btnCamera.textContent = 'Stop camera';
  ui.btnCamera.disabled = false;
  ui.btnRecord.disabled = false;
  scheduleFrame();
}

function stopCamera() {
  if (rec) stopRecording();
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  ui.btnCamera.textContent = 'Start camera';
  ui.btnRecord.disabled = true;
  setStatus('Camera off');
}

/**
 * Auto-exposure / white-balance changes inject large colour steps that
 * swamp the ~0.1% pulse signal. Lock them where the browser allows it.
 */
async function lockCameraSettings(track) {
  const caps = track.getCapabilities?.() ?? {};
  const adv = {};
  if (caps.exposureMode?.includes('manual')) adv.exposureMode = 'manual';
  if (caps.whiteBalanceMode?.includes('manual')) adv.whiteBalanceMode = 'manual';
  if (caps.focusMode?.includes('manual')) adv.focusMode = 'manual';
  if (!Object.keys(adv).length) return;
  // Give auto-exposure a moment to settle before freezing it.
  setTimeout(() => track.applyConstraints({ advanced: [adv] }).catch(() => {}), 1500);
}

function scheduleFrame() {
  if (!running) return;
  if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame((now) => onFrame(now, null));
}

function onFrame(now, meta) {
  if (!running) return;
  // Prefer the capture timestamp: frame delivery jitter otherwise shows up as HRV.
  const tMs = meta?.captureTime ?? meta?.expectedDisplayTime ?? now;
  sctx.drawImage(video, 0, 0, sampler.width, sampler.height);
  const { faceFound, regions, face } = tracker.update(video, now);
  const px = meanSkinRGB(sctx, regions);
  const ok = px.skinFraction >= MIN_SKIN_FRACTION && Number.isFinite(px.g);

  drawOverlay(regions, face, ok);

  if (ok) {
    const t = tMs / 1000;
    if (!live.t.length || t > live.t[live.t.length - 1]) {
      live.t.push(t); live.r.push(px.r); live.g.push(px.g); live.b.push(px.b);
      trimLive(t);
      if (rec) rec.frames.push({ t: t - rec.startS, r: px.r, g: px.g, b: px.b, skin: px.skinFraction, face: faceFound });
    }
  }

  if (now - lastAnalysis > 500) {
    lastAnalysis = now;
    updateLive(faceFound, ok);
  }
  if (rec) ui.recTime.textContent = `● ${fmtDuration(tMs / 1000 - rec.startS)}`;
  scheduleFrame();
}

function trimLive(tNow) {
  const cutoff = tNow - LIVE_WINDOW_S;
  let k = 0;
  while (k < live.t.length && live.t[k] < cutoff) k++;
  if (k) for (const key of ['t', 'r', 'g', 'b']) live[key].splice(0, k);
}

function updateLive(faceFound, skinOk) {
  const span = live.t.length ? live.t[live.t.length - 1] - live.t[0] : 0;
  if (!skinOk) setStatus(faceFound ? 'Face found, but too little skin visible' : 'No skin in box. Centre your forehead.');
  else if (span < 5) setStatus(`Collecting… ${Math.max(0, 5 - span).toFixed(0)} s`);
  else setStatus(faceFound ? 'Tracking face' : 'Fixed box (no face detector)');

  const res = span >= 5 ? analyze(live, FS) : null;
  if (!res) { drawChart(null); return; }

  const q = qualityLabel(res.spectral.snrDb);
  ui.quality.textContent = q;
  ui.quality.className = `value small q-${q}`;
  ui.snr.textContent = `SNR ${res.spectral.snrDb.toFixed(1)} dB`;

  if (q !== 'poor') hrHistory.push(res.spectral.bpm);
  hrHistory = hrHistory.slice(-6);
  // Median of recent estimates suppresses single-window glitches.
  const sorted = [...hrHistory].sort((a, b) => a - b);
  ui.hr.textContent = sorted.length ? Math.round(sorted[sorted.length >> 1]) : '--';

  // HRV from a 10 s window is too short to mean much. Show it only after a recording.
  if (!lastResult) { ui.rmssd.textContent = '--'; ui.sdnn.textContent = '--'; }
  drawChart(res);
}

function startRecording() {
  const t0 = live.t.length ? live.t[live.t.length - 1] : performance.now() / 1000;
  rec = { startS: t0, startedAt: new Date().toISOString(), frames: [] };
  lastResult = null;
  ui.btnRecord.textContent = 'Stop recording';
  ui.btnRecord.classList.add('recording');
  ui.btnCsv.disabled = ui.btnJson.disabled = true;
}

function stopRecording() {
  const r = rec;
  rec = null;
  ui.btnRecord.textContent = 'Start recording';
  ui.btnRecord.classList.remove('recording');
  ui.recTime.textContent = '';
  const f = r.frames;
  const res = analyze({ t: f.map((x) => x.t), r: f.map((x) => x.r), g: f.map((x) => x.g), b: f.map((x) => x.b) }, FS);
  if (!res) { setStatus('Recording too short (need ≥ 4 s of usable frames)'); return; }
  lastResult = { ...res, meta: { startedAt: r.startedAt, frames: f } };
  // At 30 fps, sensor noise alone adds ~20–40 ms of apparent RMSSD (see README),
  // so HRV is only shown when the recording is long and clean enough.
  if (hrvReliable(res)) {
    ui.rmssd.textContent = fmt(res.hrv.rmssdMs);
    ui.sdnn.textContent = fmt(res.hrv.sdnnMs);
  } else {
    ui.rmssd.textContent = ui.sdnn.textContent = 'n/a';
    setStatus('HRV withheld: needs ≥ 60 s, SNR ≥ 3 dB, ≥ 80% clean beats');
  }
  ui.btnCsv.disabled = ui.btnJson.disabled = false;
}

function hrvReliable(res) {
  return res.t.length / res.fs >= 60 && res.spectral.snrDb >= 3 && res.hrv.acceptedFraction >= 0.8;
}

function summary(res) {
  const f = res.meta.frames;
  const dur = res.t.length / res.fs;
  return {
    type: 'rPPG pulse recording (NOT an electrocardiogram)',
    method: 'POS (Wang et al. 2017), 0.7–4 Hz zero-phase band-pass, resampled to ' + res.fs + ' Hz',
    startedAt: res.meta.startedAt,
    durationS: +dur.toFixed(2),
    framesCaptured: f.length,
    effectiveFps: +(f.length / Math.max(dur, 1e-9)).toFixed(2),
    faceTrackedFraction: +(f.filter((x) => x.face).length / (f.length || 1)).toFixed(3),
    heartRate: {
      spectralBpm: +res.spectral.bpm.toFixed(1),
      peakBasedBpm: +res.hrv.bpm.toFixed(1),
      snrDb: +res.spectral.snrDb.toFixed(2),
      quality: qualityLabel(res.spectral.snrDb),
    },
    hrv: {
      sdnnMs: round1(res.hrv.sdnnMs),
      rmssdMs: round1(res.hrv.rmssdMs),
      beatsDetected: res.peaks.length,
      acceptedIntervalFraction: +res.hrv.acceptedFraction.toFixed(3),
      meetsReliabilityCriteria: hrvReliable(res),
      caveat: 'rPPG HRV is less accurate than ECG-derived HRV, especially RMSSD. Treat as indicative only.',
    },
    interBeatIntervalsS: res.hrv.ibi.map((v, i) => ({ ibi: +v.toFixed(4), accepted: res.hrv.valid[i] })),
  };
}

function downloadCsv() {
  const res = lastResult;
  const peakSet = new Set(res.peaks.map((p) => Math.round(p)));
  const rows = ['time_s,pulse_au,is_beat,r_mean,g_mean,b_mean'];
  for (let i = 0; i < res.t.length; i++) {
    rows.push([
      (i / res.fs).toFixed(4), res.bvp[i].toFixed(5), peakSet.has(i) ? 1 : 0,
      res.rgb.r[i].toFixed(4), res.rgb.g[i].toFixed(4), res.rgb.b[i].toFixed(4),
    ].join(','));
  }
  save(`rppg_${stamp(res)}.csv`, rows.join('\n'), 'text/csv');
}

function downloadJson() {
  save(`rppg_${stamp(lastResult)}_summary.json`, JSON.stringify(summary(lastResult), null, 2), 'application/json');
}

// ---------- drawing ----------

function drawOverlay(regions, face, ok) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  octx.lineWidth = 2;
  if (face) {
    octx.strokeStyle = 'rgba(255,255,255,0.5)';
    octx.strokeRect(face.x, face.y, face.w, face.h);
  }
  octx.strokeStyle = ok ? '#3ddc84' : '#ff5c5c';
  for (const r of regions) octx.strokeRect(r.x, r.y, r.w, r.h);
}

function drawChart(res) {
  const dpr = window.devicePixelRatio || 1;
  const W = chart.clientWidth, H = chart.clientHeight;
  if (chart.width !== W * dpr) { chart.width = W * dpr; chart.height = H * dpr; }
  const c = chart.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  c.strokeStyle = css.getPropertyValue('--grid');
  c.lineWidth = 1;
  for (let s = 0; s <= LIVE_WINDOW_S; s++) {
    const x = (s / LIVE_WINDOW_S) * W;
    c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke();
  }
  if (!res) return;
  const y = res.bvp, n = y.length;
  const offset = LIVE_WINDOW_S * res.fs - n; // right-align
  const X = (i) => ((i + offset) / (LIVE_WINDOW_S * res.fs)) * W;
  const lim = Math.max(3, ...y.map(Math.abs));
  const Y = (v) => H / 2 - (v / lim) * (H / 2 - 8);
  c.strokeStyle = css.getPropertyValue('--trace');
  c.lineWidth = 2;
  c.beginPath();
  y.forEach((v, i) => (i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v))));
  c.stroke();
  c.fillStyle = css.getPropertyValue('--fg');
  for (const p of res.peaks) {
    const i = Math.round(p);
    c.beginPath(); c.arc(X(p), Y(y[i]), 3, 0, 2 * Math.PI); c.fill();
  }
}

// ---------- helpers ----------

function setStatus(s) { ui.status.textContent = s; }
const fmt = (v) => (Number.isFinite(v) ? Math.round(v) : '--');
const round1 = (v) => (Number.isFinite(v) ? +v.toFixed(1) : null);
const fmtDuration = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const stamp = (res) => res.meta.startedAt.replace(/[:.]/g, '-');
function save(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
