// Region-of-interest tracking and skin-pixel averaging.

const MP_VERSION = '0.10.14';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';

/**
 * Try to load MediaPipe's face detector. Resolves to null if it can't be
 * loaded (offline, blocked CDN); the app then falls back to a fixed box.
 */
export async function loadFaceDetector() {
  try {
    const vision = await import(/* @vite-ignore */ `${MP_BASE}/vision_bundle.mjs`);
    const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    const make = (delegate) => vision.FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate },
      runningMode: 'VIDEO',
      minDetectionConfidence: 0.5,
    });
    return await make('GPU').catch(() => make('CPU'));
  } catch (err) {
    console.warn('Face detector unavailable, using fixed ROI:', err);
    return null;
  }
}

/**
 * Tracks a face box with exponential smoothing and derives forehead and
 * cheek sub-regions (the areas with the strongest rPPG signal and the least
 * non-rigid motion from talking or blinking).
 */
export class RoiTracker {
  constructor(detector) {
    this.detector = detector;
    this.box = null; // smoothed {x,y,w,h} in video pixels
    this.lastSeen = -Infinity;
    this.frame = 0;
  }

  update(video, nowMs) {
    const W = video.videoWidth, H = video.videoHeight;
    // Run detection every 3rd frame; it's the expensive step.
    if (this.detector && this.frame++ % 3 === 0) {
      const res = this.detector.detectForVideo(video, nowMs);
      const d = res.detections?.[0]?.boundingBox;
      if (d) {
        const nb = { x: d.originX, y: d.originY, w: d.width, h: d.height };
        const a = this.box ? 0.3 : 1;
        this.box = this.box
          ? Object.fromEntries(Object.keys(nb).map((k) => [k, this.box[k] * (1 - a) + nb[k] * a]))
          : nb;
        this.lastSeen = nowMs;
      }
    }
    const faceFound = this.detector && this.box && nowMs - this.lastSeen < 1000;
    if (faceFound) return { faceFound: true, regions: subRegions(this.box, W, H), face: this.box };
    // Fallback: a box in the upper centre where a forehead usually is.
    const fb = { x: W * 0.38, y: H * 0.18, w: W * 0.24, h: H * 0.14 };
    return { faceFound: false, regions: [fb], face: null };
  }
}

function subRegions(b, W, H) {
  const clamp = (r) => {
    const x = Math.max(0, r.x), y = Math.max(0, r.y);
    return { x, y, w: Math.max(1, Math.min(W - x, r.w)), h: Math.max(1, Math.min(H - y, r.h)) };
  };
  return [
    // Forehead: BlazeFace boxes start around the eyebrows, so go slightly above.
    clamp({ x: b.x + b.w * 0.25, y: b.y - b.h * 0.05, w: b.w * 0.5, h: b.h * 0.18 }),
    // Cheeks
    clamp({ x: b.x + b.w * 0.12, y: b.y + b.h * 0.5, w: b.w * 0.22, h: b.h * 0.2 }),
    clamp({ x: b.x + b.w * 0.66, y: b.y + b.h * 0.5, w: b.w * 0.22, h: b.h * 0.2 }),
  ];
}

/**
 * Mean RGB of skin-coloured pixels inside the regions.
 * Uses a broad YCbCr skin rule (Chai & Ngan 1999) to drop hair, eyebrows,
 * background and specular highlights.
 */
export function meanSkinRGB(ctx, regions) {
  let r = 0, g = 0, b = 0, n = 0, total = 0;
  for (const reg of regions) {
    const x = Math.round(reg.x), y = Math.round(reg.y);
    const w = Math.round(reg.w), h = Math.round(reg.h);
    if (w < 2 || h < 2) continue;
    const { data } = ctx.getImageData(x, y, w, h);
    for (let i = 0; i < data.length; i += 4) {
      const R = data[i], G = data[i + 1], B = data[i + 2];
      total++;
      const Y = 0.299 * R + 0.587 * G + 0.114 * B;
      const cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B;
      const cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B;
      if (Y < 40 || Y > 245 || cb < 77 || cb > 127 || cr < 133 || cr > 173) continue;
      r += R; g += G; b += B; n++;
    }
  }
  return { r: n ? r / n : NaN, g: n ? g / n : NaN, b: n ? b / n : NaN, skinFraction: total ? n / total : 0 };
}
