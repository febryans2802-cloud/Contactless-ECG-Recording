# Contactless Pulse Recorder (rPPG)

A browser app that records a **pulse waveform**, heart rate, and beat-to-beat intervals from a regular
webcam or phone camera. There's nothing to install, no build step, and no server-side processing: video never leaves the device.

## Read this first: this is not an ECG

An electrocardiogram measures the heart's **electrical** activity through electrodes on the skin. A camera
can't sense that. What a camera *can* sense is **remote photoplethysmography (rPPG)**: tiny skin colour
changes (≈0.1–1% of pixel intensity) caused by blood volume changing with each heartbeat.

| You get | You don't get |
|---|---|
| Heart rate | P wave, QRS complex, T wave |
| Beat timing / inter-beat intervals | PR, QRS, QT intervals, ST segment |
| Indicative HRV (long, clean recordings only) | Arrhythmia classification, ischaemia, anything diagnostic |

Some research papers use deep learning to turn PPG into an ECG-shaped curve. Those curves are *predicted*
from population data, not measured, so they can't reveal an abnormality the pulse wave doesn't already show.
This app deliberately doesn't do that. **It's not a medical device. Use it for research or education only.**

## Run it

The camera API needs `https://` or `localhost`.

```bash
python3 -m http.server 8000      # or: npm start
# open http://localhost:8000
```

On a phone you need HTTPS: deploy the folder to any static host (GitHub Pages works), or use a tunnel.

## Using it

1. **Start camera.** Face the camera in even light. Green boxes = skin regions being sampled.
2. Wait about 10 s for the live heart rate and signal quality to settle.
3. **Start recording**, keep still, then **Stop recording**. Record for **≥ 60 s** if you want HRV.
4. **Download CSV** (uniform 30 Hz: time, pulse, beat flag, mean R/G/B) and/or the **JSON summary**.

## How it works

```
camera frame ─► face detector (MediaPipe BlazeFace; falls back to a fixed forehead box)
            ─► forehead + both cheeks ─► YCbCr skin mask ─► mean R,G,B per frame (capture timestamps)
            ─► resample to 30 Hz ─► POS projection (Wang et al., IEEE TBME 2017)
            ─► 0.7–4 Hz zero-phase band-pass ─► spectral HR + SNR, peak detection ─► IBIs, SDNN, RMSSD
```

Code layout:

- `src/dsp.js`: all signal processing, pure functions, unit-tested
- `src/roi.js`: face tracking, ROI geometry, skin-pixel averaging
- `src/app.js`: camera, UI, recording, export
- `test/dsp.test.mjs`: `npm test` (Node ≥ 18)

## Accuracy: measured limits

Tested on synthetic data with known ground truth (unit tests, plus a synthetic video fed to headless
Chromium as a fake webcam):

- **Heart rate:** recovered within ±2 bpm across 50–130 bpm. The end-to-end browser run measured 75 bpm
  against a true 75 bpm, SNR 20 dB.
- **HRV has a significant noise floor.** A perfectly regular synthetic pulse (true RMSSD ≈ 0) came out with
  RMSSD ≈ 20–40 ms once realistic pixel noise was added. That's the same size as healthy resting RMSSD.
  Frame-timestamp jitter isn't the cause (±10 ms jitter adds only ~3 ms). Amplitude noise moving the
  detected peak is. Narrowing the filter around the heart rate halves the noise floor, but it also
  under-reads real variability by ~25%, so it isn't used. The app therefore **withholds HRV** unless the
  recording is ≥ 60 s with SNR ≥ 3 dB and ≥ 80% of beats passing outlier rejection, and the JSON reports
  `meetsReliabilityCriteria`.
- **Not yet validated on real people.** Real-world error is driven by motion, lighting flicker,
  auto-exposure, skin tone (darker skin gives a weaker signal at the same lighting) and camera compression. Before trusting any
  number, compare against a reference device (chest-strap ECG or a finger pulse oximeter) on your own setup.

Signal quality labels (`good` ≥ 3 dB, `fair` ≥ 0 dB, `poor` < 0 dB SNR) are heuristic thresholds, not calibrated.

## Known limitations

- The face detector loads from a CDN (jsDelivr + Google Storage). Offline, the app falls back to a fixed
  forehead box you have to position yourself in.
- Camera exposure/white-balance lock only works where the browser exposes it (mostly Chrome on Android).
  On other platforms auto-exposure changes can corrupt the signal.
- Mains-frequency light flicker aliased into 0.7–4 Hz can show up as a false "heart rate". Daylight or DC LED
  light is best.

## References

- W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan, "Algorithmic Principles of Remote PPG," *IEEE TBME* 64(7), 2017.
- G. de Haan, V. Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG," *IEEE TBME* 60(10), 2013.
