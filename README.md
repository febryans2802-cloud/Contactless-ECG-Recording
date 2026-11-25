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

The app includes a **rhythm view**: a generic ECG beat, averaged from other people's recordings, drawn at
each detected pulse. It shows *when* your heart beats in a familiar ECG layout. It does **not** show what
*your* ECG looks like, and it looks normal even when your real ECG isn't. (A neural network that tried to
generate a personal ECG from the pulse was built and evaluated. It did no better than this, see below.)
**It's not a medical device. Use it for research or education only.**

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
- `src/ecgTemplate.js`: rhythm view (generic beat placed at each pulse)
- `model/ecg_template.json`: the generic beat (built by `ml/export_template.py`)
- `ml/`: data preparation, the PPG→ECG network experiment and its evaluation (PyTorch)
- `test/`: `npm test` (Node ≥ 18); includes JS-vs-Python parity checks for the template and the filter

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

## Rhythm view

For each pulse peak the app detects, it draws the generic beat with its R-peak 258 ms earlier. That's the
median R→finger-PPG delay in the re-aligned training data, a convention rather than a measurement. The beat
is the average over 35 BIDMC patients after flipping records with inverted lead II, so it has the
conventional upright R wave. (The baseline evaluated below averaged without flipping. Timing is identical.)
In the cross-validation it found 92% of real R-peaks with a median timing error of 17 ms on clean finger
PPG. On face video, beat accuracy depends on the rPPG beat detection, which hasn't been validated on real
people. It's hidden while signal quality is "poor", because beats detected in noise are mostly wrong.

## PPG→ECG network (evaluated, not used in the app)

**Short version: on people it hasn't seen, the network is no better than pasting an average heartbeat at
each detected pulse, and neither resembles the person's real ECG.**

### Training

- Data: [BIDMC PPG and Respiration Dataset](https://physionet.org/content/bidmc/1.0.0/) (PhysioNet,
  ODC-By 1.0): 53 ICU patients, 8 min each, finger PPG + ECG lead II at 125 Hz.
- Input is made to look like the app's input: resampled to 30 Hz, the app's exact 0.7–4 Hz filter
  (parity-tested against `dsp.js`), z-scored, plus band-limited noise augmentation.
- **Channel offsets.** BIDMC's ECG and PPG channels carry an arbitrary per-record time offset (measured
  R→PPG delays were spread evenly over the whole beat interval, including physically impossible values
  under 50 ms). Each record is re-aligned so the PPG peak follows the R-peak by 250 ms. **The R-peak timing
  in the estimate is a convention, not a measurement.** Real face pulse transit time differs from this, and
  between people.
- 18 of 53 records were excluded **before training** because their beat timing was too inconsistent to
  align (arrhythmia, noisy PPG or ECG lead problems). 35 remain.
- Model: 1D U-Net (380k params), 8 s of 30 Hz pulse → 8 s of 120 Hz ECG, L1 loss, 20 epochs.

### Results: 5-fold cross-validation by subject (`ml/results.json`)

Each subject is tested by a model that never saw them. The baseline places the training set's average
beat 258–267 ms before each detected PPG peak.

| Input noise | Model: median r | Template: median r | Model: R-peaks found / precision | Template: R-peaks found / precision |
|---|---|---|---|---|
| none | 0.08 (IQR −0.12…0.32) | 0.20 (IQR −0.25…0.47) | 73% / 83% | 92% / 95% |
| 0.5× signal | 0.06 | 0.10 | 63% / 85% | 85% / 86% |
| 1× signal | 0.04 | 0.04 | 43% / 79% | 68% / 67% |

"r" is the Pearson correlation with the real ECG over 8 s segments. The model's r is not significantly
different from the template's (Wilcoxon p = 0.34; the model wins on 15 of 35 subjects). It finds fewer
R-peaks. About a third of subjects get **negative** r from both methods, because their real lead II shape or
polarity can't be predicted from a pulse.

### Why published PPG→ECG results look much better (`ml/ablation.py`)

Single fold, 7 test subjects, 12 epochs. Noisy, so compare rows with each other, not with the table above:

| Setting | Model r | Template r |
|---|---|---|
| App-matched input, unseen subjects | 0.25 | 0.14 |
| Wideband 0.5–12 Hz input (keeps PPG shape), unseen subjects | 0.09 | 0.16 |
| App-matched input, **same subjects** (train on the first 70% of each record, test on the last 30%) | **0.76** | 0.25 |

Richer input doesn't help. Letting the model see the test subject does. The network learns *each
person's* ECG shape, not a mapping from pulse to ECG that carries over to new people. Results reported from
same-subject splits measure memorisation.

### What this means

- For a **new person**, the network's output is a generic ECG-like drawing, timed to the pulse. The
  template places beats more reliably and is simpler, so **the app uses the template**.
- With **paired calibration data from the same person** (e.g. a few minutes of chest-strap ECG recorded
  alongside the camera), a personalised model can reproduce that person's *usual* ECG shape (r ≈ 0.76 here).
  Even then, it shows what their ECG normally looks like. It can't show a *change*, which is usually
  the reason to record an ECG in the first place.
- The face pulse at 30 Hz differs from the finger PPG used in training. The model has **not** been
  validated on face rPPG at all.

- **Without the person's own ECG, nothing better is possible.** Their own average beat, pasted at their
  pulses, matched their real ECG at median r = 0.61 (same-session data, so optimistic for an old printout).
  The population average got 0.25. Someone else's ECG downloaded online is just another population template.

Given pure noise, the network outputs a low-amplitude trace (std ≈ 0.2 vs ≈ 1 for real ECG) with sparse,
irregular blips, not a clean rhythm.

Reproduce: `ml/download_bidmc.sh && cd ml && pip install -r requirements.txt && python3 train.py --data data`
(CV + network), `python3 ablation.py data`, `python3 export_template.py --data data` (the app's template).
`export.py` exports network weights in a JS-loadable format, but the app no longer includes a loader.

## Known limitations

- **Signal quality is the better of two checks.** (1) Spectral SNR, which only works for a steady rhythm.
  (2) A rhythm-independent check that needs both pulse-band power (0.7–4 Hz vs 4–15 Hz on the unfiltered
  POS signal, ≥ 3.5 dB) *and* consistent beat shapes (template-matching SQI ≥ 0.88). Neither alone is enough:
  band-passed noise lined up on its own peaks scores SQI ≈ 0.8, and motion puts power in-band. When check (2)
  is the one that passes, heart rate is the mean beat rate, because an irregular rhythm has no single
  spectral peak. On held-out simulated 10 s windows:

  | | Old (spectral only) | Now |
  |---|---|---|
  | Irregular-rhythm windows shown | 10 / 150 | 63 / 150, none with < 50% of beats right |
  | Steady-rhythm windows shown | 80 / 150 | 80 / 150 (unchanged) |
  | Pulse-free windows accepted | 46 / 1600 (2.9%) | 55 / 1600 (3.4%), 1 as "good" |

  Thresholds were tuned on simulated data and need checking against real recordings. HRV still requires
  spectral SNR ≥ 3 dB, so it's only reported for steady rhythms.
- **Beat detection was tightened after testing on realistic pulse shapes.** The refractory period is now 60%
  of the spectral beat period, so the dicrotic wave isn't counted as a beat, and a harmonic check stops the
  spectral HR locking onto twice the true rate. On simulated realistic pulses at ~0 dB SNR, about 2–15% of
  beats are still missed (more at higher heart rates).
- **Demo:** `demo/index.html` (built by `node demo/build.mjs` from `demo/shell.html`) runs the same DSP on a
  simulated face signal with known true beats. It's useful for showing the app without a camera.
- The face detector loads from a CDN (jsDelivr + Google Storage). Offline, the app falls back to a fixed
  forehead box you have to position yourself in.
- Camera exposure/white-balance lock only works where the browser exposes it (mostly Chrome on Android).
  On other platforms auto-exposure changes can corrupt the signal.
- Mains-frequency light flicker aliased into 0.7–4 Hz can show up as a false "heart rate". Daylight or DC LED
  light is best.

## References

- W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan, "Algorithmic Principles of Remote PPG," *IEEE TBME* 64(7), 2017.
- G. de Haan, V. Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG," *IEEE TBME* 60(10), 2013.
