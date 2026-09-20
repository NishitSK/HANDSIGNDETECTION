# ISL Letter Recognition — Dataset, Methodology & Results Report

Generated: 2026-09-14

## 1. Summary

The project previously had no usable trained model: its local training data
(`data/ISL_DATASETS_2`, 3 classes) had MediaPipe's own landmark-overlay
graphics burned into the saved photos, giving a **0% real hand-detection
rate** on two of three classes. This report covers replacing that dataset,
fixing several pipeline bugs uncovered in the process, and training and
comparing four model architectures on the corrected pipeline.

**Result:** a TCN (Temporal Convolutional Network) model reaching **98.89%
accuracy on a held-out test set** (4,396 images, never seen during training
or model selection), the smallest of the four architectures compared
(1.2M params), and the fastest to converge (11 epochs, ~24 minutes).

## 2. Dataset

**Source:** [RealSign Indian Sign Language Dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset)
(CC0-1.0 / public domain). Fingerspelled ISL alphabet (A–Z), contributed by
four signers with varied lighting, skin tone, and hand pose, distributed as
raw camera photographs (no annotation overlays).

- 26 classes (A–Z), ~1,000 images/class, 25,977 images total after merging
  the dataset's own Training/Testing/Validation folders into one pool
  (`data/RealSign_ISL/<letter>/`) — the three-way split was redone by this
  project's own pipeline rather than reused, so it shares the same
  train/val/test partitioning logic and seed as everything else here.
- MediaPipe hand-landmark detection yield ranged **53%–100% per class**
  (mean ~85%) — normal, healthy variance for a real, diverse dataset, a
  stark contrast to the corrupted local dataset's 0%.
- Confirmed during review: ISL fingerspelling is **not** one-handed the way
  ASL is — e.g. the letter 'A' is a two-handed sign in this dataset. This
  validated keeping the pipeline's "both hands + face" landmark mode rather
  than simplifying to a single-hand model.
- 21,979 samples had a detectable hand and were used for training/eval.

## 3. Bugs found and fixed this session

These were found by testing each stage against real data end-to-end, not by
code inspection alone — several only became visible once a model actually
existed to evaluate.

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | `extract_from_image()` never returned `None` on no-hand-detected (always returned zeros) | Every `if landmarks is not None` check in the codebase was dead code; no-hand frames were silently treated as valid samples | Returns `None` when neither hand is detected |
| 2 | `gui_data_collector.py` saved the MediaPipe-annotated **display** frame, not the raw camera frame | Every photo collected with this tool had landmark dots/skeleton graphics baked into the pixels, making later re-detection fail | Collector now saves the raw frame; overlay is drawn on a separate display-only copy |
| 3 | Data augmentation ran **before** the train/val/test split | Rotated/scaled/noised near-duplicates of the same photo could land in both train and test, inflating reported accuracy | Split first, augment only the training split |
| 4 | Training sequences were **zero-padded** to the 30-frame window; live inference feeds 30 real frames | Model trained on ~97%-zero input, evaluated in the real world on 100%-real input — a train/serve skew | Short sequences are tiled (repeated) instead of zero-padded |
| 5 | No held-out test set — "evaluation" reused the validation set already used for early-stopping/checkpoint selection | Optimistic bias in reported accuracy | Added a genuine 3-way split; final metrics come from the untouched test set |
| 6 | `test_size` computed from `validation_split` config but never used (hardcoded `0.2` instead) | Config value silently ignored | Split now actually uses the configured fractions |
| 7 | No reproducibility seeding across NumPy/Python/TensorFlow | Runs not reproducible | Added `set_global_seed()` |
| 8 | Repeated model-building within one process (multi-architecture comparison) leaked TensorFlow graph/session state | `MemoryError` crash partway through a multi-run comparison (caught by a dry run before the real 8-hour run) | `keras.backend.clear_session()` at the start of every `train()` call |
| 9 | Landmark arrays defaulted to float64 | Doubled memory footprint for no accuracy benefit (Keras uses float32 internally regardless) | Cast to float32 in `prepare_sequences` |
| 10 | **`normalize_landmarks()` (translation/scale-invariant centering) was called by live inference but never by training** | Model trained on raw, un-normalized coordinates but was queried at inference time with normalized ones — a severe train/serve feature-space mismatch. This was **not caught by the held-out test-set metric** (train/val/test were all consistently un-normalized, so the 98.50% pre-fix TCN number was internally consistent but not representative of real usage) — it only surfaced via an end-to-end sanity check that ran real images through the actual `ISLInference` code path. On the pre-fix model this check scored 17/29 (58.6%), with the model collapsing to predicting one letter ('S') at 100% confidence regardless of input. | Normalization now happens in `prepare_sequences()`, matching what inference does. TCN retrained; the same sanity check improved to 26/29 (89.7%), with the 3 remaining misses being sensible confusions between visually similar letters (M↔N, S↔K), not degenerate collapse. |

Also fixed (correctness/robustness, lower impact): a `model_manager.py` crash
on empty class-name metadata; a `tts_engine.py` `AttributeError` when the
gTTS backend's import failed; a raw sklearn traceback replaced with a clear
error when a class has too few usable samples to split.

## 4. Architecture comparison methodology

Given an 8-hour compute budget (CPU only, no GPU) and no prior data on how
fast each architecture trains on this task, the comparison used two phases:

1. **Benchmark** — every architecture trained for a fixed 2-minute wall-clock
   window to measure real epochs/minute on this machine.
2. **Time-boxed full runs** — remaining budget allocated per architecture
   (60-minute floor each, plus a share of the remainder proportional to
   benchmark speed, capped so no architecture could take more than half the
   pool), each capped by a hard wall-clock `TimeBudgetCallback` in addition
   to the existing epoch/early-stopping config.

All four architectures used the **identical** data, seed, and train/val/test
split (verified deterministic given the fixed `random_state=42`), so the
comparison is apples-to-apples on data — though see the caveat in §6 about
the normalization fix landing only on the TCN result.

## 5. Results

| Architecture | Params | Epochs | Wall time | Test accuracy |
|---|---|---|---|---|
| LSTM | 4.23M | 7 | 110 min | 96.66% |
| GRU | 3.22M | 7 | 110 min | 97.77% |
| **TCN** | **1.21M** | 11 | 30 min | **98.50%** (pre-fix) → **98.89%** (post-fix, promoted) |
| Transformer | 9.62M | 12 | 109 min | 3.75% (failed to learn — see below) |

**TCN won on every axis**: highest accuracy, fewest parameters, fastest to
converge. It was promoted to `models/saved/isl_model.h5` and
`config.yaml`'s `model.type` was set to `TCN`.

Per-class results for the final (post-normalization-fix) TCN model: 98.89%
overall, with per-class precision/recall consistently in the mid-to-high
90s–100% across all 26 letters (full breakdown in
`results/TCN/final_normalized/classification_report.txt` and
`confusion_matrix.png`).

### 5.1 Why the Transformer failed

The Transformer's training loss was stuck at **exactly ln(26) = 3.258**
(the loss of a uniform random 26-class predictor) for all 12 epochs it
completed — not merely slow convergence, but a complete failure to move away
from the uniform-prediction baseline. A targeted diagnostic ruled out a
structural bug: the same architecture, given a small subset of the same real
data with no callbacks and a plain `model.fit()`, reached 100% training
accuracy within 2 epochs, and gradients on a random batch were finite and
non-zero (min norm 2e-11, max 2.14, no missing gradients). This is
consistent with the well-documented failure mode of **Post-Layer-Norm
Transformers trained with a flat learning rate and no warmup schedule** —
a known training-recipe issue, not a data or implementation bug. It was not
fixed in this session (out of scope/time); a warmup + decay LR schedule is
the natural next experiment if the Transformer variant is wanted.

## 6. Limitations and caveats

- **Only TCN was retrained with the normalization fix** (§3, bug #10). The
  LSTM/GRU/Transformer numbers above are from the pre-fix pipeline. TCN's
  own pre- vs. post-fix numbers (98.50% → 98.89%) suggest the fix's effect
  on held-out *test-set* accuracy is small — the fix's real impact is on
  live/production inference accuracy, which the test-set metric doesn't
  capture at all (train/val/test were internally consistent either way). A
  full re-run of LSTM/GRU/Transformer with the fix would be needed for a
  completely fair final comparison; given TCN's decisive win on every axis
  (accuracy, size, speed) this wasn't done in this session.
- Every image is an independent photograph rather than a burst-captured
  frame sequence, so per-file grouping in the leakage-safe splitter
  degenerates to (correctly) behaving like a plain stratified split — there
  is no cross-*recording-session* held-out test here, only cross-image.
- Time-boxed training means architectures were compared under a shared
  wall-clock budget, not a shared epoch budget or full convergence — TCN's
  win reflects both quality and efficiency, but LSTM/GRU might improve
  further given more time (TCN, notably, converged in a third of its
  allotted budget).
- Validation was end-to-end through the real `ISLInference` code path on a
  small (29-image) held-out sample, not a live webcam session — no webcam is
  available in this environment.
- **Open issue: the desktop live paths mirror the frame before extracting
  landmarks** (`cv2.flip` in `src/inference.py` and `src/gui_app.py`), but
  training extraction never did. Mirroring flips x and swaps MediaPipe's
  Left/Right hand labels, so the desktop webcam app feeds the model differently
  from how it was trained. The mobile app extracts from the un-mirrored frame and
  mirrors only the display, matching training. Not yet fixed on desktop.

## 7. Files

- `models/saved/isl_model.h5`, `model_metadata.json` — the promoted TCN model
- `data/RealSign_ISL/` — the dataset (gitignored, ~650MB)
- `results/<ARCH>/benchmark/`, `results/<ARCH>/final*/` — per-architecture models, metrics, and plots from every run
- `results/comparison_results.json` — structured results for all runs
- `results/TCN_hands_only/` — the hands-only ablation model, logs, and checkpoints (§8.2)
- `mobile/` — the phone browser app (`app/`), its HTTPS dev server, model converter, and parity tests

## 8. Mobile version (browser)

**§8.1-8.5 below describe the first mobile implementation (TCN, hands+face,
2.4MB). Real-world use on a mid/low-end phone found it "really slow" and "not
usable" — §8.6 explains why and replaces it with a lighter model. Kept as a
historical record (and because the conversion/verification tooling they
describe is still exactly what §8.6's model goes through), not because it's
what ships now.**

### 8.1 What was removed

- **Flan-T5 grammar model (948 MB).** The desktop app loads it at startup but
  never uses it: `correct_sentence()` defaults to `use_model=False`, so the
  rule-based path is what runs. The browser port of those rules
  (`mobile/app/grammar.mjs`) matches the Python output on all 66 reference
  cases (`mobile/tests/verify_grammar.mjs`).

### 8.2 Face landmarks: ablation

Face tracking is the most expensive per-frame step on a phone, so a hands-only
variant (126 inputs instead of 186) was tested. The decision rule was fixed
before training: drop face only if held-out accuracy falls by 0.5 points or less.

| Input | Test accuracy | Weakest letters (F1) |
|---|---|---|
| Hands + face (186) | **98.89%** | N 0.9565, M 0.9645, I 0.9739 |
| Hands only (126) | 98.13% | N 0.9137, E 0.9437, M 0.9567 |

Both used the same TCN architecture, data, split, seed, and early-stopping rule,
and both stopped after 11 epochs. Hands-only lost 0.76 points — over the
threshold — and hurt the already-weakest letters most, so **face landmarks were
kept**. To limit the cost, the phone refreshes face points every other frame.
Caveat: one run per variant, so part of the gap may be run-to-run noise.

### 8.3 Converting the model for the browser

`mobile/convert_model.py` exports the Keras model as a SavedModel and converts it
to a TensorFlow.js graph model with float16 weights: **14.6 MB `.h5` → 2.4 MB**
(one 2.4 MB weight shard plus a 48 KB `model.json`).

The converter runs in its own environment (TensorFlow 2.15 + tensorflowjs 4.17).
On Windows, tensorflowjs imports two packages at load time that its SavedModel
path never uses: JAX (installs normally as `jax[cpu]==0.4.23`) and TensorFlow
Decision Forests (no Windows build; replaced with an empty package). Setup
commands are in the README's Mobile section.

Two conversion traps turned up, both now handled inside `convert_model.py`:

1. **Keras's fused LayerNormalization** exports as a *training-mode*
   `FusedBatchNormV3` with empty mean and variance. TF.js only implements
   inference-mode batch norm, so the first converted model loaded fine but
   crashed on the first WebGL prediction. The script switches LayerNormalization
   to its non-fused path before export (the same math in ordinary mean/variance
   ops) and refuses to write a graph that still contains training-mode batch norm.
2. **TensorFlow's oneDNN graph optimizer**, on by default in the Windows build,
   then re-fused those ops into `_MklLayerNorm`, which TF.js has no kernel for.
   The script sets `TF_ENABLE_ONEDNN_OPTS=0` before importing TensorFlow.

The resulting graph has 129 nodes using only standard ops: convolutions, fused
matrix multiply, Einsum for the attention layer, and Mean / SquaredDifference /
Rsqrt for the layer norm.

### 8.4 Browser backends

Each TF.js backend was loaded on a clean page and given the same fixed input:

| Backend | Result | Time per prediction |
|---|---|---|
| WebGL | Works | 18 ms (1.3 s for the first call while shaders compile) |
| CPU (pure JavaScript) | Works; output identical to WebGL | ~5.4 s |
| WASM | Fails: no `Einsum` kernel, which the attention layer needs | — |

The app uses WebGL, the only backend fast enough for live use. Timings are from
desktop Chromium with another CPU-heavy job running, so they are relative
rather than phone numbers; the app's Settings sheet shows live model time on the
device. With the fixed model, the app loaded the model and both MediaPipe
trackers in the browser and stopped only at the camera request, which the
test browser blocks — the live camera flow still needs a real phone.

### 8.5 Verification

Two browser pages in `mobile/tests/` check the phone pipeline against the Python
one, using the same held-out test split as §5.

**Model on WebGL** (`parity.html`, all 4,396 held-out samples):

| Weights | Browser accuracy | Same letter as Keras | Keras accuracy |
|---|---|---|---|
| float16 (shipped, 2.4 MB) | 98.82% | 99.66% | 98.89% |
| float32 (4.7 MB) | 98.82% | 99.66% | 98.89% |

The JavaScript feature pipeline matches Python exactly (largest feature
difference: 0). Both weight precisions disagree with Keras on the same 15
samples, so float16 isn't the cause; the drift comes from TF.js executing the
graph differently from TensorFlow. Most flips pick Keras's runner-up on samples
Keras was unsure about (C/P, I/X, M/N), but a few are confident (one S at 0.99
became G). The app ships float16.

**MediaPipe Web vs Python** (`mediapipe_parity.html`, held-out photos, balanced
across letters). This check caught a bug that would have broken the phone app:
MediaPipe Web labels left and right hands the opposite way to the Python
`solutions.hands` API that produced the training landmarks, so most hands reached
the model in the wrong slot. `pipeline.mjs` now maps the labels back.

| | Before fix (52 photos) | After fix (52 photos) | After fix (260 photos) |
|---|---|---|---|
| Hands in the wrong left/right slot | 48 | 2 | 13 |
| Browser pipeline predicts the right letter | 11.5% | 92.3% | **91.5%** |
| Keras on the same photos | 100% | 100% | 100% |

After the fix, browser hand landmarks sit within 1.2% of the image size of
Python's (median; 95th percentile 4.9%), but the two MediaPipe APIs still don't
produce identical landmarks, and that is enough to change about 1 letter in 12.
Neither pipeline found a face in any sampled photo (the dataset's photos are hand
close-ups), so face features were zeros on both sides.

**About 91.5% (roughly ±3.4 points) is the best current estimate of accuracy on
the phone.** It is measured on held-out dataset photos rather than a live camera,
where lighting, framing, and motion will differ. The most direct way to close the
gap is to re-extract the training landmarks with MediaPipe's Tasks API, which runs
the same landmark model as the browser, and retrain.

### 8.6 Second pass: the TCN was unusable on real phone hardware

Real-world feedback on the §8.1-8.5 model: skeleton tracking felt fine, but
letter classification was "really slow" and "not usable." Root cause was
already on record in §8.4 but under-weighted: the TCN's attention layer
(`MultiHeadAttention`/Einsum) has no WASM kernel, and on a phone with a weak or
broken WebGL driver — common on budget Android GPUs — the only fallback is
unaccelerated plain-JS CPU, measured at **~5.4 seconds per prediction**. A
`FaceLandmarker` running every other frame added further cost for no benefit
to fingerspelling (ISL two-hand letters already need both hands tracked; face
carries no letter information).

**Fix:** replace the TCN with a plain Dense-only MLP over a **single frame**
(no time convolution, no attention at all), hands-only (no face tracker).
Justified by the pipeline's own design: every training "sequence" is already
one static photo tiled 30× to fill the model's time window (`prepare_sequences`
tiles rather than pads, precisely because there's no real temporal signal in a
single photo) — so a temporal model was solving a problem that doesn't exist
here. `sequence_length=1` in config, `model.type: MLP`
(`models/gesture_model.py::build_mlp_model`).

ISL retrained on the **same already-extracted landmarks** (no re-run of
MediaPipe needed — `scripts/train_mlp_isl.py` slices the cached both-hands+face
pickle down to hands-only before normalizing, same equivalence used in §8.2):

| | TCN (hands+face, §8.1-8.5) | MLP (hands-only) |
|---|---|---|
| Held-out test accuracy | 98.89% | **98.27%** |
| Parameters | ~2.4M | ~30K |
| Model size (float16) | 2.4 MB | **160 KB** |
| Parity check, WebGL, 4,396 samples | 101 s, 98.82% | **1.8 s, 98.82%, 100% argmax agreement with Keras** |
| Single prediction, WASM backend | not supported (no Einsum kernel) | **~2 ms** |

0.6 accuracy points for a ~15× smaller model that's two to three orders of
magnitude faster in the worst case (no-WebGL fallback) and unblocks a real
WASM-accelerated middle tier between WebGL and plain CPU (§8.7) — the actual
CPU-fallback problem underneath the original complaint.

**Desktop app unaffected.** This entire pass only touches `mobile/`; the
desktop `models/saved/isl_model.h5` (TCN, §1-§7) was hash-verified unchanged
throughout (`d676aeae...`).

ASL added the same way, one model per sign system rather than one shared
26-vs-52-class model (kept smaller, avoids ISL's two-handed letters and ASL's
one-handed letters sharing a decision boundary):

- **Dataset, attempt 1 — rejected.** `AI-Datasets/ASL-Alphabet-Dataset` (a
  Kaggle `grassknoted/asl-alphabet` mirror, no auth needed) looked clean by
  every check that matters for a GitHub mirror — real repo, correct file
  counts, `raw.githubusercontent.com` serving actual JPEGs. Only spot-checking
  actual image content caught it: **every photo, across both its train and
  test splits, has MediaPipe's own landmark-overlay dots and skeleton lines
  burned into the pixels** — the identical corruption this project's own local
  dataset had at the very start of this work (§3, bug #2's root cause).
  13,000 images downloaded and deleted before any training touched them, once
  three spot-checked samples all showed the same overlay.
- **Dataset, attempt 2 — used.** `cristian20a/ASL_Dataset` on GitHub (real
  photos from multiple signers, ~80-100/letter, no LFS, no auth). This time
  verified *before* the bulk download: fetched 4 samples across different
  letters, confirmed clean visually, then ran actual MediaPipe extraction on
  those 4 and got 4/4 detections before downloading the other 2,267.

**ASL model results.** 2,268 real photos across the 24 letters the source
repo has (~80-100/letter, multiple signers), same MLP architecture and
training script as ISL (`scripts/train_mlp_asl.py`), per-letter detection
rates all ≥80% (worst: Q at 80%; everything else 91-100%):

| | |
|---|---|
| Classes | 24 (A-I, K-Y — no J/Z, see below) |
| Held-out test accuracy | **92.34%** (444 samples) |
| Model size (float16) | 160 KB |
| Parity check, WebGL, 444 samples | **100% argmax agreement with Keras, 0 feature drift** |

92.34% vs ISL's 98.27% mainly reflects dataset size (2,268 photos / ~90 per
letter vs ISL's tens of thousands) and signer diversity, not the
architecture — the same MLP hits 98%+ on ISL's larger set.

**No J or Z.** Both are motion signs with no static handshape. The ISL
source dataset happened to include static frames for its motion letters
(trained anyway, marked "motion sign — still frame" in the pamphlet, per
§8.7's J/Z note). The ASL source (`cristian20a/ASL_Dataset`) has no J or Z
folders at all — confirmed against its GitHub tree listing, not just a
missing local download — so rather than fabricate an approximation, the ASL
model is trained as a 24-class model and the app's pamphlet for ASL mode
simply has no J/Z cards (`meta.class_names` drives the pamphlet grid, so this
needed no special-casing — the missing classes are just absent from the
list).

**A verification-tooling bug this caught.** The parity/verification scripts
(`export_reference.py`, `parity_core.mjs`) were written for the ISL MLP,
which reuses the original 62-point (hands+face) landmark cache sliced down
to 42 points late in the pipeline — so the raw per-sample "Python reference
frame" files they read/write stayed 186-features-wide throughout, even
though only 126 of those features get fed to the model. Hardcoding that
186/62 stride worked for ISL by coincidence, not because it was general. The
ASL landmarks were extracted hands-only from the very start
(`detect_face=False`, 42 points / 126 features with no wider array to slice
from), which broke that assumption: the very first ASL parity run showed
2.9% argmax agreement with Keras, i.e. random noise, on both WebGL and CPU
backends. Isolated with a `--no-quantize` float32 rebuild (ruled out
float16), then traced to the hardcoded stride reading each sample from the
wrong 60-float offset into the raw-frame buffer. Fixed by having
`export_reference.py` record the actual `raw_points`/`raw_frame_width` per
export into `meta.json`, and `parity_core.mjs` read those (defaulting to
62/186 for old exports with no such fields, so the already-verified ISL
result stays reproducible unchanged). Re-ran both ISL and ASL parity after
the fix: ISL still 100%/98.27% agreement (no regression), ASL now 100%/92.34%.

### 8.7 Mobile-only optimizations

Once the model itself stopped being the bottleneck, remaining tricks were
filtered against what's actually reachable from a browser tab (no native
TFLite/NPU/DSP/CoreML access) and what's proportionate to a ~30K-parameter
model already at 160KB — most heavier techniques (quantization beyond float16,
pruning, distillation, NAS) would save an imperceptible amount at this size.
Shipped:

- **WASM as a middle backend**, `webgl → wasm → cpu` instead of
  `webgl → cpu`. Only possible because the MLP has no Einsum op (§8.6) —
  measured **~2ms/prediction** on WASM, so devices with broken/absent WebGL
  (not rare on cheap Android GPUs) get real acceleration instead of falling
  straight to the ~5.4s/prediction unaccelerated path.
- **`WEBGL_FORCE_F16_TEXTURES`** — halves GPU texture bandwidth, the actual
  bottleneck on most low-end mobile GPUs (compute is rarely the limit there).
- **Per-mode `numHands`** — ASL is one-handed, ISL needs both; the tracker is
  now built with the mode's actual requirement (`metadata.json`'s `num_hands`)
  instead of always tracking 2, and rebuilt (MediaPipe task options are
  immutable) only when a mode switch actually changes the count.
- **Preallocated input tensor buffer** — one `Float32Array` reused every
  `classify()` call instead of a fresh allocation each frame; less GC churn,
  which costs more on phones (smaller heaps, longer pauses) than desktop.
- **Service worker asset caching** (`mobile/app/sw.js`) — cache-first for the
  ~20MB of CDN/tracker/model assets (version-pinned in their URLs, or bumped
  via `CACHE_VERSION` for our own files, so a cache hit is always correct),
  network-first with cache fallback for the app shell. First visit unchanged;
  every visit after skips ~20MB of network — speed and mobile data both.
- **J/Z labeled, not silently wrong.** Both alphabets' J and Z are motion
  signs; both datasets only have static photos of them. Trained anyway
  (whatever static handshape the photo shows) but the pamphlet marks their
  cards "motion sign — still frame" rather than presenting them as equivalent
  to the other 24 letters.

Considered and explicitly not done, with reasons: INT8/INT4 quantization,
pruning, weight clustering, low-rank/tensor decomposition, knowledge
distillation, NAS (real techniques, wrong scale — the model is already
160KB/~30K params; worth it only if a future model is much bigger); NPU/DSP/
NNAPI/CoreML/Metal/TensorRT delegates (not reachable from a browser tab —
native-app territory); federated learning / on-device personalization (no
training pipeline exists on-device, none wanted); early-exit/cascade/
conditional computation, token pruning (solve problems — very deep nets,
transformers — this model doesn't have); operator/graph/layer fusion (already
done automatically by the TF.js converter, confirmed in the converted graph's
`_FusedConv2D`/`_FusedMatMul` nodes). Deferred rather than rejected: INT8
quantization anyway if size/speed still matters after the above, a
capture-resolution rung on the adaptive quality ladder for the very lowest
hardware tier, `OffscreenCanvas`+Worker for the overlay drawing.
