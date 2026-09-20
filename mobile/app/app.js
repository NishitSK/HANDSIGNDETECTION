import { FilesetResolver, HandLandmarker, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs';
import { argmax, assembleLandmarks, featureLayout, normalizeFrame, PredictionSmoother, tileSequence } from './pipeline.mjs';
import { processSequence } from './grammar.mjs';

const TASKS_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const MODE_STORAGE_KEY = 'islFingerspellMode';
const MODE_LABELS = { isl: 'ISL', asl: 'ASL' };

const HOLD_TO_ADD_MS = 900;
// A letter that was just added can only be added again after the hand drops or
// changes for this long — otherwise one held sign would type AAAA.
const RELEASE_TO_REPEAT_MS = 300;
const EMPTY_HINT = 'Hold a letter steady to add it. Use Space between words.';
const TARGET_FPS = 30;

// Tracking (hand position, drawn every frame) and classification (which letter,
// smoothed over ~5 predictions and held for 900ms before it counts) don't need the
// same refresh rate. Face landmarks barely move frame to frame and the model result
// is smoothed anyway, so both start throttled and adapt from there — see
// adaptQuality(). Phone GPUs vary a lot; a fixed guess would be wrong on most of them.
const QUALITY_STEPS = [
  { face: 1, classify: 1 }, // full quality
  { face: 2, classify: 1 },
  { face: 2, classify: 2 }, // default starting point
  { face: 3, classify: 2 },
  { face: 3, classify: 3 },
  { face: 4, classify: 4 }, // most throttled
];
let qualityIndex = 2;
let lastQualityCheck = 0;

const $ = (id) => document.getElementById(id);
const els = {
  stage: $('stage'),
  video: $('video'),
  overlay: $('overlay'),
  glyph: $('glyph'),
  glyphOutline: $('glyphOutline'),
  glyphFill: $('glyphFill'),
  glyphCaption: $('glyphCaption'),
  status: $('status'),
  statusMessage: $('statusMessage'),
  retryButton: $('retryButton'),
  tapeWords: $('tapeWords'),
  tapeCurrent: $('tapeCurrent'),
  announcer: $('announcer'),
  addButton: $('addButton'),
  deleteButton: $('deleteButton'),
  spaceButton: $('spaceButton'),
  clearButton: $('clearButton'),
  speakButton: $('speakButton'),
  settingsButton: $('settingsButton'),
  settings: $('settings'),
  autoCapture: $('autoCapture'),
  showLandmarks: $('showLandmarks'),
  mirrorInput: $('mirrorInput'),
  readout: $('readout'),
  wordmark: $('wordmark'),
  pamphletButton: $('pamphletButton'),
  pamphlet: $('pamphlet'),
  pamphletSystem: $('pamphletSystem'),
  pamphletGrid: $('pamphletGrid'),
  tutorButton: $('tutorButton'),
  tutorDialog: $('tutorDialog'),
  ghostGuide: $('ghostGuide'),
  ghostGuideImg: $('ghostGuideImg'),
  ghostGuideLabel: $('ghostGuideLabel'),
  tutorPreviewImg: $('tutorPreviewImg'),
  tutorPrevBtn: $('tutorPrevBtn'),
  tutorNextBtn: $('tutorNextBtn'),
  tutorSelect: $('tutorSelect'),
  tutorTargetTitle: $('tutorTargetTitle'),
  tutorHintText: $('tutorHintText'),
  tutorGuideToggle: $('tutorGuideToggle'),
  tutorOpacitySlider: $('tutorOpacitySlider'),
  tutorOpacityVal: $('tutorOpacityVal'),
  tutorStatusBadge: $('tutorStatusBadge'),
  tutorStatusText: $('tutorStatusText'),
  tutorStatusIcon: $('tutorStatusIcon'),
};
const overlayContext = els.overlay.getContext('2d');

const state = {
  words: [],
  letters: [],
  sentence: '',
  live: null,
  hold: null,
  locked: null,
  releasedSince: null,
  fps: 0,
  modelMs: 0,
};

let model;
let meta;
let featureCount;
let seqLen;
let inputBuffer;
let includeFace;
let smoother;
let mode = 'isl';
try {
  mode = localStorage.getItem(MODE_STORAGE_KEY) === 'asl' ? 'asl' : 'isl';
} catch {
  // Private browsing or storage disabled — default to ISL.
}
let handLandmarker;
let faceLandmarker;
let faceLoadPromise = null;
let visionResolver = null;
let stream = null;
let lastFace = null;
let frameIndex = 0;
let lastVideoTime = -1;
let lastFrameAt = 0;
let classifying = false;
// Bumped whenever hands leave the frame, so a classification that was already in
// flight can't paint a letter after the hands are gone.
let presenceToken = 0;
let mirrorCanvas = null;

const ema = (previous, value, alpha = 0.15) => (previous ? previous + alpha * (value - previous) : value);

const withStage = (stage, promise) =>
  promise.catch((error) => {
    throw Object.assign(error instanceof Error ? error : new Error(String(error)), { stage });
  });

// ---------------------------------------------------------------------------
// Loading

// Both sign systems ship as separate models under model/<mode>/ — same shape
// family (single-frame, hands-only), but distinct class labels and weights, so
// each gets its own small download rather than one net guessing which alphabet
// a hand shape belongs to.
// Try backends in order of speed, falling all the way to plain JS only if
// nothing accelerated is available. WebGL is fastest where it works; WASM
// (SIMD+threads) is the important middle rung — it has no kernel for the
// attention/Einsum op the old TCN needed, which is why that model's fallback
// meant unaccelerated CPU, but this Dense-only MLP has no such gap, so WASM
// now gets real acceleration on devices with broken/absent WebGL (common on
// cheap Android GPUs) instead of them falling straight to the slow path.
async function initBackend() {
  if (tf.getBackend() != null) return;

  // Halves GPU texture bandwidth, the actual bottleneck on most low-end
  // mobile GPUs (compute is rarely the limit there) — must be set before the
  // webgl backend initializes.
  tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);

  for (const backend of ['webgl', 'wasm', 'cpu']) {
    try {
      if (backend === 'wasm') {
        tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.17.0/dist/');
      }
      if (await tf.setBackend(backend)) break;
    } catch {
      // Try the next backend down the chain.
    }
  }
  await tf.ready();
}

async function loadClassifier(forMode) {
  const response = await fetch(`model/${forMode}/metadata.json`);
  if (!response.ok) throw new Error(`model/${forMode}/metadata.json returned ${response.status}`);
  const nextMeta = await response.json();

  await initBackend();

  const nextModel = await tf.loadGraphModel(`model/${forMode}/model.json`);
  const nextFeatureCount = nextModel.inputs[0].shape.at(-1);
  // Sequence-model checkpoints (shape [batch, seq, features]) report a real
  // middle dimension; single-frame MLPs report shape [batch, features] with
  // nothing in between — treat that as seqLen 1 rather than reading `undefined`.
  const rawSeqLen = nextModel.inputs[0].shape.length > 2 ? nextModel.inputs[0].shape.at(-2) : 1;
  const nextSeqLen = rawSeqLen > 0 ? rawSeqLen : 1;

  // Compile shaders / warm up now so the first real frame isn't the slow one.
  tf.tidy(() => {
    nextModel.predict(tf.zeros([1, nextSeqLen, nextFeatureCount])).dataSync();
  });

  model?.dispose();
  model = nextModel;
  meta = nextMeta;
  featureCount = nextFeatureCount;
  seqLen = nextSeqLen;
  ({ includeFace } = featureLayout(featureCount));
  smoother = new PredictionSmoother(meta.smoothing_window);
  // One buffer reused every classify() call instead of a fresh Float32Array +
  // tensor each frame — less GC churn, which costs more on phones (smaller
  // heaps, longer pauses) than desktop.
  inputBuffer = new Float32Array(nextSeqLen * nextFeatureCount);
}

async function createTask(Task, vision, modelAssetPath, options) {
  const base = { runningMode: 'VIDEO', ...options };
  try {
    return await Task.createFromOptions(vision, { ...base, baseOptions: { modelAssetPath, delegate: 'GPU' } });
  } catch {
    return Task.createFromOptions(vision, { ...base, baseOptions: { modelAssetPath, delegate: 'CPU' } });
  }
}

let currentNumHands = null;

// numHands isn't just a resource-budget knob — MediaPipe finds at most this
// many hands per frame, so it's fixed per sign system: ASL fingerspelling is
// one-handed (tracking a second hand there is pure wasted work), ISL needs
// both. HandLandmarker options are immutable after creation, so a mode
// switch that needs a different count has to build a new tracker rather than
// reconfigure the existing one.
async function loadHandTracker(vision, numHands) {
  if (handLandmarker && currentNumHands === numHands) return;
  const previous = handLandmarker;
  handLandmarker = await createTask(HandLandmarker, vision, HAND_MODEL, {
    minTrackingConfidence: 0.5,
    numHands,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
  });
  currentNumHands = numHands;
  previous?.close();
}

function loadFaceTracker(vision) {
  // Both shipped models are hands-only, so this normally never runs — kept for
  // a future/custom model that does want face features, loaded lazily rather
  // than by default so a phone never pays for a tracker no active mode uses.
  faceLoadPromise ??= createTask(FaceLandmarker, vision, FACE_MODEL, {
    minTrackingConfidence: 0.5,
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
  }).then((tracker) => {
    faceLandmarker = tracker;
  });
  return faceLoadPromise;
}

async function loadTrackers() {
  const vision = await FilesetResolver.forVisionTasks(TASKS_WASM);
  visionResolver = vision;
  await loadHandTracker(vision, meta.num_hands ?? 2);
  if (includeFace) await loadFaceTracker(vision);
}

async function startCamera(facing) {
  stream?.getTracks().forEach((track) => track.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    // Some front cameras default to 15fps unless asked for more; ask explicitly
    // rather than tracking capped below what we're trying to hit.
    video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: TARGET_FPS } },
  });
  els.video.srcObject = stream;
  await els.video.play();
  els.stage.dataset.facing = facing;
  els.overlay.width = els.video.videoWidth;
  els.overlay.height = els.video.videoHeight;
  lastVideoTime = -1;
}

async function keepScreenAwake() {
  try {
    await navigator.wakeLock?.request('screen');
  } catch {
    // Unsupported or refused; the screen may dim during long pauses.
  }
}

// ---------------------------------------------------------------------------
// Per-frame loop

function inputFrame() {
  if (!els.mirrorInput.checked) return els.video;
  const { videoWidth: w, videoHeight: h } = els.video;
  mirrorCanvas ??= document.createElement('canvas');
  if (mirrorCanvas.width !== w || mirrorCanvas.height !== h) {
    mirrorCanvas.width = w;
    mirrorCanvas.height = h;
  }
  const context = mirrorCanvas.getContext('2d');
  context.setTransform(-1, 0, 0, 1, w, 0);
  context.drawImage(els.video, 0, 0, w, h);
  return mirrorCanvas;
}

function tick() {
  requestAnimationFrame(tick);

  const video = els.video;
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const now = performance.now();
  if (lastFrameAt) state.fps = ema(state.fps, 1000 / (now - lastFrameAt));
  lastFrameAt = now;

  const quality = QUALITY_STEPS[qualityIndex];
  const source = inputFrame();
  const hands = handLandmarker.detectForVideo(source, now);
  if (includeFace && frameIndex % quality.face === 0) {
    lastFace = faceLandmarker.detectForVideo(source, now);
  }
  frameIndex++;

  drawHands(hands);
  adaptQuality(now);

  const rows = assembleLandmarks(hands, lastFace, includeFace);
  // ISL letters are two-handed; classifying with one hand real and the other
  // zero-filled (assembleLandmarks' fallback for a missing hand) feeds the
  // model a shape it never saw in training and produces a confident-looking
  // but meaningless prediction instead of no prediction at all. Wait for the
  // mode's actual required hand count instead.
  const requiredHands = meta.num_hands ?? 2;
  const seenHands = hands.landmarks?.length ?? 0;
  if (!rows || seenHands < requiredHands) {
    presenceToken++;
    smoother.clear();
    updateLive(null, 0, false, now);
    els.glyphCaption.textContent = requiredHands > 1 && seenHands > 0 ? 'Show both hands' : 'Show a letter';
    return;
  }

  // Re-classifying isn't needed every tracked frame — a prediction is smoothed over
  // ~5 of them and has to hold for 900ms before it counts anyway. Between actual
  // classify() calls, keep the hold timer and glyph animation advancing at full frame
  // rate off the last known result, so throttling the model doesn't look like stutter.
  if (!classifying && frameIndex % quality.classify === 0) {
    classify(rows);
  } else {
    renderGlyph(updateHold(state.live?.letter ?? null, Boolean(state.live?.ready), now));
  }
}

// Every couple of seconds, nudge toward more or less tracking/classification work
// based on measured fps, so the same code settles near 30fps on both a flagship and a
// budget phone instead of assuming one fixed capability.
function adaptQuality(now) {
  if (now - lastQualityCheck < 2000 || state.fps === 0) return;
  lastQualityCheck = now;

  if (state.fps < TARGET_FPS - 6 && qualityIndex < QUALITY_STEPS.length - 1) {
    qualityIndex++;
  } else if (state.fps > TARGET_FPS - 2 && qualityIndex > 0) {
    qualityIndex--;
  }
}

async function classify(rows) {
  classifying = true;
  const token = presenceToken;
  const started = performance.now();

  tileSequence(normalizeFrame(rows), seqLen, inputBuffer);
  const output = tf.tidy(() => model.predict(tf.tensor3d(inputBuffer, [1, seqLen, featureCount])));
  const probabilities = await output.data();
  output.dispose();

  state.modelMs = ema(state.modelMs, performance.now() - started);
  classifying = false;
  if (token !== presenceToken) return;

  const index = argmax(probabilities);
  const smoothed = smoother.push(index, probabilities[index]);
  updateLive(meta.class_names[smoothed.index], smoothed.confidence, smoothed.confidence > meta.confidence_threshold, performance.now());
}

function drawHands(result) {
  const { width: w, height: h } = els.overlay;
  overlayContext.clearRect(0, 0, w, h);
  if (!els.showLandmarks.checked) return;

  // With mirrored input the landmarks live in flipped space; map them back so they
  // line up with the video underneath.
  const mirrored = els.mirrorInput.checked;
  const toCanvas = (p) => [(mirrored ? 1 - p.x : p.x) * w, p.y * h];

  overlayContext.lineWidth = Math.max(2, w / 240);
  overlayContext.strokeStyle = 'rgba(244, 246, 251, 0.55)';
  overlayContext.fillStyle = '#f4f6fb';
  const radius = Math.max(2.5, w / 180);

  for (const points of result.landmarks ?? []) {
    overlayContext.beginPath();
    for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
      const [x1, y1] = toCanvas(points[start]);
      const [x2, y2] = toCanvas(points[end]);
      overlayContext.moveTo(x1, y1);
      overlayContext.lineTo(x2, y2);
    }
    overlayContext.stroke();

    for (const point of points) {
      const [x, y] = toCanvas(point);
      overlayContext.beginPath();
      overlayContext.arc(x, y, radius, 0, Math.PI * 2);
      overlayContext.fill();
    }
  }
}

// ---------------------------------------------------------------------------
// Live letter and hold-to-add

function updateLive(letter, confidence, ready, now) {
  state.live = letter ? { letter, confidence, ready } : null;
  renderGlyph(updateHold(letter, ready, now));
  updateTutorMatch(letter, confidence);
}

function updateHold(letter, ready, now) {
  if (!ready) {
    state.hold = null;
    state.releasedSince ??= now;
    if (now - state.releasedSince >= RELEASE_TO_REPEAT_MS) state.locked = null;
    return 0;
  }

  state.releasedSince = null;
  if (state.locked && state.locked !== letter) state.locked = null;
  if (state.locked === letter || !els.autoCapture.checked) return 0;

  if (state.hold?.letter !== letter) state.hold = { letter, since: now };
  const progress = Math.min(1, (now - state.hold.since) / HOLD_TO_ADD_MS);
  if (progress === 1) {
    addLetter(letter);
    return 0;
  }
  return progress;
}

function renderGlyph(holdProgress) {
  const { live } = state;
  els.glyph.dataset.state = live ? (live.ready ? 'ready' : 'seeing') : 'idle';
  els.glyph.style.setProperty('--fill', live ? live.confidence.toFixed(3) : '0');
  els.glyph.style.setProperty('--hold', holdProgress.toFixed(3));

  const text = live?.letter ?? '';
  if (els.glyphFill.textContent !== text) {
    els.glyphFill.textContent = text;
    els.glyphOutline.textContent = text;
  }
  els.glyphCaption.textContent = live ? `${Math.round(live.confidence * 100)}% match` : 'Show a letter';

  els.addButton.disabled = !live?.ready;
  els.addButton.textContent = live?.ready ? `Add ${live.letter}` : 'Add letter';
}

// ---------------------------------------------------------------------------
// Spelling tape

function announce(text) {
  els.announcer.textContent = text;
}

// Without WebGL the model falls back to plain JavaScript, which takes seconds per
// prediction instead of ~20 ms — say so rather than letting the app look broken.
const emptyHint = () =>
  tf.getBackend() === 'webgl'
    ? EMPTY_HINT
    : "This browser can't run the model on the GPU, so each letter takes several seconds to appear.";

function renderTape() {
  const { words, letters, sentence } = state;
  if (sentence) {
    els.tapeWords.textContent = sentence;
    els.tapeWords.dataset.kind = 'sentence';
  } else if (words.length) {
    els.tapeWords.textContent = words.join(' ');
    els.tapeWords.dataset.kind = 'words';
  } else {
    els.tapeWords.textContent = letters.length ? '' : emptyHint();
    els.tapeWords.dataset.kind = 'hint';
  }
  els.tapeCurrent.textContent = letters.join('');
  const line = els.tapeCurrent.parentElement;
  line.scrollLeft = line.scrollWidth;
}

function addLetter(letter) {
  state.letters.push(letter);
  state.sentence = '';
  state.locked = letter;
  state.hold = null;
  navigator.vibrate?.(12);
  announce(letter);
  renderTape();
}

function deleteLetter() {
  if (state.letters.length) state.letters.pop();
  else if (state.words.length) state.letters = [...state.words.pop()];
  state.sentence = '';
  renderTape();
}

function addSpace() {
  if (!state.letters.length) return;
  state.words.push(state.letters.join(''));
  state.letters = [];
  state.sentence = '';
  renderTape();
}

function speak() {
  const words = [...state.words, state.letters.join('')].filter(Boolean);
  if (!words.length) return;

  state.sentence = processSequence(words);
  renderTape();
  announce(state.sentence);

  if (window.speechSynthesis) {
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(state.sentence);
    utterance.lang = 'en-IN';
    speechSynthesis.speak(utterance);
  }
}

function clearAll() {
  state.words = [];
  state.letters = [];
  state.sentence = '';
  state.locked = null;
  window.speechSynthesis?.cancel();
  renderTape();
}

// ---------------------------------------------------------------------------
// Status, settings, startup

function showStatus(message, { error = false, retry = false } = {}) {
  els.status.hidden = false;
  els.status.dataset.error = String(error);
  els.statusMessage.textContent = message;
  els.retryButton.hidden = !retry;
}

function describeError(error) {
  switch (error?.name) {
    case 'NotAllowedError':
      return "Camera access is blocked. Allow the camera for this site in your browser's site settings, then try again.";
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No usable camera was found on this device.';
    case 'NotReadableError':
      return 'The camera is being used by another app. Close that app, then try again.';
  }
  if (error?.stage === 'model') {
    return "The sign model didn't load from model/. Start the app with serve_https.py from the mobile folder.";
  }
  if (error?.stage === 'trackers') {
    return "Hand tracking didn't download. The first visit needs an internet connection (about 20 MB).";
  }
  if (error?.stage === 'switch-mode') {
    return `Couldn't switch to ${MODE_LABELS[error.mode] ?? error.mode}. Check your connection and try again.`;
  }
  return `Something went wrong: ${error?.message ?? error}`;
}

function renderReadout() {
  const inputs = includeFace ? 'hands + face' : 'hands only';
  const quality = QUALITY_STEPS[qualityIndex];
  els.readout.textContent =
    `${state.fps.toFixed(0)} fps (target ${TARGET_FPS}) · model ${state.modelMs.toFixed(1)} ms · ` +
    `face 1/${quality.face} · classify 1/${quality.classify} · ${inputs} (${featureCount} inputs) · ${tf.getBackend()}`;
}

// J and Z are motion signs in real ISL/ASL (a drawn hook, a traced Z) — the
// training photos, like the pamphlet photos, can only show one still frame of
// that motion, not the motion itself. Label them rather than presenting them
// as equivalent to the other 24 static letters.
const MOTION_LETTERS = new Set(['J', 'Z']);

const ISL_TUTOR_HINTS = {
  'A': 'Point dominant index finger to touch non-dominant thumb tip.',
  'B': 'Touch tips of both thumbs and index fingers together to form two loops.',
  'C': 'Curve dominant hand into a open "C" shape facing inward/side.',
  'D': 'Dominant index upright while thumb and fingers form a loop.',
  'E': 'Touch dominant index finger to tip of non-dominant index finger.',
  'F': 'Cross index fingers of both hands to form an "F" shape.',
  'G': 'Place both closed fists vertically one on top of the other.',
  'H': 'Lay dominant palm flat across the non-dominant open horizontal palm.',
  'I': 'Touch dominant index finger to tip of non-dominant middle finger.',
  'J': 'Touch middle finger tip and trace dominant hand downward.',
  'K': 'Form a "V" with dominant index/middle fingers touching non-dominant index.',
  'L': 'Open dominant hand into an "L" shape with thumb and index at 90 degrees.',
  'M': 'Place 3 dominant fingertips (index, middle, ring) on non-dominant palm.',
  'N': 'Place 2 dominant fingertips (index, middle) on non-dominant palm.',
  'O': 'Touch dominant index finger to tip of non-dominant ring finger.',
  'P': 'Dominant thumb and index form a circle touching non-dominant index.',
  'Q': 'Hook dominant index finger onto the thumb of the non-dominant hand.',
  'R': 'Hook dominant curved index finger over non-dominant flat palm.',
  'S': 'Hook pinky fingers of both hands together tightly.',
  'T': 'Touch dominant index finger to edge of non-dominant palm below pinky.',
  'U': 'Touch dominant index finger to tip of non-dominant pinky finger.',
  'V': 'Form a clear "V" with dominant index and middle fingers on open base palm.',
  'W': 'Interlock fingers of both hands pointing diagonally upwards.',
  'X': 'Cross both extended index fingers over each other to form an "X".',
  'Y': 'Extend dominant thumb and pinky while tucking middle three fingers.',
  'Z': 'Hold dominant open flat palm vertically facing non-dominant horizontal palm.'
};

let tutorLetter = 'A';

function initTutor() {
  if (!els.tutorSelect) return;
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
  els.tutorSelect.replaceChildren(
    ...letters.map((ltr) => {
      const opt = document.createElement('option');
      opt.value = ltr;
      opt.textContent = `Letter ${ltr}`;
      return opt;
    }),
  );
  updateTutorLetter('A');
}

function updateTutorLetter(letter) {
  tutorLetter = letter;
  if (els.tutorSelect) els.tutorSelect.value = letter;
  if (els.tutorTargetTitle) els.tutorTargetTitle.textContent = `Target: Letter ${letter}`;
  if (els.tutorPreviewImg) els.tutorPreviewImg.src = `pamphlet/${mode}/${letter}.jpg`;
  if (els.tutorHintText) els.tutorHintText.textContent = ISL_TUTOR_HINTS[letter] || `Sign the letter ${letter}.`;
  if (els.ghostGuideImg) els.ghostGuideImg.src = `pamphlet/${mode}/${letter}.jpg`;
  if (els.ghostGuideLabel) {
    els.ghostGuideLabel.classList.remove('matched');
    els.ghostGuideLabel.textContent = `Guide: ${letter}`;
  }
}

function updateTutorMatch(letter, confidence) {
  if (!els.ghostGuide || els.ghostGuide.hidden) return;
  const isMatch = letter === tutorLetter && confidence >= 0.7;
  if (isMatch) {
    els.ghostGuideLabel.classList.add('matched');
    els.ghostGuideLabel.textContent = `🎯 ${tutorLetter} (${Math.round(confidence * 100)}%)`;
    if (els.tutorStatusBadge) {
      els.tutorStatusBadge.classList.add('matched');
      els.tutorStatusIcon.textContent = '🎯';
      els.tutorStatusText.textContent = `EXCELLENT! Matched '${tutorLetter}' (${Math.round(confidence * 100)}%)`;
    }
  } else {
    els.ghostGuideLabel.classList.remove('matched');
    els.ghostGuideLabel.textContent = `Guide: ${tutorLetter}`;
    if (els.tutorStatusBadge) {
      els.tutorStatusBadge.classList.remove('matched');
      els.tutorStatusIcon.textContent = '✋';
      els.tutorStatusText.textContent = letter ? `Target: ${tutorLetter} | Detected: ${letter}` : 'Align hands with guide to practice';
    }
  }
}

function renderPamphlet() {
  els.pamphletSystem.textContent = MODE_LABELS[mode];
  els.pamphletGrid.replaceChildren(
    ...meta.class_names.map((letter) => {
      const card = document.createElement('div');
      card.className = 'pamphlet-card';
      card.title = `Tap to practice letter ${letter}`;
      const img = document.createElement('img');
      img.src = `pamphlet/${mode}/${letter}.jpg`;
      img.alt = MOTION_LETTERS.has(letter)
        ? `${letter} is a motion sign; shown here as a still frame, not the full motion`
        : `How to sign the letter ${letter}`;
      img.loading = 'lazy';
      const label = document.createElement('span');
      label.textContent = letter;
      card.append(img, label);
      if (MOTION_LETTERS.has(letter)) {
        const note = document.createElement('small');
        note.className = 'pamphlet-note';
        note.textContent = 'motion sign — still frame';
        card.append(note);
      }
      card.addEventListener('click', () => {
        els.pamphlet.close();
        updateTutorLetter(letter);
        els.ghostGuide.hidden = !els.tutorGuideToggle.checked;
        els.tutorDialog.showModal();
      });
      return card;
    }),
  );
}

// Switching modes only needs a new (small) classifier — both shipped models
// read the same hands-only landmarks, so the camera and hand tracker keep
// running through the switch instead of restarting.
async function applyMode(nextMode) {
  if (nextMode === mode && meta) return;

  const previous = mode;
  mode = nextMode;
  try {
    localStorage.setItem(MODE_STORAGE_KEY, mode);
  } catch {
    // Private browsing or storage disabled — the choice just won't persist.
  }

  showStatus(`Loading ${MODE_LABELS[mode]}…`);
  try {
    await withStage('switch-mode', loadClassifier(mode)).catch((e) => {
      throw Object.assign(e, { mode });
    });
    if (visionResolver) {
      if (currentNumHands !== (meta.num_hands ?? 2)) await loadHandTracker(visionResolver, meta.num_hands ?? 2);
      if (includeFace && !faceLandmarker) await loadFaceTracker(visionResolver);
    }

    smoother.clear();
    state.live = null;
    state.hold = null;
    state.locked = null;
    presenceToken++;

    els.wordmark.firstChild.textContent = `${MODE_LABELS[mode]} `;
    renderPamphlet();
    els.status.hidden = true;
  } catch (error) {
    mode = previous;
    document.querySelector(`input[name="signSystem"][value="${previous}"]`).checked = true;
    console.error(error);
    showStatus(describeError(error), { error: true, retry: false });
    setTimeout(() => { els.status.hidden = true; }, 2500);
  }
}

function wireControls() {
  els.addButton.addEventListener('click', () => state.live?.ready && addLetter(state.live.letter));
  els.deleteButton.addEventListener('click', deleteLetter);
  els.spaceButton.addEventListener('click', addSpace);
  els.clearButton.addEventListener('click', clearAll);
  els.speakButton.addEventListener('click', speak);
  els.retryButton.addEventListener('click', () => location.reload());

  let readoutTimer = null;
  els.settingsButton.addEventListener('click', () => {
    renderReadout();
    readoutTimer = setInterval(renderReadout, 500);
    els.settings.showModal();
  });
  els.settings.addEventListener('close', () => clearInterval(readoutTimer));

  document.querySelectorAll('input[name="facing"]').forEach((radio) =>
    radio.addEventListener('change', async () => {
      try {
        await startCamera(radio.value);
      } catch (error) {
        showStatus(describeError(error), { error: true, retry: true });
      }
    }),
  );

  document.querySelectorAll('input[name="signSystem"]').forEach((radio) =>
    radio.addEventListener('change', () => applyMode(radio.value)),
  );

  els.pamphletButton.addEventListener('click', () => {
    renderPamphlet();
    els.pamphlet.showModal();
  });

  els.tutorButton?.addEventListener('click', () => {
    els.ghostGuide.hidden = !els.tutorGuideToggle.checked;
    els.tutorDialog.showModal();
  });

  els.tutorPrevBtn?.addEventListener('click', () => {
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    const idx = letters.indexOf(tutorLetter);
    const nextIdx = (idx - 1 + letters.length) % letters.length;
    updateTutorLetter(letters[nextIdx]);
  });

  els.tutorNextBtn?.addEventListener('click', () => {
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    const idx = letters.indexOf(tutorLetter);
    const nextIdx = (idx + 1) % letters.length;
    updateTutorLetter(letters[nextIdx]);
  });

  els.tutorSelect?.addEventListener('change', (e) => {
    updateTutorLetter(e.target.value);
  });

  els.tutorGuideToggle?.addEventListener('change', () => {
    els.ghostGuide.hidden = !els.tutorGuideToggle.checked;
  });

  els.tutorOpacitySlider?.addEventListener('input', (e) => {
    const val = e.target.value;
    if (els.tutorOpacityVal) els.tutorOpacityVal.textContent = `${val}%`;
    if (els.ghostGuideImg) els.ghostGuideImg.style.opacity = (val / 100).toString();
  });

  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && stream) keepScreenAwake();
  });
}

function registerServiceWorker() {
  // Purely an optimization (caches the ~20MB of assets for instant repeat
  // visits) — never block or fail startup over it.
  if (!('serviceWorker' in navigator)) return;
  navigator.serviceWorker.register('sw.js').catch((error) => {
    console.warn('Service worker registration failed (app still works, just without asset caching):', error);
  });
}

async function start() {
  wireControls();
  registerServiceWorker();

  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showStatus('The camera only works over HTTPS. Open the https:// address printed by serve_https.py.', { error: true });
    return;
  }

  document.querySelector(`input[name="signSystem"][value="${mode}"]`).checked = true;
  els.wordmark.firstChild.textContent = `${MODE_LABELS[mode]} `;

  try {
    showStatus(`Loading ${MODE_LABELS[mode]}…`);
    await withStage('model', loadClassifier(mode));
    renderTape();
    renderPamphlet();
    initTutor();

    showStatus('Loading hand tracking… The first visit downloads about 20 MB.');
    await withStage('trackers', loadTrackers());

    showStatus('Starting the camera…');
    await startCamera('user');
    keepScreenAwake();

    els.status.hidden = true;
    // Give the tracker GPU delegates one uncounted frame to finish their first-call
    // shader compile before fps/quality adaptation starts sampling.
    lastQualityCheck = performance.now();
    requestAnimationFrame(tick);
  } catch (error) {
    console.error(error);
    showStatus(describeError(error), { error: true, retry: true });
  }
}

start();
