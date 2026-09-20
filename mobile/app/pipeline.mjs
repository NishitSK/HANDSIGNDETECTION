// Browser port of the feature pipeline in src/landmark_extraction.py and
// src/inference.py. The model only works if it gets exactly the features it was
// trained on, so this mirrors the Python step for step, including its quirks
// (e.g. normalization anchored to the left wrist even when the left hand is
// missing). tests/verify_pipeline.mjs checks it against the Python output.

export const HAND_POINTS = 21;
export const FACE_KEY_INDICES = [10, 152, 234, 454, 4, 1, 33, 263, 61, 291, 199, 6, 168, 8, 9, 151, 337, 299, 69, 104];

const HANDS_ONLY_FEATURES = HAND_POINTS * 2 * 3;
const HANDS_AND_FACE_FEATURES = HANDS_ONLY_FEATURES + FACE_KEY_INDICES.length * 3;

export function featureLayout(featureCount) {
  if (featureCount === HANDS_AND_FACE_FEATURES) return { includeFace: true };
  if (featureCount === HANDS_ONLY_FEATURES) return { includeFace: false };
  throw new Error(`Model expects ${featureCount} features per frame; this pipeline supports ${HANDS_ONLY_FEATURES} or ${HANDS_AND_FACE_FEATURES}.`);
}

const zeroRows = (count) => Array.from({ length: count }, () => [0, 0, 0]);

// Returns rows of [x, y, z]: left hand, right hand, then optionally 20 face points.
// null when no hand is visible, matching extract_from_image().
export function assembleLandmarks(handResult, faceResult, includeFace) {
  let left = null;
  let right = null;
  const hands = handResult?.landmarks ?? [];
  // Older tasks-vision releases name this field `handednesses`.
  const handedness = handResult?.handedness ?? handResult?.handednesses ?? [];

  // MediaPipe Web labels handedness the opposite way to the legacy Python
  // `solutions.hands` API that produced the training landmarks: on held-out photos
  // 48 of 52 hands landed in the other slot (tests/mediapipe_parity.html). Map the
  // web labels back to the training convention.
  const trainingLabel = { Left: 'Right', Right: 'Left' };
  hands.forEach((points, i) => {
    const rows = points.map((p) => [p.x, p.y, p.z]);
    if (trainingLabel[handedness[i]?.[0]?.categoryName] === 'Left') left = rows;
    else right = rows;
  });

  if (!left && !right) return null;

  const rows = [...(left ?? zeroRows(HAND_POINTS)), ...(right ?? zeroRows(HAND_POINTS))];
  if (includeFace) {
    const face = faceResult?.faceLandmarks?.[0];
    if (face) {
      for (const idx of FACE_KEY_INDICES) rows.push([face[idx].x, face[idx].y, face[idx].z]);
    } else {
      rows.push(...zeroRows(FACE_KEY_INDICES.length));
    }
  }
  return rows;
}

export function normalizeFrame(rows) {
  const [ox, oy, oz] = rows[0];
  const size = Math.hypot(rows[12][0] - ox, rows[12][1] - oy, rows[12][2] - oz);
  const out = new Float32Array(rows.length * 3);
  rows.forEach(([x, y, z], i) => {
    const scale = size > 0 ? size : 1;
    out[i * 3] = (x - ox) / scale;
    out[i * 3 + 1] = (y - oy) / scale;
    out[i * 3 + 2] = (z - oz) / scale;
  });
  return out;
}

// Training sequences are a single photo repeated across the window, so the live
// input is built the same way rather than from N consecutive (changing) frames.
// `length` isn't a shared constant — it's per-model (1 for the single-frame MLPs,
// 30 for the legacy TCN) — so callers must read it off the loaded model and pass
// it explicitly rather than relying on a default that could silently mismatch.
// Pass `out` (sized frame.length * length) to fill a reused buffer instead of
// allocating a fresh one — matters on a hot per-frame path on a phone, where
// GC pauses are longer and more frequent than on desktop.
export function tileSequence(frame, length, out = new Float32Array(frame.length * length)) {
  for (let t = 0; t < length; t++) out.set(frame, t * frame.length);
  return out;
}

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

// Same smoothing as ISLInference.predict_gesture: majority vote over the last few
// predictions once at least 3 exist, confidence averaged over the winning votes.
export class PredictionSmoother {
  constructor(windowSize = 5) {
    this.windowSize = windowSize;
    this.buffer = [];
  }

  push(index, confidence) {
    this.buffer.push([index, confidence]);
    if (this.buffer.length > this.windowSize) this.buffer.shift();
    if (this.buffer.length < 3) return { index, confidence };

    const counts = new Map();
    for (const [i] of this.buffer) counts.set(i, (counts.get(i) ?? 0) + 1);
    let winner = null;
    let winnerCount = -1;
    for (const [i, count] of counts) {
      if (count > winnerCount) {
        winner = i;
        winnerCount = count;
      }
    }
    const votes = this.buffer.filter(([i]) => i === winner).map(([, c]) => c);
    return { index: winner, confidence: votes.reduce((a, b) => a + b, 0) / votes.length };
  }

  clear() {
    this.buffer = [];
  }
}
