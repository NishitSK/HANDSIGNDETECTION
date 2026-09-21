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
export function assembleLandmarks(handResult, faceResult, includeFace, mode = 'isl') {
  const candidates = assembleLandmarkCandidates(handResult, faceResult, includeFace, mode);
  return candidates ? candidates[0] : null;
}

// When only 1 hand is visible:
// - In ASL (single-handed 24-letter alphabet A-Y), the model was trained with the hand
//   in the LEFT slot (rows 0..20) and zeros in the right slot. Normalization is anchored
//   to row 0 (the wrist). Placing zeros in the left slot breaks normalization (size=0,
//   scale=1.0) and creates spurious out-of-distribution logits that corrupt detection.
//   Therefore, in ASL mode, we strictly assemble single hands into the LEFT slot.
//   If multiple hands are visible, each is evaluated as a single hand candidate in the left slot.
// - In ISL, single-handed signs (e.g. C, I, L, O, U, V) were trained in the RIGHT slot,
//   while two-handed signs use both slots. Returning candidates for both slots allows
//   evaluating both in a single parallel batch so single-handed signs always match.
export function assembleLandmarkCandidates(handResult, faceResult, includeFace, mode = 'isl') {
  const hands = handResult?.landmarks ?? [];
  if (!hands.length) return null;

  const faceRows = [];
  if (includeFace) {
    const face = faceResult?.faceLandmarks?.[0];
    if (face) {
      for (const idx of FACE_KEY_INDICES) faceRows.push([face[idx].x, face[idx].y, face[idx].z]);
    } else {
      faceRows.push(...zeroRows(FACE_KEY_INDICES.length));
    }
  }

  if (mode === 'asl') {
    if (hands.length === 1) {
      const single = hands[0].map((p) => [p.x, p.y, p.z]);
      return [[...single, ...zeroRows(HAND_POINTS), ...faceRows]];
    }
    return hands.map((h) => [...h.map((p) => [p.x, p.y, p.z]), ...zeroRows(HAND_POINTS), ...faceRows]);
  }

  // ISL mode:
  if (hands.length === 1) {
    const single = hands[0];
    const wrist = single[0];
    // Scale by palm length (wrist to middle knuckle MCP, landmark 9) rather than
    // finger tip (landmark 12) so gestures with extended fingers (like V) aren't shrunk.
    const mcp = single[9] ?? single[12];
    const curPalm = Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y, (mcp.z || 0) - (wrist.z || 0));
    const scale = curPalm > 1e-4 ? (0.423 / curPalm) : 1;
    const alignedSingle = single.map((p) => [
      (p.x - wrist.x) * scale + 0.76442,
      (p.y - wrist.y) * scale + 0.78281,
      ((p.z || 0) - (wrist.z || 0)) * scale,
    ]);
    const rightSlotCandidate = [...zeroRows(HAND_POINTS), ...alignedSingle, ...faceRows];
    return [rightSlotCandidate];
  }

  let left = null;
  let right = null;
  const handedness = handResult?.handedness ?? handResult?.handednesses ?? [];
  const trainingLabel = { Left: 'Right', Right: 'Left' };
  hands.forEach((points, i) => {
    const rows = points.map((p) => [p.x, p.y, p.z]);
    if (trainingLabel[handedness[i]?.[0]?.categoryName] === 'Left') left = rows;
    else right = rows;
  });

  const twoHandedRows = [
    ...(left ?? zeroRows(HAND_POINTS)),
    ...(right ?? zeroRows(HAND_POINTS)),
    ...faceRows,
  ];
  return [twoHandedRows];
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
