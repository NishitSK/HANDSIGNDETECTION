// Scores a converted TF.js model on the held-out test split exported by
// export_reference.py and measures how far it drifts from Keras. Shared by
// verify_pipeline.mjs (Node) and parity.html (browser).
import { normalizeFrame, tileSequence, argmax } from '../app/pipeline.mjs';

const MAX_REPORTED_DISAGREEMENTS = 12;

// seqLen isn't a global constant anymore — different models declare different
// window lengths (1 for the single-frame MLPs, 30 for the legacy TCN), so read
// it off the model actually being tested rather than assuming one.
function seqLenOf(model) {
  const shape = model.inputs[0].shape;
  const len = shape.length > 2 ? shape.at(-2) : 1;
  return len > 0 ? len : 1;
}

export async function runParity({
  tf,
  model,
  meta,
  pythonFrames,
  rawLandmarks,
  kerasProbs,
  featureCount,
  limit = meta.n,
  batchSize = 128,
  onProgress = () => {},
}) {
  const n = Math.min(limit, meta.n);
  const numClasses = meta.num_classes;
  const points = featureCount / 3;
  // Legacy exports (the 62-point hands+face ISL cache) have no raw_points/
  // raw_frame_width fields on disk — default to the widths those files were
  // always written with.
  const rawPoints = meta.raw_points ?? 62;
  const rawFrameWidth = meta.raw_frame_width ?? 186;
  const seqLen = seqLenOf(model);
  const started = performance.now();

  let maxFeatureDiff = 0;
  let maxProbDiff = 0;
  let maxAbsFeature = 0;
  let agreeWithKeras = 0;
  let correctKeras = 0;
  let correctFromPythonFeatures = 0;
  let correctFullJsPipeline = 0;
  const disagreements = [];

  const predict = async (data, batch) => {
    const output = tf.tidy(() => model.predict(tf.tensor3d(data, [batch, seqLen, featureCount])));
    const values = await output.data();
    output.dispose();
    return values;
  };

  const largest = (values) => values.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

  for (let start = 0; start < n; start += batchSize) {
    const batch = Math.min(batchSize, n - start);
    const pythonBatch = new Float32Array(batch * seqLen * featureCount);
    const jsBatch = new Float32Array(batch * seqLen * featureCount);

    for (let k = 0; k < batch; k++) {
      const i = start + k;
      const pythonFrame = pythonFrames.subarray(i * rawFrameWidth, i * rawFrameWidth + featureCount);
      maxAbsFeature = Math.max(maxAbsFeature, largest(pythonFrame));

      const rows = [];
      for (let p = 0; p < points; p++) {
        const o = i * rawPoints * 3 + p * 3;
        rows.push([rawLandmarks[o], rawLandmarks[o + 1], rawLandmarks[o + 2]]);
      }
      const jsFrame = normalizeFrame(rows);
      for (let f = 0; f < featureCount; f++) {
        maxFeatureDiff = Math.max(maxFeatureDiff, Math.abs(jsFrame[f] - pythonFrame[f]));
      }

      pythonBatch.set(tileSequence(pythonFrame, seqLen), k * seqLen * featureCount);
      jsBatch.set(tileSequence(jsFrame, seqLen), k * seqLen * featureCount);
    }

    const fromPython = await predict(pythonBatch, batch);
    const fromJs = await predict(jsBatch, batch);

    for (let k = 0; k < batch; k++) {
      const i = start + k;
      const label = meta.y_test[i];
      const reference = kerasProbs.subarray(i * numClasses, (i + 1) * numClasses);
      const p = fromPython.subarray(k * numClasses, (k + 1) * numClasses);
      const j = fromJs.subarray(k * numClasses, (k + 1) * numClasses);
      for (let c = 0; c < numClasses; c++) maxProbDiff = Math.max(maxProbDiff, Math.abs(p[c] - reference[c]));

      const kerasTop = argmax(reference);
      const tfjsTop = argmax(p);
      if (tfjsTop === kerasTop) {
        agreeWithKeras++;
      } else if (disagreements.length < MAX_REPORTED_DISAGREEMENTS) {
        disagreements.push({
          sample: i,
          label: meta.class_names[label],
          keras: `${meta.class_names[kerasTop]} ${reference[kerasTop].toFixed(4)}`,
          tfjs: `${meta.class_names[tfjsTop]} ${p[tfjsTop].toFixed(4)}`,
          kerasProbOfTfjsLetter: +reference[tfjsTop].toFixed(4),
          largestInputFeature: +largest(pythonFrames.subarray(i * rawFrameWidth, i * rawFrameWidth + featureCount)).toFixed(2),
        });
      }
      if (kerasTop === label) correctKeras++;
      if (tfjsTop === label) correctFromPythonFeatures++;
      if (argmax(j) === label) correctFullJsPipeline++;
    }
    onProgress((start + batch) / n);
  }

  return {
    samples: n,
    backend: tf.getBackend(),
    featureCount,
    seqLen,
    maxFeatureDiffJsVsPython: maxFeatureDiff,
    maxProbDiffTfjsVsKeras: maxProbDiff,
    largestInputFeature: maxAbsFeature,
    argmaxAgreementWithKeras: agreeWithKeras / n,
    kerasAccuracy: correctKeras / n,
    tfjsAccuracyFromPythonFeatures: correctFromPythonFeatures / n,
    tfjsAccuracyFullJsPipeline: correctFullJsPipeline / n,
    seconds: (performance.now() - started) / 1000,
    disagreements,
  };
}
