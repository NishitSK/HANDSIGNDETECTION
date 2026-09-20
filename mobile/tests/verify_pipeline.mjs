// Node wrapper for parity_core.mjs. Node only has the pure-JS CPU backend here (the
// WASM backend lacks the Einsum kernel the attention layer needs), which takes seconds
// per prediction on this model. Use --limit for a quick check, or open parity.html in
// a desktop browser to score the whole split on WebGL, the backend phones use.
//
//   TFJS_DIR=<dir containing node_modules/@tensorflow/tfjs> \
//   node verify_pipeline.mjs --model <tfjs model dir> --data <results/mobile_verification> --features 186|126 [--limit 50]
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { runParity } from './parity_core.mjs';

const arg = (name) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 ? process.argv[i + 1] : undefined;
};

const modelDir = arg('model');
const dataDir = arg('data');
const featureCount = Number(arg('features') ?? 186);
const limit = arg('limit') ? Number(arg('limit')) : undefined;
if (!modelDir || !dataDir || !process.env.TFJS_DIR) {
  console.error('usage: TFJS_DIR=... node verify_pipeline.mjs --model <dir> --data <dir> --features 186|126 [--limit N]');
  process.exit(2);
}

const require = createRequire(path.join(process.env.TFJS_DIR, 'package.json'));
const tf = require('@tensorflow/tfjs');
await tf.setBackend('cpu');
await tf.ready();

// Copy into a fresh buffer: Node's pooled Buffers can sit at offsets a Float32Array can't view.
const readFloat32 = (file) => new Float32Array(new Uint8Array(readFileSync(file)).buffer);

function fileModelHandler(dir) {
  return {
    async load() {
      const json = JSON.parse(readFileSync(path.join(dir, 'model.json'), 'utf8'));
      const shards = json.weightsManifest.flatMap((group) => group.paths).map((p) => readFileSync(path.join(dir, p)));
      const weights = Buffer.concat(shards);
      return {
        modelTopology: json.modelTopology,
        format: json.format,
        generatedBy: json.generatedBy,
        convertedBy: json.convertedBy,
        signature: json.signature,
        userDefinedMetadata: json.userDefinedMetadata,
        modelInitializer: json.modelInitializer,
        weightSpecs: json.weightsManifest.flatMap((group) => group.weights),
        weightData: weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength),
      };
    },
  };
}

const result = await runParity({
  tf,
  model: await tf.loadGraphModel(fileModelHandler(modelDir)),
  meta: JSON.parse(readFileSync(path.join(dataDir, 'meta.json'), 'utf8')),
  pythonFrames: readFloat32(path.join(dataDir, 'x_test_frames_186.f32')),
  rawLandmarks: readFloat32(path.join(dataDir, 'raw_landmarks_62x3.f32')),
  kerasProbs: readFloat32(path.join(dataDir, featureCount === 186 ? 'keras_probs_full.f32' : 'keras_probs_hands_only.f32')),
  featureCount,
  limit,
  batchSize: 16,
});

console.log(JSON.stringify(result, null, 2));
