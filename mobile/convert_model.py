"""
Convert a trained Keras model into the TF.js graph model the mobile app loads.

Run it with the converter environment (TensorFlow 2.15 + tensorflowjs), not the
training venv:

    ../.venv-convert/Scripts/python.exe mobile/convert_model.py models/saved/isl_model.h5

Weights are stored as float16, halving the download. Class names come from the
model_metadata.json that training writes next to the .h5. Check the result against
Keras with tests/export_reference.py + tests/verify_pipeline.mjs.
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

# TensorFlow's Windows build (oneDNN) rewrites LayerNormalization into an
# `_MklLayerNorm` kernel during graph optimization, which TF.js can't run.
# Must be set before TensorFlow is imported.
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / 'app' / 'model'
DEFAULT_SETTINGS = {'confidence_threshold': 0.6, 'smoothing_window': 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('keras_model', type=Path, help='trained .h5 model, e.g. models/saved/isl_model.h5')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT, help='directory the app loads the model from')
    parser.add_argument('--no-quantize', action='store_true', help='keep float32 weights')
    args = parser.parse_args()

    sidecar = args.keras_model.parent / 'model_metadata.json'
    if not sidecar.exists():
        raise SystemExit(f'Expected class names in {sidecar} (written by training next to the model).')

    from tensorflow import keras
    from tensorflowjs.converters.converter import convert

    model = keras.models.load_model(args.keras_model, compile=False)
    feature_count = model.input_shape[-1]

    # Keras's fused LayerNormalization exports as a training-mode FusedBatchNormV3
    # with empty mean/variance. TF.js only implements inference-mode batch norm, so
    # that node crashes on WebGL and computes nonsense elsewhere. The non-fused path
    # is the same math written with ordinary mean/variance ops.
    for layer in model.submodules:
        if isinstance(layer, keras.layers.LayerNormalization):
            layer._fused = False

    # mkdtemp + ignore_errors: on Windows TensorFlow can keep SavedModel files open
    # until exit, which makes TemporaryDirectory's cleanup raise.
    workdir = Path(tempfile.mkdtemp(prefix='isl_convert_'))
    try:
        saved_model = workdir / 'saved_model'
        staging = workdir / 'tfjs'
        model.export(str(saved_model))

        flags = ['--input_format=tf_saved_model', '--output_format=tfjs_graph_model',
                 '--signature_name=serving_default', '--saved_model_tags=serve']
        if not args.no_quantize:
            flags.append('--quantize_float16=*')
        convert(flags + [str(saved_model), str(staging)])

        graph = json.loads((staging / 'model.json').read_text())
        weight_shapes = {w['name']: w['shape'] for group in graph['weightsManifest'] for w in group['weights']}

        def has_empty_mean(node):
            inputs = [name.lstrip('^').split(':')[0] for name in node['input']]
            return len(inputs) > 3 and weight_shapes.get(inputs[3]) == [0]

        training_batch_norms = [
            node['name'] for node in graph['modelTopology']['node']
            if node['op'].startswith('FusedBatchNorm')
            and (node.get('attr', {}).get('is_training', {}).get('b') or has_empty_mean(node))
        ]
        if training_batch_norms:
            raise SystemExit('Converted graph still contains training-mode batch norm, which TF.js '
                             f'cannot run: {training_batch_norms}')

        args.out.mkdir(parents=True, exist_ok=True)
        # Remove the previous model first so stale weight shards can't sit next to a new model.json.
        for stale in [args.out / 'model.json', *args.out.glob('group*-shard*of*.bin')]:
            stale.unlink(missing_ok=True)
        for produced in staging.iterdir():
            shutil.copy(produced, args.out / produced.name)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    metadata_path = args.out / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else dict(DEFAULT_SETTINGS)
    metadata['class_names'] = json.loads(sidecar.read_text())['class_names']
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')

    size_kb = sum(f.stat().st_size for f in args.out.iterdir() if f.name != 'metadata.json') / 1024
    print(f'Wrote {args.out}: {feature_count} features per frame, '
          f'{len(metadata["class_names"])} classes, {size_kb:.0f} KB')


if __name__ == '__main__':
    main()
