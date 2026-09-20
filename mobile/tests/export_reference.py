"""
Export the held-out test split plus Keras reference predictions for
verify_pipeline.mjs.

Reproduces the split in ISLTrainer.train() exactly (same expressions, same seeds),
so the browser model is scored on the very samples the Keras model was. The printed
accuracy should match the training log; if it doesn't, the split has drifted and the
comparison isn't valid.

    ../.venv/Scripts/python.exe mobile/tests/export_reference.py --model models/saved/isl_model.h5 --features 186
    ../.venv/Scripts/python.exe mobile/tests/export_reference.py --model results/ISL_MLP/isl_model.h5 --features 126 --seq-len 1
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LANDMARKS = ROOT / 'data' / 'landmarks' / 'landmarks_letter_both_hands_face.pkl'
OUT = ROOT / 'results' / 'mobile_verification'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--features', type=int, choices=[186, 126], required=True,
                        help='186 = hands + face, 126 = hands only')
    parser.add_argument('--seq-len', type=int, default=30,
                        help='sequence_length the model was trained with (30 = legacy '
                             'tiled-TCN models, 1 = single-frame MLP models)')
    parser.add_argument('--landmarks', type=Path, default=LANDMARKS,
                        help='landmarks pickle to load (default: the ISL both-hands+face cache)')
    parser.add_argument('--out-dir', type=Path, default=OUT,
                        help='directory to write reference exports to (default: results/mobile_verification)')
    args = parser.parse_args()
    out_dir = args.out_dir
    model_path = args.model.resolve()

    # ISLTrainer resolves config.yaml and data paths relative to the project root.
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from tensorflow import keras
    from src.train import ISLTrainer

    trainer = ISLTrainer('config.yaml')
    dataset = trainer.load_data(str(args.landmarks))
    labels = np.array(dataset['labels'])
    groups = np.array(trainer.groups)
    n = len(labels)

    test_frac = 1 - trainer.config.get('data', 'train_test_split', default=0.8)
    use_groups = all(len(np.unique(groups[labels == c])) >= 3 for c in np.unique(labels))
    if use_groups:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
        _, test_idx = next(splitter.split(np.zeros((n, 1)), labels, groups))
    else:
        _, test_idx = train_test_split(np.arange(n), test_size=test_frac, random_state=42, stratify=labels)

    test_landmarks = [dataset['landmarks'][i] for i in test_idx]
    X_test, y_test = trainer.prepare_sequences(test_landmarks, labels[test_idx], args.seq_len)
    if not np.all(X_test == X_test[:, :1, :]):
        raise SystemExit('Expected every test sequence to be one frame repeated across the window.')

    # The hands occupy the first 42 landmarks, and normalization is anchored to
    # hand points only, so the first 126 features equal a hands-only extraction.
    model = keras.models.load_model(model_path, compile=False)
    probs = model.predict(X_test[:, :, :args.features], verbose=0).astype(np.float32)
    accuracy = float(np.mean(np.argmax(probs, axis=1) == y_test))

    # Consumers (parity.html, mediapipe_parity.html, verify_pipeline.mjs) hardcode
    # these two filenames regardless of point/feature count — keep them fixed and
    # separate ISL vs ASL by --out-dir instead, so both parity setups keep working
    # unmodified.
    out_dir.mkdir(parents=True, exist_ok=True)
    np.stack([np.asarray(l) for l in test_landmarks]).astype(np.float32).tofile(out_dir / 'raw_landmarks_62x3.f32')
    X_test[:, 0, :].astype(np.float32).tofile(out_dir / 'x_test_frames_186.f32')
    suffix = 'full' if args.features == 186 else 'hands_only'
    probs.tofile(out_dir / f'keras_probs_{suffix}.f32')

    # Each sample's source photo is embedded in its group id as "<class>_session<path>"
    # (train.py's fallback when filenames carry no timestamp). mediapipe_parity.html
    # re-runs MediaPipe Web on those photos.
    class_names = list(dataset['class_names'])
    test_images = []
    for i in test_idx:
        prefix = f'{class_names[labels[i]]}_session'
        group = str(trainer.groups[i])
        test_images.append(group[len(prefix):] if group.startswith(prefix) else '')
    if not all(p.lower().endswith(('.jpg', '.jpeg', '.png')) for p in test_images):
        print('Note: group ids do not contain photo paths, so mediapipe_parity.html cannot run.')
        test_images = None

    meta_path = out_dir / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        'n': int(len(test_idx)),
        'num_classes': int(probs.shape[1]),
        'class_names': class_names,
        'test_images': test_images,
        'y_test': [int(v) for v in y_test],
        'test_idx': [int(v) for v in test_idx],
        'use_groups': bool(use_groups),
        'keras_test_accuracy' if args.features == 186 else 'keras_hands_only_test_accuracy': accuracy,
        # Datasets that were extracted hands-only from the start (e.g. ASL, which
        # never had a face slice to begin with) store fewer raw points per frame
        # than the legacy 62-point (hands+face) ISL cache that x_test_frames_186.f32
        # and raw_landmarks_62x3.f32 are named after. parity_core.mjs reads these
        # widths instead of assuming 62/186, defaulting to the legacy values so old
        # exports (with no such fields on disk) keep behaving exactly as before.
        'raw_points': int(np.asarray(test_landmarks[0]).shape[0]),
        'raw_frame_width': int(X_test.shape[-1]),
    })
    meta_path.write_text(json.dumps(meta))

    print(f'Keras accuracy on {len(test_idx)} held-out samples ({args.features} features): {accuracy:.6f}')


if __name__ == '__main__':
    main()
