"""
Train ASL letters as a single-frame MLP, hands-only.

Two combined datasets:
- cristian20a/ASL_Dataset on GitHub (public, no LFS, no Kaggle auth) — real
  photos from multiple signers/backgrounds, ~80-100/letter. A first attempt
  used a different mirror (AI-Datasets/ASL-Alphabet-Dataset) that turned out
  to have MediaPipe's own landmark-overlay graphics burned into every photo
  (the exact corruption this project's local dataset had at the very start)
  — caught by spot-checking samples before the full download this time, not
  after.
- Marxulia/asl_sign_languages_alphabets_v03 on Hugging Face — ~420/letter,
  spot-checked clean. Live-camera testing on the first (cristian20a-only)
  model found the classic ASL fist-shape cluster (S/T confused with M, R/U
  confused with each other) failing in practice despite scoring well on that
  dataset's own small held-out split — a sign of too little data to separate
  those letters robustly, not a pipeline bug (confirmed independently by
  running photos from a third source through the Keras model directly).
  This second dataset's images share a background/signer per letter (more
  volume, less diversity than cristian20a) — combining both trades on their
  different strengths rather than picking one.

Extracts with detect_face=False (skips FaceLandmarker entirely — not needed,
ASL fingerspelling carries no facial grammar, and it's the heaviest MediaPipe
model anyway), then trains the same architecture as scripts/train_mlp_isl.py.

J and Z are motion signs. Neither dataset has J or Z photos (cristian20a has
no J/Z folders at all, confirmed against its GitHub tree listing; the HF set
labels them but this script skips them for consistency — see SKIP_LETTERS).
Rather than fabricate or approximate a still frame as the letter, this trains
a 24-class model (A-I, K-Y) and the app marks J/Z as unavailable in ASL mode
instead of pretending a static photo stands in for a motion sign.

Rotation normalization: a first combined-dataset run (no rotation step) came
out WORSE overall (92.34% -> 83.81% on a held-out split) despite fixing G and
T, because R and U got newly confused with V. The two sources differ in hand
rotation/framing (cristian20a's shots aren't consistently upright; the HF
set's are, but at a different angle), and the existing translate+scale-only
normalization (shared with ISL, in LandmarkExtractor._normalize_single_frame)
has no rotation invariance, so the model had to learn "sideways G" and
"upright G" as unrelated patterns from limited examples of each — exactly
the failure a first live test caught (G confidently predicted as Y). This
rotates each hand block (independently, around its own wrist) so the
wrist-to-middle-MCP vector points the same direction in every sample, before
the existing scale/translate normalization runs on top. Scoped to this
script only — ISL's normalization (and its already-verified model) is
untouched; two-handed ISL signs may encode meaning in relative hand rotation
that a single-hand ASL letter doesn't, so it isn't safe to assume the same
change would help there without separately verifying it.

Writes to results/ASL_MLP/, isolated from the ISL model.
"""
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.landmark_extraction import LandmarkExtractor
from src.train import ISLTrainer

DATASETS = [
    ('cristian20a', Path('data/ASL_Alphabet_v2')),
    ('hf_marxulia', Path('data/ASL_Alphabet_hf')),
]
DST = 'data/landmarks/landmarks_asl_hands_only_mlp_rotated.pkl'
OUT = 'results/ASL_MLP'
SKIP_LETTERS = {'J', 'Z'}  # motion signs; neither source has usable static photos for these
CLASSES = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in SKIP_LETTERS]

WRIST, MIDDLE_MCP = 0, 9  # standard MediaPipe hand landmark indices, local to each 21-point block


def rotate_hand_block_to_canonical(block):
    """Rotate one 21-point (x, y, z) hand block around its own wrist so the
    wrist->middle-MCP vector points straight up in image coordinates. A
    no-op (all zeros stay zero) for the empty hand slot. Only x, y rotate —
    z (depth) is left alone, since this corrects in-image-plane rotation,
    not 3D viewpoint."""
    wrist = block[WRIST].copy()
    rel = block - wrist
    dx, dy = rel[MIDDLE_MCP, 0], rel[MIDDLE_MCP, 1]
    angle = np.arctan2(dy, dx)
    target = np.arctan2(-1.0, 0.0)  # "up" in image coords (y grows downward)
    rot = target - angle
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    out = block.copy()
    out[:, 0] = wrist[0] + rel[:, 0] * cos_r - rel[:, 1] * sin_r
    out[:, 1] = wrist[1] + rel[:, 0] * sin_r + rel[:, 1] * cos_r
    return out


def rotate_to_canonical(landmarks_42):
    landmarks_42 = np.asarray(landmarks_42)
    out = landmarks_42.copy()
    out[0:21] = rotate_hand_block_to_canonical(landmarks_42[0:21])
    out[21:42] = rotate_hand_block_to_canonical(landmarks_42[21:42])
    return out


extractor = LandmarkExtractor(static_mode=True, max_hands=2, detect_face=False)

all_landmarks, all_labels, all_groups = [], [], []
for class_idx, letter in enumerate(CLASSES):
    files = []
    for source, root in DATASETS:
        files.extend((source, f) for f in sorted((root / letter).glob('*.jpg')))
    if not files:
        print(f'WARN: no images for {letter}')
        continue

    detected = 0
    for source, f in tqdm(files, desc=letter):
        image = cv2.imread(str(f))
        if image is None:
            continue
        landmarks = extractor.extract_from_image(image)
        if landmarks is not None:
            rotated = rotate_to_canonical(landmarks[:42])  # hands only, drop the (unrequested) face slice
            all_landmarks.append(rotated)
            all_labels.append(class_idx)
            all_groups.append(f'{letter}_{source}_{f.stem}')  # independent photos -> unique per file
            detected += 1

    rate = detected / len(files)
    print(f'  {letter}: {detected}/{len(files)} detected ({rate:.0%})')
    if rate < 0.5:
        print(f'  [WARN] low detection rate for {letter}')

extractor.close()

if not all_landmarks:
    print('No hands detected in any image — aborting.')
    sys.exit(1)

Path(DST).parent.mkdir(parents=True, exist_ok=True)
with open(DST, 'wb') as f:
    pickle.dump({
        'landmarks': all_landmarks,
        'labels': np.array(all_labels),
        'groups': all_groups,
        'class_names': CLASSES,
        'num_classes': len(CLASSES),
        'input_features': 126,
        'mode': 'hands_only_mlp',
    }, f)
print(f'Saved {len(all_landmarks)} samples -> {DST}')

trainer = ISLTrainer('config.yaml')
cfg = trainer.config.config
cfg['model']['type'] = 'MLP'
cfg['model']['sequence_length'] = 1
cfg['model']['input_features'] = 126
cfg['model']['num_classes'] = len(CLASSES)
cfg['paths']['model_save_dir'] = OUT
cfg['paths']['logs_dir'] = f'{OUT}/logs'
cfg['paths']['checkpoints_dir'] = f'{OUT}/checkpoints'

history, accuracy = trainer.train(DST, time_budget_minutes=30)
print(f'ASL_MLP_EPOCHS: {len(history.history["loss"])}')
print(f'ASL_MLP_TEST_ACCURACY: {accuracy}')
