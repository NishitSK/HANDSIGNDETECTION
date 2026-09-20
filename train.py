"""
Unified ISL Training Script - Train Any Gesture/Phrase
Supports: Single hand, Both hands, Face detection
Features: 63 (single hand) or 186 (both hands + face)
"""
import sys
from pathlib import Path
from src.train import ISLTrainer
from src.landmark_extraction import LandmarkExtractor
import pickle
import numpy as np
from tqdm import tqdm
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================

# Choose training mode
TRAINING_MODE = "both_hands_face"  # Options: "single_hand", "both_hands_face"

# Choose what to train
TRAIN_TYPE = "letter"

# Define classes based on type
if TRAIN_TYPE == "letter":
    CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # All 26 letters
    
elif TRAIN_TYPE == "number":
    CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
elif TRAIN_TYPE == "phrase":
    CLASSES = ['Good morning', 'Good night', 'See you later',
               'How are you', 'Thank you', 'Thank_You', 'You', 'How',
               'Hello', 'Sorry', 'Please', 'Help', 'Good',
               'Welcome', 'Name', 'Fine', 'Happy', 'Understand',
               'What', 'Where', 'Who', 'When', 'Why']
    
elif TRAIN_TYPE == "custom":
    # Add your custom classes here
    CLASSES = ['A', 'B', 'C']

# Model configuration
EPOCHS = 40  # Bounded — TargetAccuracyCallback/EarlyStopping can still cut this short
BATCH_SIZE = 16  # Good batch size for LSTM
LEARNING_RATE = 0.001  # Standard learning rate for LSTM

# Feature count based on mode
if TRAINING_MODE == "single_hand":
    INPUT_FEATURES = 63  # 21 landmarks * 3 coords
    MAX_HANDS = 1
    DETECT_FACE = False
elif TRAINING_MODE == "both_hands_face":
    INPUT_FEATURES = 186  # (21*2 hands + 20 face) * 3 coords
    MAX_HANDS = 2
    DETECT_FACE = True

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

print("\n" + "="*70)
print("  ASL UNIFIED TRAINING SYSTEM")
print("="*70)
print(f"\nConfiguration:")
print(f"  - Training Mode: {TRAINING_MODE}")
print(f"  - Type: {TRAIN_TYPE}")
print(f"  - Classes: {len(CLASSES)}")
print(f"  - Input Features: {INPUT_FEATURES}")
print(f"  - Hands: {MAX_HANDS}")
print(f"  - Face Detection: {DETECT_FACE}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Batch Size: {BATCH_SIZE}")

print(f"\nClasses to train:")
for i, cls in enumerate(CLASSES, 1):
    print(f"  {i:2d}. {cls}")

# ============================================================================
# LANDMARK EXTRACTION
# ============================================================================

print("\n" + "="*70)
print("  EXTRACTING LANDMARKS")
print("="*70)

extractor = LandmarkExtractor(
    static_mode=True, 
    max_hands=MAX_HANDS,
    detect_face=DETECT_FACE
)

def _derive_capture_sessions(image_files, gap_threshold_sec=2.0):
    """
    Group images captured seconds apart (a single burst / hold in front of
    the camera) into the same "session". Files are named Image_<unix_ts>.jpg
    by the data collector, so a burst of ~80 photos taken in 3 seconds are
    near-duplicate frames of the SAME gesture instance.

    Splitting such bursts randomly at the frame level lets near-identical
    frames of one instance land in both train and test, which inflates
    reported accuracy without reflecting real generalization. Returning a
    session id per file lets the trainer split by session (group) instead.

    Falls back to a unique session per file (no leakage protection, but no
    crash) if filenames don't carry a parseable timestamp.
    """
    import re

    parsed = []
    for path in image_files:
        match = re.search(r'([0-9]+\.[0-9]+)', path.stem)
        parsed.append((path, float(match.group(1)) if match else None))

    if any(ts is None for _, ts in parsed):
        # Can't reliably order/cluster — every file is its own session.
        return {path: str(path) for path, _ in parsed}

    parsed.sort(key=lambda item: item[1])

    session_map = {}
    session_idx = 0
    prev_ts = None
    for path, ts in parsed:
        if prev_ts is not None and (ts - prev_ts) > gap_threshold_sec:
            session_idx += 1
        session_map[path] = session_idx
        prev_ts = ts

    return session_map


import yaml as _yaml
with open('config.yaml', 'r') as _f:
    _cfg = _yaml.safe_load(_f)
dataset_path = Path(_cfg.get('data', {}).get('dataset_path', 'data/ISL_DATASETS_2'))
print(f"\nDataset path (from config.yaml data.dataset_path): {dataset_path}")
all_landmarks = []
all_labels = []
all_groups = []  # capture-session id per sample, used for leakage-free splitting

for class_idx, class_name in enumerate(CLASSES):
    folder_path = dataset_path / class_name

    if not folder_path.exists():
        print(f"⚠ Skipping {class_name} - folder not found")
        continue

    # Get all image files
    image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))

    # Also check frames subfolder (from extracted videos)
    frames_folder = folder_path / 'frames'
    if frames_folder.exists():
        image_files.extend(list(frames_folder.glob('*.jpg')))
        image_files.extend(list(frames_folder.glob('*.png')))

    if not image_files:
        print(f"⚠ No images found for {class_name}")
        continue

    print(f"\n📸 Processing: {class_name} ({len(image_files)} images)")

    session_map = _derive_capture_sessions(image_files)
    detected_count = 0

    for img_path in tqdm(image_files, desc=f"  {class_name}"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        landmarks = extractor.extract_from_image(image)
        if landmarks is not None:
            all_landmarks.append(landmarks)
            all_labels.append(class_idx)
            all_groups.append(f"{class_name}_session{session_map[img_path]}")
            detected_count += 1

    detection_rate = detected_count / len(image_files) if image_files else 0.0
    print(f"  Hand detected in {detected_count}/{len(image_files)} images ({detection_rate:.0%})")
    if detection_rate < 0.5:
        print(f"  [WARN] Low detection rate for '{class_name}'. Common causes:")
        print(f"         - Images already have a landmark overlay drawn on them (re-detection")
        print(f"           fails on annotated pixels) — recollect with a tool that saves the")
        print(f"           RAW camera frame, not the visualization frame.")
        print(f"         - detection.min_detection_confidence in config.yaml is too strict for")
        print(f"           tightly-cropped or low-light photos — try lowering it.")

if not all_landmarks:
    print("\n❌ No hand landmarks could be extracted from ANY image.")
    print("   Nothing to train on — see the per-class detection-rate warnings above")
    print("   for likely causes. Fix the data/config issue and re-run before training.")
    sys.exit(1)

# Convert to numpy arrays
X = np.array(all_landmarks)
y = np.array(all_labels)

print(f"\n✅ Extraction Complete:")
print(f"  - Total samples: {len(X)}")
print(f"  - Shape: {X.shape}")
print(f"  - Classes found: {len(set(y))}")

# Class distribution
print(f"\nSample distribution:")
for i, cls in enumerate(CLASSES):
    count = np.sum(y == i)
    if count > 0:
        print(f"  {cls:20s}: {count:4d} samples")

# ============================================================================
# SAVE LANDMARKS
# ============================================================================

landmark_file = Path(f'data/landmarks/landmarks_{TRAIN_TYPE}_{TRAINING_MODE}.pkl')
landmark_file.parent.mkdir(parents=True, exist_ok=True)

with open(landmark_file, 'wb') as f:
    pickle.dump({
        'landmarks': X,
        'labels': y,
        'groups': all_groups,
        'class_names': CLASSES,
        'num_classes': len(CLASSES),
        'input_features': INPUT_FEATURES,
        'mode': TRAINING_MODE
    }, f)

print(f"\n[SAVED] Landmarks saved to: {landmark_file}")

# ============================================================================
# UPDATE CONFIG.YAML
# ============================================================================

import yaml

config_path = Path('config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update model config
config['model']['num_classes'] = len(CLASSES)
config['model']['input_features'] = INPUT_FEATURES
config['model']['name'] = f'ISL_{TRAIN_TYPE.capitalize()}_{TRAINING_MODE}'
config['training']['epochs'] = EPOCHS
config['training']['batch_size'] = BATCH_SIZE
config['model']['learning_rate'] = LEARNING_RATE

# Save updated config
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"✅ Updated config.yaml")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n" + "="*70)
print("  TRAINING MODEL")
print("="*70)

trainer = ISLTrainer()
trainer.train(str(landmark_file))

# ============================================================================
# TRAINING COMPLETE
# ============================================================================

print("\n" + "="*70)
print("  [SUCCESS] TRAINING COMPLETE!")
print("="*70)
print(f"\nModel Summary:")
print(f"  - Type: {TRAIN_TYPE}")
print(f"  - Mode: {TRAINING_MODE}")
print(f"  - Classes: {len(CLASSES)}")
print(f"  - Features: {INPUT_FEATURES}")
if TRAINING_MODE == "both_hands_face":
    print(f"  - Detection: Both hands + Face")
else:
    print(f"  - Detection: Single hand")
print(f"\n📁 Files:")
print(f"  • Landmarks: {landmark_file}")
print(f"  • Models: models/checkpoints/")
print(f"  • Logs: logs/")
print("\n💡 Next steps:")
print("  1. Check models/checkpoints/ for the best model")
print("  2. Update config.yaml model_path to use the best model")
print("  3. Run: python main.py (to test with GUI)")
print("="*70)
