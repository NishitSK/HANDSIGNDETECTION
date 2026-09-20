"""
Create a sampled landmarks dataset by extracting up to N images per class from
`data/ISL_DATASETS_2` and saving a landmarks pickle to `data/landmarks/landmarks_sample.pkl`.
This keeps a small dataset suitable for a quick training dry-run.
"""
from src.landmark_extraction import LandmarkExtractor
from pathlib import Path
import numpy as np
import pickle

SRC = Path(__file__).resolve().parents[1] / 'data' / 'ISL_DATASETS_2'
OUT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'landmarks'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / 'landmarks_sample.pkl'

MAX_PER_CLASS = 200  # images per class to process (adjust as needed)

extractor = LandmarkExtractor(static_mode=True, max_hands=2, detect_face=True)

landmarks = []
labels = []
class_names = []

if not SRC.exists():
    print(f"Source dataset folder not found: {SRC}")
    raise SystemExit(1)

class_dirs = sorted([d for d in SRC.iterdir() if d.is_dir()])

for idx, d in enumerate(class_dirs):
    class_names.append(d.name)
    print(f"Processing class {idx+1}/{len(class_dirs)}: {d.name}")
    # gather images
    imgs = []
    for ext in ('*.jpg','*.jpeg','*.png'):
        imgs.extend(list(d.rglob(ext)))
    imgs = sorted(imgs)[:MAX_PER_CLASS]
    proc = 0
    for img_path in imgs:
        try:
            import cv2
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            lm = extractor.extract_from_image(im)
            if lm is not None:
                # normalize and flatten
                lm_norm = extractor.normalize_landmarks(lm)
                lm_flat = extractor.flatten_landmarks(lm_norm)
                landmarks.append(lm_flat)
                labels.append(idx)
                proc += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    print(f"  -> Extracted {proc} samples for class {d.name}")

if len(landmarks) == 0:
    print("No landmarks extracted — aborting")
    raise SystemExit(1)

X = np.array(landmarks)
y = np.array(labels)

with open(OUT_FILE, 'wb') as f:
    pickle.dump({'landmarks': X, 'labels': y, 'class_names': class_names}, f)

print(f"Saved sample landmarks to: {OUT_FILE}")
print(f"Total samples: {len(y)}")
