import os
import sys
import json
import cv2
import numpy as np
import urllib.request
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
PROJECT_DIR = Path(r"C:\Users\Acer\Desktop\mini_project 22\mini_project 22")
sys.path.insert(0, str(PROJECT_DIR))

from src.landmark_extraction import LandmarkExtractor
from tensorflow import keras
from models.gesture_model import PositionalEncoding, TransformerBlock

# Model paths
model_path = PROJECT_DIR / "models" / "saved" / "isl_model.h5"
meta_path = PROJECT_DIR / "models" / "saved" / "model_metadata.json"

with open(meta_path, 'r') as f:
    meta = json.load(f)
class_names = meta['class_names']
seq_len = meta['input_shape'][0]

print(f"Loading model: {model_path}...")
model = keras.models.load_model(model_path, custom_objects={
    'PositionalEncoding': PositionalEncoding,
    'TransformerBlock': TransformerBlock
})
print("Model loaded.")

extractor = LandmarkExtractor(static_mode=True, max_hands=2, detect_face=True)

def evaluate_image(img, expected_letter, label=""):
    print(f"\n=======================================================")
    print(f"Sample: {label} (Expected: '{expected_letter}')")
    print(f"Image Resolution: {img.shape[1]}x{img.shape[0]}")

    # 1. Test UNFLIPPED (As Trained)
    pred_unflipped, conf_unflipped, hands_unflipped, probs_unflipped = predict_frame(img)
    
    # 2. Test FLIPPED (Desktop Webcam Mirroring: cv2.flip)
    flipped_img = cv2.flip(img, 1)
    pred_flipped, conf_flipped, hands_flipped, probs_flipped = predict_frame(flipped_img)

    print(f"  [UNFLIPPED - As Trained]")
    print(f"    Hands Detected: Left={hands_unflipped['left']}, Right={hands_unflipped['right']}")
    if pred_unflipped is not None:
        status = "[PASS]" if pred_unflipped == expected_letter else "[FAIL]"
        print(f"    Prediction: {status} -> '{pred_unflipped}' ({conf_unflipped*100:.1f}%)")
        # Top 3 predictions
        top3_idx = np.argsort(probs_unflipped)[::-1][:3]
        top3_str = ", ".join([f"{class_names[i]}: {probs_unflipped[i]*100:.1f}%" for i in top3_idx])
        print(f"    Top 3: {top3_str}")
    else:
        print("    Prediction: [NO HANDS DETECTED by MediaPipe]")

    print(f"  [FLIPPED - Desktop Webcam Mirror (cv2.flip)]")
    print(f"    Hands Detected: Left={hands_flipped['left']}, Right={hands_flipped['right']}")
    if pred_flipped is not None:
        status = "[PASS]" if pred_flipped == expected_letter else "[FAIL]"
        print(f"    Prediction: {status} -> '{pred_flipped}' ({conf_flipped*100:.1f}%)")
        top3_idx = np.argsort(probs_flipped)[::-1][:3]
        top3_str = ", ".join([f"{class_names[i]}: {probs_flipped[i]*100:.1f}%" for i in top3_idx])
        print(f"    Top 3: {top3_str}")
    else:
        print("    Prediction: [NO HANDS DETECTED by MediaPipe]")

def predict_frame(frame):
    landmarks = extractor.extract_from_image(frame)
    if landmarks is None:
        return None, 0.0, {'left': False, 'right': False}, None

    left_present = bool(np.any(landmarks[:21]))
    right_present = bool(np.any(landmarks[21:42]))
    hands_info = {'left': left_present, 'right': right_present}

    # Normalize
    norm = LandmarkExtractor.normalize_landmarks(landmarks)
    flattened = norm.flatten()
    seq = np.tile(flattened, (seq_len, 1)).reshape(1, seq_len, -1)

    probs = model.predict(seq, verbose=0)[0]
    best_idx = np.argmax(probs)
    return class_names[best_idx], float(probs[best_idx]), hands_info, probs

# --- TEST 1: Live Internet Photo from Wikimedia Commons ---
downloads_dir = PROJECT_DIR / "test_downloads"
downloads_dir.mkdir(exist_ok=True)

photo_sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Sign_language%2C_2014_%2801%29.jpg/640px-Sign_language%2C_2014_%2801%29.jpg"
photo_dest = downloads_dir / "wikimedia_real_photo.jpg"

try:
    print(f"\nDownloading real sign photo from Wikimedia:\n{photo_sample_url}...")
    req = urllib.request.Request(photo_sample_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as r, open(photo_dest, 'wb') as f:
        f.write(r.read())
    img = cv2.imread(str(photo_dest))
    if img is not None:
        # In this photo, signer's hand gesture is evaluated against ISL model
        evaluate_image(img, "Unknown (Internet Photo)", label="Real Sign Photo from Wikimedia Commons")
except Exception as e:
    print(f"Error downloading test photo: {e}")

# --- TEST 2: Testing Known Confused & Edge-Case Letters ---
# Specifically testing letters known to be difficult: M, N, S, K, A, C, J, Z, B, V
test_letters = ['A', 'B', 'C', 'M', 'N', 'K', 'S', 'J', 'Z', 'V']
realsign_dir = PROJECT_DIR / "data" / "RealSign_ISL"

print("\n\n=======================================================")
print("TESTING HELD-OUT DATASET SAMPLES ACROSS LETTERS")
print("=======================================================")

for letter in test_letters:
    folder = realsign_dir / letter
    if not folder.exists():
        continue
    samples = sorted(list(folder.glob("Testing_*.jpg")))
    if samples:
        # Test 2 samples per letter
        for sample_path in samples[:2]:
            img = cv2.imread(str(sample_path))
            if img is not None:
                evaluate_image(img, letter, label=f"Held-out Sample: {sample_path.name}")
