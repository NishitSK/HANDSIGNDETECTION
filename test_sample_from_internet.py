import urllib.request
import os
import cv2
import numpy as np
import json
import sys
from pathlib import Path

# Paths
ROOT = Path(r"C:\Users\Acer\Desktop\mini_project 22\mini_project 22")
sys.path.append(str(ROOT))

from src.landmark_extraction import LandmarkExtractor
from tensorflow import keras
from models.gesture_model import PositionalEncoding, TransformerBlock

# Online sample URLs for testing (Real public sign language samples from GitHub / Wikipedia)
# Let's test a sample of letter 'A' or 'B' or 'C'
test_samples = [
    {
        "letter": "A",
        "url": "https://raw.githubusercontent.com/RealSign62/RealSign-Indian-Sign-Language-Dataset/main/Training/A/A_1.jpg",
        "name": "sample_A.jpg"
    },
    {
        "letter": "B",
        "url": "https://raw.githubusercontent.com/RealSign62/RealSign-Indian-Sign-Language-Dataset/main/Training/B/B_1.jpg",
        "name": "sample_B.jpg"
    },
    {
        "letter": "C",
        "url": "https://raw.githubusercontent.com/RealSign62/RealSign-Indian-Sign-Language-Dataset/main/Training/C/C_1.jpg",
        "name": "sample_C.jpg"
    }
]

# Create output dir
out_dir = ROOT / "test_internet_samples"
out_dir.mkdir(exist_ok=True)

# Load Model
model_path = ROOT / "models" / "saved" / "isl_model.h5"
meta_path = ROOT / "models" / "saved" / "model_metadata.json"

with open(meta_path, 'r') as f:
    metadata = json.load(f)
class_names = metadata['class_names']
seq_len = metadata['input_shape'][0]

print(f"Loading model: {model_path}")
model = keras.models.load_model(model_path, custom_objects={
    'PositionalEncoding': PositionalEncoding,
    'TransformerBlock': TransformerBlock
})
print("Model loaded successfully.")

extractor = LandmarkExtractor(static_mode=True, max_hands=2, detect_face=True)

def test_image_url(url, expected_letter, filename):
    filepath = out_dir / filename
    print(f"\n==========================================")
    print(f"Downloading test sample for Letter '{expected_letter}' from:\n{url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp, open(filepath, 'wb') as f:
            f.write(resp.read())
        print(f"Saved to {filepath}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return

    img = cv2.imread(str(filepath))
    if img is None:
        print(f"Could not read image: {filepath}")
        return

    print(f"Image shape: {img.shape}")

    # Test 1: Original Unflipped Image (As trained)
    run_inference(img, expected_letter, label="Original (Unflipped)")

    # Test 2: Horizontally Flipped Image (What desktop webcam currently feeds!)
    flipped = cv2.flip(img, 1)
    run_inference(flipped, expected_letter, label="Horizontally Flipped (Webcam Mirroring)")

def run_inference(image, expected, label=""):
    print(f"\n--- Testing: {label} ---")
    landmarks = extractor.extract_from_image(image)
    if landmarks is None:
        print("❌ MediaPipe detected NO HANDS in the frame!")
        return

    # Check which hands were detected
    left_hand = landmarks[:21]
    right_hand = landmarks[21:42]
    has_left = np.any(left_hand)
    has_right = np.any(right_hand)
    print(f"Hand detection: Left Hand = {has_left}, Right Hand = {has_right}")

    # Normalization
    normalized = LandmarkExtractor.normalize_landmarks(landmarks)
    flattened = normalized.flatten()
    
    # Tile sequence to 30 frames (matching trained sequence format)
    seq = np.tile(flattened, (seq_len, 1)).reshape(1, seq_len, -1)

    preds = model.predict(seq, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:5]

    print(f"Expected: {expected}")
    print("Top 5 Predictions:")
    for rank, idx in enumerate(top_indices, 1):
        c_name = class_names[idx]
        conf = preds[idx] * 100
        match = "✅" if c_name == expected else "  "
        print(f"  {rank}. {match} {c_name}: {conf:.2f}%")

for sample in test_samples:
    test_image_url(sample["url"], sample["letter"], sample["name"])
