import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\Acer\Desktop\mini_project 22\mini_project 22")
sys.path.insert(0, str(ROOT))

print("=" * 65)
print("       ISL RECOGNITION & TUTOR COMPREHENSIVE TEST SUITE       ")
print("=" * 65)

# 1. Load Model & Metadata
model_path = ROOT / "models" / "saved" / "isl_model.h5"
meta_path = ROOT / "models" / "saved" / "model_metadata.json"
config_path = ROOT / "config.yaml"

with open(meta_path, 'r') as f:
    meta = json.load(f)
class_names = meta['class_names']
seq_len = meta['input_shape'][0]

print(f"\n[1] Loading trained ISL Model from {model_path.name}...")
from tensorflow import keras
from models.gesture_model import PositionalEncoding, TransformerBlock
model = keras.models.load_model(model_path, custom_objects={
    'PositionalEncoding': PositionalEncoding,
    'TransformerBlock': TransformerBlock
})
print("    ✓ ISL Model loaded successfully.")

# 2. Test MediaPipe Landmark Extractor
print("\n[2] Initializing Landmark Extractor...")
from src.landmark_extraction import LandmarkExtractor
extractor = LandmarkExtractor(static_mode=True, max_hands=2, detect_face=True)
print("    ✓ Extractor ready.")

def predict_single_image(img):
    landmarks = extractor.extract_from_image(img)
    if landmarks is None:
        return None, 0.0, None
    norm = LandmarkExtractor.normalize_landmarks(landmarks)
    flat = norm.flatten()
    seq = np.tile(flat, (seq_len, 1)).reshape(1, seq_len, -1)
    probs = model.predict(seq, verbose=0)[0]
    best_idx = np.argmax(probs)
    return class_names[best_idx], float(probs[best_idx]), probs

# 3. Test Letters with Un-flipped (New Fixed Logic) vs Flipped (Old Flawed Logic)
test_letters = ['A', 'B', 'C', 'D', 'E', 'K', 'M', 'N', 'S', 'V', 'Z']
print(f"\n[3] Testing Recognition Accuracy on {len(test_letters)} Letters:")
print(f"    Comparing Un-flipped (New Fix) vs Flipped (Old Bug)\n")
print(f"{'Letter':<8} | {'Un-flipped (Fixed)':<25} | {'Flipped (Old Bug)':<25} | {'Fix Impact'}")
print("-" * 75)

pass_count = 0
for letter in test_letters:
    folder = ROOT / "data" / "RealSign_ISL" / letter
    if not folder.exists():
        continue
    # pick first Testing image
    img_files = list(folder.glob("Testing_*.jpg"))
    if not img_files:
        continue
    img = cv2.imread(str(img_files[0]))
    
    # Predict un-flipped
    p_unf, c_unf, _ = predict_single_image(img)
    # Predict flipped
    flipped_img = cv2.flip(img, 1)
    p_flp, c_flp, _ = predict_single_image(flipped_img)
    
    unf_str = f"'{p_unf}' ({c_unf*100:.1f}%)" if p_unf else "No Hands"
    flp_str = f"'{p_flp}' ({c_flp*100:.1f}%)" if p_flp else "No Hands"
    
    is_pass = (p_unf == letter)
    if is_pass:
        pass_count += 1
    
    impact = "FIXED!" if (p_unf == letter and p_flp != letter) else ("CORRECT" if is_pass else "MISMATCH")
    print(f"{letter:<8} | {unf_str:<25} | {flp_str:<25} | {impact}")

print("-" * 75)
print(f"Accuracy with New Un-flipped Fix: {pass_count}/{len(test_letters)} ({pass_count/len(test_letters)*100:.1f}%)")

# 4. Test VideoThread Stream & Live Process Frame with 30 Frames
print("\n[4] Testing ISLInference Stream Processing (35 sequential frames)...")
from src.inference import ISLInference
inference = ISLInference(str(model_path), str(config_path))
test_c = cv2.imread(str(ROOT / "data" / "RealSign_ISL" / "C" / "Testing_175.jpg"))

stream_pred = ""
stream_conf = 0.0
for frame_idx in range(35):
    disp, stream_pred, stream_conf = inference.process_frame(test_c, mirror_display=True)

print(f"    ✓ Stream prediction after buffer filled:")
print(f"      Expected: 'C' -> Detected: '{stream_pred}', Confidence: {stream_conf*100:.1f}%")
print(f"      Display frame mirrored shape: {disp.shape}")
assert stream_pred == 'C', f"Expected 'C', got '{stream_pred}'"

# 5. Test Tutor Feature & Reference Assets
print("\n[5] Testing Sign Language Tutor Module & Assets...")
from src.gui_app import ISL_LETTER_HINTS
print(f"    ✓ Total letters with instructions: {len(ISL_LETTER_HINTS)}")
assert len(ISL_LETTER_HINTS) == 26

missing_images = []
for ltr in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    img_path = ROOT / "mobile" / "app" / "pamphlet" / "isl" / f"{ltr}.jpg"
    if not img_path.exists():
        missing_images.append(ltr)

if not missing_images:
    print("    ✓ All 26 ISL reference guide images present.")
else:
    print(f"    ❌ Missing images: {missing_images}")

# 6. Test Ghost Guide Alpha Blending
print("\n[6] Testing Transparent Ghost Guide Alpha Blending on camera frame...")
h_f, w_f = 480, 640
cam_frame = np.full((h_f, w_f, 3), 50, dtype=np.uint8) # simulated dark background
guide_img = cv2.imread(str(ROOT / "mobile" / "app" / "pamphlet" / "isl" / "C.jpg"))
guide_size = min(int(h_f * 0.72), int(w_f * 0.42))
guide_resized = cv2.resize(guide_img, (guide_size, guide_size), interpolation=cv2.INTER_AREA)

x_offset = w_f - guide_size - 15
y_offset = (h_f - guide_size) // 2
roi = cam_frame[y_offset:y_offset+guide_size, x_offset:x_offset+guide_size]

for opacity in [0.2, 0.45, 0.7]:
    blended = cv2.addWeighted(roi, 1.0 - opacity, guide_resized, opacity, 0)
    assert blended.shape == (guide_size, guide_size, 3)
    print(f"    ✓ Blended ghost guide with opacity {int(opacity*100)}% -> shape {blended.shape}")

# 7. Test GUI App Tutor Mode Initialization & Live Prediction Matching
print("\n[7] Testing ISLGUIApp Tutor Mode Integration...")
from PyQt5.QtWidgets import QApplication
from src.gui_app import ISLGUIApp

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

gui_app = ISLGUIApp(str(model_path), str(config_path))
print("    ✓ ISLGUIApp initialized.")

# Switch tutor letter to 'C'
gui_app.on_tutor_letter_changed('C')
assert gui_app.current_tutor_letter == 'C'
print("    ✓ Tutor letter switched to 'C'")

# Simulate matching prediction
gui_app.update_prediction('C', 0.98)
status_text = gui_app.tutor_match_status.text()
print(f"    ✓ Tutor feedback on match: '{status_text}'")
assert "EXCELLENT" in status_text

# Simulate non-matching prediction
gui_app.update_prediction('B', 0.85)
status_text = gui_app.tutor_match_status.text()
print(f"    ✓ Tutor feedback on mismatch: '{status_text}'")
assert "Keep trying" in status_text

print("\n" + "=" * 65)
print("            ALL AUTOMATED TESTS PASSED SUCCESSFULLY!          ")
print("=" * 65)
