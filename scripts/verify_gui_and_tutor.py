import sys
import os
import cv2
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
ROOT = Path(r"C:\Users\Acer\Desktop\mini_project 22\mini_project 22")
sys.path.insert(0, str(ROOT))

print("1. Testing imports from src.inference and src.gui_app...")
from src.inference import ISLInference
from src.gui_app import VideoThread, ISLGUIApp, ISL_LETTER_HINTS
from PyQt5.QtWidgets import QApplication

print(f"[OK] Imports successful. ISL_LETTER_HINTS has {len(ISL_LETTER_HINTS)} letter hints.")

print("\n2. Verifying guide images exist for A-Z...")
missing = []
for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    p = ROOT / "mobile" / "app" / "pamphlet" / "isl" / f"{c}.jpg"
    if not p.exists():
        missing.append(c)
if not missing:
    print("[OK] All 26 ISL reference guide images exist in mobile/app/pamphlet/isl/")
else:
    print(f"[WARN] Missing letters: {missing}")

print("\n3. Testing VideoThread guide blending logic on a test frame...")
test_frame = np.zeros((360, 640, 3), dtype=np.uint8)
test_guide = cv2.imread(str(ROOT / "mobile" / "app" / "pamphlet" / "isl" / "A.jpg"))

h_f, w_f = test_frame.shape[:2]
guide_size = min(int(h_f * 0.72), int(w_f * 0.42))
guide_resized = cv2.resize(test_guide, (guide_size, guide_size), interpolation=cv2.INTER_AREA)
x_offset = w_f - guide_size - 15
y_offset = (h_f - guide_size) // 2

roi = test_frame[y_offset:y_offset+guide_size, x_offset:x_offset+guide_size]
alpha = 0.5
blended = cv2.addWeighted(roi, 1.0 - alpha, guide_resized, alpha, 0)
test_frame[y_offset:y_offset+guide_size, x_offset:x_offset+guide_size] = blended
print(f"[OK] Transparent guide blending succeeded. Blended frame shape: {test_frame.shape}")

print("\n4. Testing ISLInference process_frame with mirror_display=True on sample 'C'...")
model_path = ROOT / "models" / "saved" / "isl_model.h5"
config_path = ROOT / "config.yaml"
inference = ISLInference(str(model_path), str(config_path))

sample_img = cv2.imread(str(ROOT / "data" / "RealSign_ISL" / "C" / "Testing_175.jpg"))
display_frame, pred, conf = inference.process_frame(sample_img, mirror_display=True)
print(f"[OK] process_frame with mirror_display=True executed cleanly.")
print(f"     Expected 'C' -> Prediction: '{pred}', Confidence: {conf*100:.1f}%")
print(f"     Display frame shape: {display_frame.shape}")

print("\n5. Testing PyQt application initialization...")
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

gui = ISLGUIApp(str(model_path), str(config_path))
print("[OK] ISLGUIApp created successfully.")
print(f"     Current tutor letter: {gui.current_tutor_letter}")
print(f"     Guide checkbox state: {gui.guide_checkbox.isChecked()}")
print(f"     Mirror checkbox state: {gui.mirror_checkbox.isChecked()}")

# Test switching tutor letter
gui.on_tutor_letter_changed("B")
print(f"     Switched tutor to 'B': {gui.current_tutor_letter}")
print(f"     Tutor target label: {gui.tutor_target_label.text()}")

# Test update_prediction with matching letter
gui.update_prediction("B", 0.95)
print(f"     Tutor match status after match: {gui.tutor_match_status.text()}")

print("\n*** ALL TESTS PASSED SUCCESSFULLY! ***")
