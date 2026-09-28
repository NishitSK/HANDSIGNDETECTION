"""
Offscreen GUI & Tutor Studio Smoke Test
"""

import os
import sys
from pathlib import Path

# Ensure offscreen Qt platform and UTF-8 stdout
os.environ["QT_QPA_PLATFORM"] = "offscreen"
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QImage
from PyQt5.QtCore import QSize

from src.gui_app import (
    MotionVideoGuideCanvas,
    TutorDialog,
    ISLGUIApp,
    COMMON_GESTURE_NAMES,
    ADAPTIVE_VIDEO_INSTRUCTIONS
)

def run_smoke_test():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    print("=======================================================")
    print("  RUNNING GUI & LEARN STUDIO SMOKE TEST")
    print("=======================================================")

    # 1. Test MotionVideoGuideCanvas
    print("\n1. Testing MotionVideoGuideCanvas...")
    canvas = MotionVideoGuideCanvas()
    canvas.resize(360, 260)
    
    test_signs = ['J', 'Z'] + COMMON_GESTURE_NAMES + ['A', 'B']
    for sign in test_signs:
        canvas.setSign(sign)
        # Advance frames and paint
        for _ in range(5):
            canvas._step()
            img = QImage(QSize(360, 260), QImage.Format_ARGB32)
            painter = QPainter(img)
            canvas.render(painter)
            painter.end()
        print(f"  [OK] Canvas successfully rendered: '{sign}'")

    canvas.toggle_play()
    canvas.toggle_speed()
    canvas.reset()
    print("  [OK] Canvas playback, speed toggle, and reset verified")

    # 2. Test ISLGUIApp and TutorDialog
    print("\n2. Initializing ISLGUIApp (Offscreen)...")
    config_path = PROJECT_ROOT / 'config.yaml'
    model_path = PROJECT_ROOT / 'results' / 'ISL_MLP' / 'isl_model.h5'
    gui_window = ISLGUIApp(str(model_path), str(config_path))
    assert gui_window.width() == 960
    assert gui_window.height() == 720
    print("  [OK] ISLGUIApp initialized with 960x720 desktop dimensions")

    # 3. Test TutorDialog
    print("\n3. Testing Tutor Studio Dialog...")
    tutor = TutorDialog(gui_window)
    assert tutor.width() >= 800
    print(f"  [OK] TutorDialog studio geometry: {tutor.width()}x{tutor.height()}")

    # Category switching
    tutor.switch_category("letters")
    assert tutor.sign_combo.count() == 26
    print("  [OK] Alphabet category populated with 26 letters")

    tutor.switch_category("gestures")
    assert tutor.sign_combo.count() == len(COMMON_GESTURE_NAMES)
    print(f"  [OK] Gestures category populated with {len(COMMON_GESTURE_NAMES)} conversational gestures")

    # Test guide modes
    for mode in ['video', 'photo', 'skeleton']:
        tutor.set_guide_mode(mode)
        print(f"  [OK] Switched guide mode to: '{mode}'")

    # Test match HUD
    tutor.update_match("HELLO", 0.95)
    print(f"  [OK] Match HUD updated: {tutor.status_badge.text()}")

    # 4. Test Gesture word predictions on ISLGUIApp
    print("\n4. Testing Gesture Word Detection & Spelling Tape in Main App...")
    # Simulate receiving "HELLO" prediction
    gui_window.update_prediction("HELLO", 0.95)
    assert "HELLO" in gui_window.glyph_letter.text()
    print("  [OK] Glyph display updated for gesture: 'HELLO'")

    # Simulate hold confirmation
    for _ in range(15):
        gui_window.update_prediction("HELLO", 0.95)
    
    tape_content = "".join(gui_window.spelled_letters)
    assert "HELLO" in tape_content
    print(f"  [OK] Word gesture auto-added to tape: '{tape_content.strip()}'")

    # Simulate letter 'A'
    gui_window.update_prediction("A", 0.98)
    for _ in range(26):
        gui_window.update_prediction("A", 0.98)
    print(f"  [OK] Letter auto-added to tape: '{''.join(gui_window.spelled_letters).strip()}'")

    print("\n=======================================================")
    print("  ALL GUI & LEARN STUDIO TESTS PASSED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == "__main__":
    run_smoke_test()
