"""
ISL Fingerspell - Desktop Application
Identical UI & Design System to the Mobile Web/Android App
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QSlider,
    QCheckBox, QComboBox, QDialog, QFrame, QSizePolicy, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QPainter, QBrush, QPen

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference import ISLInference
from utils.config_loader import get_config
from utils.camera import open_camera

ISL_LETTER_HINTS = {
    'A': "Both hands: Non-dominant palm open facing inward; touch thumb tip with dominant index finger.",
    'B': "Both hands: Non-dominant palm open; touch index finger tip with dominant index finger.",
    'C': "Right hand: Form curved 'C' shape with thumb and fingers facing left.",
    'D': "Both hands: Non-dominant index extended straight up; dominant hand curves against it.",
    'E': "Both hands: Non-dominant hand open; touch middle finger tip with dominant index finger.",
    'F': "Both hands: Cross dominant index and middle fingers over non-dominant index and middle.",
    'G': "Both hands: Both hands form fists held side-by-side with thumbs tucked.",
    'H': "Both hands: Flat dominant palm brushes forward across non-dominant flat palm.",
    'I': "Both hands: Non-dominant hand open; touch ring finger tip with dominant index finger.",
    'J': "Both hands: Non-dominant hand open; dominant index traces 'J' downwards from ring finger.",
    'K': "Both hands: Non-dominant index straight up; dominant index touches knuckle at an angle.",
    'L': "Right hand: Thumb and index extended at 90 degrees forming an 'L' shape.",
    'M': "Both hands: Dominant index, middle, and ring fingers rest on non-dominant flat palm.",
    'N': "Both hands: Dominant index and middle fingers rest on non-dominant flat palm.",
    'O': "Both hands: Non-dominant hand open; touch pinky finger tip with dominant index finger.",
    'P': "Both hands: Non-dominant index pointing up; dominant hand forms a loop/circle at the top.",
    'Q': "Both hands: Dominant hand forms a loop hooked around the base of non-dominant thumb.",
    'R': "Both hands: Dominant index finger hooked over non-dominant palm/index.",
    'S': "Both hands: Dominant pinky finger linked with non-dominant pinky finger.",
    'T': "Both hands: Dominant index finger held horizontally against non-dominant index edge.",
    'U': "Both hands: Non-dominant hand open; dominant index and middle fingertips touch palm.",
    'V': "Right hand: Dominant index and middle fingers extended in a 'V' peace shape.",
    'W': "Both hands: Fingers of both hands interlaced together pointing upward.",
    'X': "Both hands: Dominant and non-dominant index fingers crossed over each other.",
    'Y': "Right hand: Dominant thumb and pinky finger extended ('hang loose' sign).",
    'Z': "Both hands: Non-dominant palm flat; dominant index finger traces 'Z' path."
}


class VideoThread(QThread):
    """Camera capture thread matching mobile pipeline"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    prediction_signal = pyqtSignal(str, float)

    def __init__(self, inference_engine, camera_id=0):
        super().__init__()
        self.inference_engine = inference_engine
        self.camera_id = camera_id
        self.running = True
        self.mirror_display = True

        # Ghost guide overlay properties
        self.guide_enabled = False
        self.guide_image = None
        self.guide_letter = "A"
        self.guide_opacity = 0.5
        self.show_landmarks = True

    def run(self):
        try:
            cap = open_camera(self.camera_id, width=640, height=480)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return

        last_processed = None
        last_prediction = None
        last_confidence = 0.0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Model inference on raw frame
            try:
                processed_frame, prediction, confidence = self.inference_engine.process_frame(
                    frame, mirror_display=self.mirror_display
                )
                last_processed = processed_frame
                last_prediction = prediction
                last_confidence = confidence
            except Exception:
                processed_frame = cv2.flip(frame, 1) if self.mirror_display else frame.copy()
                prediction = last_prediction
                confidence = last_confidence

            if processed_frame is None:
                processed_frame = cv2.flip(frame, 1) if self.mirror_display else frame.copy()

            # Blend transparent ghost guide overlay if enabled
            if self.guide_enabled and self.guide_image is not None:
                try:
                    h_f, w_f = processed_frame.shape[:2]
                    guide_size = min(int(h_f * 0.75), int(w_f * 0.45))
                    if guide_size > 50:
                        guide_resized = cv2.resize(self.guide_image, (guide_size, guide_size), interpolation=cv2.INTER_AREA)
                        x_offset = w_f - guide_size - 16
                        y_offset = (h_f - guide_size) // 2

                        roi = processed_frame[y_offset:y_offset + guide_size, x_offset:x_offset + guide_size]
                        alpha = float(np.clip(self.guide_opacity, 0.1, 0.95))
                        blended = cv2.addWeighted(roi, 1.0 - alpha, guide_resized, alpha, 0)
                        processed_frame[y_offset:y_offset + guide_size, x_offset:x_offset + guide_size] = blended

                        # Clean guide border matching mobile --marigold / --ink-line
                        cv2.rectangle(processed_frame, (x_offset, y_offset), (x_offset + guide_size, y_offset + guide_size), (63, 178, 255), 2)
                        cv2.putText(processed_frame, f"GUIDE: {self.guide_letter}", (x_offset + 6, y_offset - 8),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (63, 178, 255), 1, cv2.LINE_AA)
                except Exception:
                    pass

            self.change_pixmap_signal.emit(processed_frame)
            self.prediction_signal.emit(prediction if prediction else "", confidence if prediction else 0.0)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class TutorDialog(QDialog):
    """Sign Language Tutor Dialog matching mobile sheet-tutor"""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Sign Language Tutor")
        self.setFixedWidth(520)
        self.setStyleSheet("""
            QDialog {
                background-color: #14234b;
                color: #f4f6fb;
                border-radius: 16px;
            }
            QLabel {
                color: #f4f6fb;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            QPushButton.btn-nav {
                background-color: #0b1633;
                color: #f4f6fb;
                border: 1px solid #26386a;
                border-radius: 8px;
                font-weight: bold;
                padding: 6px 12px;
                min-width: 36px;
            }
            QPushButton.btn-nav:hover {
                background-color: #26386a;
            }
            QComboBox {
                background-color: #0b1633;
                color: #f4f6fb;
                border: 1px solid #26386a;
                border-radius: 8px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QCheckBox {
                color: #f4f6fb;
                font-size: 13px;
                font-weight: 500;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #0b1633;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #ffb23f;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #f4f6fb;
                width: 16px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(14)

        # Header
        title = QLabel("SIGN LANGUAGE TUTOR")
        title.setFont(QFont('Segoe UI', 15, QFont.Bold))
        title.setStyleSheet("letter-spacing: 1px; color: #f4f6fb;")
        layout.addWidget(title)

        subtitle = QLabel("Practice handshapes with a live guide overlay on your camera")
        subtitle.setStyleSheet("color: #93a1c6; font-size: 12px;")
        layout.addWidget(subtitle)

        # Tutor Card
        card = QFrame()
        card.setStyleSheet("background-color: #0b1633; border: 1px solid #26386a; border-radius: 12px; padding: 10px;")
        card_layout = QHBoxLayout()
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(16)

        # Preview image & nav
        left_box = QVBoxLayout()
        left_box.setSpacing(8)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(110, 110)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background: #050b1c; border: 1px solid #26386a; border-radius: 8px;")
        left_box.addWidget(self.preview_label)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        prev_btn = QPushButton("◀")
        prev_btn.setProperty("class", "btn-nav")
        prev_btn.clicked.connect(self.prev_letter)
        nav_row.addWidget(prev_btn)

        self.letter_combo = QComboBox()
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.letter_combo.addItem(f"Letter {c}", c)
        self.letter_combo.currentIndexChanged.connect(self.on_combo_changed)
        nav_row.addWidget(self.letter_combo)

        next_btn = QPushButton("▶")
        next_btn.setProperty("class", "btn-nav")
        next_btn.clicked.connect(self.next_letter)
        nav_row.addWidget(next_btn)

        left_box.addLayout(nav_row)
        card_layout.addLayout(left_box)

        # Details
        right_box = QVBoxLayout()
        right_box.setSpacing(8)

        self.target_title = QLabel("Target: Letter A")
        self.target_title.setFont(QFont('Segoe UI', 13, QFont.Bold))
        self.target_title.setStyleSheet("color: #ffb23f;")
        right_box.addWidget(self.target_title)

        self.hint_label = QLabel(ISL_LETTER_HINTS.get('A', ''))
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #f4f6fb; font-size: 11px; line-height: 1.4;")
        right_box.addWidget(self.hint_label)
        right_box.addStretch()

        card_layout.addLayout(right_box, stretch=1)
        card.setLayout(card_layout)
        layout.addWidget(card)

        # Ghost guide toggle
        self.guide_checkbox = QCheckBox("Show guide overlay on camera")
        self.guide_checkbox.setChecked(self.parent_app.guide_enabled)
        self.guide_checkbox.toggled.connect(self.toggle_guide)
        layout.addWidget(self.guide_checkbox)

        # Opacity slider
        opacity_row = QHBoxLayout()
        self.opacity_title = QLabel("Guide Opacity: 50%")
        self.opacity_title.setStyleSheet("color: #93a1c6; font-size: 12px;")
        opacity_row.addWidget(self.opacity_title)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(15, 90)
        self.opacity_slider.setValue(int(self.parent_app.guide_opacity * 100))
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_row.addWidget(self.opacity_slider, stretch=1)
        layout.addLayout(opacity_row)

        # Status badge
        self.status_badge = QLabel("Align hand with guide to practice")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setStyleSheet("""
            background-color: rgba(38, 56, 106, 0.4);
            border: 1px solid #26386a;
            border-radius: 8px;
            padding: 10px;
            color: #93a1c6;
            font-weight: 600;
        """)
        layout.addWidget(self.status_badge)

        # Done button
        done_btn = QPushButton("Done")
        done_btn.setFixedHeight(42)
        done_btn.setStyleSheet("""
            QPushButton {
                background-color: #f4f6fb;
                color: #0b1633;
                border: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #ffb23f;
            }
        """)
        done_btn.clicked.connect(self.accept)
        layout.addWidget(done_btn)

        self.setLayout(layout)
        self.sync_letter(self.parent_app.current_tutor_letter)

    def sync_letter(self, letter):
        idx = ord(letter) - ord('A')
        self.letter_combo.blockSignals(True)
        self.letter_combo.setCurrentIndex(idx)
        self.letter_combo.blockSignals(False)

        self.target_title.setText(f"Target: Letter {letter}")
        self.hint_label.setText(ISL_LETTER_HINTS.get(letter, "Observe the guide image."))

        img = self.parent_app.load_guide_image(letter)
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(pix)
        else:
            self.preview_label.setText("No Image")

    def on_combo_changed(self, idx):
        letter = chr(ord('A') + idx)
        self.parent_app.on_tutor_letter_changed(letter)
        self.sync_letter(letter)

    def prev_letter(self):
        curr = self.parent_app.current_tutor_letter
        new_l = chr(ord('A') + (ord(curr) - ord('A') - 1) % 26)
        self.parent_app.on_tutor_letter_changed(new_l)
        self.sync_letter(new_l)

    def next_letter(self):
        curr = self.parent_app.current_tutor_letter
        new_l = chr(ord('A') + (ord(curr) - ord('A') + 1) % 26)
        self.parent_app.on_tutor_letter_changed(new_l)
        self.sync_letter(new_l)

    def toggle_guide(self, checked):
        self.parent_app.guide_enabled = checked
        if self.parent_app.video_thread:
            self.parent_app.video_thread.guide_enabled = checked

    def on_opacity_changed(self, val):
        self.parent_app.guide_opacity = val / 100.0
        self.opacity_title.setText(f"Guide Opacity: {val}%")
        if self.parent_app.video_thread:
            self.parent_app.video_thread.guide_opacity = self.parent_app.guide_opacity

    def update_match(self, prediction, confidence):
        target = self.parent_app.current_tutor_letter
        if prediction == target and confidence >= 0.60:
            self.status_badge.setText(f"🎯 EXCELLENT! Matched '{prediction}' ({confidence:.1%})")
            self.status_badge.setStyleSheet("""
                background-color: rgba(63, 185, 80, 0.25);
                border: 1px solid #3fb950;
                border-radius: 8px;
                padding: 10px;
                color: #3fb950;
                font-weight: bold;
            """)
        else:
            self.status_badge.setText(f"Target: {target} | Detected: {prediction if prediction else '...'} ({confidence:.0%})")
            self.status_badge.setStyleSheet("""
                background-color: rgba(255, 178, 63, 0.15);
                border: 1px solid #ffb23f;
                border-radius: 8px;
                padding: 10px;
                color: #ffb23f;
                font-weight: 600;
            """)


class PamphletDialog(QDialog):
    """Letters Reference Grid matching mobile sheet-pamphlet"""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("ISL Letters Reference")
        self.resize(600, 680)
        self.setStyleSheet("""
            QDialog {
                background-color: #0b1633;
                color: #f4f6fb;
            }
            QLabel {
                color: #f4f6fb;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("LETTERS")
        title.setFont(QFont('Segoe UI', 16, QFont.Bold))
        title.setStyleSheet("letter-spacing: 1px;")
        layout.addWidget(title)

        hint = QLabel("How to sign each letter in Indian Sign Language (ISL).")
        hint.setStyleSheet("color: #93a1c6; font-size: 12px;")
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout()
        grid.setSpacing(12)

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, c in enumerate(letters):
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #14234b;
                    border: 1px solid #26386a;
                    border-radius: 10px;
                }
                QFrame:hover {
                    border-color: #ffb23f;
                }
            """)
            card_layout = QVBoxLayout()
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(6)

            img_label = QLabel()
            img_label.setFixedSize(95, 95)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("background: #050b1c; border-radius: 6px;")

            img = self.parent_app.load_guide_image(c)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(95, 95, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_label.setPixmap(pix)
            else:
                img_label.setText(c)

            card_layout.addWidget(img_label)

            lbl = QLabel(c)
            lbl.setFont(QFont('Segoe UI', 14, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #f4f6fb;")
            card_layout.addWidget(lbl)

            card.setLayout(card_layout)
            row = i // 4
            col = i % 4
            grid.addWidget(card, row, col)

        container.setLayout(grid)
        scroll.setWidget(container)
        layout.addWidget(scroll, stretch=1)

        done_btn = QPushButton("Done")
        done_btn.setFixedHeight(42)
        done_btn.setStyleSheet("""
            QPushButton {
                background-color: #f4f6fb;
                color: #0b1633;
                border: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #ffb23f;
            }
        """)
        done_btn.clicked.connect(self.accept)
        layout.addWidget(done_btn)

        self.setLayout(layout)


class SettingsDialog(QDialog):
    """Settings modal matching mobile settings sheet"""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Settings")
        self.setFixedWidth(440)
        self.setStyleSheet("""
            QDialog {
                background-color: #14234b;
                color: #f4f6fb;
                border-radius: 16px;
            }
            QLabel {
                color: #f4f6fb;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            QCheckBox {
                color: #f4f6fb;
                font-size: 13px;
                font-weight: 500;
                spacing: 8px;
            }
            QComboBox {
                background-color: #0b1633;
                color: #f4f6fb;
                border: 1px solid #26386a;
                border-radius: 8px;
                padding: 6px 12px;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)

        title = QLabel("SETTINGS")
        title.setFont(QFont('Segoe UI', 15, QFont.Bold))
        title.setStyleSheet("letter-spacing: 1px;")
        layout.addWidget(title)

        # Auto capture toggle
        self.auto_capture_check = QCheckBox("Add a letter automatically when it's held steady")
        self.auto_capture_check.setChecked(self.parent_app.auto_capture_enabled)
        layout.addWidget(self.auto_capture_check)

        # Show landmarks
        self.landmarks_check = QCheckBox("Show hand tracking overlay")
        self.landmarks_check.setChecked(self.parent_app.show_landmarks_enabled)
        layout.addWidget(self.landmarks_check)

        # Mirror camera
        self.mirror_check = QCheckBox("Mirror camera input (Selfie View)")
        self.mirror_check.setChecked(self.parent_app.mirror_camera_enabled)
        layout.addWidget(self.mirror_check)

        # Recognition engine
        engine_label = QLabel("Recognition Pipeline:")
        engine_label.setFont(QFont('Segoe UI', 10, QFont.Bold))
        engine_label.setStyleSheet("color: #93a1c6;")
        layout.addWidget(engine_label)

        self.engine_combo = QComboBox()
        self.engine_combo.addItems([
            "⚡ Auto (MLP Static + GRU Motion)",
            "✋ MLP Only (Fast Static Handsigns)",
            "🌊 GRU Only (Motion Gestures)"
        ])
        mode = getattr(self.parent_app.inference_engine, 'ensemble_mode', 'auto_ensemble')
        if mode == 'mlp_only':
            self.engine_combo.setCurrentIndex(1)
        elif mode == 'gru_only':
            self.engine_combo.setCurrentIndex(2)
        else:
            self.engine_combo.setCurrentIndex(0)
        layout.addWidget(self.engine_combo)

        # Readout info
        readout = QLabel("Model: HighAccuracy Residual MLP (99.76% accuracy)\nInput: 126 normalized hand landmarks (42x3)")
        readout.setStyleSheet("color: #56648c; font-size: 11px; padding: 6px; background: #0b1633; border-radius: 6px;")
        layout.addWidget(readout)

        done_btn = QPushButton("Done")
        done_btn.setFixedHeight(42)
        done_btn.setStyleSheet("""
            QPushButton {
                background-color: #f4f6fb;
                color: #0b1633;
                border: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #ffb23f;
            }
        """)
        done_btn.clicked.connect(self.save_and_close)
        layout.addWidget(done_btn)

        self.setLayout(layout)

    def save_and_close(self):
        self.parent_app.auto_capture_enabled = self.auto_capture_check.isChecked()
        self.parent_app.show_landmarks_enabled = self.landmarks_check.isChecked()
        self.parent_app.mirror_camera_enabled = self.mirror_check.isChecked()

        if self.parent_app.video_thread:
            self.parent_app.video_thread.mirror_display = self.parent_app.mirror_camera_enabled

        idx = self.engine_combo.currentIndex()
        modes = ['auto_ensemble', 'mlp_only', 'gru_only']
        self.parent_app.inference_engine.ensemble_mode = modes[idx]

        self.accept()


class ISLGUIApp(QMainWindow):
    """Main Application with Mobile-Identical Design System"""

    def __init__(self, model_path, config_path='config.yaml'):
        super().__init__()
        self.config = get_config(config_path)
        self.inference_engine = ISLInference(model_path, config_path)
        self.video_thread = None

        # Recognition & Hold-to-add state
        self.auto_capture_enabled = True
        self.show_landmarks_enabled = True
        self.mirror_camera_enabled = True
        self.current_tutor_letter = "A"
        self.guide_enabled = False
        self.guide_opacity = 0.5
        self.guide_images_cache = {}

        # Tape spelling state
        self.spelled_letters = []
        self.translated_sentence = ""
        self.last_prediction = None
        self.last_confidence = 0.0

        # Auto-capture hold counters (at 30 FPS, 25 frames ~ 850ms)
        self.hold_count = 0
        self.hold_target = 25
        self.last_held_char = None
        self.cooldown_frames = 0

        # Active dialogs
        self.tutor_dialog = None

        self.init_ui()

    def load_guide_image(self, letter):
        letter = str(letter).upper()
        if letter in self.guide_images_cache:
            return self.guide_images_cache[letter]

        pamphlet_path = PROJECT_ROOT / 'mobile' / 'app' / 'pamphlet' / 'isl' / f"{letter}.jpg"
        if pamphlet_path.exists():
            img = cv2.imread(str(pamphlet_path))
            self.guide_images_cache[letter] = img
            return img

        return None

    def init_ui(self):
        self.setWindowTitle("ISL Fingerspell")
        self.resize(760, 880)
        self.setMinimumSize(640, 720)

        # Global stylesheet matching mobile/app/styles.css tokens
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b1633;
                color: #f4f6fb;
            }
            QWidget#appRoot {
                background-color: #0b1633;
            }
            QPushButton.pill {
                min-height: 32px;
                padding: 0 14px;
                border: 1px solid #26386a;
                border-radius: 16px;
                background-color: transparent;
                color: #f4f6fb;
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton.pill:hover {
                background-color: #14234b;
                border-color: #ffb23f;
            }
            QPushButton.pill:pressed {
                background-color: #050b1c;
            }
            /* Mobile control buttons */
            QPushButton.btn-ctrl {
                min-height: 44px;
                padding: 0 12px;
                border: 1px solid #26386a;
                border-radius: 10px;
                background-color: #14234b;
                color: #f4f6fb;
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton.btn-ctrl:hover {
                background-color: #26386a;
            }
            QPushButton.btn-ctrl:pressed {
                transform: translateY(1px);
            }
            QPushButton#addButton {
                background-color: #f4f6fb;
                color: #0b1633;
                font-weight: 800;
                font-size: 14px;
                border: 1px solid #f4f6fb;
            }
            QPushButton#addButton:hover {
                background-color: #ffb23f;
                border-color: #ffb23f;
            }
            QPushButton#clearButton {
                background-color: transparent;
                color: #93a1c6;
            }
            QPushButton#speakButton {
                background-color: #14234b;
                border: 1px solid #f4f6fb;
                color: #f4f6fb;
                font-weight: 700;
            }
            QPushButton#speakButton:hover {
                background-color: #26386a;
                border-color: #ffb23f;
            }
        """)

        central_widget = QWidget()
        central_widget.setObjectName("appRoot")
        self.setCentralWidget(central_widget)

        app_layout = QVBoxLayout()
        app_layout.setContentsMargins(16, 12, 16, 16)
        app_layout.setSpacing(10)
        central_widget.setLayout(app_layout)

        # -------------------------------------------------------------
        # 1. TOPBAR: Wordmark + Pill Actions [Learn] [Letters] [Settings]
        # -------------------------------------------------------------
        topbar = QHBoxLayout()
        topbar.setContentsMargins(4, 0, 4, 4)

        # Wordmark: ISL Fingerspell
        wordmark_container = QWidget()
        wm_layout = QHBoxLayout()
        wm_layout.setContentsMargins(0, 0, 0, 0)
        wm_layout.setSpacing(6)

        wm_isl = QLabel("ISL")
        wm_isl.setFont(QFont('Segoe UI', 15, QFont.Black))
        wm_isl.setStyleSheet("color: #f4f6fb; letter-spacing: 2px;")
        wm_layout.addWidget(wm_isl)

        wm_sub = QLabel("FINGERSPELL")
        wm_sub.setFont(QFont('Segoe UI', 15, QFont.Bold))
        wm_sub.setStyleSheet("color: #93a1c6; letter-spacing: 1px;")
        wm_layout.addWidget(wm_sub)

        wordmark_container.setLayout(wm_layout)
        topbar.addWidget(wordmark_container)
        topbar.addStretch()

        # Action pills
        learn_btn = QPushButton("Learn")
        learn_btn.setProperty("class", "pill")
        learn_btn.clicked.connect(self.open_tutor)
        topbar.addWidget(learn_btn)

        letters_btn = QPushButton("Letters")
        letters_btn.setProperty("class", "pill")
        letters_btn.clicked.connect(self.open_pamphlet)
        topbar.addWidget(letters_btn)

        settings_btn = QPushButton("Settings")
        settings_btn.setProperty("class", "pill")
        settings_btn.clicked.connect(self.open_settings)
        topbar.addWidget(settings_btn)

        app_layout.addLayout(topbar)

        # -------------------------------------------------------------
        # 2. STAGE: Video feed + Floating Glyph Card (Top-Right)
        # -------------------------------------------------------------
        stage_container = QFrame()
        stage_container.setObjectName("stage")
        stage_container.setStyleSheet("""
            #stage {
                background-color: #050b1c;
                border: 1px solid #26386a;
                border-radius: 14px;
            }
        """)
        stage_layout = QGridLayout()
        stage_layout.setContentsMargins(0, 0, 0, 0)

        # Central video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setStyleSheet("background: transparent; border-radius: 14px;")
        stage_layout.addWidget(self.video_label, 0, 0)

        # Floating Glyph Card (overlayed at top-right inside stage)
        glyph_overlay_container = QWidget()
        glyph_overlay_container.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        glyph_overlay_layout = QHBoxLayout()
        glyph_overlay_layout.setContentsMargins(0, 12, 14, 0)
        glyph_overlay_layout.addStretch()

        # In-camera glyph box
        self.glyph_box = QFrame()
        self.glyph_box.setFixedSize(110, 140)
        self.glyph_box.setStyleSheet("""
            QFrame {
                background-color: rgba(11, 22, 51, 0.82);
                border: 2px solid #26386a;
                border-radius: 12px;
            }
        """)
        glyph_box_layout = QVBoxLayout()
        glyph_box_layout.setContentsMargins(6, 6, 6, 8)
        glyph_box_layout.setSpacing(4)

        # Giant letter label
        self.glyph_letter = QLabel("—")
        self.glyph_letter.setFont(QFont('Segoe UI', 40, QFont.Bold))
        self.glyph_letter.setAlignment(Qt.AlignCenter)
        self.glyph_letter.setStyleSheet("color: #f4f6fb; background: transparent; border: none;")
        glyph_box_layout.addWidget(self.glyph_letter, stretch=1)

        # Golden hold progress bar
        self.glyph_hold_bar = QProgressBar()
        self.glyph_hold_bar.setRange(0, 100)
        self.glyph_hold_bar.setValue(0)
        self.glyph_hold_bar.setFixedHeight(5)
        self.glyph_hold_bar.setTextVisible(False)
        self.glyph_hold_bar.setStyleSheet("""
            QProgressBar {
                background-color: #26386a;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #ffb23f;
                border-radius: 2px;
            }
        """)
        glyph_box_layout.addWidget(self.glyph_hold_bar)

        # Caption
        self.glyph_caption = QLabel("Show a letter")
        self.glyph_caption.setFont(QFont('Segoe UI', 8))
        self.glyph_caption.setAlignment(Qt.AlignCenter)
        self.glyph_caption.setStyleSheet("color: #93a1c6; background: transparent; border: none;")
        glyph_box_layout.addWidget(self.glyph_caption)

        self.glyph_box.setLayout(glyph_box_layout)
        glyph_overlay_layout.addWidget(self.glyph_box, alignment=Qt.AlignTop)
        glyph_overlay_container.setLayout(glyph_overlay_layout)

        stage_layout.addWidget(glyph_overlay_container, 0, 0)
        stage_container.setLayout(stage_layout)
        app_layout.addWidget(stage_container, stretch=1)

        # -------------------------------------------------------------
        # 3. SPELLING TAPE: Crisp Paper Card (#f4f6fb)
        # -------------------------------------------------------------
        tape_card = QFrame()
        tape_card.setObjectName("tapeCard")
        tape_card.setStyleSheet("""
            #tapeCard {
                background-color: #f4f6fb;
                border-radius: 14px;
            }
        """)
        tape_layout = QVBoxLayout()
        tape_layout.setContentsMargins(16, 12, 16, 12)
        tape_layout.setSpacing(4)

        # Hint / translated sentence
        self.tape_words = QLabel("Hold a letter steady to add it. Use Space between words.")
        self.tape_words.setFont(QFont('Segoe UI', 10))
        self.tape_words.setStyleSheet("color: #56648c; background: transparent;")
        self.tape_words.setWordWrap(True)
        tape_layout.addWidget(self.tape_words)

        # Current spelling letters + Caret
        self.tape_current = QLabel("")
        self.tape_current.setFont(QFont('Segoe UI', 24, QFont.Bold))
        self.tape_current.setStyleSheet("color: #0b1633; background: transparent; letter-spacing: 2px;")
        tape_layout.addWidget(self.tape_current)

        tape_card.setLayout(tape_layout)
        app_layout.addWidget(tape_card)

        # -------------------------------------------------------------
        # 4. CONTROLS: [Delete] [Add letter] [Space] [Clear all] [Speak]
        # -------------------------------------------------------------
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setProperty("class", "btn-ctrl")
        self.delete_btn.clicked.connect(self.on_delete_clicked)
        controls_layout.addWidget(self.delete_btn, stretch=1)

        self.add_btn = QPushButton("Add letter")
        self.add_btn.setObjectName("addButton")
        self.add_btn.setProperty("class", "btn-ctrl")
        self.add_btn.clicked.connect(self.on_add_clicked)
        controls_layout.addWidget(self.add_btn, stretch=2)

        self.space_btn = QPushButton("Space")
        self.space_btn.setProperty("class", "btn-ctrl")
        self.space_btn.clicked.connect(self.on_space_clicked)
        controls_layout.addWidget(self.space_btn, stretch=1)

        self.clear_btn = QPushButton("Clear all")
        self.clear_btn.setObjectName("clearButton")
        self.clear_btn.setProperty("class", "btn-ctrl")
        self.clear_btn.clicked.connect(self.on_clear_clicked)
        controls_layout.addWidget(self.clear_btn, stretch=1)

        self.speak_btn = QPushButton("Speak")
        self.speak_btn.setObjectName("speakButton")
        self.speak_btn.setProperty("class", "btn-ctrl")
        self.speak_btn.clicked.connect(self.on_speak_clicked)
        controls_layout.addWidget(self.speak_btn, stretch=1)

        app_layout.addLayout(controls_layout)

        # Caret blink timer
        self.caret_visible = True
        self.caret_timer = QTimer(self)
        self.caret_timer.timeout.connect(self.toggle_caret)
        self.caret_timer.start(550)

    def keyPressEvent(self, event):
        """Keyboard shortcuts matching mobile experience"""
        if event.key() == Qt.Key_Space:
            if self.last_prediction and self.last_confidence >= 0.55:
                self.on_add_clicked()
            else:
                self.on_space_clicked()
        elif event.key() == Qt.Key_Backspace:
            self.on_delete_clicked()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_T):
            self.on_speak_clicked()
        elif event.key() in (Qt.Key_C, Qt.Key_Escape):
            self.on_clear_clicked()
        elif event.key() == Qt.Key_L:
            self.open_tutor()
        elif event.key() == Qt.Key_P:
            self.open_pamphlet()
        elif event.key() == Qt.Key_S:
            self.open_settings()
        else:
            super().keyPressEvent(event)

    def start_camera(self):
        if self.video_thread is None or not self.video_thread.isRunning():
            camera_id = self.config.get('camera', 'device_id', default=0)
            self.video_thread = VideoThread(self.inference_engine, camera_id)
            self.video_thread.mirror_display = self.mirror_camera_enabled
            self.video_thread.guide_enabled = self.guide_enabled
            self.video_thread.guide_image = self.load_guide_image(self.current_tutor_letter)
            self.video_thread.guide_letter = self.current_tutor_letter
            self.video_thread.guide_opacity = self.guide_opacity
            self.video_thread.change_pixmap_signal.connect(self.update_frame)
            self.video_thread.prediction_signal.connect(self.update_prediction)
            self.video_thread.start()

    def stop_camera(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def update_prediction(self, prediction, confidence):
        self.last_prediction = prediction
        self.last_confidence = confidence

        if not prediction:
            self.glyph_letter.setText("—")
            self.glyph_box.setStyleSheet("""
                QFrame {
                    background-color: rgba(11, 22, 51, 0.82);
                    border: 2px dashed #26386a;
                    border-radius: 12px;
                }
            """)
            self.glyph_caption.setText("Show a letter")
            self.glyph_hold_bar.setValue(0)
            self.hold_count = 0
            return

        # Active glyph display
        self.glyph_letter.setText(prediction)
        self.glyph_box.setStyleSheet("""
            QFrame {
                background-color: rgba(11, 22, 51, 0.88);
                border: 2px solid #ffb23f;
                border-radius: 12px;
            }
        """)
        self.glyph_caption.setText(f"{confidence:.0%} Match")

        # Auto-capture hold-to-add logic (identical to mobile HOLD_TO_ADD_MS = 900ms)
        if self.auto_capture_enabled and confidence >= 0.60:
            if self.cooldown_frames > 0:
                self.cooldown_frames -= 1
            elif prediction == self.last_held_char:
                self.hold_count += 1
                progress = min(1.0, self.hold_count / float(self.hold_target))
                self.glyph_hold_bar.setValue(int(progress * 100))

                if self.hold_count >= self.hold_target:
                    # Hold completed: Add letter to tape!
                    self.append_letter(prediction)
                    self.hold_count = 0
                    self.glyph_hold_bar.setValue(0)
                    self.cooldown_frames = 12  # ~400ms cooldown so it doesn't duplicate
            else:
                self.last_held_char = prediction
                self.hold_count = 1
                self.glyph_hold_bar.setValue(int(100 / self.hold_target))
        else:
            self.hold_count = 0
            self.glyph_hold_bar.setValue(0)

        # Notify Tutor dialog if open
        if self.tutor_dialog and self.tutor_dialog.isVisible():
            self.tutor_dialog.update_match(prediction, confidence)

    def append_letter(self, letter):
        self.spelled_letters.append(letter)
        self.update_tape_display()

    def update_tape_display(self):
        text = "".join(self.spelled_letters)
        cursor = "|" if self.caret_visible else " "
        self.tape_current.setText(f"{text}{cursor}")

    def toggle_caret(self):
        self.caret_visible = not self.caret_visible
        self.update_tape_display()

    def on_add_clicked(self):
        if self.last_prediction and self.last_confidence >= 0.50:
            self.append_letter(self.last_prediction)

    def on_delete_clicked(self):
        if self.spelled_letters:
            self.spelled_letters.pop()
            self.update_tape_display()

    def on_space_clicked(self):
        if self.spelled_letters and self.spelled_letters[-1] != " ":
            self.spelled_letters.append(" ")
            self.update_tape_display()

            # Process sentence through grammar correction
            raw_text = "".join(self.spelled_letters).strip()
            words = raw_text.split()
            if words:
                try:
                    sentence = self.inference_engine.grammar_corrector.correct(words)
                    if sentence:
                        self.translated_sentence = sentence
                        self.tape_words.setText(f"✓ \"{sentence}\"")
                        self.tape_words.setStyleSheet("color: #0b1633; font-weight: bold; font-size: 11pt;")
                except Exception:
                    pass

    def on_clear_clicked(self):
        self.spelled_letters = []
        self.translated_sentence = ""
        self.tape_words.setText("Hold a letter steady to add it. Use Space between words.")
        self.tape_words.setStyleSheet("color: #56648c; font-size: 10pt;")
        self.update_tape_display()

    def on_speak_clicked(self):
        text_to_speak = self.translated_sentence if self.translated_sentence else "".join(self.spelled_letters).strip()
        if text_to_speak:
            try:
                self.inference_engine.tts_engine.speak(text_to_speak)
            except Exception as e:
                print(f"[TTS Error] {e}")

    def on_tutor_letter_changed(self, letter):
        self.current_tutor_letter = str(letter).upper()
        if self.video_thread:
            self.video_thread.guide_letter = self.current_tutor_letter
            self.video_thread.guide_image = self.load_guide_image(self.current_tutor_letter)

    def open_tutor(self):
        self.tutor_dialog = TutorDialog(self)
        self.tutor_dialog.exec_()

    def open_pamphlet(self):
        dialog = PamphletDialog(self)
        dialog.exec_()

    def open_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec_()

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


def main():
    import argparse
    from utils.config_loader import get_config

    config = get_config('config.yaml')
    default_model_path = config['paths']['model_path']

    parser = argparse.ArgumentParser(description='ISL Fingerspell Application')
    parser.add_argument('--model', type=str, default=default_model_path,
                        help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = ISLGUIApp(args.model, args.config)
    window.show()
    window.start_camera()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
