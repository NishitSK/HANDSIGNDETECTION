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
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QPointF
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QPalette, QColor, QPainter, QBrush, QPen,
    QPainterPath, QLinearGradient, QRadialGradient
)

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

                        if guide_resized.ndim == 3 and guide_resized.shape[2] == 4:
                            # 4-channel transparent RGBA skeleton overlay
                            alpha_channel = (guide_resized[:, :, 3] / 255.0) * alpha
                            bgr_guide = guide_resized[:, :, :3]
                            for ch_i in range(3):
                                roi[:, :, ch_i] = np.clip(
                                    (1.0 - alpha_channel) * roi[:, :, ch_i] + alpha_channel * bgr_guide[:, :, ch_i],
                                    0, 255
                                ).astype(np.uint8)
                            processed_frame[y_offset:y_offset + guide_size, x_offset:x_offset + guide_size] = roi
                        else:
                            bgr_guide = guide_resized[:, :, :3] if guide_resized.ndim == 3 else guide_resized
                            blended = cv2.addWeighted(roi, 1.0 - alpha, bgr_guide, alpha, 0)
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


ADAPTIVE_VIDEO_INSTRUCTIONS = {
    'J': {
        'title': "Letter J — Adaptive Motion Gesture",
        'is_adaptive': True,
        'badge': "🌊 ADAPTIVE MOTION SIGN (Sequence GRU Tracked)",
        'steps': [
            ("Phase 1: Starting Anchor", "Hold non-dominant palm flat facing you with fingers upright. Poise dominant index fingertip touching the tip of non-dominant ring finger."),
            ("Phase 2: Downward Stroke", "Sweep dominant index finger straight downward across the length of the palm in a smooth, continuous vertical vector."),
            ("Phase 3: Curved Terminal Hook", "At the lower edge of the palm, hook fingertip smoothly outward and upward to the left, carving the distinct tail of 'J'."),
            ("Phase 4: Adaptive GRU Detection", "The 99.39% GRU sequence model tracks this continuous 30-frame spatial-temporal curvature trajectory in real time.")
        ],
        'kinematic_tip': "Maintain steady finger speed. A 0.8-second motion arc delivers the highest recognition confidence in the GRU model."
    },
    'Z': {
        'title': "Letter Z — Adaptive Motion Gesture",
        'is_adaptive': True,
        'badge': "🌊 ADAPTIVE MOTION SIGN (Sequence GRU Tracked)",
        'steps': [
            ("Phase 1: Starting Anchor", "Hold non-dominant hand flat as a horizontal baseline. Poise dominant index finger extended at top-left (~15 cm in front of chest)."),
            ("Phase 2: Stroke 1 (Top Bar)", "Draw a crisp horizontal line from left to right (~12-15 cm across the camera frame)."),
            ("Phase 3: Stroke 2 (Diagonal Slash)", "Cut sharply diagonally downward and to the left at a 45° angle back to the vertical origin line."),
            ("Phase 4: Stroke 3 (Bottom Bar)", "Trace a final horizontal line from left to right along the bottom plane to complete the letter 'Z'."),
            ("Phase 5: Adaptive GRU Detection", "The GRU recurrent network detects the distinct sharp inflection angles and velocities across all 30 frames.")
        ],
        'kinematic_tip': "Pause momentarily at each corner vertex to emphasize the directional angle changes for the temporal tracker."
    }
}


class MotionVideoGuideCanvas(QWidget):
    """
    Animated video trajectory guide widget for dynamic and adaptive ISL signs.
    Simulates a high-frame-rate motion video instruction showing trajectory,
    directional vectors, keyframe waypoints, and animated hand contact beacons.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(220, 160)
        self.letter = 'J'
        self.is_playing = True
        self.speed = 1.0  # 1.0x or 0.5x
        self.t = 0.0  # 0.0 to 1.0
        self.trail = []  # list of (x, y)

        # 30 FPS smooth animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._step)
        self.timer.start(33)

    def setLetter(self, letter):
        self.letter = str(letter).upper()
        self.t = 0.0
        self.trail.clear()
        self.update()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.update()

    def toggle_speed(self):
        self.speed = 0.5 if self.speed == 1.0 else 1.0
        self.update()

    def reset(self):
        self.t = 0.0
        self.trail.clear()
        self.update()

    def _step(self):
        if not self.is_playing:
            return
        dt = 0.016 * self.speed
        self.t += dt
        if self.t > 1.25:  # Brief pause at loop completion
            self.t = 0.0
            self.trail.clear()
        self.update()

    def _get_coords(self, t_val):
        w = float(self.width())
        h = float(self.height())
        t_clamped = min(1.0, max(0.0, t_val))

        if self.letter == 'J':
            if t_clamped <= 0.60:
                u = t_clamped / 0.60
                x = 0.58 * w
                y = (0.20 + u * (0.64 - 0.20)) * h
            else:
                u = (t_clamped - 0.60) / 0.40
                p0 = np.array([0.58 * w, 0.64 * h])
                p1 = np.array([0.58 * w, 0.88 * h])
                p2 = np.array([0.36 * w, 0.88 * h])
                p3 = np.array([0.26 * w, 0.68 * h])
                pt = (1 - u)**3 * p0 + 3 * (1 - u)**2 * u * p1 + 3 * (1 - u) * u**2 * p2 + u**3 * p3
                x, y = pt[0], pt[1]
            return x, y

        elif self.letter == 'Z':
            if t_clamped <= 0.33:
                u = t_clamped / 0.33
                x = (0.25 + u * 0.50) * w
                y = 0.25 * h
            elif t_clamped <= 0.66:
                u = (t_clamped - 0.33) / 0.33
                x = (0.75 - u * 0.50) * w
                y = (0.25 + u * 0.50) * h
            else:
                u = (t_clamped - 0.66) / 0.34
                x = (0.25 + u * 0.50) * w
                y = 0.75 * h
            return x, y

        else:
            # Static signs: Pulsing anchor
            x = 0.50 * w
            y = 0.50 * h
            return x, y

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Canvas background (dark obsidian matching UI)
        painter.setBrush(QBrush(QColor(5, 11, 28)))
        painter.setPen(QPen(QColor(38, 56, 106), 1.5))
        painter.drawRoundedRect(0, 0, w, h, 10, 10)

        # Subtle coordinate grid
        grid_pen = QPen(QColor(15, 28, 64), 1, Qt.DotLine)
        painter.setPen(grid_pen)
        for gx in range(30, w, 40):
            painter.drawLine(gx, 0, gx, h)
        for gy in range(25, h, 35):
            painter.drawLine(0, gy, w, gy)

        # Video instruction watermark
        painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
        painter.setPen(QPen(QColor(86, 100, 140)))
        mode_tag = "MOTION VIDEO (GRU)" if self.letter in ('J', 'Z') else "STATIC POSE GUIDE"
        painter.drawText(8, 16, f"● {mode_tag}")
        pct = int(min(1.0, self.t) * 100)
        speed_lbl = f"{self.speed:.1f}x"
        painter.drawText(w - 75, 16, f"{pct}% | {speed_lbl}")

        if self.letter == 'J':
            # Draw Palm reference silhouette
            palm_pen = QPen(QColor(25, 42, 85), 1.5, Qt.DashLine)
            painter.setPen(palm_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(int(w * 0.42), int(h * 0.22), int(w * 0.38), int(h * 0.60), 8, 8)
            painter.drawText(int(w * 0.46), int(h * 0.52), "PALM")

            # Static trajectory path
            path = QPainterPath()
            path.moveTo(w * 0.58, h * 0.20)
            path.lineTo(w * 0.58, h * 0.64)
            path.cubicTo(w * 0.58, h * 0.88, w * 0.36, h * 0.88, w * 0.26, h * 0.68)
            painter.setPen(QPen(QColor(63, 178, 255, 90), 3, Qt.DashLine))
            painter.drawPath(path)

            # Waypoint markers
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(63, 178, 255)))
            painter.drawEllipse(QPointF(w * 0.58, h * 0.20), 4, 4)
            painter.drawEllipse(QPointF(w * 0.26, h * 0.68), 4, 4)

            painter.setFont(QFont('Segoe UI', 7, QFont.Bold))
            painter.setPen(QPen(QColor(147, 161, 198)))
            painter.drawText(int(w * 0.62), int(h * 0.22), "1. Start")
            painter.drawText(int(w * 0.12), int(h * 0.68), "2. Hook")

        elif self.letter == 'Z':
            # Static trajectory lines
            painter.setPen(QPen(QColor(63, 178, 255, 90), 3, Qt.DashLine))
            painter.drawLine(int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.25))
            painter.drawLine(int(w * 0.75), int(h * 0.25), int(w * 0.25), int(h * 0.75))
            painter.drawLine(int(w * 0.25), int(h * 0.75), int(w * 0.75), int(h * 0.75))

            # Waypoints 1, 2, 3, 4
            pts = [
                (w * 0.25, h * 0.25, "1"), (w * 0.75, h * 0.25, "2"),
                (w * 0.25, h * 0.75, "3"), (w * 0.75, h * 0.75, "4")
            ]
            painter.setFont(QFont('Segoe UI', 7, QFont.Bold))
            for px, py, tag in pts:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(63, 178, 255)))
                painter.drawEllipse(QPointF(px, py), 4, 4)
                painter.setPen(QPen(QColor(147, 161, 198)))
                painter.drawText(int(px - 10 if px > w * 0.5 else px + 6), int(py - 4), tag)

        else:
            # Static letters: pulsing concentric alignment rings
            cx = w * 0.50
            cy = h * 0.50
            pulse_r = 16 + int(8 * np.sin(self.t * 6.28))
            painter.setPen(QPen(QColor(255, 178, 63, 80), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), pulse_r, pulse_r)
            painter.drawEllipse(QPointF(cx, cy), pulse_r + 14, pulse_r + 14)
            painter.setFont(QFont('Segoe UI', 9, QFont.Bold))
            painter.setPen(QPen(QColor(244, 246, 251)))
            painter.drawText(int(cx - 38), int(cy + 4), f"SIGN '{self.letter}'")

        # Current Animated Beacon
        cur_x, cur_y = self._get_coords(self.t)
        self.trail.append((cur_x, cur_y))
        if len(self.trail) > 14:
            self.trail.pop(0)

        # Draw fading neon trail
        for i, (tx, ty) in enumerate(self.trail[:-1]):
            trail_alpha = int(180 * (i / float(len(self.trail))))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 178, 63, trail_alpha)))
            rad = 2 + int(3 * (i / float(len(self.trail))))
            painter.drawEllipse(QPointF(tx, ty), rad, rad)

        # Draw lead golden beacon
        glow = QRadialGradient(cur_x, cur_y, 16)
        glow.setColorAt(0.0, QColor(255, 178, 63, 230))
        glow.setColorAt(0.5, QColor(255, 178, 63, 110))
        glow.setColorAt(1.0, QColor(255, 178, 63, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(QPointF(cur_x, cur_y), 16, 16)

        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QPointF(cur_x, cur_y), 3.5, 3.5)

        # Phase label at bottom
        painter.setFont(QFont('Segoe UI', 8))
        painter.setPen(QPen(QColor(244, 246, 251)))
        if self.letter == 'J':
            lbl = "Phase: Downward Sweep" if self.t <= 0.60 else "Phase: Curved Terminal Hook"
        elif self.letter == 'Z':
            if self.t <= 0.33:
                lbl = "Phase: Stroke 1 (Top Bar)"
            elif self.t <= 0.66:
                lbl = "Phase: Stroke 2 (Diagonal Slash)"
            else:
                lbl = "Phase: Stroke 3 (Bottom Bar)"
        else:
            lbl = "Pose: Align hand steady"
        painter.drawText(8, h - 8, lbl)


class TutorDialog(QDialog):
    """Sign Language Tutor Dialog with 3 Guide Modes: Photo, Skeleton & Motion Video Instructions"""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Sign Language Tutor")
        self.setFixedWidth(540)
        self.guide_style = getattr(self.parent_app, 'guide_style', 'photo')
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
            QPushButton.btn-mode {
                background-color: #0b1633;
                color: #93a1c6;
                border: 1px solid #26386a;
                border-radius: 8px;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 10px;
            }
            QPushButton.btn-mode:hover {
                background-color: #26386a;
                color: #f4f6fb;
            }
            QPushButton.btn-mode-active {
                background-color: #ffb23f;
                color: #0b1633;
                border: 1px solid #ffb23f;
                border-radius: 8px;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 10px;
            }
            QPushButton.btn-vid-ctrl {
                background-color: #0b1633;
                color: #f4f6fb;
                border: 1px solid #26386a;
                border-radius: 6px;
                font-size: 10px;
                font-weight: bold;
                padding: 4px 8px;
            }
            QPushButton.btn-vid-ctrl:hover {
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
        layout.setContentsMargins(22, 18, 22, 20)
        layout.setSpacing(12)

        # Header
        title = QLabel("SIGN LANGUAGE TUTOR & VIDEO INSTRUCTIONS")
        title.setFont(QFont('Segoe UI', 14, QFont.Bold))
        title.setStyleSheet("letter-spacing: 1px; color: #f4f6fb;")
        layout.addWidget(title)

        subtitle = QLabel("Practice handshapes with Photo, Skeleton, or Live Video Trajectory Instructions")
        subtitle.setStyleSheet("color: #93a1c6; font-size: 11px;")
        layout.addWidget(subtitle)

        # Guide Mode Selector (Photo / Skeleton / Motion Video)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        mode_lbl = QLabel("Guide Style:")
        mode_lbl.setStyleSheet("color: #93a1c6; font-size: 11px; font-weight: bold;")
        mode_row.addWidget(mode_lbl)

        self.btn_photo = QPushButton("📷 Photo")
        self.btn_photo.clicked.connect(lambda: self.set_guide_mode('photo'))
        mode_row.addWidget(self.btn_photo)

        self.btn_skel = QPushButton("🦴 Skeleton")
        self.btn_skel.clicked.connect(lambda: self.set_guide_mode('skeleton'))
        mode_row.addWidget(self.btn_skel)

        self.btn_video = QPushButton("🎬 Motion Video")
        self.btn_video.clicked.connect(lambda: self.set_guide_mode('video'))
        mode_row.addWidget(self.btn_video)

        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Main Tutor Card
        card = QFrame()
        card.setStyleSheet("background-color: #0b1633; border: 1px solid #26386a; border-radius: 12px;")
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(10)

        # Navigation and Letter selector row
        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)

        prev_btn = QPushButton("◀")
        prev_btn.setProperty("class", "btn-nav")
        prev_btn.clicked.connect(self.prev_letter)
        nav_row.addWidget(prev_btn)

        self.letter_combo = QComboBox()
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            suffix = " 🌊 [Motion]" if c in ('J', 'Z') else ""
            self.letter_combo.addItem(f"Letter {c}{suffix}", c)
        self.letter_combo.currentIndexChanged.connect(self.on_combo_changed)
        nav_row.addWidget(self.letter_combo, stretch=1)

        next_btn = QPushButton("▶")
        next_btn.setProperty("class", "btn-nav")
        next_btn.clicked.connect(self.next_letter)
        nav_row.addWidget(next_btn)

        card_layout.addLayout(nav_row)

        # Display Stage: Dual views (Canvas for Video vs Image Label for Photo/Skeleton)
        stage_row = QHBoxLayout()
        stage_row.setSpacing(12)

        # Left: Visual Media Container
        left_box = QVBoxLayout()
        left_box.setSpacing(6)

        self.video_canvas = MotionVideoGuideCanvas(self)
        left_box.addWidget(self.video_canvas)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(220, 160)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background: #050b1c; border: 1px solid #26386a; border-radius: 10px;")
        left_box.addWidget(self.preview_label)

        # Video Player Controls
        self.vid_ctrl_row = QHBoxLayout()
        self.vid_ctrl_row.setSpacing(4)
        self.play_btn = QPushButton("⏸ Pause")
        self.play_btn.setProperty("class", "btn-vid-ctrl")
        self.play_btn.clicked.connect(self.toggle_canvas_play)
        self.vid_ctrl_row.addWidget(self.play_btn)

        self.speed_btn = QPushButton("⚡ 1.0x")
        self.speed_btn.setProperty("class", "btn-vid-ctrl")
        self.speed_btn.clicked.connect(self.toggle_canvas_speed)
        self.vid_ctrl_row.addWidget(self.speed_btn)

        self.replay_btn = QPushButton("↺ Replay")
        self.replay_btn.setProperty("class", "btn-vid-ctrl")
        self.replay_btn.clicked.connect(self.video_canvas.reset)
        self.vid_ctrl_row.addWidget(self.replay_btn)

        left_box.addLayout(self.vid_ctrl_row)
        stage_row.addLayout(left_box)

        # Right: Details & Video Instructions
        right_box = QVBoxLayout()
        right_box.setSpacing(6)

        self.target_title = QLabel("Target: Letter A")
        self.target_title.setFont(QFont('Segoe UI', 13, QFont.Bold))
        self.target_title.setStyleSheet("color: #ffb23f;")
        right_box.addWidget(self.target_title)

        # Adaptive Motion Tag Badge
        self.adaptive_badge = QLabel("✋ STATIC POSE (Residual MLP Tracked)")
        self.adaptive_badge.setStyleSheet("""
            background-color: rgba(63, 178, 255, 0.15);
            border: 1px solid #3fb2ff;
            border-radius: 6px;
            padding: 4px 8px;
            color: #3fb2ff;
            font-size: 10px;
            font-weight: bold;
        """)
        right_box.addWidget(self.adaptive_badge)

        # Instructions Scroll Area
        inst_scroll = QScrollArea()
        inst_scroll.setFixedHeight(120)
        inst_scroll.setWidgetResizable(True)
        inst_scroll.setStyleSheet("background: transparent; border: none;")

        self.instructions_widget = QWidget()
        self.instructions_layout = QVBoxLayout(self.instructions_widget)
        self.instructions_layout.setContentsMargins(0, 0, 4, 0)
        self.instructions_layout.setSpacing(4)

        self.hint_label = QLabel(ISL_LETTER_HINTS.get('A', ''))
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #f4f6fb; font-size: 11px; line-height: 1.3;")
        self.instructions_layout.addWidget(self.hint_label)

        inst_scroll.setWidget(self.instructions_widget)
        right_box.addWidget(inst_scroll)

        stage_row.addLayout(right_box, stretch=1)
        card_layout.addLayout(stage_row)

        card.setLayout(card_layout)
        layout.addWidget(card)

        # Ghost guide overlay controls
        self.guide_checkbox = QCheckBox("Show live guide overlay on camera")
        self.guide_checkbox.setChecked(self.parent_app.guide_enabled)
        self.guide_checkbox.toggled.connect(self.toggle_guide)
        layout.addWidget(self.guide_checkbox)

        # Opacity slider
        opacity_row = QHBoxLayout()
        self.opacity_title = QLabel("Guide Opacity: 50%")
        self.opacity_title.setStyleSheet("color: #93a1c6; font-size: 11px;")
        opacity_row.addWidget(self.opacity_title)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(15, 90)
        self.opacity_slider.setValue(int(self.parent_app.guide_opacity * 100))
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_row.addWidget(self.opacity_slider, stretch=1)
        layout.addLayout(opacity_row)

        # Real-time Match status badge
        self.status_badge = QLabel("Align hand with guide to practice")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setStyleSheet("""
            background-color: rgba(38, 56, 106, 0.4);
            border: 1px solid #26386a;
            border-radius: 8px;
            padding: 8px;
            color: #93a1c6;
            font-weight: 600;
            font-size: 11px;
        """)
        layout.addWidget(self.status_badge)

        # Done button
        done_btn = QPushButton("Done")
        done_btn.setFixedHeight(38)
        done_btn.setStyleSheet("""
            QPushButton {
                background-color: #f4f6fb;
                color: #0b1633;
                border: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #ffb23f;
            }
        """)
        done_btn.clicked.connect(self.accept)
        layout.addWidget(done_btn)

        self.setLayout(layout)
        self.set_guide_mode(self.guide_style, update_view_only=True)
        self.sync_letter(self.parent_app.current_tutor_letter)

    def set_guide_mode(self, mode, update_view_only=False):
        self.guide_style = mode
        if not update_view_only:
            self.parent_app.guide_style = mode
            if self.parent_app.video_thread:
                self.parent_app.video_thread.guide_image = self.parent_app.load_guide_image(
                    self.parent_app.current_tutor_letter, style=mode
                )

        # Update button visual styling
        for btn, m in [(self.btn_photo, 'photo'), (self.btn_skel, 'skeleton'), (self.btn_video, 'video')]:
            btn.setProperty("class", "btn-mode-active" if self.guide_style == m else "btn-mode")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # Toggle canvas vs image label
        if self.guide_style == 'video':
            self.video_canvas.show()
            self.preview_label.hide()
            for i in range(self.vid_ctrl_row.count()):
                w = self.vid_ctrl_row.itemAt(i).widget()
                if w:
                    w.show()
        else:
            self.video_canvas.hide()
            self.preview_label.show()
            for i in range(self.vid_ctrl_row.count()):
                w = self.vid_ctrl_row.itemAt(i).widget()
                if w:
                    w.hide()

        self.sync_letter(self.parent_app.current_tutor_letter)

    def toggle_canvas_play(self):
        self.video_canvas.toggle_play()
        self.play_btn.setText("▶ Play" if not self.video_canvas.is_playing else "⏸ Pause")

    def toggle_canvas_speed(self):
        self.video_canvas.toggle_speed()
        self.speed_btn.setText("🐢 0.5x" if self.video_canvas.speed == 0.5 else "⚡ 1.0x")

    def sync_letter(self, letter):
        idx = ord(letter) - ord('A')
        self.letter_combo.blockSignals(True)
        self.letter_combo.setCurrentIndex(idx)
        self.letter_combo.blockSignals(False)

        self.target_title.setText(f"Target: Letter {letter}")
        self.video_canvas.setLetter(letter)

        # Clear existing instructions widgets
        while self.instructions_layout.count():
            item = self.instructions_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Check if letter has structured adaptive video instructions
        if letter in ADAPTIVE_VIDEO_INSTRUCTIONS:
            info = ADAPTIVE_VIDEO_INSTRUCTIONS[letter]
            self.adaptive_badge.setText(info['badge'])
            self.adaptive_badge.setStyleSheet("""
                background-color: rgba(255, 178, 63, 0.2);
                border: 1px solid #ffb23f;
                border-radius: 6px;
                padding: 4px 8px;
                color: #ffb23f;
                font-size: 10px;
                font-weight: bold;
            """)

            # Build step-by-step video instruction cards
            for step_title, step_desc in info['steps']:
                s_box = QFrame()
                s_box.setStyleSheet("background: #050b1c; border-left: 3px solid #ffb23f; border-radius: 4px; padding: 4px;")
                s_layout = QVBoxLayout(s_box)
                s_layout.setContentsMargins(6, 4, 6, 4)
                s_layout.setSpacing(2)

                t_lbl = QLabel(step_title)
                t_lbl.setFont(QFont('Segoe UI', 9, QFont.Bold))
                t_lbl.setStyleSheet("color: #ffb23f;")
                s_layout.addWidget(t_lbl)

                d_lbl = QLabel(step_desc)
                d_lbl.setFont(QFont('Segoe UI', 8))
                d_lbl.setWordWrap(True)
                d_lbl.setStyleSheet("color: #f4f6fb;")
                s_layout.addWidget(d_lbl)

                self.instructions_layout.addWidget(s_box)

            # Kinematic velocity tip
            tip_lbl = QLabel(f"💡 Kinematic Tip: {info['kinematic_tip']}")
            tip_lbl.setWordWrap(True)
            tip_lbl.setFont(QFont('Segoe UI', 8, QFont.StyleItalic))
            tip_lbl.setStyleSheet("color: #93a1c6; padding-top: 4px;")
            self.instructions_layout.addWidget(tip_lbl)
        else:
            self.adaptive_badge.setText("✋ STATIC POSE (Residual MLP Tracked)")
            self.adaptive_badge.setStyleSheet("""
                background-color: rgba(63, 178, 255, 0.15);
                border: 1px solid #3fb2ff;
                border-radius: 6px;
                padding: 4px 8px;
                color: #3fb2ff;
                font-size: 10px;
                font-weight: bold;
            """)

            hint_text = ISL_LETTER_HINTS.get(letter, "Observe the guide image and align hand posture.")
            h_lbl = QLabel(f"<b>Execution Guide:</b><br>{hint_text}")
            h_lbl.setWordWrap(True)
            h_lbl.setFont(QFont('Segoe UI', 9))
            h_lbl.setStyleSheet("color: #f4f6fb; line-height: 1.4;")
            self.instructions_layout.addWidget(h_lbl)

            tip_lbl = QLabel("💡 Tip: Hold handshape steady for 800ms. The 99.76% MLP classifier will auto-confirm.")
            tip_lbl.setWordWrap(True)
            tip_lbl.setFont(QFont('Segoe UI', 8, QFont.StyleItalic))
            tip_lbl.setStyleSheet("color: #93a1c6; padding-top: 6px;")
            self.instructions_layout.addWidget(tip_lbl)

        # Image preview for Photo / Skeleton modes
        if self.guide_style != 'video':
            img = self.parent_app.load_guide_image(letter, style=self.guide_style)
            if img is not None:
                if img.ndim == 3 and img.shape[2] == 4:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                    h_i, w_i, _ = rgb.shape
                    qimg = QImage(rgb.data, w_i, h_i, 4 * w_i, QImage.Format_RGBA8888)
                else:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h_i, w_i, ch = rgb.shape
                    qimg = QImage(rgb.data, w_i, h_i, ch * w_i, QImage.Format_RGB888)

                pix = QPixmap.fromImage(qimg).scaled(220, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_label.setPixmap(pix)
            else:
                self.preview_label.setText(f"Sign {letter}")

    def on_combo_changed(self, idx):
        letter = chr(ord('A') + idx)
        # For adaptive motion gestures (J, Z), auto-switch to Motion Video guide
        if letter in ('J', 'Z') and self.guide_style != 'video':
            self.set_guide_mode('video', update_view_only=True)

        self.parent_app.on_tutor_letter_changed(letter)
        self.sync_letter(letter)

    def prev_letter(self):
        curr = self.parent_app.current_tutor_letter
        new_l = chr(ord('A') + (ord(curr) - ord('A') - 1) % 26)
        if new_l in ('J', 'Z') and self.guide_style != 'video':
            self.set_guide_mode('video', update_view_only=True)
        self.parent_app.on_tutor_letter_changed(new_l)
        self.sync_letter(new_l)

    def next_letter(self):
        curr = self.parent_app.current_tutor_letter
        new_l = chr(ord('A') + (ord(curr) - ord('A') + 1) % 26)
        if new_l in ('J', 'Z') and self.guide_style != 'video':
            self.set_guide_mode('video', update_view_only=True)
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
                padding: 8px;
                color: #3fb950;
                font-weight: bold;
                font-size: 11px;
            """)
        else:
            self.status_badge.setText(f"Target: {target} | Detected: {prediction if prediction else '...'} ({confidence:.0%})")
            self.status_badge.setStyleSheet("""
                background-color: rgba(255, 178, 63, 0.15);
                border: 1px solid #ffb23f;
                border-radius: 8px;
                padding: 8px;
                color: #ffb23f;
                font-weight: 600;
                font-size: 11px;
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
            "⚡ Task-Adaptive (MLP Static 99.7% + GRU Motion 99.4%)",
            "✋ MLP Only (Fast Static Handsigns)",
            "🌊 GRU Only (Sequential Motion Gestures)"
        ])
        mode = getattr(self.parent_app.inference_engine, 'ensemble_mode', 'task_adaptive')
        if mode == 'mlp_only':
            self.engine_combo.setCurrentIndex(1)
        elif mode == 'gru_only':
            self.engine_combo.setCurrentIndex(2)
        else:
            self.engine_combo.setCurrentIndex(0)
        layout.addWidget(self.engine_combo)

        # Readout info
        readout = QLabel("Dual Models: 99.76% Deep Residual MLP + 99.39% Sequential GRU\nAdaptive Pipeline: Kinematic velocity & curvature automatic gating")
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
        modes = ['task_adaptive', 'mlp_only', 'gru_only']
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
        self.guide_style = 'photo'
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

    def load_guide_image(self, letter, style='photo'):
        letter = str(letter).upper()
        cache_key = (letter, style)
        if cache_key in self.guide_images_cache:
            return self.guide_images_cache[cache_key]

        if style == 'skeleton':
            pamphlet_path = PROJECT_ROOT / 'mobile' / 'app' / 'pamphlet' / 'isl-skeleton' / f"{letter}.png"
            if pamphlet_path.exists():
                img = cv2.imread(str(pamphlet_path), cv2.IMREAD_UNCHANGED)
                self.guide_images_cache[cache_key] = img
                return img

        # Default photo
        pamphlet_path = PROJECT_ROOT / 'mobile' / 'app' / 'pamphlet' / 'isl' / f"{letter}.jpg"
        if pamphlet_path.exists():
            img = cv2.imread(str(pamphlet_path))
            self.guide_images_cache[cache_key] = img
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
        engine_str = getattr(self.inference_engine, 'active_engine', 'MLP')
        self.glyph_caption.setText(f"{confidence:.0%} Match • {engine_str}")

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
            self.video_thread.guide_image = self.load_guide_image(
                self.current_tutor_letter, style=getattr(self, 'guide_style', 'photo')
            )

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
