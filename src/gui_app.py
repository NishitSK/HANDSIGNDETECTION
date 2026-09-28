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
        'title': "Letter J — Dynamic Tracing Gesture",
        'is_adaptive': True,
        'badge': "🌊 DYNAMIC MOTION SIGN (99.39% GRU Tracked)",
        'steps': [
            ("Phase 1: Starting Anchor", "Hold non-dominant palm flat facing you with fingers upright. Poise dominant index fingertip touching the tip of non-dominant ring finger."),
            ("Phase 2: Downward Stroke", "Sweep dominant index finger straight downward across the length of the palm in a smooth, continuous vertical vector."),
            ("Phase 3: Curved Terminal Hook", "At the lower edge of the palm, hook fingertip smoothly outward and upward to the left, carving the distinct tail of 'J'."),
            ("Phase 4: GRU Temporal Tracking", "The 99.39% GRU sequence model tracks this continuous 30-frame spatial-temporal curvature trajectory in real time.")
        ],
        'kinematic_tip': "Maintain steady finger speed. A 0.8-second motion arc delivers the highest recognition confidence in the GRU model."
    },
    'Z': {
        'title': "Letter Z — Dynamic Zigzag Gesture",
        'is_adaptive': True,
        'badge': "🌊 DYNAMIC MOTION SIGN (99.39% GRU Tracked)",
        'steps': [
            ("Phase 1: Starting Anchor", "Hold non-dominant hand flat as a horizontal baseline. Poise dominant index finger extended at top-left (~15 cm in front of chest)."),
            ("Phase 2: Stroke 1 (Top Bar)", "Draw a crisp horizontal line from left to right (~12-15 cm across the camera frame)."),
            ("Phase 3: Stroke 2 (Diagonal Slash)", "Cut sharply diagonally downward and to the left at a 45° angle back to the vertical origin line."),
            ("Phase 4: Stroke 3 (Bottom Bar)", "Trace a final horizontal line from left to right along the bottom plane to complete the letter 'Z'."),
            ("Phase 5: GRU Temporal Tracking", "The GRU recurrent network detects the distinct sharp inflection angles and velocities across all 30 frames.")
        ],
        'kinematic_tip': "Pause momentarily at each corner vertex to emphasize the directional angle changes for the temporal tracker."
    },
    'HELLO': {
        'title': "Hello / Greeting — Lateral Waving Gesture",
        'is_adaptive': True,
        'badge': "🌊 CONVERSATIONAL GESTURE (100% GRU Tracked)",
        'steps': [
            ("Phase 1: Temple Anchor", "Raise dominant hand to right temple height, palm flat facing forward with fingers upright and relaxed."),
            ("Phase 2: Lateral Waving Arc", "Sweep the hand smoothly side-to-side in a 12-15 cm horizontal waving motion across the camera frame."),
            ("Phase 3: Rhythmic Oscillation", "Perform a continuous 2-cycle wave with steady hand velocity at eye/temple level."),
            ("Phase 4: GRU Temporal Tracking", "The GRU sequence model detects the lateral hand oscillation, height anchor, and upright palm orientation.")
        ],
        'kinematic_tip': "Keep hand at forehead/temple height. A smooth double-wave triggers immediate high-confidence detection."
    },
    'THANK YOU': {
        'title': "Thank You — Forward Sweeping Gesture",
        'is_adaptive': True,
        'badge': "🌊 CONVERSATIONAL GESTURE (100% GRU Tracked)",
        'steps': [
            ("Phase 1: Chin / Lip Anchor", "Touch the fingertips of your dominant flat hand gently to your chin or lips, palm facing inward toward your face."),
            ("Phase 2: Forward Outward Glide", "Sweep your hand smoothly forward and slightly downward away from your chin toward the camera."),
            ("Phase 3: Open Offering Posture", "End with palm facing upward/forward at mid-chest level in a relaxed, open offering stance."),
            ("Phase 4: GRU Temporal Tracking", "The GRU network captures the forward depth vector (Z/Y displacement) and finger extension posture.")
        ],
        'kinematic_tip': "Ensure a decisive forward sweep away from the chin towards the camera for instantaneous classification."
    },
    'PLEASE': {
        'title': "Please — Circular Chest Rub Gesture",
        'is_adaptive': True,
        'badge': "🌊 CONVERSATIONAL GESTURE (100% GRU Tracked)",
        'steps': [
            ("Phase 1: Heart Anchor", "Place the flat palm of your dominant hand over the center of your chest (sternum/heart level) with fingers angled slightly upward."),
            ("Phase 2: Clockwise Circular Orbit", "Rub your palm in smooth, continuous clockwise circles across your chest (~12 cm diameter)."),
            ("Phase 3: Consistent Contact", "Maintain gentle planar contact with the chest throughout the circular trajectory."),
            ("Phase 4: GRU Temporal Tracking", "The recurrent network matches the continuous circular orbital velocity and planar flat hand posture.")
        ],
        'kinematic_tip': "Keep all fingers extended and together flat against the chest plane rather than curving your fingers."
    },
    'YES': {
        'title': "Yes — Affirmative Fist Nodding Gesture",
        'is_adaptive': True,
        'badge': "🌊 CONVERSATIONAL GESTURE (100% GRU Tracked)",
        'steps': [
            ("Phase 1: Fist Orientation", "Form a closed fist with dominant hand held at mid-chest height, knuckle plane facing the camera."),
            ("Phase 2: Downward Wrist Nod", "Nod the fist downward at the wrist joint, mimicking a head nodding 'yes'."),
            ("Phase 3: Rhythmic Bounce", "Tilt the fist back up and perform 2-3 rhythmic vertical nods with consistent cadence."),
            ("Phase 4: GRU Temporal Tracking", "The GRU temporal cells capture the vertical harmonic pitch oscillations of the wrist and knuckles.")
        ],
        'kinematic_tip': "Pivot primarily at the wrist joint rather than moving your entire arm up and down."
    },
    'NO': {
        'title': "No — Decisive Beak Snap Gesture",
        'is_adaptive': True,
        'badge': "🌊 CONVERSATIONAL GESTURE (100% GRU Tracked)",
        'steps': [
            ("Phase 1: Open Beak Posture", "Extend dominant thumb, index, and middle fingers together, held 4-5 cm apart like an open beak."),
            ("Phase 2: Rapid Closing Snap", "Snap index and middle fingertips sharply downward to pinch firmly against the thumb tip."),
            ("Phase 3: Rhythmic Repeat", "Slightly release and snap closed once more in a crisp, decisive negation cadence."),
            ("Phase 4: GRU Temporal Tracking", "The model detects the rapid finger-convergence velocity and ring/pinky curled retraction.")
        ],
        'kinematic_tip': "Keep ring and pinky fingers tightly curled against palm to highlight the index-middle-thumb convergence."
    },
    'HELP': {
        'title': "Help — Dual-Hand Tandem Lift Gesture",
        'is_adaptive': True,
        'badge': "🌊 CONVERSATIONAL GESTURE (100% GRU Tracked)",
        'steps': [
            ("Phase 1: Base Palm Support", "Hold non-dominant hand flat, palm facing upward, at lower chest height (~20 cm from body)."),
            ("Phase 2: Thumbs-Up Placement", "Form a thumbs-up fist with dominant hand and place the bottom of the fist directly onto the flat palm."),
            ("Phase 3: Tandem Upward Lift", "Lift both hands together smoothly upward by 12-15 cm in a synchronized rescue motion."),
            ("Phase 4: GRU Temporal Tracking", "The GRU network verifies bilateral hand proximity and synchronized upward translation.")
        ],
        'kinematic_tip': "Keep both hands in steady contact throughout the upward lifting stroke."
    },
    'GOOD': {
        'title': "Good / Fine — Chin-to-Palm Gesture",
        'is_adaptive': True,
        'badge': "🌊 CONVERSATIONAL GESTURE (100% GRU Tracked)",
        'steps': [
            ("Phase 1: Chin Touch", "Touch the fingertips of your dominant flat hand to your chin while resting non-dominant open palm facing up below."),
            ("Phase 2: Downward Descent", "Bring dominant hand forward and downward in a steady diagonal trajectory."),
            ("Phase 3: Palm Landing", "Land the back or palm of the dominant hand cleanly into the open non-dominant flat palm."),
            ("Phase 4: GRU Temporal Tracking", "Tracks sequential hand-to-face release followed by bilateral hand convergence.")
        ],
        'kinematic_tip': "A crisp landing of the dominant hand onto the non-dominant palm ensures instant classification."
    },
    'NAMASTE': {
        'title': "Namaste / Greeting — Symmetrical Prayer Gesture",
        'is_adaptive': True,
        'badge': "🌊 CONVERSATIONAL GESTURE (100% GRU Tracked)",
        'steps': [
            ("Phase 1: Lateral Preparation", "Bring both hands upward in front of chest, palms facing inward toward each other."),
            ("Phase 2: Centered Convergence", "Press palms and all 10 fingers flat against each other along the body's vertical midline."),
            ("Phase 3: Heart Posture Hold", "Hold the symmetrical prayer hands upright at chest level with fingers pointing upward."),
            ("Phase 4: GRU Temporal Tracking", "The model detects bilateral palm symmetry, zero lateral velocity, and vertical finger alignment.")
        ],
        'kinematic_tip': "Keep forearms horizontal and elbows relaxed for a clean symmetrical posture."
    }
}

COMMON_GESTURE_NAMES = [
    'HELLO', 'THANK YOU', 'PLEASE', 'YES', 'NO', 'HELP', 'GOOD', 'NAMASTE'
]


class MotionVideoGuideCanvas(QWidget):
    """
    State-of-the-art interactive animated video trajectory guide widget.
    Simulates a high-frame-rate motion video instruction showing trajectory,
    directional vectors, keyframe waypoints, animated hand beacons, and gesture blueprints.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(360, 260)
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
        self.setSign(letter)

    def setSign(self, sign):
        self.letter = str(sign).upper()
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
                y = (0.22 + u * (0.64 - 0.22)) * h
            else:
                u = (t_clamped - 0.60) / 0.40
                p0 = np.array([0.58 * w, 0.64 * h])
                p1 = np.array([0.58 * w, 0.88 * h])
                p2 = np.array([0.34 * w, 0.88 * h])
                p3 = np.array([0.24 * w, 0.68 * h])
                pt = (1 - u)**3 * p0 + 3 * (1 - u)**2 * u * p1 + 3 * (1 - u) * u**2 * p2 + u**3 * p3
                x, y = pt[0], pt[1]
            return x, y

        elif self.letter == 'Z':
            if t_clamped <= 0.33:
                u = t_clamped / 0.33
                x = (0.25 + u * 0.50) * w
                y = 0.26 * h
            elif t_clamped <= 0.66:
                u = (t_clamped - 0.33) / 0.33
                x = (0.75 - u * 0.50) * w
                y = (0.26 + u * 0.48) * h
            else:
                u = (t_clamped - 0.66) / 0.34
                x = (0.25 + u * 0.50) * w
                y = 0.74 * h
            return x, y

        elif self.letter == 'HELLO':
            wave = np.sin(t_clamped * 4 * np.pi)
            x = (0.50 + wave * 0.24) * w
            y = (0.34 + np.sin(t_clamped * 2 * np.pi) * 0.04) * h
            return x, y

        elif self.letter == 'THANK YOU':
            # Starts at chin, sweeps forward and down
            x = (0.50 + 0.08 * t_clamped) * w
            y = (0.30 + 0.44 * t_clamped) * h
            return x, y

        elif self.letter == 'PLEASE':
            # Circular chest rubbing orbit
            angle = t_clamped * 3.5 * np.pi
            x = (0.50 + 0.22 * np.cos(angle)) * w
            y = (0.52 + 0.22 * np.sin(angle)) * h
            return x, y

        elif self.letter == 'YES':
            # Fist nodding up and down at wrist
            nod = np.abs(np.sin(t_clamped * 4 * np.pi))
            x = 0.50 * w
            y = (0.36 + 0.32 * nod) * h
            return x, y

        elif self.letter == 'NO':
            # Index/middle finger closing snap onto thumb
            snap = 0.5 + 0.5 * np.cos(t_clamped * 4 * np.pi)
            x = (0.50 + 0.16 * snap) * w
            y = (0.45 + 0.16 * (1.0 - snap)) * h
            return x, y

        elif self.letter == 'HELP':
            # Tandem upward lift
            x = 0.50 * w
            y = (0.76 - 0.44 * t_clamped) * h
            return x, y

        elif self.letter == 'GOOD':
            if t_clamped <= 0.35:
                x = 0.50 * w
                y = 0.30 * h
            else:
                u = (t_clamped - 0.35) / 0.65
                x = 0.50 * w
                y = (0.30 + u * 0.44) * h
            return x, y

        elif self.letter == 'NAMASTE':
            converge = min(1.0, t_clamped * 1.5)
            x = (0.28 + converge * 0.22) * w
            y = 0.52 * h
            return x, y

        else:
            # Static sign alignment target
            x = 0.50 * w
            y = 0.50 * h
            return x, y

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Canvas background (sleek gradient matching mobile design tokens)
        bg_grad = QLinearGradient(0, 0, w, h)
        bg_grad.setColorAt(0.0, QColor(7, 16, 38))
        bg_grad.setColorAt(1.0, QColor(13, 27, 62))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor(38, 56, 106), 2))
        painter.drawRoundedRect(0, 0, w, h, 14, 14)

        # Subtle coordinate grid
        grid_pen = QPen(QColor(22, 38, 82, 110), 1, Qt.DotLine)
        painter.setPen(grid_pen)
        for gx in range(40, w, 40):
            painter.drawLine(gx, 0, gx, h)
        for gy in range(35, h, 35):
            painter.drawLine(0, gy, w, gy)

        # Video instruction watermark
        painter.setFont(QFont('Segoe UI', 9, QFont.Bold))
        painter.setPen(QPen(QColor(147, 161, 198)))
        is_dynamic = (self.letter in ('J', 'Z') or self.letter in COMMON_GESTURE_NAMES)
        mode_tag = "MOTION VIDEO GUIDE (GRU)" if is_dynamic else "STATIC HANDSHAPE GUIDE"
        painter.drawText(14, 22, f"● {mode_tag}")
        pct = int(min(1.0, self.t) * 100)
        speed_lbl = f"{self.speed:.1f}x"
        painter.drawText(w - 95, 22, f"{pct}% | {speed_lbl}")

        # Vector blueprints for each gesture
        if self.letter == 'J':
            # Palm reference silhouette
            palm_pen = QPen(QColor(38, 56, 106), 1.5, Qt.DashLine)
            painter.setPen(palm_pen)
            painter.setBrush(QBrush(QColor(11, 22, 51, 160)))
            painter.drawRoundedRect(int(w * 0.44), int(h * 0.22), int(w * 0.32), int(h * 0.58), 10, 10)
            painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
            painter.setPen(QPen(QColor(86, 100, 140)))
            painter.drawText(int(w * 0.50), int(h * 0.50), "PALM")

            # Trajectory path
            path = QPainterPath()
            path.moveTo(w * 0.58, h * 0.22)
            path.lineTo(w * 0.58, h * 0.64)
            path.cubicTo(w * 0.58, h * 0.88, w * 0.34, h * 0.88, w * 0.24, h * 0.68)
            painter.setPen(QPen(QColor(63, 178, 255, 120), 4, Qt.DashLine))
            painter.drawPath(path)

            # Waypoint markers
            for pt, lbl in [((w * 0.58, h * 0.22), "1. Start Anchor"), ((w * 0.58, h * 0.64), "2. Glide Down"), ((w * 0.24, h * 0.68), "3. Hook Up")]:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(63, 178, 255)))
                painter.drawEllipse(QPointF(pt[0], pt[1]), 5, 5)
                painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
                painter.setPen(QPen(QColor(244, 246, 251)))
                painter.drawText(int(pt[0] + 8), int(pt[1] + 4), lbl)

        elif self.letter == 'Z':
            painter.setPen(QPen(QColor(63, 178, 255, 120), 4, Qt.DashLine))
            painter.drawLine(int(w * 0.25), int(h * 0.26), int(w * 0.75), int(h * 0.26))
            painter.drawLine(int(w * 0.75), int(h * 0.26), int(w * 0.25), int(h * 0.74))
            painter.drawLine(int(w * 0.25), int(h * 0.74), int(w * 0.75), int(h * 0.74))

            pts = [
                (w * 0.25, h * 0.26, "1. Top Bar"), (w * 0.75, h * 0.26, "2. Turn"),
                (w * 0.25, h * 0.74, "3. Diagonal Cut"), (w * 0.75, h * 0.74, "4. Bottom Base")
            ]
            for px, py, tag in pts:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(63, 178, 255)))
                painter.drawEllipse(QPointF(px, py), 5, 5)
                painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
                painter.setPen(QPen(QColor(244, 246, 251)))
                painter.drawText(int(px - 65 if px > w * 0.5 else px + 8), int(py - 6), tag)

        elif self.letter == 'HELLO':
            # Head/Temple reference box
            painter.setPen(QPen(QColor(38, 56, 106), 1.5, Qt.DashLine))
            painter.setBrush(QBrush(QColor(11, 22, 51, 140)))
            painter.drawRoundedRect(int(w * 0.38), int(h * 0.20), int(w * 0.24), int(h * 0.32), 12, 12)
            painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
            painter.setPen(QPen(QColor(86, 100, 140)))
            painter.drawText(int(w * 0.42), int(h * 0.37), "TEMPLE")

            # Waving arc trajectory
            wave_path = QPainterPath()
            wave_path.moveTo(w * 0.26, h * 0.34)
            for step_x in range(int(w * 0.26), int(w * 0.74), 4):
                u = (step_x - w * 0.26) / (w * 0.48)
                wy = (0.34 + np.sin(u * 4 * np.pi) * 0.04) * h
                wave_path.lineTo(step_x, wy)
            painter.setPen(QPen(QColor(63, 178, 255, 120), 4, Qt.DashLine))
            painter.drawPath(wave_path)

            # Waving ripple rings
            for r_i in range(3):
                ring_r = 16 + r_i * 12
                painter.setPen(QPen(QColor(63, 178, 255, 60 - r_i * 18), 2))
                painter.drawArc(int(w * 0.74 - ring_r), int(h * 0.34 - ring_r), ring_r * 2, ring_r * 2, -45 * 16, 90 * 16)

        elif self.letter == 'THANK YOU':
            # Chin anchor silhouette
            painter.setPen(QPen(QColor(38, 56, 106), 1.5, Qt.DashLine))
            painter.setBrush(QBrush(QColor(11, 22, 51, 140)))
            painter.drawEllipse(int(w * 0.42), int(h * 0.18), int(w * 0.16), int(h * 0.20))
            painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
            painter.setPen(QPen(QColor(86, 100, 140)))
            painter.drawText(int(w * 0.45), int(h * 0.29), "CHIN")

            # Forward sweeping cone and arrow
            sweep_path = QPainterPath()
            sweep_path.moveTo(w * 0.50, h * 0.30)
            sweep_path.lineTo(w * 0.58, h * 0.74)
            painter.setPen(QPen(QColor(63, 178, 255, 130), 4, Qt.DashLine))
            painter.drawPath(sweep_path)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(63, 178, 255)))
            painter.drawEllipse(QPointF(w * 0.50, h * 0.30), 5, 5)
            painter.drawEllipse(QPointF(w * 0.58, h * 0.74), 5, 5)
            painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
            painter.setPen(QPen(QColor(244, 246, 251)))
            painter.drawText(int(w * 0.54), int(h * 0.32), "1. Chin Touch")
            painter.drawText(int(w * 0.62), int(h * 0.76), "2. Forward Sweep")

        elif self.letter == 'PLEASE':
            # Chest outline
            painter.setPen(QPen(QColor(38, 56, 106), 1.5, Qt.DashLine))
            painter.setBrush(QBrush(QColor(11, 22, 51, 140)))
            painter.drawRoundedRect(int(w * 0.28), int(h * 0.30), int(w * 0.44), int(h * 0.44), 16, 16)
            painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
            painter.setPen(QPen(QColor(86, 100, 140)))
            painter.drawText(int(w * 0.45), int(h * 0.53), "CHEST")

            # Orbital rubbing ring
            orbit_r = w * 0.22
            painter.setPen(QPen(QColor(63, 178, 255, 120), 4, Qt.DashLine))
            painter.drawEllipse(QPointF(w * 0.50, h * 0.52), orbit_r, orbit_r)

            # Circular rotation arrowheads
            painter.setPen(QPen(QColor(255, 178, 63), 2))
            painter.drawLine(int(w * 0.50 + orbit_r), int(h * 0.52), int(w * 0.50 + orbit_r - 8), int(h * 0.52 - 8))
            painter.drawLine(int(w * 0.50 + orbit_r), int(h * 0.52), int(w * 0.50 + orbit_r + 8), int(h * 0.52 - 8))

        elif self.letter == 'YES':
            # Vertical nod trajectory line
            painter.setPen(QPen(QColor(63, 178, 255, 120), 4, Qt.DashLine))
            painter.drawLine(int(w * 0.50), int(h * 0.36), int(w * 0.50), int(h * 0.68))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(63, 178, 255)))
            painter.drawEllipse(QPointF(w * 0.50, h * 0.36), 5, 5)
            painter.drawEllipse(QPointF(w * 0.50, h * 0.68), 5, 5)

            painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
            painter.setPen(QPen(QColor(244, 246, 251)))
            painter.drawText(int(w * 0.54), int(h * 0.38), "Top Flexion")
            painter.drawText(int(w * 0.54), int(h * 0.70), "Bottom Flexion (Nod)")

        elif self.letter == 'NO':
            # Snap pinch angle lines
            painter.setPen(QPen(QColor(63, 178, 255, 120), 4, Qt.DashLine))
            painter.drawLine(int(w * 0.40), int(h * 0.38), int(w * 0.50), int(h * 0.55))
            painter.drawLine(int(w * 0.60), int(h * 0.38), int(w * 0.50), int(h * 0.55))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 178, 63)))
            painter.drawEllipse(QPointF(w * 0.50, h * 0.55), 6, 6)
            painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
            painter.setPen(QPen(QColor(244, 246, 251)))
            painter.drawText(int(w * 0.52), int(h * 0.58), "Snap Pinch Point")

        elif self.letter == 'HELP':
            # Non-dominant flat palm baseline
            painter.setPen(QPen(QColor(63, 178, 255), 3))
            painter.drawLine(int(w * 0.32), int(h * 0.76), int(w * 0.68), int(h * 0.76))
            painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
            painter.setPen(QPen(QColor(86, 100, 140)))
            painter.drawText(int(w * 0.38), int(h * 0.82), "BASE FLAT PALM")

            # Upward tandem lift arrows
            painter.setPen(QPen(QColor(63, 178, 255, 120), 4, Qt.DashLine))
            painter.drawLine(int(w * 0.50), int(h * 0.76), int(w * 0.50), int(h * 0.32))
            painter.setPen(QPen(QColor(255, 178, 63), 3))
            painter.drawLine(int(w * 0.50), int(h * 0.32), int(w * 0.46), int(h * 0.36))
            painter.drawLine(int(w * 0.50), int(h * 0.32), int(w * 0.54), int(h * 0.36))

        elif self.letter == 'GOOD':
            # Chin to palm landing
            painter.setPen(QPen(QColor(63, 178, 255, 120), 4, Qt.DashLine))
            painter.drawLine(int(w * 0.50), int(h * 0.30), int(w * 0.50), int(h * 0.74))
            painter.setPen(QPen(QColor(63, 178, 255), 3))
            painter.drawLine(int(w * 0.35), int(h * 0.74), int(w * 0.65), int(h * 0.74))
            painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
            painter.setPen(QPen(QColor(244, 246, 251)))
            painter.drawText(int(w * 0.54), int(h * 0.32), "1. Chin Touch")
            painter.drawText(int(w * 0.54), int(h * 0.72), "2. Palm Landing")

        elif self.letter == 'NAMASTE':
            # Twin symmetrical convergence paths
            painter.setPen(QPen(QColor(63, 178, 255, 120), 4, Qt.DashLine))
            painter.drawLine(int(w * 0.28), int(h * 0.52), int(w * 0.50), int(h * 0.52))
            painter.drawLine(int(w * 0.72), int(h * 0.52), int(w * 0.50), int(h * 0.52))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 178, 63)))
            painter.drawEllipse(QPointF(w * 0.50, h * 0.52), 6, 6)
            painter.setFont(QFont('Segoe UI', 8, QFont.Bold))
            painter.setPen(QPen(QColor(244, 246, 251)))
            painter.drawText(int(w * 0.44), int(h * 0.62), "CENTERLINE")

        else:
            # Static letters: pulsing concentric alignment rings
            cx = w * 0.50
            cy = h * 0.50
            pulse_r = 22 + int(10 * np.sin(self.t * 6.28))
            painter.setPen(QPen(QColor(255, 178, 63, 90), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), pulse_r, pulse_r)
            painter.drawEllipse(QPointF(cx, cy), pulse_r + 18, pulse_r + 18)
            painter.setFont(QFont('Segoe UI', 12, QFont.Bold))
            painter.setPen(QPen(QColor(244, 246, 251)))
            painter.drawText(int(cx - 48), int(cy + 6), f"SIGN '{self.letter}'")

        # Current Animated Beacon & Trajectory Trail
        cur_x, cur_y = self._get_coords(self.t)
        self.trail.append((cur_x, cur_y))
        if len(self.trail) > 18:
            self.trail.pop(0)

        # Draw fading neon trail
        for i, (tx, ty) in enumerate(self.trail[:-1]):
            trail_alpha = int(220 * (i / float(len(self.trail))))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 178, 63, trail_alpha)))
            rad = 3 + int(4 * (i / float(len(self.trail))))
            painter.drawEllipse(QPointF(tx, ty), rad, rad)

        # Draw lead golden beacon
        glow = QRadialGradient(cur_x, cur_y, 20)
        glow.setColorAt(0.0, QColor(255, 178, 63, 240))
        glow.setColorAt(0.5, QColor(255, 178, 63, 110))
        glow.setColorAt(1.0, QColor(255, 178, 63, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(QPointF(cur_x, cur_y), 20, 20)

        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QPointF(cur_x, cur_y), 4.5, 4.5)

        # Phase label at bottom banner
        painter.setFont(QFont('Segoe UI', 9, QFont.Bold))
        painter.setPen(QPen(QColor(255, 178, 63)))
        if self.letter == 'J':
            lbl = "Phase: Downward Sweep" if self.t <= 0.60 else "Phase: Curved Terminal Hook"
        elif self.letter == 'Z':
            if self.t <= 0.33:
                lbl = "Phase: Top Bar Glide"
            elif self.t <= 0.66:
                lbl = "Phase: Diagonal Slash"
            else:
                lbl = "Phase: Bottom Base Bar"
        elif self.letter == 'HELLO':
            lbl = "Phase: Lateral Waving Arc"
        elif self.letter == 'THANK YOU':
            lbl = "Phase: Forward Sweeping Release"
        elif self.letter == 'PLEASE':
            lbl = "Phase: Circular Chest Orbit"
        elif self.letter == 'YES':
            lbl = "Phase: Downward Fist Nod"
        elif self.letter == 'NO':
            lbl = "Phase: Decisive Beak Snap"
        elif self.letter == 'HELP':
            lbl = "Phase: Tandem Upward Lift"
        elif self.letter == 'GOOD':
            lbl = "Phase: Chin to Palm Landing"
        elif self.letter == 'NAMASTE':
            lbl = "Phase: Symmetrical Convergence"
        else:
            lbl = "Pose: Steady Alignment"
        painter.drawText(14, h - 12, f"▶ {lbl}")


class TutorDialog(QDialog):
    """Spacious, State-of-the-Art Sign Language Tutor & Video Studio"""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Sign Language Tutor Studio")
        self.resize(860, 640)
        self.setMinimumSize(800, 580)
        self.guide_style = getattr(self.parent_app, 'guide_style', 'photo')
        self.current_category = "letters"  # "letters" or "gestures"

        self.setStyleSheet("""
            QDialog {
                background-color: #0b1633;
                color: #f4f6fb;
            }
            QLabel {
                color: #f4f6fb;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            QPushButton.btn-nav {
                background-color: #14234b;
                color: #f4f6fb;
                border: 1px solid #26386a;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                padding: 6px 14px;
                min-width: 36px;
            }
            QPushButton.btn-nav:hover {
                background-color: #26386a;
                border-color: #ffb23f;
            }
            QPushButton.btn-mode {
                background-color: #14234b;
                color: #93a1c6;
                border: 1px solid #26386a;
                border-radius: 8px;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 12px;
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
                font-weight: 800;
                padding: 6px 12px;
            }
            QPushButton.btn-cat {
                background-color: #14234b;
                color: #93a1c6;
                border: 1px solid #26386a;
                border-radius: 8px;
                font-size: 12px;
                font-weight: bold;
                padding: 7px 14px;
            }
            QPushButton.btn-cat:hover {
                background-color: #26386a;
                color: #f4f6fb;
            }
            QPushButton.btn-cat-active {
                background-color: #26386a;
                color: #ffb23f;
                border: 1px solid #ffb23f;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 800;
                padding: 7px 14px;
            }
            QPushButton.btn-vid-ctrl {
                background-color: #14234b;
                color: #f4f6fb;
                border: 1px solid #26386a;
                border-radius: 6px;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 12px;
            }
            QPushButton.btn-vid-ctrl:hover {
                background-color: #26386a;
                border-color: #3fb2ff;
            }
            QComboBox {
                background-color: #14234b;
                color: #f4f6fb;
                border: 1px solid #26386a;
                border-radius: 8px;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 13px;
            }
            QCheckBox {
                color: #f4f6fb;
                font-size: 12px;
                font-weight: 600;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #14234b;
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
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Header Row
        header_row = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("SIGN LANGUAGE TUTOR & GESTURE STUDIO")
        title.setFont(QFont('Segoe UI', 15, QFont.Bold))
        title.setStyleSheet("letter-spacing: 1.5px; color: #f4f6fb;")
        title_box.addWidget(title)

        subtitle = QLabel("Interactive Practice with High-Res Photos, Skeletons, and Live Animated Video Guides")
        subtitle.setStyleSheet("color: #93a1c6; font-size: 11px;")
        title_box.addWidget(subtitle)
        header_row.addLayout(title_box)
        header_row.addStretch()

        # Category Switcher Pills: [ 🔤 Alphabet A–Z ] [ 💬 Common Gestures ]
        self.btn_cat_letters = QPushButton("🔤 Alphabet (A–Z)")
        self.btn_cat_letters.clicked.connect(lambda: self.switch_category("letters"))
        header_row.addWidget(self.btn_cat_letters)

        self.btn_cat_gestures = QPushButton("💬 Common Gestures")
        self.btn_cat_gestures.clicked.connect(lambda: self.switch_category("gestures"))
        header_row.addWidget(self.btn_cat_gestures)

        layout.addLayout(header_row)

        # Main Studio Two-Column Layout
        studio_row = QHBoxLayout()
        studio_row.setSpacing(16)

        # ============================================================
        # LEFT COLUMN: VISUAL MEDIA STUDIO (370px width)
        # ============================================================
        left_studio = QFrame()
        left_studio.setFixedWidth(380)
        left_studio.setStyleSheet("background-color: #14234b; border: 1px solid #26386a; border-radius: 14px;")
        left_layout = QVBoxLayout(left_studio)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        # Guide Style Pills row
        style_row = QHBoxLayout()
        style_lbl = QLabel("Guide Style:")
        style_lbl.setStyleSheet("color: #93a1c6; font-size: 11px; font-weight: bold;")
        style_row.addWidget(style_lbl)

        self.btn_photo = QPushButton("📷 Photo")
        self.btn_photo.clicked.connect(lambda: self.set_guide_mode('photo'))
        style_row.addWidget(self.btn_photo)

        self.btn_skel = QPushButton("🦴 Skeleton")
        self.btn_skel.clicked.connect(lambda: self.set_guide_mode('skeleton'))
        style_row.addWidget(self.btn_skel)

        self.btn_video = QPushButton("🎬 Motion Video")
        self.btn_video.clicked.connect(lambda: self.set_guide_mode('video'))
        style_row.addWidget(self.btn_video)
        left_layout.addLayout(style_row)

        # Visual Display Viewport (Holds Canvas for Video vs Image Label for Photo/Skeleton)
        self.video_canvas = MotionVideoGuideCanvas(self)
        left_layout.addWidget(self.video_canvas)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(360, 260)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background: #050b1c; border: 1px solid #26386a; border-radius: 14px;")
        left_layout.addWidget(self.preview_label)

        # Video Player Controls
        self.vid_ctrl_row = QHBoxLayout()
        self.vid_ctrl_row.setSpacing(6)
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
        left_layout.addLayout(self.vid_ctrl_row)

        # Ghost Guide Overlay controls on webcam
        overlay_box = QFrame()
        overlay_box.setStyleSheet("background: #050b1c; border-radius: 8px; padding: 6px;")
        overlay_layout = QVBoxLayout(overlay_box)
        overlay_layout.setContentsMargins(8, 6, 8, 6)
        overlay_layout.setSpacing(6)

        self.guide_checkbox = QCheckBox("Show ghost guide overlay on camera")
        self.guide_checkbox.setChecked(self.parent_app.guide_enabled)
        self.guide_checkbox.toggled.connect(self.toggle_guide)
        overlay_layout.addWidget(self.guide_checkbox)

        op_row = QHBoxLayout()
        self.opacity_title = QLabel("Opacity: 50%")
        self.opacity_title.setStyleSheet("color: #93a1c6; font-size: 11px;")
        op_row.addWidget(self.opacity_title)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(15, 90)
        self.opacity_slider.setValue(int(self.parent_app.guide_opacity * 100))
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        op_row.addWidget(self.opacity_slider, stretch=1)
        overlay_layout.addLayout(op_row)
        left_layout.addWidget(overlay_box)

        studio_row.addWidget(left_studio)

        # ============================================================
        # RIGHT COLUMN: PRACTICE DECK & INSTRUCTION STUDIO
        # ============================================================
        right_studio = QFrame()
        right_studio.setStyleSheet("background-color: #14234b; border: 1px solid #26386a; border-radius: 14px;")
        right_layout = QVBoxLayout(right_studio)
        right_layout.setContentsMargins(14, 12, 14, 12)
        right_layout.setSpacing(10)

        # Navigation row: [◀] [Letter / Gesture Dropdown] [▶]
        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)

        prev_btn = QPushButton("◀")
        prev_btn.setProperty("class", "btn-nav")
        prev_btn.clicked.connect(self.prev_sign)
        nav_row.addWidget(prev_btn)

        self.sign_combo = QComboBox()
        self.sign_combo.currentIndexChanged.connect(self.on_combo_changed)
        nav_row.addWidget(self.sign_combo, stretch=1)

        next_btn = QPushButton("▶")
        next_btn.setProperty("class", "btn-nav")
        next_btn.clicked.connect(self.next_sign)
        nav_row.addWidget(next_btn)
        right_layout.addLayout(nav_row)

        # Target Title & Telemetry Badge
        meta_row = QHBoxLayout()
        self.target_title = QLabel("Target: Letter A")
        self.target_title.setFont(QFont('Segoe UI', 14, QFont.Bold))
        self.target_title.setStyleSheet("color: #ffb23f;")
        meta_row.addWidget(self.target_title)
        meta_row.addStretch()

        self.adaptive_badge = QLabel("✋ STATIC POSE (Residual MLP)")
        self.adaptive_badge.setStyleSheet("""
            background-color: rgba(63, 178, 255, 0.15);
            border: 1px solid #3fb2ff;
            border-radius: 6px;
            padding: 4px 10px;
            color: #3fb2ff;
            font-size: 10px;
            font-weight: bold;
        """)
        meta_row.addWidget(self.adaptive_badge)
        right_layout.addLayout(meta_row)

        # Structured Step-by-Step Instructions Area (Spacious, modern card stack)
        self.instructions_scroll = QScrollArea()
        self.instructions_scroll.setWidgetResizable(True)
        self.instructions_scroll.setStyleSheet("background: transparent; border: none;")

        self.instructions_widget = QWidget()
        self.instructions_layout = QVBoxLayout(self.instructions_widget)
        self.instructions_layout.setContentsMargins(0, 0, 4, 0)
        self.instructions_layout.setSpacing(8)
        self.instructions_scroll.setWidget(self.instructions_widget)
        right_layout.addWidget(self.instructions_scroll, stretch=1)

        # Live Practice Match Status HUD
        self.status_badge = QLabel("Target: A | Perform sign in camera to practice")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setStyleSheet("""
            background-color: rgba(11, 22, 51, 0.8);
            border: 1px solid #26386a;
            border-radius: 10px;
            padding: 10px;
            color: #93a1c6;
            font-weight: 700;
            font-size: 12px;
        """)
        right_layout.addWidget(self.status_badge)

        # Done Button
        done_btn = QPushButton("Done")
        done_btn.setFixedHeight(40)
        done_btn.setStyleSheet("""
            QPushButton {
                background-color: #f4f6fb;
                color: #0b1633;
                border: none;
                border-radius: 10px;
                font-weight: 800;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #ffb23f;
            }
        """)
        done_btn.clicked.connect(self.accept)
        right_layout.addWidget(done_btn)

        studio_row.addWidget(right_studio, stretch=1)
        layout.addLayout(studio_row)

        self.setLayout(layout)

        # Initialize with letters
        self.switch_category("letters", initial_sync=False)
        self.set_guide_mode(self.guide_style, update_view_only=True)
        self.sync_target(self.parent_app.current_tutor_letter)

    def accept(self):
        if hasattr(self.parent_app, 'inference_engine') and self.parent_app.inference_engine:
            self.parent_app.inference_engine.target_gesture_focus = None
        super().accept()

    def closeEvent(self, event):
        if hasattr(self.parent_app, 'inference_engine') and self.parent_app.inference_engine:
            self.parent_app.inference_engine.target_gesture_focus = None
        super().closeEvent(event)

    def switch_category(self, cat, initial_sync=True):
        self.current_category = cat
        if hasattr(self.parent_app, 'set_app_mode'):
            self.parent_app.set_app_mode(cat)
        self.btn_cat_letters.setProperty("class", "btn-cat-active" if cat == "letters" else "btn-cat")
        self.btn_cat_gestures.setProperty("class", "btn-cat-active" if cat == "gestures" else "btn-cat")
        self.btn_cat_letters.style().unpolish(self.btn_cat_letters)
        self.btn_cat_letters.style().polish(self.btn_cat_letters)
        self.btn_cat_gestures.style().unpolish(self.btn_cat_gestures)
        self.btn_cat_gestures.style().polish(self.btn_cat_gestures)

        self.sign_combo.blockSignals(True)
        self.sign_combo.clear()

        if cat == "letters":
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                suffix = " 🌊 [Motion]" if c in ('J', 'Z') else ""
                self.sign_combo.addItem(f"Letter {c}{suffix}", c)
            target = self.parent_app.current_tutor_letter if self.parent_app.current_tutor_letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" else "A"
        else:
            for g in COMMON_GESTURE_NAMES:
                self.sign_combo.addItem(f"Gesture: {g} 🌊", g)
            target = COMMON_GESTURE_NAMES[0]
            # Gestures default to motion video mode
            self.set_guide_mode('video')

        self.sign_combo.blockSignals(False)

        if initial_sync:
            self.sync_target(target)

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

        self.sync_target(self.parent_app.current_tutor_letter)

    def toggle_canvas_play(self):
        self.video_canvas.toggle_play()
        self.play_btn.setText("▶ Play" if not self.video_canvas.is_playing else "⏸ Pause")

    def toggle_canvas_speed(self):
        self.video_canvas.toggle_speed()
        self.speed_btn.setText("🐢 0.5x" if self.video_canvas.speed == 0.5 else "⚡ 1.0x")

    def sync_target(self, target):
        target = str(target).upper()
        self.parent_app.current_tutor_letter = target
        if hasattr(self.parent_app, 'inference_engine') and self.parent_app.inference_engine:
            if target in COMMON_GESTURE_NAMES:
                self.parent_app.inference_engine.target_gesture_focus = target
            else:
                self.parent_app.inference_engine.target_gesture_focus = None

        # Set combo index
        self.sign_combo.blockSignals(True)
        idx = self.sign_combo.findData(target)
        if idx >= 0:
            self.sign_combo.setCurrentIndex(idx)
        self.sign_combo.blockSignals(False)

        # Title
        label_prefix = "Gesture" if target in COMMON_GESTURE_NAMES else "Letter"
        self.target_title.setText(f"Target: {label_prefix} {target}")
        self.video_canvas.setSign(target)

        # Clear previous instruction cards
        while self.instructions_layout.count():
            item = self.instructions_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Build structured step cards
        if target in ADAPTIVE_VIDEO_INSTRUCTIONS:
            info = ADAPTIVE_VIDEO_INSTRUCTIONS[target]
            self.adaptive_badge.setText(info['badge'])
            self.adaptive_badge.setStyleSheet("""
                background-color: rgba(255, 178, 63, 0.2);
                border: 1px solid #ffb23f;
                border-radius: 6px;
                padding: 4px 10px;
                color: #ffb23f;
                font-size: 10px;
                font-weight: bold;
            """)

            for step_title, step_desc in info['steps']:
                s_box = QFrame()
                s_box.setStyleSheet("""
                    QFrame {
                        background-color: #050b1c;
                        border-left: 3px solid #ffb23f;
                        border-radius: 6px;
                    }
                """)
                s_layout = QVBoxLayout(s_box)
                s_layout.setContentsMargins(10, 8, 10, 8)
                s_layout.setSpacing(3)

                t_lbl = QLabel(step_title)
                t_lbl.setFont(QFont('Segoe UI', 10, QFont.Bold))
                t_lbl.setStyleSheet("color: #ffb23f;")
                s_layout.addWidget(t_lbl)

                d_lbl = QLabel(step_desc)
                d_lbl.setFont(QFont('Segoe UI', 9))
                d_lbl.setWordWrap(True)
                d_lbl.setStyleSheet("color: #f4f6fb; line-height: 1.3;")
                s_layout.addWidget(d_lbl)

                self.instructions_layout.addWidget(s_box)

            # Kinematic tip callout
            tip_box = QFrame()
            tip_box.setStyleSheet("background: rgba(63, 178, 255, 0.1); border: 1px solid rgba(63, 178, 255, 0.3); border-radius: 6px;")
            tip_layout = QVBoxLayout(tip_box)
            tip_layout.setContentsMargins(10, 8, 10, 8)
            tip_lbl = QLabel(f"💡 <b>Kinematic Pro Tip:</b> {info['kinematic_tip']}")
            tip_lbl.setWordWrap(True)
            tip_lbl.setFont(QFont('Segoe UI', 9))
            tip_lbl.setStyleSheet("color: #3fb2ff;")
            tip_layout.addWidget(tip_lbl)
            self.instructions_layout.addWidget(tip_box)

        else:
            self.adaptive_badge.setText("✋ STATIC HANDSHAPE (99.76% MLP)")
            self.adaptive_badge.setStyleSheet("""
                background-color: rgba(63, 178, 255, 0.15);
                border: 1px solid #3fb2ff;
                border-radius: 6px;
                padding: 4px 10px;
                color: #3fb2ff;
                font-size: 10px;
                font-weight: bold;
            """)

            hint_text = ISL_LETTER_HINTS.get(target, "Align your handshape according to the visual guide.")
            h_box = QFrame()
            h_box.setStyleSheet("background-color: #050b1c; border-left: 3px solid #3fb2ff; border-radius: 6px;")
            h_layout = QVBoxLayout(h_box)
            h_layout.setContentsMargins(10, 8, 10, 8)

            t_lbl = QLabel("Execution Guide")
            t_lbl.setFont(QFont('Segoe UI', 10, QFont.Bold))
            t_lbl.setStyleSheet("color: #3fb2ff;")
            h_layout.addWidget(t_lbl)

            d_lbl = QLabel(hint_text)
            d_lbl.setFont(QFont('Segoe UI', 9))
            d_lbl.setWordWrap(True)
            d_lbl.setStyleSheet("color: #f4f6fb; line-height: 1.4;")
            h_layout.addWidget(d_lbl)
            self.instructions_layout.addWidget(h_box)

            tip_box = QFrame()
            tip_box.setStyleSheet("background: rgba(255, 178, 63, 0.1); border: 1px solid rgba(255, 178, 63, 0.3); border-radius: 6px;")
            tip_layout = QVBoxLayout(tip_box)
            tip_layout.setContentsMargins(10, 8, 10, 8)
            tip_lbl = QLabel("💡 <b>Pro Tip:</b> Hold handshape steady for 800ms. The 99.76% Deep Residual MLP will auto-confirm.")
            tip_lbl.setWordWrap(True)
            tip_lbl.setFont(QFont('Segoe UI', 9))
            tip_lbl.setStyleSheet("color: #ffb23f;")
            tip_layout.addWidget(tip_lbl)
            self.instructions_layout.addWidget(tip_box)

        # Image preview for Photo / Skeleton modes (for letters)
        if self.guide_style != 'video':
            img = self.parent_app.load_guide_image(target, style=self.guide_style)
            if img is not None:
                if img.ndim == 3 and img.shape[2] == 4:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                    h_i, w_i, _ = rgb.shape
                    qimg = QImage(rgb.data, w_i, h_i, 4 * w_i, QImage.Format_RGBA8888)
                else:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h_i, w_i, ch = rgb.shape
                    qimg = QImage(rgb.data, w_i, h_i, ch * w_i, QImage.Format_RGB888)

                pix = QPixmap.fromImage(qimg).scaled(360, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_label.setPixmap(pix)
            else:
                self.preview_label.setText(f"Sign: {target}\n(Select 🎬 Motion Video)")
                self.preview_label.setFont(QFont('Segoe UI', 12, QFont.Bold))
                self.preview_label.setStyleSheet("color: #93a1c6; background: #050b1c; border-radius: 14px;")

    def on_combo_changed(self, idx):
        target = self.sign_combo.itemData(idx)
        if not target:
            return

        # For gestures and J/Z, auto switch to video mode
        if (target in COMMON_GESTURE_NAMES or target in ('J', 'Z')) and self.guide_style != 'video':
            self.set_guide_mode('video', update_view_only=True)

        self.parent_app.on_tutor_letter_changed(target)
        self.sync_target(target)

    def prev_sign(self):
        cur_idx = self.sign_combo.currentIndex()
        count = self.sign_combo.count()
        if count > 0:
            new_idx = (cur_idx - 1) % count
            self.sign_combo.setCurrentIndex(new_idx)

    def next_sign(self):
        cur_idx = self.sign_combo.currentIndex()
        count = self.sign_combo.count()
        if count > 0:
            new_idx = (cur_idx + 1) % count
            self.sign_combo.setCurrentIndex(new_idx)

    def toggle_guide(self, checked):
        self.parent_app.guide_enabled = checked
        if self.parent_app.video_thread:
            self.parent_app.video_thread.guide_enabled = checked

    def on_opacity_changed(self, val):
        self.parent_app.guide_opacity = val / 100.0
        self.opacity_title.setText(f"Opacity: {val}%")
        if self.parent_app.video_thread:
            self.parent_app.video_thread.guide_opacity = self.parent_app.guide_opacity

    def update_match(self, prediction, confidence):
        target = self.parent_app.current_tutor_letter
        if prediction and prediction.upper() == target.upper() and confidence >= 0.60:
            self.status_badge.setText(f"🎯 EXCELLENT! Matched '{prediction}' ({confidence:.1%})")
            self.status_badge.setStyleSheet("""
                background-color: rgba(63, 185, 80, 0.25);
                border: 2px solid #3fb950;
                border-radius: 10px;
                padding: 10px;
                color: #3fb950;
                font-weight: 800;
                font-size: 13px;
            """)
        else:
            self.status_badge.setText(f"Target: {target} | Detected: {prediction if prediction else '...'} ({confidence:.0%})")
            self.status_badge.setStyleSheet("""
                background-color: rgba(11, 22, 51, 0.85);
                border: 1px solid #26386a;
                border-radius: 10px;
                padding: 10px;
                color: #93a1c6;
                font-weight: 600;
                font-size: 12px;
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

    def set_app_mode(self, mode):
        if hasattr(self, 'inference_engine') and self.inference_engine:
            self.inference_engine.app_mode = mode

        for btn, m in [(self.btn_mode_auto, 'auto'), (self.btn_mode_gestures, 'gestures'), (self.btn_mode_letters, 'letters')]:
            btn.setProperty("class", "seg-btn-active" if mode == m else "seg-btn")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if mode == 'gestures':
            self.glyph_caption.setText("Perform gesture (Hello, Thank You, Namaste...)")
            self.glyph_letter.setText("??")
        elif mode == 'letters':
            self.glyph_caption.setText("Hold letter steady (A-Z)")
            self.glyph_letter.setText("??")
        else:
            self.glyph_caption.setText("Show a letter or gesture")
            self.glyph_letter.setText("?")
        self.glyph_hold_bar.setValue(0)
        self.hold_count = 0

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
        self.setWindowTitle("ISL Fingerspell & Gesture Studio")
        self.resize(960, 720)
        self.setMinimumSize(800, 600)

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
            QPushButton.seg-btn {
                min-height: 28px;
                padding: 0 12px;
                border: 1px solid #26386a;
                border-radius: 14px;
                background-color: transparent;
                color: #93a1c6;
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 11px;
                font-weight: 700;
            }
            QPushButton.seg-btn:hover {
                background-color: #14234b;
                color: #f4f6fb;
                border-color: #ffb23f;
            }
            QPushButton.seg-btn-active {
                min-height: 28px;
                padding: 0 12px;
                border: 1px solid #ffb23f;
                border-radius: 14px;
                background-color: #ffb23f;
                color: #0b1633;
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 11px;
                font-weight: 800;
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

        # Mode Selection Pills: [ ? Auto ] [ ?? Gestures ] [ ?? Alphabet ]
        self.btn_mode_auto = QPushButton("? Auto")
        self.btn_mode_auto.setProperty("class", "seg-btn-active")
        self.btn_mode_auto.clicked.connect(lambda: self.set_app_mode('auto'))
        topbar.addWidget(self.btn_mode_auto)

        self.btn_mode_gestures = QPushButton("?? Gestures")
        self.btn_mode_gestures.setProperty("class", "seg-btn")
        self.btn_mode_gestures.clicked.connect(lambda: self.set_app_mode('gestures'))
        topbar.addWidget(self.btn_mode_gestures)

        self.btn_mode_letters = QPushButton("?? Alphabet")
        self.btn_mode_letters.setProperty("class", "seg-btn")
        self.btn_mode_letters.clicked.connect(lambda: self.set_app_mode('letters'))
        topbar.addWidget(self.btn_mode_letters)
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

        # Ensure buttons do not steal keyboard focus from window
        for btn in [learn_btn, letters_btn, settings_btn, self.btn_mode_auto, self.btn_mode_gestures, self.btn_mode_letters, self.delete_btn, self.add_btn, self.space_btn, self.clear_btn, self.speak_btn]:
            btn.setFocusPolicy(Qt.NoFocus)

        # Global keyboard shortcuts
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence

        QShortcut(QKeySequence(Qt.Key_Space), self, self.on_shortcut_space)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, self.on_delete_clicked)
        QShortcut(QKeySequence(Qt.Key_Return), self, self.on_speak_clicked)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self.on_speak_clicked)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.on_clear_clicked)
        QShortcut(QKeySequence("Ctrl+C"), self, self.on_clear_clicked)
        QShortcut(QKeySequence("Ctrl+T"), self, self.on_speak_clicked)
        QShortcut(QKeySequence("Ctrl+L"), self, self.open_tutor)
        QShortcut(QKeySequence("Ctrl+P"), self, self.open_pamphlet)
        QShortcut(QKeySequence("Ctrl+S"), self, self.open_settings)
        QShortcut(QKeySequence("Ctrl+G"), self, self.toggle_gesture_letter_mode)
        QShortcut(QKeySequence(Qt.Key_F1), self, self.open_tutor)
        QShortcut(QKeySequence(Qt.Key_F2), self, self.toggle_gesture_letter_mode)

        # Caret blink timer
        self.caret_visible = True
        self.caret_timer = QTimer(self)
        self.caret_timer.timeout.connect(self.toggle_caret)
        self.caret_timer.start(550)

    def on_shortcut_space(self):
        """Space key handler: captures current camera sign if confident, else inserts space"""
        if self.last_prediction and self.last_confidence >= 0.50:
            self.on_add_clicked()
        else:
            self.on_space_clicked()

    def toggle_gesture_letter_mode(self):
        """Toggle between pure Gestures and Alphabet modes via 'G' key"""
        cur = getattr(self.inference_engine, 'app_mode', 'auto')
        next_mode = 'gestures' if cur != 'gestures' else 'letters'
        self.set_app_mode(next_mode)

    def keyPressEvent(self, event):
        """Comprehensive keyboard handling: shortcuts + direct physical key typing"""
        key = event.key()
        text = event.text()
        modifiers = event.modifiers()
        has_ctrl = bool(modifiers & Qt.ControlModifier)

        if key == Qt.Key_Space:
            self.on_shortcut_space()
        elif key == Qt.Key_Backspace:
            self.on_delete_clicked()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self.on_speak_clicked()
        elif key == Qt.Key_Escape:
            self.on_clear_clicked()
        elif has_ctrl and key == Qt.Key_C:
            self.on_clear_clicked()
        elif has_ctrl and key == Qt.Key_T:
            self.on_speak_clicked()
        elif has_ctrl and key == Qt.Key_L:
            self.open_tutor()
        elif has_ctrl and key == Qt.Key_P:
            self.open_pamphlet()
        elif has_ctrl and key == Qt.Key_S:
            self.open_settings()
        elif has_ctrl and key == Qt.Key_G:
            self.toggle_gesture_letter_mode()
        elif key == Qt.Key_F1:
            self.open_tutor()
        elif key == Qt.Key_F2:
            self.toggle_gesture_letter_mode()
        elif text and text.isalpha() and len(text) == 1:
            # Direct typing support for all keyboard letter keys (A-Z)
            self.append_letter(text.upper())
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
            self.glyph_letter.setFont(QFont('Segoe UI', 38, QFont.Bold))
            self.glyph_box.setFixedSize(110, 140)
            self.glyph_box.setStyleSheet("""
                QFrame {
                    background-color: rgba(11, 22, 51, 0.82);
                    border: 2px dashed #26386a;
                    border-radius: 12px;
                }
            """)
            self.glyph_caption.setText("Show a letter or gesture")
            self.glyph_hold_bar.setValue(0)
            self.hold_count = 0
            return

        # Active glyph display with adaptive font scaling
        self.glyph_letter.setText(prediction)
        is_gesture = len(prediction) > 1

        if is_gesture:
            self.glyph_letter.setFont(QFont('Segoe UI', 15, QFont.Bold))
            self.glyph_box.setFixedSize(145, 140)
            self.glyph_box.setStyleSheet("""
                QFrame {
                    background-color: rgba(11, 22, 51, 0.92);
                    border: 2px solid #3fb2ff;
                    border-radius: 12px;
                }
            """)
            target_hold = 12  # Fast ~400ms confirmation for dynamic gestures
        else:
            self.glyph_letter.setFont(QFont('Segoe UI', 38, QFont.Bold))
            self.glyph_box.setFixedSize(110, 140)
            self.glyph_box.setStyleSheet("""
                QFrame {
                    background-color: rgba(11, 22, 51, 0.88);
                    border: 2px solid #ffb23f;
                    border-radius: 12px;
                }
            """)
            target_hold = 25  # ~850ms hold for steady static letters

        engine_str = getattr(self.inference_engine, 'active_engine', 'MLP')
        self.glyph_caption.setText(f"{confidence:.0%} • {engine_str}")

        # Auto-capture hold-to-add logic
        if self.auto_capture_enabled and confidence >= 0.60:
            if self.cooldown_frames > 0:
                self.cooldown_frames -= 1
            elif prediction == self.last_held_char:
                self.hold_count += 1
                progress = min(1.0, self.hold_count / float(target_hold))
                self.glyph_hold_bar.setValue(int(progress * 100))

                if self.hold_count >= target_hold:
                    # Hold completed: Add letter or gesture word to tape!
                    self.append_letter(prediction)
                    self.hold_count = 0
                    self.glyph_hold_bar.setValue(0)
                    self.cooldown_frames = 15  # Cooldown prevents accidental repeat
            else:
                self.last_held_char = prediction
                self.hold_count = 1
                self.glyph_hold_bar.setValue(int(100 / target_hold))
        else:
            self.hold_count = 0
            self.glyph_hold_bar.setValue(0)

        # Notify Tutor dialog if open
        if self.tutor_dialog and self.tutor_dialog.isVisible():
            self.tutor_dialog.update_match(prediction, confidence)

    def append_letter(self, letter):
        if len(letter) > 1:
            # Multi-character gesture word (e.g. "HELLO", "THANK YOU")
            if self.spelled_letters and self.spelled_letters[-1] != " ":
                self.spelled_letters.append(" ")
            self.spelled_letters.append(letter)
            self.spelled_letters.append(" ")
            try:
                self.inference_engine.tts_engine.speak(letter)
            except Exception:
                pass
        else:
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
