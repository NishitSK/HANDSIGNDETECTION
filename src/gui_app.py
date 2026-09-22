"""
Professional GUI Application for ISL Translation System
Advanced interface suitable for news channels and professional use
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """Thread for video capture and processing"""
    
    change_pixmap_signal = pyqtSignal(np.ndarray)
    prediction_signal = pyqtSignal(str, float)
    
    def __init__(self, inference_engine, camera_id=0):
        super().__init__()
        self.inference_engine = inference_engine
        self.camera_id = camera_id
        self.running = True
        self.mirror_display = True
        
        # Transparent Guide Overlay settings
        self.guide_enabled = False
        self.guide_image = None
        self.guide_letter = ""
        self.guide_opacity = 0.5
    
    def run(self):
        # Lower capture resolution to reduce CPU/GPU load for faster GUI.
        # open_camera() also works around the default Windows backend, which
        # opens the device but never delivers frames — that would leave the
        # loop below spinning on `continue` forever with a blank window.
        try:
            cap = open_camera(self.camera_id, width=640, height=360)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return

        # Frame skipping: 1 = process every frame for smooth non-flickering display
        self.frame_counter = 0
        self.frame_skip = getattr(self, 'frame_skip', 1)
        last_processed = None
        last_prediction = None
        last_confidence = 0.0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # CRITICAL: Keep raw frame UN-FLIPPED for model inference
            raw_frame = frame

            # Run inference on un-flipped frame, with mirror_display applied to the returned visualization
            if (self.frame_counter % self.frame_skip) == 0:
                try:
                    processed_frame, prediction, confidence = self.inference_engine.process_frame(
                        raw_frame, mirror_display=self.mirror_display
                    )
                    last_processed = processed_frame
                    last_prediction = prediction
                    last_confidence = confidence
                except Exception:
                    processed_frame = cv2.flip(raw_frame, 1) if self.mirror_display else raw_frame.copy()
                    prediction = last_prediction
                    confidence = last_confidence
            else:
                # Maintain last processed overlay so screen never flickers or drops HUD
                processed_frame = last_processed.copy() if last_processed is not None else (cv2.flip(raw_frame, 1) if self.mirror_display else raw_frame.copy())
                prediction = last_prediction
                confidence = last_confidence

            # Scale processed frame back to display resolution if needed
            if processed_frame is None:
                processed_frame = cv2.flip(raw_frame, 1) if self.mirror_display else raw_frame.copy()

            # Blend transparent tutor guide overlay if enabled
            if self.guide_enabled and self.guide_image is not None:
                try:
                    h_f, w_f = processed_frame.shape[:2]
                    guide_size = min(int(h_f * 0.72), int(w_f * 0.42))
                    if guide_size > 40:
                        guide_resized = cv2.resize(self.guide_image, (guide_size, guide_size), interpolation=cv2.INTER_AREA)
                        # Position on right side of frame so user can see it while signing
                        x_offset = w_f - guide_size - 15
                        y_offset = (h_f - guide_size) // 2
                        
                        roi = processed_frame[y_offset:y_offset+guide_size, x_offset:x_offset+guide_size]
                        alpha = float(np.clip(self.guide_opacity, 0.1, 0.95))
                        blended = cv2.addWeighted(roi, 1.0 - alpha, guide_resized, alpha, 0)
                        processed_frame[y_offset:y_offset+guide_size, x_offset:x_offset+guide_size] = blended
                        
                        # High-tech guide box and label
                        cv2.rectangle(processed_frame, (x_offset, y_offset), (x_offset + guide_size, y_offset + guide_size), (0, 255, 180), 2)
                        cv2.putText(processed_frame, f"GUIDE: {self.guide_letter}", (x_offset + 6, y_offset - 8),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 180), 1, cv2.LINE_AA)
                except Exception:
                    pass

            # Emit signals for UI update
            self.change_pixmap_signal.emit(processed_frame)
            self.prediction_signal.emit(prediction if prediction else "", confidence if prediction else 0.0)

            self.frame_counter += 1

        cap.release()
    
    def stop(self):
        self.running = False
        self.wait()


class ISLGUIApp(QMainWindow):
    """Main GUI application"""
    
    def __init__(self, model_path, config_path='config.yaml'):
        super().__init__()
        
        self.config = get_config(config_path)
        
        # Initialize inference engine
        self.inference_engine = ISLInference(model_path, config_path)
        
        # Video thread
        self.video_thread = None
        # UI state
        self.is_recording = False
        self.recorded_frames = []
        self.manual_mode = True  # Start in manual mode
        self.last_prediction = None
        self.last_confidence = 0.0

        # Advanced settings (defaults tuned for speed)
        self.confidence_threshold = 0.7
        self.smoothing_enabled = True
        # Turning off landmarks by default reduces drawing overhead
        self.show_landmarks = False
        self.show_connections = True
        self.record_with_audio = False
        self.camera_brightness = 0
        self.camera_contrast = 0
        self.camera_resolution = "640x360"

        # Session tracking
        self.total_captures = 0
        self.total_sentences = 0
        self.session_start = QDateTime.currentDateTime()

        # Tutor & Ghost Guide state
        self.mirror_camera_enabled = True
        self.guide_enabled = False
        self.guide_opacity = 0.5
        self.current_tutor_letter = "A"
        self.guide_images_cache = {}

        # Initialize UI
        self.init_ui()
        self.on_tutor_letter_changed("A")
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Space:
            # Space bar triggers manual capture
            self.manual_capture()
        elif event.key() == Qt.Key_C:
            # C key clears sequence
            self.clear_sequence()
        elif event.key() == Qt.Key_T:
            # T key translates
            self.translate_sequence()
        elif event.key() == Qt.Key_M:
            # M key toggles mode
            self.toggle_mode()
    
    def manual_capture(self):
        """Manually capture current gesture"""
        if self.last_prediction and self.last_confidence >= 0.55:
            # Add to sequence
            self.inference_engine.detected_words.append(self.last_prediction)
            
            # Update display
            sequence = ' → '.join(self.inference_engine.detected_words)
            self.sequence_text.setPlainText(sequence)
            
            # Visual feedback
            self.capture_btn.setStyleSheet("background-color: #00ff00; color: black;")
            QTimer.singleShot(200, lambda: self.capture_btn.setStyleSheet("background-color: #00aa00; color: white;"))
            
            # Status update
            self.statusBar().showMessage(f'Captured: {self.last_prediction} ({self.last_confidence:.1%})', 2000)
            print(f"📸 Manual capture: {self.last_prediction} ({self.last_confidence:.2f})")
        else:
            self.statusBar().showMessage('No clear gesture detected. Please hold gesture steady.', 2000)
    
    def toggle_mode(self):
        """Toggle between auto and manual mode"""
        self.manual_mode = not self.mode_toggle_btn.isChecked()
        
        if self.manual_mode:
            self.mode_indicator.setText("● MANUAL MODE")
            self.mode_indicator.setStyleSheet("color: #00ff00; padding: 5px; font-weight: bold;")
            self.mode_toggle_btn.setText("⇄ Auto")
            self.capture_btn.setEnabled(True)
            self.statusBar().showMessage('Manual Mode - Press Space to capture', 2000)
        else:
            self.mode_indicator.setText("● AUTO MODE")
            self.mode_indicator.setStyleSheet("color: #ff9900; padding: 5px; font-weight: bold;")
            self.mode_toggle_btn.setText("⇄ Manual")
            self.capture_btn.setEnabled(False)
            self.statusBar().showMessage('Auto Mode - Signs captured automatically', 2000)
        
        # Update inference engine
        self.inference_engine.manual_mode = self.manual_mode

    def load_guide_image(self, letter):
        """Load and cache guide image for a given letter"""
        letter = str(letter).upper()
        if letter in self.guide_images_cache:
            return self.guide_images_cache[letter]

        pamphlet_path = Path(__file__).resolve().parent.parent / 'mobile' / 'app' / 'pamphlet' / 'isl' / f"{letter}.jpg"
        if pamphlet_path.exists():
            img = cv2.imread(str(pamphlet_path))
            self.guide_images_cache[letter] = img
            return img

        dataset_path = Path(__file__).resolve().parent.parent / 'data' / 'RealSign_ISL' / letter / "Testing_175.jpg"
        if dataset_path.exists():
            img = cv2.imread(str(dataset_path))
            self.guide_images_cache[letter] = img
            return img

        return None

    def on_tutor_letter_changed(self, letter):
        """Handle user selecting a new practice letter"""
        if not letter or not hasattr(self, 'tutor_preview_label'):
            return
        self.current_tutor_letter = str(letter)
        guide_img = self.load_guide_image(self.current_tutor_letter)

        if self.video_thread:
            self.video_thread.guide_letter = self.current_tutor_letter
            self.video_thread.guide_image = guide_img

        if guide_img is not None:
            rgb_preview = cv2.cvtColor(guide_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_preview.shape
            qimg = QImage(rgb_preview.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.tutor_preview_label.setPixmap(pix)
        else:
            self.tutor_preview_label.setText("No Image")

        hint = ISL_LETTER_HINTS.get(self.current_tutor_letter, "Observe the guide image and form the sign.")
        if hasattr(self, 'tutor_hint_label'):
            self.tutor_hint_label.setText(hint)
        if hasattr(self, 'tutor_target_label'):
            self.tutor_target_label.setText(f"Target Sign: Letter {self.current_tutor_letter}")
        if hasattr(self, 'tutor_match_status'):
            self.tutor_match_status.setText(f"Ready to practice Letter {self.current_tutor_letter}")
            self.tutor_match_status.setStyleSheet("color: #00ddff; padding: 6px; background: rgba(0, 150, 255, 0.15); border-radius: 6px;")

    def on_tutor_prev_letter(self):
        idx = ord(self.current_tutor_letter) - ord('A')
        new_letter = chr(ord('A') + (idx - 1) % 26)
        if hasattr(self, 'tutor_letter_combo'):
            self.tutor_letter_combo.setCurrentIndex(ord(new_letter) - ord('A'))

    def on_tutor_next_letter(self):
        idx = ord(self.current_tutor_letter) - ord('A')
        new_letter = chr(ord('A') + (idx + 1) % 26)
        if hasattr(self, 'tutor_letter_combo'):
            self.tutor_letter_combo.setCurrentIndex(ord(new_letter) - ord('A'))

    def on_guide_toggle(self, checked):
        self.guide_enabled = checked
        if self.video_thread:
            self.video_thread.guide_enabled = checked
            self.video_thread.guide_image = self.load_guide_image(self.current_tutor_letter)
            self.video_thread.guide_letter = self.current_tutor_letter

    def on_guide_opacity_changed(self, value):
        self.guide_opacity = value / 100.0
        if hasattr(self, 'guide_opacity_label'):
            self.guide_opacity_label.setText(f"{value}%")
        if self.video_thread:
            self.video_thread.guide_opacity = self.guide_opacity

    def on_mirror_toggled(self, checked):
        self.mirror_camera_enabled = checked
        if self.video_thread:
            self.video_thread.mirror_display = checked
        self.statusBar().showMessage(f"Camera mirroring: {'ON (Selfie View)' if checked else 'OFF (Natural View)'}", 2000)

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("ISL Translation System v2.0 Pro - [Grammar | Real-time Hands | Tutor]")

        # Fit the window to the screen actually available rather than a fixed
        # 1600x950, which overflows smaller displays and clips the right-hand
        # control panel off-screen.
        #
        # Windows display scaling complicates this: at 125% scaling Qt may
        # report the full physical resolution (e.g. 1920x1080) while its own
        # output still gets scaled up by 1.25 before hitting the screen, so a
        # "1600px" window really occupies 2000px. Divide the reported geometry
        # by the DPI ratio to get the space we can actually use. When Qt *is*
        # DPI-aware the ratio is 1.0 and this is a no-op.
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry()
        dpi_scale = max(1.0, screen.logicalDotsPerInch() / 96.0)

        usable_w = int(available.width() / dpi_scale)
        usable_h = int(available.height() / dpi_scale)

        width = min(1600, usable_w - 60)
        height = min(950, usable_h - 60)
        self.resize(width, height)
        self.move(
            available.x() + max(0, (usable_w - width) // 2),
            available.y() + max(0, (usable_h - height) // 2),
        )
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        central_widget.setLayout(main_layout)
        
        # Left panel - Video feed (60% width)
        left_panel = self.create_video_panel()
        main_layout.addWidget(left_panel, stretch=3)
        
        # Right panel - Controls (40% width)
        right_panel = self.create_control_panel()
        main_layout.addWidget(right_panel, stretch=2)
        
        # Apply modern theme
        self.apply_theme()
        
        # Status bar
        self.statusBar().showMessage('Ready - Press Space to capture gestures')
        self.statusBar().setStyleSheet("background-color: #1e1e1e; color: #00ff00; padding: 5px;")
        
        # Menu bar
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        start_action = QAction('Start Camera', self)
        start_action.triggered.connect(self.start_camera)
        file_menu.addAction(start_action)
        
        stop_action = QAction('Stop Camera', self)
        stop_action.triggered.connect(self.stop_camera)
        file_menu.addAction(stop_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu('Settings')
        
        tts_settings = QAction('TTS Settings', self)
        tts_settings.triggered.connect(self.show_tts_settings)
        settings_menu.addAction(tts_settings)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_video_panel(self):
        """Create video feed panel with sleek HUD status card"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        panel.setLayout(layout)
        
        # Video label with sleek rounded border
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 270)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #0d1117;
                border: 2px solid #30363d;
                border-radius: 12px;
            }
        """)
        layout.addWidget(self.video_label, stretch=1)
        
        # Detection HUD Card
        hud_card = QFrame()
        hud_card.setObjectName("hudCard")
        hud_card.setStyleSheet("""
            #hudCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #161b22, stop:0.5 #1c2128, stop:1 #161b22);
                border: 1px solid #30363d;
                border-radius: 12px;
            }
        """)
        hud_layout = QVBoxLayout()
        hud_layout.setContentsMargins(16, 12, 16, 12)
        hud_layout.setSpacing(8)
        
        # Top row: Sign Badge & Confidence percentage
        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        
        sign_container = QWidget()
        sign_container_layout = QHBoxLayout()
        sign_container_layout.setContentsMargins(0, 0, 0, 0)
        sign_container_layout.setSpacing(10)
        
        sign_icon = QLabel("✋")
        sign_icon.setFont(QFont('Segoe UI Emoji', 16))
        sign_container_layout.addWidget(sign_icon)
        
        self.current_sign_label = QLabel("NO SIGN DETECTED")
        self.current_sign_label.setFont(QFont('Segoe UI', 18, QFont.Bold))
        self.current_sign_label.setStyleSheet("color: #58a6ff; letter-spacing: 1px;")
        sign_container_layout.addWidget(self.current_sign_label)
        sign_container.setLayout(sign_container_layout)
        top_row.addWidget(sign_container)
        
        top_row.addStretch()
        
        self.confidence_label = QLabel("Confidence: 0.0%")
        self.confidence_label.setFont(QFont('Segoe UI', 15, QFont.Bold))
        self.confidence_label.setStyleSheet("color: #3fb950;")
        top_row.addWidget(self.confidence_label)
        
        hud_layout.addLayout(top_row)
        
        # Middle: High-tech progress bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMinimum(0)
        self.confidence_bar.setMaximum(100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setFixedHeight(8)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #21262d;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #58a6ff, stop:0.7 #3fb950, stop:1 #2ea043);
            }
        """)
        hud_layout.addWidget(self.confidence_bar)
        
        # Bottom row: Quick action hints
        bottom_hints = QHBoxLayout()
        hints_label = QLabel("⚡ Shortcuts: [Space] Capture Sign  •  [T] Translate & Speak  •  [C] Clear  •  [M] Auto/Manual")
        hints_label.setFont(QFont('Segoe UI', 9))
        hints_label.setStyleSheet("color: #8b949e;")
        bottom_hints.addWidget(hints_label)
        bottom_hints.addStretch()
        hud_layout.addLayout(bottom_hints)
        
        hud_card.setLayout(hud_layout)
        layout.addWidget(hud_card)
        
        return panel
    
    def create_control_panel(self):
        """Create tabbed control panel for sleek navigation and zero clutter"""
        panel = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(0, 0, 0, 0)
        panel.setLayout(main_layout)
        
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        
        # ==========================================
        # TAB 1: 🗣️ LIVE TRANSLATION & SPEECH
        # ==========================================
        tab_translate = QWidget()
        tab_trans_layout = QVBoxLayout()
        tab_trans_layout.setSpacing(10)
        tab_trans_layout.setContentsMargins(10, 12, 10, 10)
        
        # Mode indicator with toggle
        mode_container = QWidget()
        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mode_indicator = QLabel("● MANUAL MODE")
        self.mode_indicator.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.mode_indicator.setStyleSheet("color: #58a6ff; padding: 4px;")
        mode_layout.addWidget(self.mode_indicator)
        
        self.mode_toggle_btn = QPushButton("⇄ Switch to Auto")
        self.mode_toggle_btn.setMinimumHeight(28)
        self.mode_toggle_btn.setCheckable(True)
        self.mode_toggle_btn.clicked.connect(self.toggle_mode)
        mode_layout.addWidget(self.mode_toggle_btn)
        
        mode_container.setLayout(mode_layout)
        tab_trans_layout.addWidget(mode_container)
        
        # Capture button (prominent hero button)
        self.capture_btn = QPushButton("📸 CAPTURE SIGN (Space)")
        self.capture_btn.setMinimumHeight(50)
        self.capture_btn.setFont(QFont('Segoe UI', 12, QFont.Bold))
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #238636, stop:1 #2ea043);
                color: white;
                border: 1px solid #3fb950;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2ea043, stop:1 #3fb950);
                border: 1px solid #56d364;
            }
            QPushButton:pressed {
                background: #1b6528;
            }
        """)
        self.capture_btn.clicked.connect(self.manual_capture)
        tab_trans_layout.addWidget(self.capture_btn)
        
        # Sequence box
        sequence_group = QGroupBox("📝 Detected Signs Sequence")
        sequence_layout = QVBoxLayout()
        self.sequence_text = QTextEdit()
        self.sequence_text.setReadOnly(True)
        self.sequence_text.setMaximumHeight(70)
        self.sequence_text.setFont(QFont('Courier New', 11, QFont.Bold))
        self.sequence_text.setPlaceholderText("Captured signs: A → B → C...")
        sequence_layout.addWidget(self.sequence_text)
        sequence_group.setLayout(sequence_layout)
        tab_trans_layout.addWidget(sequence_group)
        
        # Translation output box
        translation_group = QGroupBox("✅ Grammar AI Translation")
        translation_layout = QVBoxLayout()
        self.translation_text = QTextEdit()
        self.translation_text.setReadOnly(True)
        self.translation_text.setMaximumHeight(80)
        self.translation_text.setFont(QFont('Segoe UI', 11))
        self.translation_text.setPlaceholderText("English translation will appear here...")
        translation_layout.addWidget(self.translation_text)
        
        grammar_hint = QLabel("💡 'who you' → 'Who are you?'  |  'name you what' → 'What is your name?'")
        grammar_hint.setFont(QFont('Segoe UI', 8))
        grammar_hint.setStyleSheet("color: #8b949e; padding: 2px;")
        translation_layout.addWidget(grammar_hint)
        translation_group.setLayout(translation_layout)
        tab_trans_layout.addWidget(translation_group)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)
        
        self.translate_btn = QPushButton("🔊 Speak & Translate (T)")
        self.translate_btn.setMinimumHeight(40)
        self.translate_btn.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.translate_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1f6feb, stop:1 #388bfd);
                color: white;
                border: 1px solid #58a6ff;
                border-radius: 8px;
            }
            QPushButton:hover { background: #388bfd; }
        """)
        self.translate_btn.clicked.connect(self.translate_sequence)
        actions_layout.addWidget(self.translate_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear (C)")
        self.clear_btn.setMinimumHeight(40)
        self.clear_btn.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                color: #f85149;
                border: 1px solid #da3633;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #b62324; color: white; }
        """)
        self.clear_btn.clicked.connect(self.clear_sequence)
        actions_layout.addWidget(self.clear_btn)
        tab_trans_layout.addLayout(actions_layout)
        
        # Dual Ensemble Engine selector card
        engine_group = QGroupBox("🤖 Recognition Engine")
        engine_layout = QVBoxLayout()
        engine_layout.setSpacing(6)
        
        mode_select_layout = QHBoxLayout()
        mode_label = QLabel("Active Pipeline:")
        mode_label.setFont(QFont('Segoe UI', 9, QFont.Bold))
        mode_select_layout.addWidget(mode_label)
        
        self.engine_mode_combo = QComboBox()
        self.engine_mode_combo.addItems([
            "⚡ Auto (MLP Static + GRU Motion)",
            "✋ MLP Only (Static Handsigns)",
            "🌊 GRU Only (Motion Gestures)"
        ])
        current_mode = self.config.get('detection', 'mode', default='auto_ensemble')
        if current_mode == 'mlp_only':
            self.engine_mode_combo.setCurrentIndex(1)
        elif current_mode == 'gru_only':
            self.engine_mode_combo.setCurrentIndex(2)
        else:
            self.engine_mode_combo.setCurrentIndex(0)
            
        self.engine_mode_combo.currentIndexChanged.connect(self.on_engine_mode_changed)
        mode_select_layout.addWidget(self.engine_mode_combo)
        engine_layout.addLayout(mode_select_layout)
        
        self.engine_status_label = QLabel("Active: MLP (Static Signs) | v: 0.000")
        self.engine_status_label.setFont(QFont('Segoe UI', 9))
        self.engine_status_label.setStyleSheet("color: #3fb950; padding: 2px;")
        engine_layout.addWidget(self.engine_status_label)
        
        engine_group.setLayout(engine_layout)
        tab_trans_layout.addWidget(engine_group)
        
        tab_trans_layout.addStretch()
        tab_translate.setLayout(tab_trans_layout)
        self.tabs.addTab(tab_translate, "🗣️ Live Translate")
        
        # ==========================================
        # TAB 2: 🎓 TUTOR & PRACTICE (Learn ISL)
        # ==========================================
        tab_tutor = QWidget()
        tab_tutor_layout = QVBoxLayout()
        tab_tutor_layout.setSpacing(10)
        tab_tutor_layout.setContentsMargins(10, 12, 10, 10)
        
        # Camera Mirroring
        self.mirror_checkbox = QCheckBox("🪞 Mirror Camera Feed (Selfie View)")
        self.mirror_checkbox.setChecked(True)
        self.mirror_checkbox.setFont(QFont('Segoe UI', 9, QFont.Bold))
        self.mirror_checkbox.setStyleSheet("color: #58a6ff;")
        self.mirror_checkbox.toggled.connect(self.on_mirror_toggled)
        tab_tutor_layout.addWidget(self.mirror_checkbox)
        
        # Letter selector
        letter_select_widget = QWidget()
        letter_select_layout = QHBoxLayout()
        letter_select_layout.setContentsMargins(0, 0, 0, 0)
        letter_select_layout.setSpacing(6)
        
        prev_btn = QPushButton("◀")
        prev_btn.setFixedWidth(36)
        prev_btn.setFixedHeight(34)
        prev_btn.clicked.connect(self.on_tutor_prev_letter)
        letter_select_layout.addWidget(prev_btn)
        
        tutor_label = QLabel("Practice Sign:")
        tutor_label.setFont(QFont('Segoe UI', 9, QFont.Bold))
        letter_select_layout.addWidget(tutor_label)
        
        self.tutor_letter_combo = QComboBox()
        self.tutor_letter_combo.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.tutor_letter_combo.setFixedHeight(34)
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.tutor_letter_combo.addItem(f"Letter {c}", c)
        self.tutor_letter_combo.currentIndexChanged.connect(lambda i: self.on_tutor_letter_changed(self.tutor_letter_combo.currentData()))
        letter_select_layout.addWidget(self.tutor_letter_combo, stretch=1)
        
        next_btn = QPushButton("▶")
        next_btn.setFixedWidth(36)
        next_btn.setFixedHeight(34)
        next_btn.clicked.connect(self.on_tutor_next_letter)
        letter_select_layout.addWidget(next_btn)
        
        letter_select_widget.setLayout(letter_select_layout)
        tab_tutor_layout.addWidget(letter_select_widget)
        
        # Ghost guide overlay
        guide_group = QGroupBox("👻 Ghost Guide Overlay")
        guide_layout = QVBoxLayout()
        guide_layout.setSpacing(6)
        
        self.guide_checkbox = QCheckBox("Show Ghost Guide on Camera Video")
        self.guide_checkbox.setFont(QFont('Segoe UI', 9, QFont.Bold))
        self.guide_checkbox.setStyleSheet("color: #3fb950;")
        self.guide_checkbox.setChecked(False)
        self.guide_checkbox.toggled.connect(self.on_guide_toggle)
        guide_layout.addWidget(self.guide_checkbox)
        
        opacity_widget = QWidget()
        opacity_layout = QHBoxLayout()
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_title = QLabel("Opacity:")
        opacity_title.setFont(QFont('Segoe UI', 8))
        opacity_title.setStyleSheet("color: #8b949e;")
        opacity_layout.addWidget(opacity_title)
        
        self.guide_opacity_slider = QSlider(Qt.Horizontal)
        self.guide_opacity_slider.setRange(10, 95)
        self.guide_opacity_slider.setValue(50)
        self.guide_opacity_slider.valueChanged.connect(self.on_guide_opacity_changed)
        opacity_layout.addWidget(self.guide_opacity_slider, stretch=1)
        
        self.guide_opacity_label = QLabel("50%")
        self.guide_opacity_label.setFont(QFont('Segoe UI', 8, QFont.Bold))
        self.guide_opacity_label.setStyleSheet("color: #3fb950; min-width: 32px;")
        opacity_layout.addWidget(self.guide_opacity_label)
        opacity_widget.setLayout(opacity_layout)
        guide_layout.addWidget(opacity_widget)
        guide_group.setLayout(guide_layout)
        tab_tutor_layout.addWidget(guide_group)
        
        # Reference card
        card_widget = QFrame()
        card_widget.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 10px;
            }
        """)
        card_layout = QHBoxLayout()
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(12)
        
        self.tutor_preview_label = QLabel()
        self.tutor_preview_label.setFixedSize(90, 90)
        self.tutor_preview_label.setAlignment(Qt.AlignCenter)
        self.tutor_preview_label.setStyleSheet("background-color: #0d1117; border: 1px solid #30363d; border-radius: 8px;")
        card_layout.addWidget(self.tutor_preview_label)
        
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)
        self.tutor_target_label = QLabel("Target Sign: Letter A")
        self.tutor_target_label.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.tutor_target_label.setStyleSheet("color: #58a6ff; border: none; background: transparent;")
        info_layout.addWidget(self.tutor_target_label)
        
        self.tutor_hint_label = QLabel(ISL_LETTER_HINTS.get('A', ''))
        self.tutor_hint_label.setFont(QFont('Segoe UI', 8))
        self.tutor_hint_label.setStyleSheet("color: #c9d1d9; border: none; background: transparent;")
        self.tutor_hint_label.setWordWrap(True)
        info_layout.addWidget(self.tutor_hint_label)
        
        card_layout.addLayout(info_layout, stretch=1)
        card_widget.setLayout(card_layout)
        tab_tutor_layout.addWidget(card_widget)
        
        # Match status badge
        self.tutor_match_status = QLabel("Ready to practice Letter A")
        self.tutor_match_status.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.tutor_match_status.setStyleSheet("color: #58a6ff; padding: 10px; background: rgba(88, 166, 255, 0.12); border: 1px solid #58a6ff; border-radius: 8px;")
        self.tutor_match_status.setAlignment(Qt.AlignCenter)
        self.tutor_match_status.setWordWrap(True)
        tab_tutor_layout.addWidget(self.tutor_match_status)
        
        tab_tutor_layout.addStretch()
        tab_tutor.setLayout(tab_tutor_layout)
        self.tabs.addTab(tab_tutor, "🎓 Tutor & Guide")
        
        # ==========================================
        # TAB 3: 📸 DATA STUDIO & TRAINING
        # ==========================================
        tab_collector = QWidget()
        tab_coll_scroll = QScrollArea()
        tab_coll_scroll.setWidgetResizable(True)
        tab_coll_scroll.setStyleSheet("border: none; background: transparent;")
        tab_coll_widget = QWidget()
        tab_coll_layout = QVBoxLayout()
        tab_coll_layout.setSpacing(10)
        tab_coll_layout.setContentsMargins(8, 10, 8, 10)
        
        # Quick collector
        collect_quick_group = QGroupBox("📸 Data Capture")
        collect_quick_layout = QVBoxLayout()
        collect_quick_layout.setSpacing(8)
        
        quick_class_layout = QHBoxLayout()
        quick_class_label = QLabel("Sign Name:")
        quick_class_label.setFont(QFont('Segoe UI', 9, QFont.Bold))
        quick_class_label.setStyleSheet("color: #58a6ff;")
        quick_class_layout.addWidget(quick_class_label)
        
        self.quick_class_input = QLineEdit()
        self.quick_class_input.setPlaceholderText("Enter sign name (e.g., hello, thank_you)...")
        self.quick_class_input.setFont(QFont('Segoe UI', 10))
        self.quick_class_input.setMinimumHeight(32)
        self.quick_class_input.textChanged.connect(self.on_quick_class_changed)
        quick_class_layout.addWidget(self.quick_class_input)
        collect_quick_layout.addLayout(quick_class_layout)
        
        # Counter cards
        self.data_counter_widget = QWidget()
        counter_layout = QHBoxLayout()
        counter_layout.setContentsMargins(0, 4, 0, 4)
        
        photos_box = QWidget()
        photos_box.setStyleSheet("background: #1c2128; border: 1px solid #3fb950; border-radius: 8px; padding: 6px;")
        photos_layout = QVBoxLayout()
        photos_title = QLabel("📷 PHOTOS")
        photos_title.setFont(QFont('Segoe UI', 8, QFont.Bold))
        photos_title.setStyleSheet("color: #3fb950; border: none; background: transparent;")
        photos_title.setAlignment(Qt.AlignCenter)
        photos_layout.addWidget(photos_title)
        self.photos_count_label = QLabel("0")
        self.photos_count_label.setFont(QFont('Segoe UI', 20, QFont.Bold))
        self.photos_count_label.setStyleSheet("color: #3fb950; border: none; background: transparent;")
        self.photos_count_label.setAlignment(Qt.AlignCenter)
        photos_layout.addWidget(self.photos_count_label)
        photos_box.setLayout(photos_layout)
        counter_layout.addWidget(photos_box)
        
        videos_box = QWidget()
        videos_box.setStyleSheet("background: #1c2128; border: 1px solid #f778ba; border-radius: 8px; padding: 6px;")
        videos_layout = QVBoxLayout()
        videos_title = QLabel("🎥 VIDEOS")
        videos_title.setFont(QFont('Segoe UI', 8, QFont.Bold))
        videos_title.setStyleSheet("color: #f778ba; border: none; background: transparent;")
        videos_title.setAlignment(Qt.AlignCenter)
        videos_layout.addWidget(videos_title)
        self.videos_count_label = QLabel("0")
        self.videos_count_label.setFont(QFont('Segoe UI', 20, QFont.Bold))
        self.videos_count_label.setStyleSheet("color: #f778ba; border: none; background: transparent;")
        self.videos_count_label.setAlignment(Qt.AlignCenter)
        videos_layout.addWidget(self.videos_count_label)
        videos_box.setLayout(videos_layout)
        counter_layout.addWidget(videos_box)
        
        self.data_counter_widget.setLayout(counter_layout)
        collect_quick_layout.addWidget(self.data_counter_widget)
        
        self.quick_stats_label = QLabel("💡 Enter a sign name above to begin recording")
        self.quick_stats_label.setFont(QFont('Segoe UI', 8))
        self.quick_stats_label.setStyleSheet("color: #8b949e; padding: 4px; background: rgba(139, 148, 158, 0.1); border-radius: 6px;")
        self.quick_stats_label.setAlignment(Qt.AlignCenter)
        collect_quick_layout.addWidget(self.quick_stats_label)
        
        # Action capture buttons
        capture_buttons_layout = QHBoxLayout()
        self.start_capture_btn = QPushButton("📸 CAPTURE PHOTO")
        self.start_capture_btn.setMinimumHeight(44)
        self.start_capture_btn.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.start_capture_btn.clicked.connect(self.start_photo_capture)
        self.start_capture_btn.setEnabled(False)
        capture_buttons_layout.addWidget(self.start_capture_btn)
        
        self.start_video_btn = QPushButton("🎥 RECORD VIDEO")
        self.start_video_btn.setMinimumHeight(44)
        self.start_video_btn.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.start_video_btn.clicked.connect(self.start_video_capture)
        self.start_video_btn.setEnabled(False)
        capture_buttons_layout.addWidget(self.start_video_btn)
        collect_quick_layout.addLayout(capture_buttons_layout)
        
        self.live_counter_label = QLabel("📊 Session: 0 captured")
        self.live_counter_label.setFont(QFont('Segoe UI', 9, QFont.Bold))
        self.live_counter_label.setStyleSheet("color: #58a6ff; padding: 8px; background: #161b22; border: 1px solid #30363d; border-radius: 6px;")
        self.live_counter_label.setAlignment(Qt.AlignCenter)
        self.live_counter_label.setVisible(False)
        collect_quick_layout.addWidget(self.live_counter_label)
        
        self.quick_collect_btn = QPushButton("🔧 Launch Standalone Collector Tool")
        self.quick_collect_btn.setMinimumHeight(34)
        self.quick_collect_btn.clicked.connect(self.open_data_collector)
        self.quick_collect_btn.setEnabled(False)
        collect_quick_layout.addWidget(self.quick_collect_btn)
        collect_quick_group.setLayout(collect_quick_layout)
        tab_coll_layout.addWidget(collect_quick_group)
        
        # Training group
        training_group = QGroupBox("🚀 Model Training")
        training_layout = QVBoxLayout()
        training_layout.setSpacing(6)
        
        self.class_name_input = QLineEdit()
        self.class_name_input.setPlaceholderText("Class name to train...")
        self.class_name_input.textChanged.connect(self.on_class_name_changed)
        training_layout.addWidget(self.class_name_input)
        
        self.collection_stats_label = QLabel("No class selected")
        self.collection_stats_label.setFont(QFont('Segoe UI', 8))
        self.collection_stats_label.setStyleSheet("color: #8b949e;")
        training_layout.addWidget(self.collection_stats_label)
        
        self.collect_data_btn = QPushButton("📸 Collect Batch Data")
        self.collect_data_btn.clicked.connect(self.open_data_collector)
        self.collect_data_btn.setEnabled(False)
        training_layout.addWidget(self.collect_data_btn)
        
        self.train_model_btn = QPushButton("⚡ RETRAIN MODEL NOW")
        self.train_model_btn.setMinimumHeight(40)
        self.train_model_btn.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.train_model_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #8957e5, stop:1 #a371f7);
                color: white;
                border: 1px solid #bc8cff;
                border-radius: 8px;
            }
            QPushButton:hover { background: #a371f7; }
        """)
        self.train_model_btn.clicked.connect(self.train_model)
        self.train_model_btn.setEnabled(False)
        training_layout.addWidget(self.train_model_btn)
        
        self.collected_stats_label = QLabel("📦 Loading dataset...")
        self.collected_stats_label.setFont(QFont('Courier New', 8))
        self.collected_stats_label.setStyleSheet("color: #58a6ff;")
        training_layout.addWidget(self.collected_stats_label)
        training_group.setLayout(training_layout)
        tab_coll_layout.addWidget(training_group)
        
        tab_coll_widget.setLayout(tab_coll_layout)
        tab_coll_scroll.setWidget(tab_coll_widget)
        tab_collector_layout = QVBoxLayout()
        tab_collector_layout.setContentsMargins(0, 0, 0, 0)
        tab_collector_layout.addWidget(tab_coll_scroll)
        tab_collector.setLayout(tab_collector_layout)
        self.tabs.addTab(tab_collector, "📸 Data Studio")
        
        # ==========================================
        # TAB 4: 📊 SYSTEM & ANALYTICS
        # ==========================================
        tab_stats = QWidget()
        tab_stats_layout = QVBoxLayout()
        tab_stats_layout.setSpacing(10)
        tab_stats_layout.setContentsMargins(10, 12, 10, 10)
        
        stats_group = QGroupBox("📊 Session Metrics")
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(6)
        
        self.fps_label = QLabel("⚡ Video FPS: 0.0")
        self.fps_label.setFont(QFont('Segoe UI', 9))
        self.fps_label.setStyleSheet("color: #3fb950;")
        stats_layout.addWidget(self.fps_label)
        
        self.captures_label = QLabel("📸 Total Captures: 0")
        self.captures_label.setFont(QFont('Segoe UI', 9))
        stats_layout.addWidget(self.captures_label)
        
        self.sentences_label = QLabel("📝 Sentences Spoken: 0")
        self.sentences_label.setFont(QFont('Segoe UI', 9))
        stats_layout.addWidget(self.sentences_label)
        
        self.uptime_label = QLabel("⏱️ Session Uptime: 00:00")
        self.uptime_label.setFont(QFont('Segoe UI', 9))
        stats_layout.addWidget(self.uptime_label)
        stats_group.setLayout(stats_layout)
        tab_stats_layout.addWidget(stats_group)
        
        # Active features
        features_group = QGroupBox("⚡ Active Capabilities")
        features_layout = QVBoxLayout()
        features_text = "✓ Dual Ensemble: MLP (126) + GRU (186)\n✓ Kinematic Velocity Routing\n✓ Real-time Hands Tracking (MediaPipe)\n✓ Rule-based Grammar Engine\n✓ Real-time Speech Synthesis (pyttsx3)"
        features_label = QLabel(features_text)
        features_label.setFont(QFont('Segoe UI', 9))
        features_label.setStyleSheet("color: #3fb950; line-height: 1.4;")
        features_layout.addWidget(features_label)
        features_group.setLayout(features_layout)
        tab_stats_layout.addWidget(features_group)
        
        tab_stats_layout.addStretch()
        tab_stats.setLayout(tab_stats_layout)
        self.tabs.addTab(tab_stats, "📊 Analytics")
        
        main_layout.addWidget(self.tabs)
        
        # Trigger stats loader
        QTimer.singleShot(500, self.update_collected_stats)
        
        return panel
    
    def apply_theme(self):
        """Apply modern dark obsidian theme with clean Vercel/GitHub aesthetic"""
        self.setStyleSheet("""
            /* Main Window */
            QMainWindow {
                background-color: #0d1117;
                color: #f0f6fc;
            }
            
            /* Tabs */
            QTabWidget::pane {
                border: 1px solid #30363d;
                background: #161b22;
                border-radius: 10px;
                top: -1px;
            }
            QTabBar::tab {
                background: #0d1117;
                color: #8b949e;
                border: 1px solid #30363d;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 8px 14px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: bold;
                font-size: 11px;
                margin-right: 3px;
            }
            QTabBar::tab:selected {
                background: #161b22;
                color: #58a6ff;
                border-top: 2px solid #58a6ff;
            }
            QTabBar::tab:hover:!selected {
                background: #1c2128;
                color: #c9d1d9;
            }
            
            /* Group Boxes */
            QGroupBox {
                color: #f0f6fc;
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 10px;
                margin-top: 14px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: bold;
                font-size: 11px;
                padding: 14px 10px 10px 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 14px;
                top: 2px;
                padding: 2px 8px;
                background-color: #161b22;
                color: #58a6ff;
                border-radius: 4px;
            }
            
            /* Buttons */
            QPushButton {
                background-color: #21262d;
                color: #f0f6fc;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 6px 12px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10px;
                font-weight: bold;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #8b949e;
            }
            QPushButton:pressed {
                background-color: #161b22;
            }
            QPushButton:disabled {
                background-color: #161b22;
                color: #484f58;
                border: 1px solid #21262d;
            }
            
            /* Text Inputs */
            QTextEdit, QPlainTextEdit {
                background-color: #0d1117;
                color: #f0f6fc;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 8px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                selection-background-color: #1f6feb;
            }
            QTextEdit:focus, QPlainTextEdit:focus {
                border: 1px solid #58a6ff;
            }
            
            QLineEdit {
                background-color: #0d1117;
                color: #f0f6fc;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 6px 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10px;
            }
            QLineEdit:focus {
                border: 1px solid #58a6ff;
            }
            
            QComboBox {
                background-color: #0d1117;
                color: #58a6ff;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 5px 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10px;
                font-weight: bold;
            }
            QComboBox:focus {
                border: 1px solid #58a6ff;
            }
            QComboBox QAbstractItemView {
                background-color: #161b22;
                color: #f0f6fc;
                selection-background-color: #1f6feb;
                border: 1px solid #30363d;
                border-radius: 6px;
            }
            
            /* Scrollbars */
            QScrollBar:vertical {
                background: #0d1117;
                width: 8px;
                border-radius: 4px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #30363d;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #58a6ff;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            /* Menu & Status Bar */
            QMenuBar {
                background-color: #161b22;
                color: #f0f6fc;
                border-bottom: 1px solid #30363d;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background-color: #21262d;
                border-radius: 4px;
            }
            QMenu {
                background-color: #161b22;
                color: #f0f6fc;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item:selected {
                background-color: #1f6feb;
                border-radius: 4px;
            }
            
            QStatusBar {
                background-color: #0d1117;
                color: #3fb950;
                border-top: 1px solid #30363d;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: bold;
            }
        """)
    
    def start_camera(self):
        """Start camera feed"""
        if self.video_thread is None or not self.video_thread.isRunning():
            camera_id = self.config.get('camera', 'device_id', default=0)
            self.video_thread = VideoThread(self.inference_engine, camera_id)
            self.video_thread.mirror_display = getattr(self, 'mirror_camera_enabled', True)
            self.video_thread.guide_enabled = getattr(self, 'guide_enabled', False)
            self.video_thread.guide_image = self.load_guide_image(getattr(self, 'current_tutor_letter', 'A'))
            self.video_thread.guide_letter = getattr(self, 'current_tutor_letter', 'A')
            self.video_thread.guide_opacity = getattr(self, 'guide_opacity', 0.5)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.prediction_signal.connect(self.update_prediction)
            self.video_thread.start()
            self.statusBar().showMessage('Camera started')
    
    def stop_camera(self):
        """Stop camera feed"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.statusBar().showMessage('Camera stopped')
    
    def update_image(self, frame):
        """Update video frame"""
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
        """Update prediction display"""
        if not prediction:
            self.last_prediction = None
            self.last_confidence = 0.0
            self.current_sign_label.setText("No sign detected")
            self.confidence_label.setText("Confidence: 0%")
            self.confidence_bar.setValue(0)
            if hasattr(self, 'tutor_match_status') and hasattr(self, 'current_tutor_letter'):
                target = self.current_tutor_letter
                self.tutor_match_status.setText(f"Form the sign for Letter {target}")
                self.tutor_match_status.setStyleSheet(
                    "color: #aaaaaa; padding: 6px; background: rgba(80, 80, 80, 0.15); border-radius: 6px;"
                )
            return

        # Store for manual mode
        self.last_prediction = prediction
        self.last_confidence = confidence
        
        self.current_sign_label.setText(prediction)
        self.confidence_label.setText(f"Confidence: {confidence:.1%}")
        self.confidence_bar.setValue(int(confidence * 100))
        
        # In manual mode, DON'T auto-add to sequence
        # Only update sequence display
        if not self.manual_mode:
            # Auto mode - add to sequence automatically
            words = self.inference_engine.detected_words
            self.sequence_text.setPlainText(' → '.join(words))
        
        # Update Tutor practice match indicator
        if hasattr(self, 'tutor_match_status') and hasattr(self, 'current_tutor_letter'):
            target = self.current_tutor_letter
            if prediction == target and confidence >= 0.60:
                self.tutor_match_status.setText(f"🎯 EXCELLENT! Matched '{prediction}' ({confidence:.1%})")
                self.tutor_match_status.setStyleSheet(
                    "color: #00ff88; font-weight: bold; padding: 6px; "
                    "background: rgba(0, 255, 136, 0.25); border: 1px solid #00ff88; border-radius: 6px;"
                )
            else:
                self.tutor_match_status.setText(f"Target: {target} | Detected: {prediction} ({confidence:.0%})")
                self.tutor_match_status.setStyleSheet(
                    "color: #ffaa00; padding: 6px; background: rgba(255, 170, 0, 0.15); border-radius: 6px;"
                )

        # Update engine status badge
        if hasattr(self, 'engine_status_label'):
            active = getattr(self.inference_engine, 'active_engine', 'MLP')
            vel = getattr(self.inference_engine.velocity_tracker, 'last_velocity', 0.0)
            self.engine_status_label.setText(f"Active: {active} {'(Motion)' if active == 'GRU' else '(Static)'} | v: {vel:.3f}")

        # Update stats
        self.update_statistics()
    
    def on_engine_mode_changed(self, index):
        """Handle manual or auto switching between MLP static and GRU motion engines"""
        modes = ['auto_ensemble', 'mlp_only', 'gru_only']
        mode = modes[index]
        self.inference_engine.ensemble_mode = mode
        mode_names = ['Auto (MLP + GRU)', 'MLP (Static Signs)', 'GRU (Motion Signs)']
        self.statusBar().showMessage(f"Recognition Engine set to: {mode_names[index]}", 3000)
    
    def translate_sequence(self):
        """Translate and speak current sequence"""
        sentence = self.inference_engine.translate_sequence()
        if sentence:
            self.translation_text.append(f"• {sentence}\n")
            self.statusBar().showMessage('Translation spoken', 3000)
    
    def clear_sequence(self):
        """Clear detected sequence"""
        self.inference_engine.clear_sequence()
        self.sequence_text.clear()
        self.current_sign_label.setText("No sign detected")
        self.confidence_label.setText("Confidence: 0%")
        self.confidence_bar.setValue(0)
        self.statusBar().showMessage('Sequence cleared', 2000)
    
    def toggle_recording(self):
        """Toggle recording"""
        if self.record_btn.isChecked():
            self.is_recording = True
            self.record_btn.setText("⏹️ Stop Recording")
            self.record_btn.setStyleSheet("background-color: #cc0000; color: white;")
            self.statusBar().showMessage('Recording...')
        else:
            self.is_recording = False
            self.record_btn.setText("⏺️ Start Recording")
            self.record_btn.setStyleSheet("")
            self.statusBar().showMessage('Recording stopped', 2000)
            
            if self.recorded_frames:
                self.save_recording()
    
    def save_recording(self):
        """Save recorded video"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Recording", "", "Video Files (*.mp4)"
        )
        
        if filename:
            # Save video (implementation depends on requirements)
            self.statusBar().showMessage(f'Recording saved to {filename}', 3000)
            self.recorded_frames = []
    
    def update_statistics(self):
        """Update statistics display"""
        words_count = len(self.inference_engine.detected_words)
        sentences_count = self.translation_text.toPlainText().count('•')
        
        stats_text = f"Signs: {words_count} | Sentences: {sentences_count} | "
        stats_text += f"Accuracy: {self.inference_engine.prediction_confidence:.1%}"
        
        self.stats_label.setText(stats_text)
    
    def on_class_name_changed(self, text):
        """Handle class name input change"""
        if text.strip():
            self.collect_data_btn.setEnabled(True)
            # Check if data exists for this class
            from pathlib import Path
            data_dir = Path('data/ISL_DATASETS_2') / text.strip()
            
            photo_count = 0
            video_count = 0
            
            if data_dir.exists():
                photo_count = len(list(data_dir.glob('*.jpg')))
                video_count = len(list(data_dir.glob('*.avi')))
            
            if photo_count > 0 or video_count > 0:
                self.collection_stats_label.setText(
                    f"📊 '{text}': {photo_count} photos, {video_count} videos"
                )
                self.collection_stats_label.setStyleSheet("color: #00ff00; padding: 5px;")
            else:
                self.collection_stats_label.setText(
                    f"📊 '{text}': No data yet (Recommended: 80 photos or 25 videos)"
                )
                self.collection_stats_label.setStyleSheet("color: #ffaa00; padding: 5px;")
        else:
            self.collect_data_btn.setEnabled(False)
            self.collection_stats_label.setText("📊 No class selected")
            self.collection_stats_label.setStyleSheet("color: #ffaa00; padding: 5px;")
    
    def on_quick_class_changed(self, text):
        """Handle quick class input change - Update data counters"""
        if text.strip():
            self.quick_collect_btn.setEnabled(True)
            self.start_capture_btn.setEnabled(True)
            self.start_video_btn.setEnabled(True)
            
            # Check if data exists for this class
            from pathlib import Path
            data_dir = Path('data/ISL_DATASETS_2') / text.strip()
            
            photo_count = 0
            video_count = 0
            
            if data_dir.exists():
                photo_count = len(list(data_dir.glob('*.jpg')))
                video_count = len(list(data_dir.glob('*.avi')))
            
            # Update the large counter displays
            self.photos_count_label.setText(str(photo_count))
            self.videos_count_label.setText(str(video_count))
            
            # Update status message
            if photo_count > 0 or video_count > 0:
                total = photo_count + video_count
                recommended = "Ready to train!" if photo_count >= 80 or video_count >= 25 else "Collect more data"
                self.quick_stats_label.setText(
                    f"✅ '{text}': {total} samples collected | {recommended}"
                )
                self.quick_stats_label.setStyleSheet("""
                    color: #00ff88; 
                    padding: 8px; 
                    background: rgba(0, 255, 136, 0.15); 
                    border-radius: 6px;
                    border-left: 4px solid #00ff88;
                    font-weight: bold;
                """)
                
                # Change counter colors to green if sufficient data
                if photo_count >= 80:
                    self.photos_count_label.setStyleSheet("color: #00ff00; border: none; background: transparent;")
                else:
                    self.photos_count_label.setStyleSheet("color: #00ff88; border: none; background: transparent;")
                    
                if video_count >= 25:
                    self.videos_count_label.setStyleSheet("color: #00ff00; border: none; background: transparent;")
                else:
                    self.videos_count_label.setStyleSheet("color: #ff00ff; border: none; background: transparent;")
            else:
                self.quick_stats_label.setText(
                    f"⚠️ '{text}': No data yet | Recommended: 80 photos OR 25 videos"
                )
                self.quick_stats_label.setStyleSheet("""
                    color: #ffaa00; 
                    padding: 8px; 
                    background: rgba(255, 170, 0, 0.15); 
                    border-radius: 6px;
                    border-left: 4px solid #ffaa00;
                    font-weight: bold;
                """)
                self.photos_count_label.setStyleSheet("color: #888888; border: none; background: transparent;")
                self.videos_count_label.setStyleSheet("color: #888888; border: none; background: transparent;")
            
            # Sync with main training input
            self.class_name_input.setText(text)
        else:
            # Reset when empty
            self.quick_collect_btn.setEnabled(False)
            self.start_capture_btn.setEnabled(False)
            self.start_video_btn.setEnabled(False)
            self.photos_count_label.setText("0")
            self.videos_count_label.setText("0")
            self.photos_count_label.setStyleSheet("color: #00ff88; border: none; background: transparent;")
            self.videos_count_label.setStyleSheet("color: #ff00ff; border: none; background: transparent;")
            self.quick_stats_label.setText("💡 Enter a symbol name above to begin")
            self.quick_stats_label.setStyleSheet("""
                color: #aaaaaa; 
                padding: 8px; 
                background: rgba(80, 80, 80, 0.3); 
                border-radius: 6px;
            """)
            self.live_counter_label.setVisible(False)
    
        
        # Update total collected stats whenever class changes
        self.update_collected_stats()
    
    def update_collected_stats(self):
        """Update total collected data statistics across all classes"""
        from pathlib import Path
        isl_datasets_2_path = Path('data/ISL_DATASETS_2')
        
        if not isl_datasets_2_path.exists():
            self.collected_stats_label.setText("📦 Collected Data: No data folder found")
            self.collected_stats_label.setStyleSheet("color: #888888; padding: 5px;")
            return
        
        try:
            total_classes = 0
            total_photos = 0
            total_videos = 0
            
            # Scan all class folders
            for class_dir in isl_datasets_2_path.iterdir():
                if class_dir.is_dir():
                    total_classes += 1
                    
                    # Count photos
                    total_photos += len(list(class_dir.glob('*.jpg')))
                    
                    # Count videos
                    total_videos += len(list(class_dir.glob('*.avi')))
            
            if total_classes > 0:
                stats_text = f"📦 Total: {total_classes} classes | {total_photos} photos | {total_videos} videos"
                self.collected_stats_label.setStyleSheet("color: #00ddff; padding: 5px; font-weight: bold;")
            else:
                stats_text = "📦 Collected Data: No classes found yet"
                self.collected_stats_label.setStyleSheet("color: #888888; padding: 5px;")
            
            self.collected_stats_label.setText(stats_text)
            
        except Exception as e:
            self.collected_stats_label.setText(f"📦 Error loading stats: {str(e)}")
            self.collected_stats_label.setStyleSheet("color: #ff5555; padding: 5px;")
    
    def start_photo_capture(self):
        """Start capturing photos in GUI with live counter"""
        class_name = self.quick_class_input.text().strip()
        if not class_name:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Symbol", "Please enter a symbol/word name first!")
            return
        
        # Initialize session counter
        if not hasattr(self, 'session_photo_count'):
            self.session_photo_count = 0
        
        self.session_photo_count = 0
        self.live_counter_label.setText(f"📊 Session: {self.session_photo_count} photos captured")
        self.live_counter_label.setVisible(True)
        
        # Change button to STOP
        self.start_capture_btn.setText("⏸️ STOP CAPTURE")
        self.start_capture_btn.clicked.disconnect()
        self.start_capture_btn.clicked.connect(self.stop_photo_capture)
        
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Photo Capture Started",
            f"📸 Capturing photos for '{class_name}'\n\n"
            "Press SPACE BAR to capture each photo\n"
            "The counter will update after each capture\n\n"
            "Click 'STOP CAPTURE' when done."
        )
    
    def stop_photo_capture(self):
        """Stop photo capture session"""
        self.start_capture_btn.setText("📸 START CAPTURE")
        self.start_capture_btn.clicked.disconnect()
        self.start_capture_btn.clicked.connect(self.start_photo_capture)
        
        # Refresh the counters
        self.on_quick_class_changed(self.quick_class_input.text())
        
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Capture Stopped",
            f"✅ Photo capture session ended!\n\n"
            f"Total photos captured this session: {self.session_photo_count}"
        )
    
    def start_video_capture(self):
        """Start video recording"""
        class_name = self.quick_class_input.text().strip()
        if not class_name:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Symbol", "Please enter a symbol/word name first!")
            return
        
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Video Capture",
            f"🎥 Video recording for '{class_name}'\n\n"
            "Press R to start/stop recording\n"
            "Use the full collection tool for video features"
        )
        
        # Open the full tool for video
        self.open_data_collector()
    
    def open_data_collector(self):

        """Open enhanced data collection tool"""
        class_name = self.class_name_input.text().strip()
        if not class_name:
            QMessageBox.warning(self, "No Class Name", "Please enter a symbol/class name first.")
            return
        
        reply = QMessageBox.question(
            self,
            "Open Data Collector",
            f"This will open the enhanced data collection tool for '{class_name}'.\n\n"
            "The tool provides:\n"
            "• 3-second countdown timer\n"
            "• Photo and video capture modes\n"
            "• Real-time statistics\n"
            "• Both hands + face detection\n\n"
            "Close this window and run:\n"
            f"python collect_training_data.py\n"
            "Then press 'T' and enter '{class_name}'\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            import subprocess
            subprocess.Popen(['python', 'collect_training_data.py'], cwd=os.getcwd())
            self.statusBar().showMessage(f"Opening data collector for '{class_name}'...", 3000)
    
    def train_model(self):
        """Train model with collected data"""
        from pathlib import Path
        import json
        
        # Check for collected data
        data_dir = Path('data/collected_data')
        if not data_dir.exists():
            QMessageBox.warning(
                self,
                "No Data Found",
                "No training data found!\n\n"
                "Please collect training data first:\n"
                "1. Enter a symbol/class name\n"
                "2. Click 'Collect Training Data'\n"
                "3. Capture 80 photos or 25 videos per class\n"
                "4. Repeat for 10-50 different classes"
            )
            return
        
        # Count classes and samples
        classes = []
        total_photos = 0
        total_videos = 0
        
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                photo_dir = class_dir / 'photos'
                video_dir = class_dir / 'videos'
                
                photo_count = len(list(photo_dir.glob('*.jpg'))) if photo_dir.exists() else 0
                video_count = len(list(video_dir.glob('*.avi'))) if video_dir.exists() else 0
                
                if photo_count > 0 or video_count > 0:
                    classes.append(class_dir.name)
                    total_photos += photo_count
                    total_videos += video_count
        
        if len(classes) == 0:
            QMessageBox.warning(
                self,
                "No Data Found",
                "No valid training data found!\n\n"
                "Each class needs at least some photos or videos."
            )
            return
        
        # Confirm training
        reply = QMessageBox.question(
            self,
            "Start Training",
            f"Ready to train model!\n\n"
            f"Classes: {len(classes)}\n"
            f"Total Photos: {total_photos}\n"
            f"Total Videos: {total_videos}\n\n"
            f"Classes to train:\n{', '.join(classes[:10])}"
            f"{'...' if len(classes) > 10 else ''}\n\n"
            f"This will:\n"
            f"1. Update train.py configuration\n"
            f"2. Start training in background\n"
            f"3. Save model to models/saved/\n\n"
            f"Training may take 10-60 minutes.\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Update train.py
                self.update_train_config(classes)
                
                # Start training
                import subprocess
                subprocess.Popen(['python', 'train.py'], cwd=os.getcwd())
                
                QMessageBox.information(
                    self,
                    "Training Started",
                    f"Training started in background!\n\n"
                    f"Training {len(classes)} classes with:\n"
                    f"• {total_photos} photos\n"
                    f"• {total_videos} videos\n\n"
                    f"Check terminal for progress.\n"
                    f"Model will be saved to models/saved/"
                )
                
                self.statusBar().showMessage(f"Training {len(classes)} classes...", 5000)
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Training Error",
                    f"Failed to start training:\n{str(e)}"
                )
    
    def update_train_config(self, class_names):
        """Update train.py configuration"""
        train_path = Path('train.py')
        if not train_path.exists():
            raise FileNotFoundError("train.py not found!")
        
        with open(train_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip().startswith('TRAIN_TYPE ='):
                new_lines.append('TRAIN_TYPE = "custom"\n')
            elif line.strip().startswith('CLASSES ='):
                new_lines.append(f'CLASSES = {class_names}\n')
            elif line.strip().startswith('DATA_PATH ='):
                new_lines.append('DATA_PATH = "data/collected_data"\n')
            else:
                new_lines.append(line)
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    
    def show_tts_settings(self):
        """Show TTS settings dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("TTS Settings")
        dialog.setModal(True)
        
        layout = QFormLayout()
        
        # Rate slider
        rate_slider = QSlider(Qt.Horizontal)
        rate_slider.setMinimum(50)
        rate_slider.setMaximum(300)
        rate_slider.setValue(self.config.get('tts', 'rate', default=150))
        rate_slider.valueChanged.connect(
            lambda v: self.inference_engine.tts_engine.set_rate(v)
        )
        layout.addRow("Speaking Rate:", rate_slider)
        
        # Volume slider
        volume_slider = QSlider(Qt.Horizontal)
        volume_slider.setMinimum(0)
        volume_slider.setMaximum(100)
        volume_slider.setValue(int(self.config.get('tts', 'volume', default=1.0) * 100))
        volume_slider.valueChanged.connect(
            lambda v: self.inference_engine.tts_engine.set_volume(v / 100.0)
        )
        layout.addRow("Volume:", volume_slider)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addRow(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About ASL Translation System",
            "<h2>American Sign Language Translation System</h2>"
            "<p><b>Version:</b> 2.0 Enhanced</p>"
            "<p><b>Description:</b> Advanced real-time ASL detection and translation "
            "with grammar correction and voice synthesis.</p>"
            "<p><b>Key Features:</b></p>"
            "<ul>"
            "<li>✓ Both Hands + Face Detection (186 features)</li>"
            "<li>✓ Full MediaPipe Face Mesh (478 landmarks)</li>"
            "<li>✓ Grammar Correction (100% accuracy on common patterns)</li>"
            "<li>✓ Manual Mode with Space Bar capture</li>"
            "<li>✓ Real-time Text-to-Speech</li>"
            "<li>✓ Professional news channel quality</li>"
            "</ul>"
            "<p><b>Grammar Examples:</b></p>"
            "<ul>"
            "<li>'who you' → 'Who are you?'</li>"
            "<li>'what name' → 'What is your name?'</li>"
            "<li>'i go home' → 'I am going home.'</li>"
            "</ul>"
            "<p><b>Technology:</b> MediaPipe 0.10.8, TensorFlow 2.15, PyQt5 5.15.11</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close"""
        self.stop_camera()
        self.inference_engine.shutdown()
        event.accept()


def main():
    """Main entry point"""
    import argparse
    from utils.config_loader import get_config
    
    # Load config to get the correct model path
    config = get_config('config.yaml')
    default_model_path = config['paths']['model_path']
    
    parser = argparse.ArgumentParser(description='ISL GUI Application')
    parser.add_argument('--model', type=str, default=default_model_path,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first: python src/train.py")
        sys.exit(1)
    
    # Create application.
    # Must be set BEFORE the QApplication exists: without High-DPI awareness
    # Qt reports a display scaled at 125% as its full unscaled pixel count
    # (e.g. 1920x1080 for a 1536x864 screen), so any window sized from that
    # geometry is rendered ~25% larger than the screen and gets clipped.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = ISLGUIApp(args.model, args.config)
    window.show()
    window.start_camera()  # Auto-start camera
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
