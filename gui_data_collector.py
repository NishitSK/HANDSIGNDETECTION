"""
Professional Data Collection GUI - Matching Main GUI Style
Same interface quality as Option 1
"""

import sys
import cv2
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.landmark_extraction import LandmarkExtractor


class VideoThread(QThread):
    """Thread for video capture"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, extractor, camera_id=0):
        super().__init__()
        self.extractor = extractor
        self.camera_id = camera_id
        self.running = True
    
    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Mirror frame
                frame = cv2.flip(frame, 1)

                # Emit the RAW frame — landmark overlay is drawn separately,
                # on a display-only copy, in update_image(). Emitting the
                # annotated frame here would make it the one saved to disk
                # by capture_photo(), baking the landmark dots/skeleton into
                # every training photo and corrupting hand re-detection when
                # landmarks are extracted from it later.
                self.change_pixmap_signal.emit(frame)
        
        cap.release()
    
    def stop(self):
        self.running = False
        self.wait()


class CountdownOverlay(QWidget):
    """Countdown overlay widget"""
    countdown_finished = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.countdown_value = 0
        self.is_active = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_countdown)
        
    def start_countdown(self, seconds=3):
        """Start countdown"""
        self.countdown_value = seconds
        self.is_active = True
        self.timer.start(1000)  # 1 second interval
        self.update()
        
    def update_countdown(self):
        """Update countdown"""
        self.countdown_value -= 1
        if self.countdown_value <= 0:
            self.timer.stop()
            self.is_active = False
            self.countdown_finished.emit()
        self.update()
    
    def paintEvent(self, event):
        """Draw countdown"""
        if self.is_active and self.countdown_value > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Semi-transparent background
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
            
            # Draw countdown number
            painter.setPen(QPen(QColor(255, 255, 0), 5))
            font = QFont('Arial', 120, QFont.Bold)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, str(self.countdown_value))


class DataCollectorGUI(QMainWindow):
    """Professional Data Collection GUI"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize landmark extractor
        self.extractor = LandmarkExtractor(static_mode=False, max_hands=2, detect_face=True)
        
        # Video thread
        self.video_thread = None
        
        # Data collection state
        self.current_class = ""
        self.photo_count = 0
        self.video_count = 0
        self.is_recording = False
        self.recorded_frames = []
        self.current_frame = None
        
        # Session tracking
        self.session_start = QDateTime.currentDateTime()
        self.total_photos_captured = 0
        self.total_videos_recorded = 0
        
        # Data directory
        self.data_dir = Path("data/ISL_DATASETS_2")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize UI
        self.init_ui()
        
        # Start camera automatically
        QTimer.singleShot(500, self.start_camera)
    
    def init_ui(self):
        """Initialize user interface - Match main GUI style"""
        self.setWindowTitle("ISL Data Collection System v2.0 Pro - [Professional Quality Training Data]")
        self.setGeometry(50, 50, 1600, 950)
        
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
        
        # Apply theme matching main GUI
        self.apply_theme()
        
        # Status bar
        self.statusBar().showMessage('Ready - Enter sign name and press Space to capture')
        self.statusBar().setStyleSheet("background-color: #1e1e1e; color: #00ff00; padding: 5px;")
        
        # Create menu bar
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        save_action = QAction('Save & Exit', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_and_exit)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        shortcuts_action = QAction('Keyboard Shortcuts', self)
        shortcuts_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcuts_action)
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_video_panel(self):
        """Create video feed panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Video container
        video_container = QWidget()
        video_layout = QVBoxLayout()
        video_container.setLayout(video_layout)
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 720)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #555;")
        video_layout.addWidget(self.video_label)
        
        # Countdown overlay
        self.countdown_overlay = CountdownOverlay(self.video_label)
        self.countdown_overlay.setGeometry(self.video_label.geometry())
        self.countdown_overlay.countdown_finished.connect(self.capture_photo)
        
        layout.addWidget(video_container)
        
        # Status indicators
        status_group = QGroupBox("Capture Status")
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("● Ready")
        self.status_label.setFont(QFont('Arial', 14, QFont.Bold))
        self.status_label.setStyleSheet("color: #00ff00;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.recording_indicator = QLabel()
        self.recording_indicator.setFont(QFont('Arial', 12, QFont.Bold))
        status_layout.addWidget(self.recording_indicator)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        return panel
    
    def create_control_panel(self):
        """Create control panel - Matching main GUI"""
        panel = QWidget()
        main_layout = QVBoxLayout()
        panel.setLayout(main_layout)
        
        # Scroll area for all controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_content.setLayout(scroll_layout)
        
        # 1. Class Input Section
        class_group = QGroupBox("📝 SIGN/WORD INPUT")
        class_layout = QVBoxLayout()
        
        label1 = QLabel("Enter sign or word to collect:")
        label1.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        class_layout.addWidget(label1)
        
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("e.g., hello, thank you, yes, no...")
        self.class_input.setFont(QFont('Arial', 12))
        self.class_input.setMinimumHeight(40)
        self.class_input.textChanged.connect(self.on_class_changed)
        class_layout.addWidget(self.class_input)
        
        self.class_status = QLabel("⚠️ Enter a sign name to begin")
        self.class_status.setStyleSheet("color: #ff9900; font-size: 10px; padding: 5px;")
        self.class_status.setWordWrap(True)
        class_layout.addWidget(self.class_status)
        
        class_group.setLayout(class_layout)
        scroll_layout.addWidget(class_group)
        
        # 2. Data Counters Section
        counters_group = QGroupBox("📊 COLLECTION PROGRESS")
        counters_layout = QVBoxLayout()
        
        # Photos counter
        photo_widget = QWidget()
        photo_widget.setMinimumHeight(80)
        photo_layout = QVBoxLayout()
        photo_layout.setContentsMargins(15, 10, 15, 10)
        
        photo_label = QLabel("📸 Photos Collected")
        photo_label.setFont(QFont('Arial', 10, QFont.Bold))
        photo_label.setStyleSheet("color: #00ddff;")
        photo_layout.addWidget(photo_label)
        
        self.photo_counter = QLabel("0")
        self.photo_counter.setFont(QFont('Arial', 32, QFont.Bold))
        self.photo_counter.setAlignment(Qt.AlignCenter)
        photo_layout.addWidget(self.photo_counter)
        
        photo_widget.setLayout(photo_layout)
        photo_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1a4d1a, stop:1 #0d260d);
            border: 2px solid #00ff00;
            border-radius: 8px;
        """)
        counters_layout.addWidget(photo_widget)
        
        # Videos counter
        video_widget = QWidget()
        video_widget.setMinimumHeight(80)
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(15, 10, 15, 10)
        
        video_label = QLabel("🎥 Videos Recorded")
        video_label.setFont(QFont('Arial', 10, QFont.Bold))
        video_label.setStyleSheet("color: #ff00ff;")
        video_layout.addWidget(video_label)
        
        self.video_counter = QLabel("0")
        self.video_counter.setFont(QFont('Arial', 32, QFont.Bold))
        self.video_counter.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_counter)
        
        video_widget.setLayout(video_layout)
        video_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #4d1a4d, stop:1 #260d26);
            border: 2px solid #ff00ff;
            border-radius: 8px;
        """)
        counters_layout.addWidget(video_widget)
        
        # Recommendation label
        self.recommendation_label = QLabel("💡 Recommended: 80-100 photos or 20-30 videos per sign")
        self.recommendation_label.setStyleSheet("color: #ffaa00; font-size: 10px; padding: 5px;")
        self.recommendation_label.setWordWrap(True)
        counters_layout.addWidget(self.recommendation_label)
        
        counters_group.setLayout(counters_layout)
        scroll_layout.addWidget(counters_group)
        
        # 3. Capture Controls Section
        capture_group = QGroupBox("📸 CAPTURE CONTROLS")
        capture_layout = QVBoxLayout()
        
        # Photo capture button
        self.photo_btn = QPushButton("📸 CAPTURE PHOTO (Space)")
        self.photo_btn.setFont(QFont('Arial', 12, QFont.Bold))
        self.photo_btn.setMinimumHeight(60)
        self.photo_btn.clicked.connect(self.start_photo_countdown)
        self.photo_btn.setEnabled(False)
        self.photo_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #00aa00, stop:1 #006600);
                color: white;
                border: 2px solid #00ff00;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #00cc00, stop:1 #008800);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #008800, stop:1 #004400);
            }
            QPushButton:disabled {
                background: #2a2a2a;
                color: #666666;
                border: 2px solid #444444;
            }
        """)
        capture_layout.addWidget(self.photo_btn)
        
        # Video capture button
        self.video_btn = QPushButton("🎥 START VIDEO (R)")
        self.video_btn.setFont(QFont('Arial', 12, QFont.Bold))
        self.video_btn.setMinimumHeight(60)
        self.video_btn.clicked.connect(self.toggle_video_recording)
        self.video_btn.setEnabled(False)
        self.video_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #cc0066, stop:1 #880044);
                color: white;
                border: 2px solid #ff00aa;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ee0088, stop:1 #aa0066);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #aa0055, stop:1 #660033);
            }
            QPushButton:disabled {
                background: #2a2a2a;
                color: #666666;
                border: 2px solid #444444;
            }
        """)
        capture_layout.addWidget(self.video_btn)
        
        capture_group.setLayout(capture_layout)
        scroll_layout.addWidget(capture_group)
        
        # 4. Session Info Section
        session_group = QGroupBox("📈 SESSION STATISTICS")
        session_layout = QVBoxLayout()
        
        self.session_info = QLabel()
        self.session_info.setFont(QFont('Arial', 10))
        self.session_info.setStyleSheet("color: #aaaaaa; padding: 5px;")
        self.session_info.setWordWrap(True)
        self.update_session_info()
        session_layout.addWidget(self.session_info)
        
        session_group.setLayout(session_layout)
        scroll_layout.addWidget(session_group)
        
        # 5. Actions Section
        scroll_layout.addStretch()
        
        actions_group = QGroupBox("⚡ ACTIONS")
        actions_layout = QVBoxLayout()
        
        save_btn = QPushButton("💾 Save & Exit")
        save_btn.setMinimumHeight(40)
        save_btn.clicked.connect(self.save_and_exit)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0088ee;
            }
        """)
        actions_layout.addWidget(save_btn)
        
        exit_btn = QPushButton("❌ Exit Without Saving")
        exit_btn.setMinimumHeight(40)
        exit_btn.clicked.connect(self.close)
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #cc3333;
                color: white;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #ee4444;
            }
        """)
        actions_layout.addWidget(exit_btn)
        
        actions_group.setLayout(actions_layout)
        scroll_layout.addWidget(actions_group)
        
        # Add scroll content to scroll area
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        
        return panel
    
    def apply_theme(self):
        """Apply modern dark theme matching main GUI"""
        self.setStyleSheet("""
            /* Main Window */
            QMainWindow {
                background-color: #1a1a1a;
            }
            
            /* Group Boxes */
            QGroupBox {
                color: #ffffff;
                background-color: #252525;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                font-size: 11px;
                padding: 15px 10px 10px 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                top: 3px;
                padding: 0 8px;
                background-color: #252525;
                color: #00ddff;
            }
            
            /* Line Edits */
            QLineEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 2px solid #0088ff;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 2px solid #00bbff;
                background-color: #252525;
            }
            
            /* Labels */
            QLabel {
                color: #e0e0e0;
                background: transparent;
            }
            
            /* Scroll Area */
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #505050;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #606060;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            /* Menu Bar */
            QMenuBar {
                background-color: #2a2a2a;
                color: white;
                border-bottom: 1px solid #404040;
            }
            QMenuBar::item {
                padding: 5px 12px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background-color: #0088ff;
            }
            
            /* Menu */
            QMenu {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #404040;
            }
            QMenu::item:selected {
                background-color: #0088ff;
            }
        """)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Space and not self.is_recording:
            # Space bar for photo capture
            self.start_photo_countdown()
        elif event.key() == Qt.Key_R:
            # R key for video recording
            self.toggle_video_recording()
        elif event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q:
            # Escape or Q to exit
            self.close()
    
    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        if hasattr(self, 'countdown_overlay'):
            self.countdown_overlay.setGeometry(self.video_label.geometry())
    
    def start_camera(self):
        """Start camera feed"""
        if self.video_thread is None:
            self.video_thread = VideoThread(self.extractor, camera_id=0)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.start()
            self.statusBar().showMessage('Camera started', 2000)
    
    def stop_camera(self):
        """Stop camera feed"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            self.statusBar().showMessage('Camera stopped', 2000)
    
    def update_image(self, frame):
        """Update video display"""
        # `frame` is the raw camera frame — keep it untouched as the capture
        # source, and draw the landmark overlay only on a separate display
        # copy so saved photos/videos never contain the overlay graphics.
        self.current_frame = frame.copy()

        if self.is_recording:
            # Store frame
            self.recorded_frames.append(self.current_frame.copy())

        display_frame = self.extractor.draw_landmarks_on_image(frame, None)

        # Add recording indicator if recording
        if self.is_recording:
            cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (60, 40), cv2.FONT_HERSHEY_BOLD, 1, (0, 0, 255), 2)

        # Convert to Qt format
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def on_class_changed(self):
        """Handle class name change"""
        self.current_class = self.class_input.text().strip()
        
        if self.current_class:
            # Update counters
            self.update_counters()
            
            # Enable buttons
            self.photo_btn.setEnabled(True)
            self.video_btn.setEnabled(True)
            
            # Update status
            self.class_status.setText(f"✅ Collecting data for: '{self.current_class}'")
            self.class_status.setStyleSheet("color: #00ff00; font-size: 10px; padding: 5px;")
            self.statusBar().showMessage(f"Ready to collect data for '{self.current_class}'")
        else:
            # Disable buttons
            self.photo_btn.setEnabled(False)
            self.video_btn.setEnabled(False)
            
            # Update status
            self.class_status.setText("⚠️ Enter a sign name to begin")
            self.class_status.setStyleSheet("color: #ff9900; font-size: 10px; padding: 5px;")
            self.statusBar().showMessage('Enter a sign name to begin')
    
    def update_counters(self):
        """Update photo/video counters"""
        if not self.current_class:
            return
        
        class_dir = self.data_dir / self.current_class
        
        # Count photos
        if class_dir.exists():
            photos = list(class_dir.glob("photo_*.jpg"))
            videos = list(class_dir.glob("video_*.avi"))
            self.photo_count = len(photos)
            self.video_count = len(videos)
        else:
            self.photo_count = 0
            self.video_count = 0
        
        # Update display
        self.photo_counter.setText(str(self.photo_count))
        self.video_counter.setText(str(self.video_count))
        
        # Update recommendation
        if self.photo_count >= 80:
            self.recommendation_label.setText(f"✅ Excellent! {self.photo_count} photos collected. You can move to next sign.")
            self.recommendation_label.setStyleSheet("color: #00ff00; font-size: 10px; padding: 5px;")
        elif self.photo_count >= 50:
            self.recommendation_label.setText(f"👍 Good progress! {self.photo_count}/80 photos. Keep going!")
            self.recommendation_label.setStyleSheet("color: #ffaa00; font-size: 10px; padding: 5px;")
        else:
            self.recommendation_label.setText(f"💡 {self.photo_count}/80 photos. Recommended: 80-100 photos per sign")
            self.recommendation_label.setStyleSheet("color: #ffaa00; font-size: 10px; padding: 5px;")
    
    def start_photo_countdown(self):
        """Start 3-second countdown for photo capture"""
        if not self.current_class or self.is_recording:
            return
        
        self.status_label.setText("● Countdown...")
        self.status_label.setStyleSheet("color: #ffaa00;")
        self.statusBar().showMessage('Get ready! 3... 2... 1...')
        
        # Start countdown overlay
        self.countdown_overlay.start_countdown(3)
    
    def capture_photo(self):
        """Capture single photo"""
        if self.current_frame is None or not self.current_class:
            return
        
        # Create class directory
        class_dir = self.data_dir / self.current_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = class_dir / f"photo_{timestamp}.jpg"
        
        # Save photo
        cv2.imwrite(str(filename), self.current_frame)
        
        # Update counters
        self.photo_count += 1
        self.total_photos_captured += 1
        self.photo_counter.setText(str(self.photo_count))
        
        # Update session info
        self.update_session_info()
        
        # Update recommendation
        self.update_counters()
        
        # Visual feedback
        self.status_label.setText("● Photo Captured!")
        self.status_label.setStyleSheet("color: #00ff00;")
        self.statusBar().showMessage(f'Photo captured! Total: {self.photo_count}', 2000)
        
        # Reset status after 1 second
        QTimer.singleShot(1000, lambda: self.status_label.setText("● Ready") if not self.is_recording else None)
        QTimer.singleShot(1000, lambda: self.status_label.setStyleSheet("color: #00ff00;"))
        
        print(f"📸 Photo saved: {filename}")
    
    def toggle_video_recording(self):
        """Toggle video recording"""
        if not self.current_class:
            return
        
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.recorded_frames = []
            
            self.video_btn.setText("⏹️ STOP VIDEO (R)")
            self.photo_btn.setEnabled(False)
            self.status_label.setText("● Recording...")
            self.status_label.setStyleSheet("color: #ff0000;")
            self.recording_indicator.setText("🔴 REC")
            self.recording_indicator.setStyleSheet("color: #ff0000;")
            self.statusBar().showMessage('Recording video... Press R to stop')
            
        else:
            # Stop recording
            self.is_recording = False
            
            # Save video
            if len(self.recorded_frames) > 0:
                self.save_video()
            
            self.video_btn.setText("🎥 START VIDEO (R)")
            self.photo_btn.setEnabled(True)
            self.status_label.setText("● Ready")
            self.status_label.setStyleSheet("color: #00ff00;")
            self.recording_indicator.setText("")
            self.statusBar().showMessage(f'Video saved! Total: {self.video_count}', 2000)
    
    def save_video(self):
        """Save recorded video"""
        if not self.recorded_frames or not self.current_class:
            return
        
        # Create class directory
        class_dir = self.data_dir / self.current_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = class_dir / f"video_{timestamp}.avi"
        
        # Get frame dimensions
        height, width, _ = self.recorded_frames[0].shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(filename), fourcc, 30.0, (width, height))
        
        # Write frames
        for frame in self.recorded_frames:
            out.write(frame)
        
        out.release()
        
        # Update counters
        self.video_count += 1
        self.total_videos_recorded += 1
        self.video_counter.setText(str(self.video_count))
        
        # Update session info
        self.update_session_info()
        
        print(f"🎥 Video saved: {filename} ({len(self.recorded_frames)} frames)")
        
        # Clear recorded frames
        self.recorded_frames = []
    
    def update_session_info(self):
        """Update session statistics"""
        duration = self.session_start.secsTo(QDateTime.currentDateTime())
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        
        info_text = f"""
        Session Duration: {hours}h {minutes}m
        Total Photos: {self.total_photos_captured}
        Total Videos: {self.total_videos_recorded}
        Current Sign: {self.current_class if self.current_class else 'None'}
        """
        
        self.session_info.setText(info_text)
    
    def show_shortcuts(self):
        """Show keyboard shortcuts dialog"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Keyboard Shortcuts")
        msg.setIcon(QMessageBox.Information)
        msg.setText("""
        <b>Keyboard Shortcuts:</b><br><br>
        <b>SPACE</b> - Capture photo (3-second countdown)<br>
        <b>R</b> - Start/Stop video recording<br>
        <b>Q / ESC</b> - Exit<br>
        <b>Ctrl+S</b> - Save & Exit<br>
        <b>Ctrl+Q</b> - Exit without saving<br>
        """)
        msg.exec_()
    
    def show_about(self):
        """Show about dialog"""
        msg = QMessageBox(self)
        msg.setWindowTitle("About")
        msg.setIcon(QMessageBox.Information)
        msg.setText("""
        <b>ISL Data Collection System v2.0 Pro</b><br><br>
        Professional quality training data collection<br>
        For Indian Sign Language Translation<br><br>
        Features:<br>
        • Real-time hand + face landmark visualization<br>
        • 3-second countdown timer<br>
        • Session tracking<br>
        • Professional GUI design<br>
        """)
        msg.exec_()
    
    def save_and_exit(self):
        """Save session summary and exit"""
        # Create session summary
        summary = {
            'session_date': self.session_start.toString(Qt.ISODate),
            'duration_minutes': self.session_start.secsTo(QDateTime.currentDateTime()) // 60,
            'total_photos': self.total_photos_captured,
            'total_videos': self.total_videos_recorded,
            'last_class': self.current_class
        }
        
        # Save summary
        summary_file = self.data_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"SESSION SUMMARY")
        print(f"{'='*50}")
        print(f"Photos captured: {self.total_photos_captured}")
        print(f"Videos recorded: {self.total_videos_recorded}")
        print(f"Summary saved: {summary_file}")
        print(f"{'='*50}\n")
        
        # Show summary dialog
        msg = QMessageBox(self)
        msg.setWindowTitle("Session Complete")
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"""
        <b>Session Summary:</b><br><br>
        Photos Captured: {self.total_photos_captured}<br>
        Videos Recorded: {self.total_videos_recorded}<br><br>
        Data saved to: {self.data_dir}<br>
        """)
        msg.exec_()
        
        self.close()
    
    def closeEvent(self, event):
        """Handle window close"""
        # Stop camera
        if self.video_thread:
            self.video_thread.stop()
        
        # Close extractor
        self.extractor.close()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = DataCollectorGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
