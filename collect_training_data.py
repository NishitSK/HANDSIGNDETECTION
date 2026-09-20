"""
Enhanced ISL Data Collection Tool
Features: Timer countdown, organized storage, training integration
"""
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pickle
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.landmark_extraction import LandmarkExtractor


class ISLDataCollector:
    """Professional data collection with timer and organized storage"""
    
    def __init__(self, output_dir='data/ISL_DATASETS_2'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize landmark extractor
        self.extractor = LandmarkExtractor(
            static_mode=False,
            max_hands=2,
            min_detection_confidence=0.5,
            detect_face=True
        )
        
        # MediaPipe for visualization
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Collection state
        self.current_class = ""
        self.capture_mode = 'photo'  # 'photo' or 'video'
        self.is_recording = False
        self.recorded_frames = []
        self.recorded_landmarks = []
        
        # Timer settings
        self.timer_seconds = 3
        self.countdown_active = False
        self.countdown_start = 0
        self.timer_type = None  # 'photo' or 'video'
        
        # Input state
        self.input_mode = False
        self.input_text = ""
        
        # Statistics
        self.load_session_stats()
        
        print("\n" + "="*80)
        print("ISL DATA COLLECTION - PROFESSIONAL TOOL")
        print("="*80)
        print("\n✨ Features:")
        print("  • 3-second countdown timer")
        print("  • Auto-capture after timer")
        print(f"  • Data saved to: {self.output_dir}")
        print("  • Both hands + face detection (186 features)")
        print("  • Ready for training!")
        print("="*80 + "\n")
    
    def load_session_stats(self):
        """Load existing session statistics"""
        self.session_data = {
            'classes': {},
            'session_start': datetime.now().isoformat(),
            'total_photos': 0,
            'total_videos': 0
        }
        
        # Load existing data
        for class_dir in self.output_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                photo_count = len(list(class_dir.glob('*.jpg')))
                video_count = len(list(class_dir.glob('*.avi')))
                
                if photo_count > 0 or video_count > 0:
                    self.session_data['classes'][class_name] = {
                        'photos': photo_count,
                        'videos': video_count
                    }
                    self.session_data['total_photos'] += photo_count
                    self.session_data['total_videos'] += video_count
    
    def get_class_stats(self, class_name):
        """Get statistics for a specific class"""
        if class_name in self.session_data['classes']:
            return self.session_data['classes'][class_name]
        return {'photos': 0, 'videos': 0}
    
    def update_stats(self, class_name, capture_type):
        """Update statistics after capture"""
        if class_name not in self.session_data['classes']:
            self.session_data['classes'][class_name] = {'photos': 0, 'videos': 0}
        
        if capture_type == 'photo':
            self.session_data['classes'][class_name]['photos'] += 1
            self.session_data['total_photos'] += 1
        else:
            self.session_data['classes'][class_name]['videos'] += 1
            self.session_data['total_videos'] += 1
    
    def draw_ui(self, frame):
        """Draw enhanced UI with timer"""
        h, w = frame.shape[:2]
        
        # Top bar gradient
        overlay = frame.copy()
        for i in range(180):
            alpha = 0.8 * (1 - i/180)
            cv2.rectangle(overlay, (0, i), (w, i+1), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "ISL DATA COLLECTION", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Mode indicator
        mode_text = f"Mode: {'PHOTO' if self.capture_mode == 'photo' else 'VIDEO'}"
        mode_color = (0, 255, 100) if self.capture_mode == 'photo' else (100, 100, 255)
        cv2.rectangle(frame, (w-250, 10), (w-10, 60), (30, 30, 30), -1)
        cv2.rectangle(frame, (w-250, 10), (w-10, 60), mode_color, 2)
        cv2.putText(frame, mode_text, (w-240, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2, cv2.LINE_AA)
        
        # Class input box
        input_y = 90
        input_h = 60
        
        cv2.rectangle(frame, (20, input_y), (w-20, input_y + input_h), (40, 40, 40), -1)
        
        if self.input_mode:
            border_color = (0, 255, 255) if int(time.time() * 2) % 2 else (0, 200, 200)
            cv2.rectangle(frame, (20, input_y), (w-20, input_y + input_h), border_color, 3)
            label = "Type class name (ENTER to confirm):"
        else:
            cv2.rectangle(frame, (20, input_y), (w-20, input_y + input_h), (100, 100, 100), 2)
            label = "Current Class (Press 'T' to edit):"
        
        cv2.putText(frame, label, (30, input_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        display_text = self.input_text if self.input_mode else (self.current_class or "Not Set")
        cursor = "|" if self.input_mode and int(time.time() * 2) % 2 else ""
        
        cv2.putText(frame, display_text + cursor, (35, input_y + 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Class statistics
        stats_y = input_y + input_h + 20
        if self.current_class:
            stats = self.get_class_stats(self.current_class)
            
            cv2.rectangle(frame, (20, stats_y), (w-20, stats_y + 80), (30, 30, 30), -1)
            cv2.rectangle(frame, (20, stats_y), (w-20, stats_y + 80), (0, 200, 255), 2)
            
            cv2.putText(frame, f"'{self.current_class}' Statistics:", (35, stats_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(frame, f"Photos: {stats['photos']}", (35, stats_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1, cv2.LINE_AA)
            
            cv2.putText(frame, f"Videos: {stats['videos']}", (250, stats_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1, cv2.LINE_AA)
        
        # Global statistics
        global_y = h - 250 if self.current_class else h - 190
        cv2.rectangle(frame, (20, global_y), (w-20, global_y + 100), (30, 30, 30), -1)
        cv2.rectangle(frame, (20, global_y), (w-20, global_y + 100), (255, 200, 0), 2)
        
        cv2.putText(frame, "TOTAL COLLECTED DATA:", (35, global_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"Classes: {len(self.session_data['classes'])}", (35, global_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Photos: {self.session_data['total_photos']}", (250, global_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Videos: {self.session_data['total_videos']}", (450, global_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1, cv2.LINE_AA)
        
        # COUNTDOWN TIMER - Large and centered
        if self.countdown_active:
            elapsed = time.time() - self.countdown_start
            remaining = max(0, self.timer_seconds - elapsed)
            
            if remaining > 0:
                # Countdown number
                countdown_num = int(remaining) + 1
                
                # Pulsing scale effect
                scale = 1.0 + (0.5 * (1 - (remaining % 1)))
                
                # Large number in center
                text = str(countdown_num)
                font_scale = 12 * scale
                thickness = 25
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_BOLD, font_scale, thickness)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                
                # Glow effect
                for offset in range(5, 0, -1):
                    alpha = 0.2 * (6 - offset)
                    glow_color = (0, int(255 * alpha), int(255 * alpha))
                    cv2.putText(frame, text, (text_x - offset, text_y - offset),
                               cv2.FONT_HERSHEY_BOLD, font_scale, glow_color, thickness + offset * 2, cv2.LINE_AA)
                
                # Main number
                cv2.putText(frame, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_BOLD, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
                
                # "GET READY!" message
                ready_text = "GET READY!"
                ready_size = cv2.getTextSize(ready_text, cv2.FONT_HERSHEY_DUPLEX, 2.5, 4)[0]
                ready_x = (w - ready_size[0]) // 2
                ready_y = text_y - 250
                
                cv2.putText(frame, ready_text, (ready_x, ready_y),
                           cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 255), 4, cv2.LINE_AA)
                
                # Countdown message
                action = "PHOTO" if self.timer_type == 'photo' else "VIDEO"
                action_text = f"{action} in {countdown_num}..."
                action_size = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                action_x = (w - action_size[0]) // 2
                action_y = text_y + 250
                
                cv2.putText(frame, action_text, (action_x, action_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3, cv2.LINE_AA)
        
        # Recording indicator
        if self.is_recording and int(time.time() * 2) % 2:
            cv2.circle(frame, (w - 100, 100), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 150, 110),
                       cv2.FONT_HERSHEY_BOLD, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Controls panel
        controls_y = h - 120
        cv2.rectangle(frame, (0, controls_y), (w, h), (20, 20, 20), -1)
        cv2.line(frame, (0, controls_y), (w, controls_y), (0, 255, 255), 2)
        
        controls = [
            "T: Edit Class",
            "SPACE: Start Timer (Photo)",
            "R: Start Timer (Video)",
            "M: Switch Mode",
            "H: Help",
            "Q: Quit"
        ]
        
        x_pos = 20
        for i, ctrl in enumerate(controls):
            y_offset = 30 + (i % 2) * 30
            cv2.putText(frame, ctrl, (x_pos, controls_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            x_pos = 20 if i % 2 == 1 else w // 2 + 20
    
    def draw_landmarks(self, frame):
        """Draw hand and face landmarks"""
        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.extractor.hands.process(image_rgb)
        face_results = self.extractor.face_mesh.process(image_rgb)
        
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = hand_results.multi_handedness[hand_idx].classification[0].label
                
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                wrist = hand_landmarks.landmark[0]
                label_x = int(wrist.x * w) - 40
                label_y = int(wrist.y * h) - 30
                color = (0, 255, 0) if handedness == "Left" else (255, 100, 0)
                
                cv2.rectangle(frame, (label_x - 5, label_y - 25), 
                            (label_x + 95, label_y + 5), (30, 30, 30), -1)
                cv2.rectangle(frame, (label_x - 5, label_y - 25), 
                            (label_x + 95, label_y + 5), color, 2)
                cv2.putText(frame, handedness, (label_x, label_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame, landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
    
    def start_countdown(self, timer_type):
        """Start countdown timer"""
        self.countdown_active = True
        self.countdown_start = time.time()
        self.timer_type = timer_type
        print(f"\n⏱️  {self.timer_seconds}-second countdown started for {timer_type.upper()}...")
    
    def check_countdown(self):
        """Check if countdown finished and trigger action"""
        if self.countdown_active:
            elapsed = time.time() - self.countdown_start
            if elapsed >= self.timer_seconds:
                self.countdown_active = False
                return True
        return False
    
    def capture_photo(self, frame, landmarks):
        """Capture photo after timer"""
        if not self.current_class:
            print("❌ Set class name first (press 'T')")
            return False
        
        if landmarks is None:
            print("❌ No landmarks detected! Please ensure hands are visible.")
            return False
        
        # Save to ISL_DATASETS_2/{class_name}/
        class_dir = self.output_dir / self.current_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
        
        # Save image
        img_path = class_dir / f"{timestamp}.jpg"
        cv2.imwrite(str(img_path), frame)
        
        # Save landmarks
        landmark_path = class_dir / f"{timestamp}_landmarks.npy"
        np.save(str(landmark_path), landmarks)
        
        # Update stats
        self.update_stats(self.current_class, 'photo')
        stats = self.get_class_stats(self.current_class)
        
        print(f"✅ Photo captured! '{self.current_class}' - Total: {stats['photos']} photos")
        return True
    
    def start_video_recording(self):
        """Start video recording after timer"""
        if not self.current_class:
            print("❌ Set class name first (press 'T')")
            return False
        
        self.is_recording = True
        self.recorded_frames = []
        self.recorded_landmarks = []
        print(f"🎥 Recording '{self.current_class}'... Press R to stop.")
        return True
    
    def stop_video_recording(self):
        """Stop and save video"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if len(self.recorded_frames) == 0:
            print("❌ No frames recorded")
            return
        
        # Save to ISL_DATASETS_2/{class_name}/
        class_dir = self.output_dir / self.current_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save video
        video_path = class_dir / f"{timestamp}.avi"
        h, w = self.recorded_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
        
        for frame in self.recorded_frames:
            out.write(frame)
        out.release()
        
        # Save landmarks
        landmark_path = class_dir / f"{timestamp}_landmarks.pkl"
        with open(landmark_path, 'wb') as f:
            pickle.dump(self.recorded_landmarks, f)
        
        # Update stats
        self.update_stats(self.current_class, 'video')
        stats = self.get_class_stats(self.current_class)
        
        print(f"✅ Video saved! '{self.current_class}' - Frames: {len(self.recorded_frames)}, Total: {stats['videos']} videos")
        
        self.recorded_frames = []
        self.recorded_landmarks = []
    
    def show_help(self):
        """Show help"""
        print("\n" + "="*80)
        print("KEYBOARD CONTROLS")
        print("="*80)
        print("\nT - Type class name")
        print("SPACE - Start 3-second timer for PHOTO capture")
        print("R - Start 3-second timer for VIDEO (press R again to stop recording)")
        print("M - Switch between Photo/Video mode")
        print("H - Show this help")
        print("Q - Quit and save session\n")
        print("📁 Data saved to:", self.output_dir)
        print("Recommended: 80 photos or 25 videos per class")
        print("="*80 + "\n")
    
    def run(self):
        """Main collection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("📹 Camera started. Press 'H' for help.\n")
        self.show_help()
        
        current_frame = None
        current_landmarks = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            landmarks = self.extractor.extract_from_image(frame)
            
            # Store current frame and landmarks
            current_frame = frame.copy()
            current_landmarks = landmarks.copy() if landmarks is not None else None
            
            if landmarks is not None:
                self.draw_landmarks(frame)
                
                if self.is_recording:
                    self.recorded_frames.append(frame.copy())
                    self.recorded_landmarks.append(landmarks.copy())
            
            # Check if countdown finished - AUTO CAPTURE!
            if self.check_countdown():
                if self.timer_type == 'photo':
                    self.capture_photo(current_frame, current_landmarks)
                elif self.timer_type == 'video':
                    self.start_video_recording()
            
            self.draw_ui(frame)
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if self.input_mode:
                if key == 13:  # ENTER
                    self.current_class = self.input_text
                    self.input_mode = False
                    self.input_text = ""
                    print(f"✅ Class set to: '{self.current_class}'")
                elif key == 27:  # ESC
                    self.input_mode = False
                    self.input_text = ""
                elif key == 8:  # BACKSPACE
                    self.input_text = self.input_text[:-1]
                elif 32 <= key <= 126:
                    self.input_text += chr(key)
            else:
                if key == ord('q') or key == ord('Q'):
                    if self.is_recording:
                        self.stop_video_recording()
                    break
                elif key == ord('t') or key == ord('T'):
                    self.input_mode = True
                    self.input_text = self.current_class
                    print("✏️  Type class name, press ENTER")
                elif key == ord('m') or key == ord('M'):
                    self.capture_mode = 'video' if self.capture_mode == 'photo' else 'photo'
                    print(f"🔄 Mode: {self.capture_mode.upper()}")
                elif key == 32:  # SPACE - Start timer for photo
                    if not self.countdown_active and not self.is_recording:
                        self.start_countdown('photo')
                elif key == ord('r') or key == ord('R'):
                    if not self.countdown_active:
                        if not self.is_recording:
                            # Start timer for video
                            self.start_countdown('video')
                        else:
                            # Stop recording
                            self.stop_video_recording()
                elif key == ord('h') or key == ord('H'):
                    self.show_help()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save session
        session_path = self.output_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(session_path, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        print(f"\n✅ Session saved: {session_path}")
        print(f"📊 Summary:")
        print(f"   Classes: {len(self.session_data['classes'])}")
        print(f"   Photos: {self.session_data['total_photos']}")
        print(f"   Videos: {self.session_data['total_videos']}")
        print(f"📁 Data location: {self.output_dir}\n")


if __name__ == "__main__":
    collector = ISLDataCollector()
    collector.run()
