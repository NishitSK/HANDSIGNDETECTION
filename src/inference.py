"""
Real-time ISL Inference Engine
Detects and translates sign language in real-time using webcam
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
from collections import deque
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.landmark_extraction import LandmarkExtractor
from src.grammar_correction import GrammarCorrector
from src.tts_engine import EnhancedTTS
from utils.config_loader import get_config
from utils.camera import open_camera
# Import custom model layers so they can be registered when loading saved models
from models.gesture_model import PositionalEncoding, TransformerBlock


class PredictionSmoother:
    """Moving-window majority vote smoother with confidence averaging"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = []

    def push(self, index, confidence):
        self.buffer.append((index, confidence))
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        if len(self.buffer) < 2:
            return index, confidence

        from collections import Counter
        counts = Counter(idx for idx, _ in self.buffer)
        winner_idx, count = counts.most_common(1)[0]
        # Require clear majority to eliminate jitter and fluctuating predictions
        if count < max(2, (len(self.buffer) // 2) + 1):
            return None, 0.0
        winning_confs = [c for idx, c in self.buffer if idx == winner_idx]
        avg_conf = float(np.mean(winning_confs))
        return winner_idx, avg_conf

    def clear(self):
        self.buffer.clear()


class KinematicVelocityTracker:
    """Tracks hand centroid displacement across recent frames to detect motion vs static signs"""
    def __init__(self, window_size=8, threshold=0.018):
        self.window_size = window_size
        self.threshold = threshold
        self.positions = deque(maxlen=window_size)
        self.last_velocity = 0.0

    def update(self, landmarks):
        """
        landmarks: (62, 3) or None
        Returns: is_motion (bool), velocity (float)
        """
        if landmarks is None:
            self.positions.clear()
            self.last_velocity = 0.0
            return False, 0.0

        hand_pts = landmarks[:42]
        active_pts = hand_pts[np.any(hand_pts != 0, axis=1)]
        if len(active_pts) == 0:
            self.positions.clear()
            self.last_velocity = 0.0
            return False, 0.0

        centroid = np.mean(active_pts[:, :2], axis=0)
        self.positions.append(centroid)

        if len(self.positions) < 3:
            return False, 0.0

        deltas = [np.linalg.norm(self.positions[i] - self.positions[i - 1]) for i in range(1, len(self.positions))]
        self.last_velocity = float(np.mean(deltas))
        is_motion = self.last_velocity >= self.threshold
        return is_motion, self.last_velocity

    def clear(self):
        self.positions.clear()
        self.last_velocity = 0.0


class ISLInference:
    """Real-time ISL detection and translation with Dual Ensemble (MLP + GRU)"""
    
    def __init__(self, model_path, config_path='config.yaml'):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        # Load model (register custom objects used by saved models)
        print("Loading model...")
        try:
            self.model = keras.models.load_model(model_path, custom_objects={
                'PositionalEncoding': PositionalEncoding,
                'TransformerBlock': TransformerBlock
            })
            print("[OK] Model loaded")
        except ValueError:
            # Fallback: try loading without custom_objects (will raise if unknown types exist)
            self.model = keras.models.load_model(model_path)
            print("[OK] Model loaded (fallback)")
        
        # Load metadata - try model-specific metadata first, then fallback
        model_path_obj = Path(model_path)
        metadata_path = model_path_obj.parent / 'model_metadata.json'
        if not metadata_path.exists():
            metadata_path = Path('models/saved/model_metadata.json')
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.class_names = self.metadata['class_names']
                self.num_classes = self.metadata['num_classes']
                # Determine sequence length from model input shape or metadata
                if 'input_shape' in self.metadata:
                    self.sequence_length = self.metadata['input_shape'][0]
                elif 'input_shape' in self.metadata.get('model_info', {}):
                    self.sequence_length = self.metadata['model_info']['input_shape'][0]
                else:
                    self.sequence_length = self.config.get('model', 'sequence_length', default=30)
        else:
            # Default classes
            self.class_names = [chr(i) for i in range(ord('A'), ord('Z')+1)]
            self.num_classes = len(self.class_names)
            self.sequence_length = self.config.get('model', 'sequence_length', default=30)

        # Dual Ensemble Models: MLP (Static Handsigns) & GRU (Motion Gestures)
        self.ensemble_mode = self.config.get('detection', 'mode', default='auto_ensemble')
        self.mlp_model = None
        self.gru_model = None
        
        # Load MLP model (specialized for static handsigns)
        mlp_path = Path(self.config.get('model', 'mlp_model_path', default='results/ISL_MLP/isl_model.h5'))
        if mlp_path.exists():
            try:
                self.mlp_model = keras.models.load_model(str(mlp_path), compile=False)
                print(f"[OK] Dual Ensemble: MLP static model loaded from {mlp_path}")
            except Exception as e:
                print(f"[WARN] Failed loading MLP model: {e}")
                
        # Load GRU model (specialized for motion gestures)
        gru_path = Path(self.config.get('model', 'gru_model_path', default='results/GRU/final/isl_model.h5'))
        if gru_path.exists():
            try:
                self.gru_model = keras.models.load_model(str(gru_path), compile=False)
                print(f"[OK] Dual Ensemble: GRU motion model loaded from {gru_path}")
            except Exception as e:
                print(f"[WARN] Failed loading GRU model: {e}")

        # Fallback to base model if either is missing
        if self.mlp_model is None:
            self.mlp_model = self.model
        if self.gru_model is None:
            self.gru_model = self.model

        # Motion / Velocity Tracker
        vel_thresh = self.config.get('detection', 'motion_velocity_threshold', default=0.018)
        self.velocity_tracker = KinematicVelocityTracker(window_size=8, threshold=vel_thresh)
        self.gru_sequence_buffer = deque(maxlen=30)
        self.active_engine = "MLP"
        self.is_motion = False
        
        # Initialize components
        detect_face = self.config.get('detection', 'detect_face', default=False)
        self.landmark_extractor = LandmarkExtractor(
            detect_face=detect_face,
            min_detection_confidence=self.config.get('detection', 'min_detection_confidence', default=0.5),
            min_tracking_confidence=self.config.get('detection', 'min_tracking_confidence', default=0.5)
        )
        self.grammar_corrector = GrammarCorrector()
        
        # Initialize TTS Engine with config settings
        self.tts_engine = EnhancedTTS(
            engine=self.config.get('tts', 'engine', default='pyttsx3'),
            rate=self.config.get('tts', 'rate', default=150),
            volume=self.config.get('tts', 'volume', default=1.0),
            voice_index=self.config.get('tts', 'voice_index', default=1)
        )
        
        # Prediction smoothing
        smoothing_window = self.config.get('detection', 'smoothing_window', default=5)
        self.smoother = PredictionSmoother(window_size=smoothing_window)
        self.prediction_buffer = deque(maxlen=smoothing_window)
        
        # Gesture buffering
        self.landmark_sequence = deque(maxlen=self.sequence_length)
        self.detected_words = []
        self.last_prediction = None
        self.prediction_confidence = 0.0
        self.gesture_hold_frames = self.config.get('detection', 'gesture_hold_frames', default=10)
        self.hold_counter = 0
        
        # Performance metrics
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        
        # Manual mode support
        self.manual_mode = True  # Default to manual mode for better accuracy
        
        print(f"[OK] Inference engine initialized")
        print(f"  Classes: {self.num_classes}")
        print(f"  Ensemble Mode: {self.ensemble_mode} (MLP + GRU)")
        print(f"  Mode: Manual (capture on demand)")
    
    def predict_gesture(self, candidates, is_motion=False):
        """
        Predict gesture using either MLP (static) or GRU (motion), or auto-ensemble.
        
        Args:
            candidates: list of (62, 3) arrays or single (62, 3) array
            is_motion: Boolean from KinematicVelocityTracker
        
        Returns:
            smoothed_pred_idx, smoothed_confidence
        """
        if candidates is None:
            self.smoother.clear()
            return None, 0.0
            
        if isinstance(candidates, np.ndarray) and candidates.ndim == 2:
            candidates = [candidates]
        elif not candidates:
            self.smoother.clear()
            return None, 0.0

        # Determine which engine to use based on mode & motion detection
        use_gru = False
        if self.ensemble_mode == 'gru_only':
            use_gru = True
        elif self.ensemble_mode == 'mlp_only':
            use_gru = False
        else: # auto_ensemble
            use_gru = is_motion

        self.active_engine = "GRU" if use_gru else "MLP"
        target_model = self.gru_model if use_gru else self.mlp_model
        
        if use_gru:
            # GRU expects shape (batch, 30, 186)
            primary_c = candidates[0]
            self.gru_sequence_buffer.append(primary_c.flatten())
            
            if len(self.gru_sequence_buffer) < 30:
                cur_list = list(self.gru_sequence_buffer)
                padded_seq = [cur_list[0]] * (30 - len(cur_list)) + cur_list
            else:
                padded_seq = list(self.gru_sequence_buffer)
                
            gru_input = np.array(padded_seq, dtype=np.float32).reshape(1, 30, -1)
            probs = target_model.predict(gru_input, verbose=0)[0]
            best_idx = int(np.argmax(probs))
            best_conf = float(probs[best_idx])
        else:
            # MLP expects shape (batch, 1, 126) - hands only (42 landmarks * 3 coords)
            mlp_batch = []
            for c in candidates:
                hands_126 = c[:42].flatten()
                mlp_batch.append(hands_126.reshape(1, 126))
            mlp_batch = np.array(mlp_batch, dtype=np.float32)
            
            probs_batch = target_model.predict(mlp_batch, verbose=0)
            best_idx = 0
            best_conf = -1.0
            for probs in probs_batch:
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                if conf > best_conf:
                    best_conf = conf
                    best_idx = idx

        # Smooth predictions
        smoothed_idx, smoothed_conf = self.smoother.push(best_idx, best_conf)
        return smoothed_idx, smoothed_conf
    
    def process_frame(self, frame, mirror_display=False):
        """
        Process a single frame
        
        Args:
            frame: Raw BGR image from camera (un-flipped)
            mirror_display: If True, returns a horizontally flipped display frame while
                           ensuring model inference runs on un-flipped raw landmarks.
        
        Returns:
            display_frame with overlays, prediction, confidence
        """
        start_time = time.time()
        
        # Extract candidates and display landmarks from RAW UN-FLIPPED frame
        candidates, landmarks = self.landmark_extractor.extract_candidates_from_image(frame)
        
        # Update kinematic velocity tracker
        self.is_motion, cur_velocity = self.velocity_tracker.update(landmarks)
        if not candidates:
            self.gru_sequence_buffer.clear()
            self.velocity_tracker.clear()
        
        # Face mesh visualization only if detect_face is enabled
        full_face_results = None
        if self.landmark_extractor.detect_face and self.landmark_extractor.face_mesh is not None:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            full_face_results = self.landmark_extractor.face_mesh.process(image_rgb)
        
        prediction_text = ""
        confidence = 0.0
        
        if candidates:
            pred_idx, conf = self.predict_gesture(candidates, is_motion=self.is_motion)
            min_thresh = self.config.get('detection', 'confidence_threshold', default=0.6)
            
            if pred_idx is not None and conf >= min_thresh:
                prediction_text = self.class_names[pred_idx]
                confidence = conf
                
                # Auto mode - add to sequence automatically
                # Manual mode - only show prediction, don't auto-add
                if not self.manual_mode:
                    # Gesture holding logic (auto mode only)
                    if prediction_text != self.last_prediction:
                        self.hold_counter = 0
                        self.last_prediction = prediction_text
                    else:
                        self.hold_counter += 1
                    
                    # Add to detected words if held long enough
                    if self.hold_counter == self.gesture_hold_frames:
                        if not self.detected_words or self.detected_words[-1] != prediction_text:
                            self.detected_words.append(prediction_text)
                            print(f"Detected: {prediction_text} ({confidence:.2f})")
                else:
                    # Manual mode - just update last prediction
                    self.last_prediction = prediction_text
        else:
            self.smoother.clear()
        
        # Prepare display frame
        if mirror_display:
            display_frame = cv2.flip(frame, 1)
        else:
            display_frame = frame.copy()

        # Draw landmarks with FULL face mesh
        if landmarks is not None and self.config.get('gui', 'show_landmarks', default=True):
            self._draw_landmarks_advanced(display_frame, landmarks, full_face_results, mirrored=mirror_display)
        
        # Calculate FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        self.fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
        
        # Draw overlays
        self._draw_overlays(display_frame, prediction_text, confidence)
        
        return display_frame, prediction_text, confidence
    
    def _draw_landmarks(self, frame, landmarks):
        """Draw both hands + face mesh with advanced visualization (backward compatibility)"""
        self._draw_landmarks_advanced(frame, landmarks, None)
    
    def _draw_landmarks_advanced(self, frame, landmarks, full_face_results=None, mirrored=False):
        """Draw both hands + FULL MediaPipe face mesh (468 points) like reference image"""
        h, w = frame.shape[:2]
        
        # Import MediaPipe drawing utilities
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        
        # Split combined landmarks: left_hand (21) + right_hand (21) + face (20)
        left_hand = landmarks[:21]
        right_hand = landmarks[21:42]
        face_points = landmarks[42:62]  # Only 20 key points used for prediction
        
        # Hand connections with specific colors
        HAND_CONNECTIONS = {
            'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
            'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
            'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
            'ring': [(0, 13), (13, 14), (14, 15), (15, 16)],
            'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)],
            'palm': [(5, 9), (9, 13), (13, 17)]
        }
        
        # Color schemes
        LEFT_HAND_COLORS = {
            'thumb': (0, 100, 255),    # Orange
            'index': (0, 255, 255),    # Yellow
            'middle': (0, 255, 100),   # Green
            'ring': (255, 200, 0),     # Cyan
            'pinky': (255, 100, 0),    # Blue
            'palm': (200, 200, 200)    # Light gray
        }
        
        RIGHT_HAND_COLORS = {
            'thumb': (0, 150, 255),    # Bright orange
            'index': (0, 255, 200),    # Lime
            'middle': (150, 255, 0),   # Green-yellow
            'ring': (255, 150, 100),   # Light cyan
            'pinky': (255, 0, 100),    # Magenta
            'palm': (255, 255, 255)    # White
        }
        
        # Draw left hand
        has_left = np.any(left_hand != 0)
        if has_left:
            self._draw_single_hand(frame, left_hand, HAND_CONNECTIONS, LEFT_HAND_COLORS, "LEFT", mirrored=mirrored)
        
        # Draw right hand
        has_right = np.any(right_hand != 0)
        if has_right:
            self._draw_single_hand(frame, right_hand, HAND_CONNECTIONS, RIGHT_HAND_COLORS, "RIGHT", mirrored=mirrored)
        
        # Draw FULL MediaPipe face mesh (468 landmarks) like reference image
        if full_face_results and full_face_results.multi_face_landmarks:
            for face_landmarks in full_face_results.multi_face_landmarks:
                # If frame was mirrored, flip face landmarks x-coordinates for drawing
                if mirrored:
                    for lm in face_landmarks.landmark:
                        lm.x = 1.0 - lm.x
                # Draw tesselation (the mesh grid - creates that dense pattern)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Draw contours (face outline, eyes, lips, eyebrows)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Draw irises (eye details)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )
                if mirrored:
                    # Restore original x coordinates
                    for lm in face_landmarks.landmark:
                        lm.x = 1.0 - lm.x
    
    def _draw_single_hand(self, frame, hand_landmarks, connections, colors, label, mirrored=False):
        """Draw a single hand with enhanced visualization"""
        h, w = frame.shape[:2]

        # Same 1280px design basis as _draw_overlays — scale marker sizes so
        # the skeleton doesn't smother a smaller frame.
        s = w / 1280.0

        def px(value):
            return max(1, int(round(value * s)))

        def map_x(x_norm):
            return (1.0 - x_norm) if mirrored else x_norm

        # Draw connections
        for finger, finger_connections in connections.items():
            color = colors[finger]
            for connection in finger_connections:
                start_idx, end_idx = connection
                start_point = (int(map_x(hand_landmarks[start_idx, 0]) * w),
                             int(hand_landmarks[start_idx, 1] * h))
                end_point = (int(map_x(hand_landmarks[end_idx, 0]) * w),
                           int(hand_landmarks[end_idx, 1] * h))

                # Outer glow
                cv2.line(frame, start_point, end_point, (255, 255, 255), px(6), cv2.LINE_AA)
                # Main line with thickness
                cv2.line(frame, start_point, end_point, color, px(4), cv2.LINE_AA)

        # Draw landmarks
        for idx, landmark in enumerate(hand_landmarks):
            x, y = int(map_x(landmark[0]) * w), int(landmark[1] * h)

            if idx == 0:
                # Wrist - large yellow circle
                cv2.circle(frame, (x, y), px(14), (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), px(16), (255, 255, 255), px(3), cv2.LINE_AA)
            elif idx in [4, 8, 12, 16, 20]:
                # Fingertips - bright green
                cv2.circle(frame, (x, y), px(12), (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), px(14), (255, 255, 255), px(3), cv2.LINE_AA)
            else:
                # Joints - cyan
                cv2.circle(frame, (x, y), px(8), (255, 200, 100), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), px(10), (255, 255, 255), px(2), cv2.LINE_AA)

        # Add hand label
        wrist = hand_landmarks[0]
        label_x = int(map_x(wrist[0]) * w) - px(40)
        label_y = int(wrist[1] * h) - px(30)

        # Label background
        cv2.rectangle(frame, (label_x - px(5), label_y - px(25)),
                     (label_x + px(95), label_y + px(5)), (30, 30, 30), -1)
        cv2.rectangle(frame, (label_x - px(5), label_y - px(25)),
                     (label_x + px(95), label_y + px(5)), colors['palm'], px(2))
        cv2.putText(frame, label, (label_x, label_y - px(5)),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7 * s, (255, 255, 255), px(2), cv2.LINE_AA)
    
    def _draw_overlays(self, frame, prediction, confidence):
        """Draw advanced text overlays and UI elements on frame"""
        h, w = frame.shape[:2]

        # All overlay geometry below was authored against a 1280px-wide frame.
        # Scale it to whatever we actually got, otherwise the GUI's 640x360
        # capture renders every element at double size and the chrome swallows
        # the video.
        s = w / 1280.0

        def px(value):
            """Scale a design pixel value, keeping it at least 1px."""
            return max(1, int(round(value * s)))

        top_bar_h = px(150)

        # Create modern UI overlay
        # Top bar with gradient
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Top bar border
        cv2.line(frame, (0, top_bar_h), (w, top_bar_h), (0, 255, 255), px(2))

        # System title with professional styling
        cv2.putText(frame, "ISL TRANSLATION SYSTEM", (px(20), px(35)),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0 * s, (255, 255, 255), px(2), cv2.LINE_AA)
        cv2.putText(frame, "Professional Edition", (px(20), px(60)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (100, 255, 255), px(1), cv2.LINE_AA)

        # FPS counter with icon
        fps_color = (0, 255, 0) if self.fps > 20 else (0, 165, 255) if self.fps > 10 else (0, 0, 255)
        cv2.rectangle(frame, (w - px(150), px(10)), (w - px(10), px(55)), (30, 30, 30), -1)
        cv2.rectangle(frame, (w - px(150), px(10)), (w - px(10), px(55)), fps_color, px(2))
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - px(140), px(40)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, fps_color, px(2), cv2.LINE_AA)

        # Engine indicator badge (Dual Ensemble)
        engine_label = f"ENGINE: {self.active_engine} {'(MOTION)' if self.active_engine == 'GRU' else '(STATIC)'}"
        engine_color = (0, 200, 255) if self.active_engine == 'GRU' else (0, 255, 150)
        badge_w = px(220)
        badge_x = w - px(160) - badge_w - px(10)
        cv2.rectangle(frame, (badge_x, px(10)), (badge_x + badge_w, px(55)), (30, 30, 30), -1)
        cv2.rectangle(frame, (badge_x, px(10)), (badge_x + badge_w, px(55)), engine_color, px(2))
        cv2.putText(frame, engine_label, (badge_x + px(10), px(35)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42 * s, engine_color, px(1), cv2.LINE_AA)
        cv2.putText(frame, f"Velocity: {self.velocity_tracker.last_velocity:.3f}", (badge_x + px(10), px(49)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * s, (180, 180, 180), px(1), cv2.LINE_AA)

        # Detection status panel
        if prediction:
            # Confidence bar
            bar_width = int(px(300) * confidence)
            bar_color = (0, 255, 0) if confidence > 0.9 else (0, 255, 255) if confidence > 0.7 else (0, 165, 255)

            # Panel background
            panel_y = px(80)
            panel_h = px(55)
            cv2.rectangle(frame, (px(15), panel_y), (px(615), panel_y + panel_h), (30, 30, 30), -1)
            cv2.rectangle(frame, (px(15), panel_y), (px(615), panel_y + panel_h), bar_color, px(3))

            # Detected sign text
            cv2.putText(frame, f"DETECTED: {prediction.upper()}", (px(25), panel_y + px(35)),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0 * s, (255, 255, 255), px(2), cv2.LINE_AA)

            # Confidence bar
            cv2.rectangle(frame, (px(25), panel_y + px(45)), (px(25) + px(300), panel_y + px(50)), (50, 50, 50), -1)
            cv2.rectangle(frame, (px(25), panel_y + px(45)), (px(25) + bar_width, panel_y + px(50)), bar_color, -1)
            cv2.putText(frame, f"{confidence:.1%}", (px(335), panel_y + px(50)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, bar_color, px(1), cv2.LINE_AA)

            # Status indicator (blinking for high confidence)
            if confidence > 0.9:
                import time
                if int(time.time() * 2) % 2:
                    cv2.circle(frame, (px(595), panel_y + px(28)), px(12), (0, 255, 0), -1)
                    cv2.circle(frame, (px(595), panel_y + px(28)), px(14), (255, 255, 255), px(2))

        # Bottom bar - Detected sequence
        if self.detected_words:
            bottom_bar_h = px(80)

            # Bottom panel
            bottom_overlay = frame.copy()
            cv2.rectangle(bottom_overlay, (0, h - bottom_bar_h), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(bottom_overlay, 0.7, frame, 0.3, 0, frame)

            cv2.line(frame, (0, h - bottom_bar_h), (w, h - bottom_bar_h), (0, 255, 255), px(2))

            # Sequence label
            cv2.putText(frame, "SEQUENCE:", (px(20), h - px(50)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (100, 255, 255), px(1), cv2.LINE_AA)

            # Words with boxes
            sequence_text = ' > '.join(self.detected_words[-8:])  # Last 8 words
            cv2.putText(frame, sequence_text, (px(150), h - px(50)),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7 * s, (255, 255, 255), px(2), cv2.LINE_AA)

            # Word count
            cv2.putText(frame, f"[{len(self.detected_words)} signs]", (w - px(150), h - px(50)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (150, 150, 150), px(1), cv2.LINE_AA)
    
    def translate_sequence(self):
        """Translate detected word sequence to speech"""
        if not self.detected_words:
            return ""
        
        # Convert to sentence
        sentence = self.grammar_corrector.process_sequence(self.detected_words)
        
        # Speak
        if sentence:
            print(f"\nTranslated: {sentence}")
            self.tts_engine.speak(sentence, async_mode=True)
        
        return sentence
    
    def clear_sequence(self):
        """Clear detected word sequence"""
        self.detected_words = []
        self.landmark_sequence.clear()
        self.prediction_buffer.clear()
        self.smoother.clear()
        self.last_prediction = None
        self.hold_counter = 0
    
    def run_camera(self, camera_id=0):
        """
        Run real-time detection from camera
        
        Args:
            camera_id: Camera device ID
        """
        # Initialize camera (tries DirectShow/MSMF/ANY — the default backend
        # silently fails to deliver frames on many Windows machines)
        cap = open_camera(
            camera_id,
            width=self.config.get('camera', 'width', default=1280),
            height=self.config.get('camera', 'height', default=720),
        )
        cap.set(cv2.CAP_PROP_FPS, self.config.get('camera', 'fps', default=30))
        
        print("\n" + "="*60)
        print("REAL-TIME ISL DETECTION")
        print("="*60)
        print("Controls:")
        print("  SPACE - Translate current sequence")
        print("  C - Clear sequence")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process raw frame (process_frame extracts un-flipped landmarks and mirrors display)
                display_frame, prediction, confidence = self.process_frame(frame, mirror_display=True)
                
                # Display
                cv2.imshow('ISL Real-time Detection', display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space - translate
                    self.translate_sequence()
                elif key == ord('c'):  # Clear
                    self.clear_sequence()
                    print("Sequence cleared")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmark_extractor.close()
            self.tts_engine.shutdown()
    
    def shutdown(self):
        """Cleanup resources"""
        self.landmark_extractor.close()
        self.tts_engine.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ISL Real-time Inference')
    parser.add_argument('--model', type=str, default='models/saved/isl_model.h5',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first: python src/train.py")
        sys.exit(1)
    
    # Run inference
    inference = ISLInference(args.model, args.config)
    inference.run_camera(args.camera)
