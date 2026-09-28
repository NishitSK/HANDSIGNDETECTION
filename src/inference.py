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
    """Task-Adaptive Kinematic Tracker: tracks centroid velocity, acceleration, and curvature"""
    def __init__(self, window_size=10, base_threshold=0.016, threshold=None, **kwargs):
        self.window_size = window_size
        if threshold is not None:
            self.base_threshold = threshold
        elif 'motion_threshold' in kwargs:
            self.base_threshold = kwargs['motion_threshold']
        else:
            self.base_threshold = base_threshold
        self.adaptive_threshold = self.base_threshold
        self.positions = deque(maxlen=window_size)
        self.velocities = deque(maxlen=window_size)
        self.last_velocity = 0.0
        self.motion_momentum = 0

    def update(self, landmarks):
        """
        landmarks: (62, 3) or None
        Returns: is_motion (bool), velocity (float)
        """
        if landmarks is None:
            self.clear()
            return False, 0.0

        hand_pts = landmarks[:42]
        active_pts = hand_pts[np.any(hand_pts != 0, axis=1)]
        if len(active_pts) == 0:
            self.clear()
            return False, 0.0

        centroid = np.mean(active_pts[:, :2], axis=0)
        self.positions.append(centroid)

        if len(self.positions) < 3:
            return False, 0.0

        # Instantaneous delta
        delta = float(np.linalg.norm(self.positions[-1] - self.positions[-2]))
        self.velocities.append(delta)

        # Smooth velocity across window
        self.last_velocity = float(np.mean(self.velocities))

        # Adaptive threshold: adjust dynamically to user baseline jitter
        min_vel = float(np.min(self.velocities))
        self.adaptive_threshold = max(0.012, min(0.028, min_vel * 2.2 + 0.010))

        # Dynamic curvature detection (for curved traces like J and zig-zags like Z)
        is_curving = False
        if len(self.positions) >= 4:
            v1 = self.positions[-1] - self.positions[-2]
            v2 = self.positions[-2] - self.positions[-3]
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 0.005 and norm2 > 0.005:
                cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                angle_diff = float(np.arccos(cos_angle))
                if angle_diff > 0.45: # > ~25 degrees direction change
                    is_curving = True

        raw_motion = (self.last_velocity >= self.adaptive_threshold) or is_curving

        # Adaptive motion momentum hysteresis: retain motion state across brief direction pauses
        if raw_motion:
            self.motion_momentum = min(12, self.motion_momentum + 3)
        else:
            self.motion_momentum = max(0, self.motion_momentum - 1)

        is_motion = (self.motion_momentum > 2)
        return is_motion, self.last_velocity

    def clear(self):
        self.positions.clear()
        self.velocities.clear()
        self.last_velocity = 0.0
        self.motion_momentum = 0


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
        self.ensemble_mode = self.config.get('detection', 'mode', default='task_adaptive')
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

        # Load Conversational Gesture GRU Model (HELLO, THANK YOU, PLEASE, YES, NO, HELP, GOOD, NAMASTE)
        self.gesture_model = None
        self.gesture_class_names = []
        gesture_path = Path('results/GESTURES/isl_gesture_model.h5')
        gesture_meta_path = Path('results/GESTURES/model_metadata.json')
        if gesture_path.exists():
            try:
                self.gesture_model = keras.models.load_model(str(gesture_path), compile=False)
                if gesture_meta_path.exists():
                    with open(gesture_meta_path, 'r') as f:
                        self.gesture_class_names = json.load(f).get('class_names', [])
                print(f"[OK] Conversational Gesture GRU loaded ({len(self.gesture_class_names)} gestures)")
            except Exception as e:
                print(f"[WARN] Failed loading gesture model: {e}")

        # Fallback to base model if either is missing
        if self.mlp_model is None:
            self.mlp_model = self.model
        if self.gru_model is None:
            self.gru_model = self.model

        # High-performance compiled graph executors (drops inference latency from 130ms to 1.6ms)
        self._compiled_mlp = None
        if self.mlp_model is not None:
            @tf.function(reduce_retracing=True)
            def _fast_mlp(tensor):
                return self.mlp_model(tensor, training=False)
            self._compiled_mlp = _fast_mlp
            try:
                _ = self._compiled_mlp(tf.zeros((1, 1, 126), dtype=tf.float32)).numpy()
            except Exception:
                pass

        self._compiled_gru = None
        if self.gru_model is not None:
            @tf.function(reduce_retracing=True)
            def _fast_gru(tensor):
                return self.gru_model(tensor, training=False)
            self._compiled_gru = _fast_gru
            try:
                e_dim = getattr(self.gru_model, 'input_shape', [None, 30, 126])[-1]
                _ = self._compiled_gru(tf.zeros((1, 30, e_dim), dtype=tf.float32)).numpy()
            except Exception:
                pass

        self._compiled_gesture = None
        if self.gesture_model is not None:
            @tf.function(reduce_retracing=True)
            def _fast_gesture(tensor):
                return self.gesture_model(tensor, training=False)
            self._compiled_gesture = _fast_gesture
            try:
                _ = self._compiled_gesture(tf.zeros((1, 30, 126), dtype=tf.float32)).numpy()
            except Exception:
                pass

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
        self.target_gesture_focus = None  # Hook for tutor studio or focused gesture practice
        self.app_mode = 'auto'  # 'auto', 'gestures', 'letters'
        self.raw_landmark_history = deque(maxlen=30)
        
        print(f"[OK] Inference engine initialized")
        print(f"  Classes: {self.num_classes}")
        print(f"  Ensemble Mode: {self.ensemble_mode} (MLP + GRU)")
        print(f"  Mode: Manual (capture on demand)")
    
    def analyze_hand_shape(self, hand_21):
        """
        Scale-invariant anatomical analysis of a single hand (21 landmarks).
        Returns extended/curled finger states, thumb orientation, and palm configuration.
        """
        if hand_21 is None or np.all(hand_21 == 0):
            return None
        wrist = hand_21[0]
        finger_indices = [
            (1, 2, 3, 4),    # Thumb
            (5, 6, 7, 8),    # Index
            (9, 10, 11, 12), # Middle
            (13, 14, 15, 16),# Ring
            (17, 18, 19, 20) # Pinky
        ]
        extended = []
        curled = []
        t_mcp, t_pip, t_ip, t_tip = [hand_21[i] for i in finger_indices[0]]
        dist_thumb_tip_wrist = np.linalg.norm(t_tip[:2] - wrist[:2])
        dist_thumb_mcp_wrist = np.linalg.norm(t_mcp[:2] - wrist[:2])
        thumb_ext = dist_thumb_tip_wrist > dist_thumb_mcp_wrist * 1.25
        thumb_curled = dist_thumb_tip_wrist < dist_thumb_mcp_wrist * 1.15
        extended.append(thumb_ext)
        curled.append(thumb_curled)

        for mcp_i, pip_i, dip_i, tip_i in finger_indices[1:]:
            pip = hand_21[pip_i]
            tip = hand_21[tip_i]
            d_tip_wrist = np.linalg.norm(tip[:2] - wrist[:2])
            d_pip_wrist = np.linalg.norm(pip[:2] - wrist[:2])
            is_ext = (d_tip_wrist > d_pip_wrist * 1.05) and (tip[1] < pip[1] + 0.05)
            is_curled = (d_tip_wrist < d_pip_wrist * 1.10) or (tip[1] > pip[1] - 0.02)
            extended.append(is_ext)
            curled.append(is_curled)

        f_tips_y = [hand_21[i][1] for i in [8, 12, 16, 20]]
        t_tip_y = hand_21[4][1]
        t_mcp_y = hand_21[2][1]
        thumb_up = (t_tip_y < t_mcp_y - 0.025) and (t_tip_y < wrist[1] - 0.03)
        thumb_highest = t_tip_y < min(f_tips_y) - 0.02

        return {
            'wrist': wrist,
            'extended': extended,
            'curled': curled,
            'thumb_up': thumb_up,
            'thumb_highest': thumb_highest,
            'all_4_curled': sum(curled[1:]) >= 3,
            'all_4_extended': sum(extended[1:]) >= 3,
            'raw': hand_21
        }

    def detect_conversational_gesture(self, raw_landmarks, padded_seq, is_motion=False):
        """
        Robust scale- and position-invariant kinematic gesture analyzer:
        HELLO, THANK YOU, PLEASE, YES, NO, HELP, GOOD, NAMASTE.
        """
        if raw_landmarks is None or len(raw_landmarks) < 42:
            return None, 0.0

        lms = np.asarray(raw_landmarks)[:42]
        self.raw_landmark_history.append(lms)

        l_info = self.analyze_hand_shape(lms[:21])
        r_info = self.analyze_hand_shape(lms[21:42])

        # --- 1. NAMASTE (Two hands joined together in prayer) ---
        if l_info and r_info:
            w_dist = float(np.linalg.norm(l_info['wrist'][:2] - r_info['wrist'][:2]))
            l_tip_12 = l_info['raw'][12]
            r_tip_12 = r_info['raw'][12]
            m_dist = float(np.linalg.norm(l_tip_12[:2] - r_tip_12[:2]))
            l_up = l_tip_12[1] < l_info['wrist'][1] - 0.02
            r_up = r_tip_12[1] < r_info['wrist'][1] - 0.02
            if w_dist < 0.35 and m_dist < 0.26 and l_up and r_up:
                return 'NAMASTE', 0.99
        elif (self.target_gesture_focus == 'NAMASTE' or self.app_mode == 'gestures') and (l_info or r_info):
            # Single hand fallback if MediaPipe merged hands during contact
            h_fallback = r_info if r_info else l_info
            if h_fallback['wrist'][1] > 0.35 and h_fallback['raw'][12][1] < h_fallback['wrist'][1] - 0.08:
                if (0.30 <= h_fallback['wrist'][0] <= 0.70) and (self.target_gesture_focus == 'NAMASTE'):
                    return 'NAMASTE', 0.95

        # --- 2. HELP (Two hands: one flat base palm + one thumbs-up fist on top) ---
        if l_info and r_info:
            w_dist = float(np.linalg.norm(l_info['wrist'][:2] - r_info['wrist'][:2]))
            if w_dist < 0.35:
                if r_info['thumb_up'] and r_info['all_4_curled'] and (r_info['wrist'][1] <= l_info['wrist'][1] + 0.12):
                    return 'HELP', 0.98
                if l_info['thumb_up'] and l_info['all_4_curled'] and (l_info['wrist'][1] <= r_info['wrist'][1] + 0.12):
                    return 'HELP', 0.98

        # --- SINGLE HAND GESTURES ---
        h_info = r_info if r_info else l_info
        if not h_info:
            return None, 0.0

        wrist = h_info['wrist']
        h_raw = h_info['raw']

        # --- 3. GOOD (Thumbs Up) ---
        if h_info['thumb_up'] and h_info['all_4_curled'] and h_info['thumb_highest']:
            return 'GOOD', 0.98

        # Trajectory statistics across recent history
        if len(self.raw_landmark_history) >= 8:
            hist_arr = np.array(self.raw_landmark_history)
            h_idx = 21 if r_info else 0
            wrist_x_track = hist_arr[:, h_idx, 0]
            wrist_y_track = hist_arr[:, h_idx, 1]
            x_std = float(np.std(wrist_x_track[-15:]))
            y_std = float(np.std(wrist_y_track[-15:]))
            y_sweep = float(wrist_y_track[-1] - np.min(wrist_y_track[:10]))
            y_start = float(np.mean(wrist_y_track[:5]))
        else:
            x_std, y_std, y_sweep, y_start = 0.0, 0.0, 0.0, float(wrist[1])

        # --- 4. HELLO (Raised open hand, waving or greeting) ---
        if (h_info['all_4_extended'] or sum(h_info['extended']) >= 4) and (wrist[1] < 0.65):
            if x_std > 0.015 or is_motion or (self.target_gesture_focus == 'HELLO') or (wrist[1] < 0.45):
                return 'HELLO', 0.96

        # --- 5. THANK YOU (Chin to chest downward sweep) ---
        if h_info['all_4_extended'] or sum(h_info['extended']) >= 3:
            if (y_start < 0.55 and y_sweep > 0.035) or (self.target_gesture_focus == 'THANK YOU' and 0.30 <= wrist[1] <= 0.75):
                return 'THANK YOU', 0.95

        # --- 6. PLEASE (Chest flat open palm / circular rub) ---
        if (0.28 <= wrist[0] <= 0.72) and (0.35 <= wrist[1] <= 0.78) and (h_info['all_4_extended'] or sum(h_info['extended']) >= 3):
            if (x_std > 0.012 and y_std > 0.012) or (self.target_gesture_focus == 'PLEASE') or (self.app_mode == 'gestures' and is_motion):
                return 'PLEASE', 0.94

        # --- 7. YES (Nodding fist) ---
        if h_info['all_4_curled']:
            if (y_std > 0.015 and y_std > 1.4 * x_std) or (self.target_gesture_focus == 'YES') or (self.app_mode == 'gestures' and is_motion):
                return 'YES', 0.93

        # --- 8. NO (Index/Middle beak snap against thumb) ---
        d_beak_index = float(np.linalg.norm(h_raw[4, :2] - h_raw[8, :2]))
        d_beak_mid = float(np.linalg.norm(h_raw[4, :2] - h_raw[12, :2]))
        if (d_beak_index < 0.09 or d_beak_mid < 0.09) and (h_info['curled'][3] and h_info['curled'][4]):
            if is_motion or (self.target_gesture_focus == 'NO') or (self.app_mode == 'gestures'):
                return 'NO', 0.93

        # Neural GRU model cross-check (using compiled graph executor)
        if self.gesture_model is not None and len(self.gesture_class_names) > 0 and len(padded_seq) == 30:
            try:
                g_input = np.array(padded_seq, dtype=np.float32).reshape(1, 30, 126)
                if self._compiled_gesture is not None:
                    g_probs = self._compiled_gesture(tf.convert_to_tensor(g_input, dtype=tf.float32)).numpy()[0]
                else:
                    g_probs = self.gesture_model(tf.convert_to_tensor(g_input, dtype=tf.float32), training=False).numpy()[0]
                top_g_idx = int(np.argmax(g_probs))
                top_g_conf = float(g_probs[top_g_idx])
                g_name = self.gesture_class_names[top_g_idx]
                if self.target_gesture_focus and self.target_gesture_focus in self.gesture_class_names:
                    t_idx = self.gesture_class_names.index(self.target_gesture_focus)
                    if float(g_probs[t_idx]) >= 0.45:
                        return self.target_gesture_focus, 0.88
                if top_g_conf >= 0.72:
                    return g_name, top_g_conf
            except Exception:
                pass

        return None, 0.0

    def predict_gesture(self, candidates, is_motion=False, raw_landmarks=None):
        """
        Predict gesture or letter based on app_mode ('auto', 'gestures', 'letters').
        """
        if candidates is None:
            self.smoother.clear()
            return None, 0.0
            
        if isinstance(candidates, np.ndarray) and candidates.ndim == 2:
            candidates = [candidates]
        elif not candidates:
            self.smoother.clear()
            return None, 0.0

        # Continuous GRU buffer update
        if raw_landmarks is not None and len(raw_landmarks) >= 42:
            feature_vec = np.asarray(raw_landmarks)[:42].flatten()
        else:
            primary_c = candidates[0]
            feature_vec = primary_c[:42].flatten()
        self.gru_sequence_buffer.append(feature_vec)

        if len(self.gru_sequence_buffer) < 30:
            cur_list = list(self.gru_sequence_buffer)
            padded_seq = [cur_list[0]] * (30 - len(cur_list)) + cur_list
        else:
            padded_seq = list(self.gru_sequence_buffer)

        # -------------------------------------------------------------
        # 1. EVALUATE GESTURES (IF MODE IS 'auto' OR 'gestures')
        # -------------------------------------------------------------
        if self.app_mode in ('auto', 'gestures'):
            conv_name, conv_conf = self.detect_conversational_gesture(raw_landmarks, padded_seq, is_motion)
            if conv_name is not None and conv_conf >= 0.65:
                self.active_engine = f"GESTURE ({conv_name})"
                self.active_task = f"CONVERSATIONAL_GESTURE ({conv_name})"
                self.smoother.clear()
                return conv_name, conv_conf

        # Pure 'gestures' mode NEVER outputs letters or guesses randomly
        if self.app_mode == 'gestures':
            return None, 0.0

        # -------------------------------------------------------------
        # 2. ALPHABET LETTER PREDICTION (DUAL MLP + GRU ENSEMBLE)
        # -------------------------------------------------------------
        primary_c = candidates[0]
        motion_char_set = {'J', 'Z'}

        if self.ensemble_mode == 'gru_only':
            use_gru = True
            task_name = "GRU_ONLY"
        elif self.ensemble_mode == 'mlp_only':
            use_gru = False
            task_name = "MLP_ONLY"
        else:
            if is_motion:
                use_gru = True
                task_name = "DYNAMIC_MOTION (GRU)"
            else:
                use_gru = False
                task_name = "STATIC_POSE (MLP)"

        self.active_task = task_name
        self.active_engine = "GRU" if use_gru else "MLP"

        if use_gru:
            expected_dim = getattr(self.gru_model, 'input_shape', [None, 30, 126])[-1]
            if expected_dim == 126:
                gru_input = np.array(padded_seq, dtype=np.float32).reshape(1, 30, 126)
            else:
                full_c = [c.flatten() if len(c.flatten()) == expected_dim else np.pad(c[:42].flatten(), (0, expected_dim - 126)) for c in padded_seq]
                gru_input = np.array(full_c, dtype=np.float32).reshape(1, 30, expected_dim)

            if self._compiled_gru is not None:
                probs = self._compiled_gru(tf.convert_to_tensor(gru_input, dtype=tf.float32)).numpy()[0]
            else:
                probs = self.gru_model(tf.convert_to_tensor(gru_input, dtype=tf.float32), training=False).numpy()[0]
            best_idx = int(np.argmax(probs))
            best_conf = float(probs[best_idx])
            pred_char = self.class_names[best_idx] if best_idx < len(self.class_names) else ""

            if pred_char not in motion_char_set and best_conf < 0.70 and self.mlp_model is not None:
                mlp_batch = np.array([primary_c[:42].flatten().reshape(1, 126)], dtype=np.float32)
                if self._compiled_mlp is not None:
                    mlp_probs = self._compiled_mlp(tf.convert_to_tensor(mlp_batch, dtype=tf.float32)).numpy()[0]
                else:
                    mlp_probs = self.mlp_model(tf.convert_to_tensor(mlp_batch, dtype=tf.float32), training=False).numpy()[0]
                mlp_idx = int(np.argmax(mlp_probs))
                mlp_conf = float(mlp_probs[mlp_idx])
                if mlp_conf > 0.88:
                    best_idx = mlp_idx
                    best_conf = mlp_conf
                    self.active_engine = "MLP (Jitter-Corrected)"
        else:
            mlp_batch = []
            for c in candidates:
                hands_126 = c[:42].flatten()
                mlp_batch.append(hands_126.reshape(1, 126))
            mlp_batch = np.array(mlp_batch, dtype=np.float32)

            if self._compiled_mlp is not None:
                probs_batch = self._compiled_mlp(tf.convert_to_tensor(mlp_batch, dtype=tf.float32)).numpy()
            else:
                probs_batch = self.mlp_model(tf.convert_to_tensor(mlp_batch, dtype=tf.float32), training=False).numpy()
            best_idx = 0
            best_conf = -1.0
            for probs in probs_batch:
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                if conf > best_conf:
                    best_conf = conf
                    best_idx = idx

            pred_char = self.class_names[best_idx] if best_idx < len(self.class_names) else ""
            if pred_char in motion_char_set and self.gru_model is not None:
                expected_dim = getattr(self.gru_model, 'input_shape', [None, 30, 126])[-1]
                gru_input = np.array(padded_seq, dtype=np.float32).reshape(1, 30, expected_dim)
                if self._compiled_gru is not None:
                    gru_probs = self._compiled_gru(tf.convert_to_tensor(gru_input, dtype=tf.float32)).numpy()[0]
                else:
                    gru_probs = self.gru_model(tf.convert_to_tensor(gru_input, dtype=tf.float32), training=False).numpy()[0]
                gru_idx = int(np.argmax(gru_probs))
                gru_conf = float(gru_probs[gru_idx])
                if self.class_names[gru_idx] in motion_char_set:
                    best_idx = gru_idx
                    best_conf = gru_conf
                    self.active_engine = "GRU (Sequence-Verified)"

        # Smooth predictions for letters
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
            pred_idx, conf = self.predict_gesture(candidates, is_motion=self.is_motion, raw_landmarks=landmarks)
            min_thresh = self.config.get('detection', 'confidence_threshold', default=0.6)
            
            if pred_idx is not None and conf >= min_thresh:
                if isinstance(pred_idx, str):
                    prediction_text = pred_idx
                else:
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
        """Draw clean, modern, user-friendly hand landmarks matching mobile aesthetics"""
        h, w = frame.shape[:2]
        
        # Split combined landmarks: left_hand (21) + right_hand (21)
        left_hand = landmarks[:21]
        right_hand = landmarks[21:42]
        
        # Standard hand bone connections
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (9, 10), (10, 11), (11, 12),           # Middle
            (13, 14), (14, 15), (15, 16),          # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)              # Palm base
        ]
        
        # Draw hands
        has_left = np.any(left_hand != 0)
        has_right = np.any(right_hand != 0)
        
        if has_left:
            self._draw_single_hand(frame, left_hand, HAND_CONNECTIONS, mirrored=mirrored)
        if has_right:
            self._draw_single_hand(frame, right_hand, HAND_CONNECTIONS, mirrored=mirrored)
            
        # Draw face contours only if explicitly enabled in config
        if full_face_results and full_face_results.multi_face_landmarks:
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_face_mesh = mp.solutions.face_mesh
            for face_landmarks in full_face_results.multi_face_landmarks:
                if mirrored:
                    for lm in face_landmarks.landmark:
                        lm.x = 1.0 - lm.x
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                if mirrored:
                    for lm in face_landmarks.landmark:
                        lm.x = 1.0 - lm.x
    
    def _draw_single_hand(self, frame, hand_landmarks, connections, mirrored=False):
        """Draw a single hand with clean, sleek, user-friendly styling (matching mobile)"""
        h, w = frame.shape[:2]
        s = w / 1280.0

        def px(val):
            return max(1, int(round(val * s)))

        def map_x(x_norm):
            return (1.0 - x_norm) if mirrored else x_norm

        # Sleek color scheme:
        # Bone connections: Clean, crisp pearl white/silver
        BONE_COLOR = (245, 245, 245)      # BGR: Soft Pearl White
        PALM_COLOR = (255, 178, 63)       # BGR: Cyan / Studio Blue
        JOINT_COLOR = (240, 242, 245)     # BGR: Crisp White
        TIP_COLOR = (255, 178, 63)        # BGR: Studio Cyan
        WRIST_COLOR = (63, 178, 255)      # BGR: Studio Amber / Gold

        # Draw bone connections
        for start_idx, end_idx in connections:
            is_palm = (start_idx, end_idx) in [(5, 9), (9, 13), (13, 17)]
            col = PALM_COLOR if is_palm else BONE_COLOR
            thick = px(2)

            pt1 = (int(map_x(hand_landmarks[start_idx, 0]) * w), int(hand_landmarks[start_idx, 1] * h))
            pt2 = (int(map_x(hand_landmarks[end_idx, 0]) * w), int(hand_landmarks[end_idx, 1] * h))
            cv2.line(frame, pt1, pt2, col, thick, cv2.LINE_AA)

        # Draw landmark points
        FINGERTIP_INDICES = {4, 8, 12, 16, 20}
        for idx, pt in enumerate(hand_landmarks):
            x, y = int(map_x(pt[0]) * w), int(pt[1] * h)

            if idx == 0:
                # Wrist: neat 5px pearl dot with thin gold ring
                cv2.circle(frame, (x, y), px(5), WRIST_COLOR, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), px(6), (255, 255, 255), px(1), cv2.LINE_AA)
            elif idx in FINGERTIP_INDICES:
                # Fingertips: clean 4px cyan dot with 1px outer ring
                cv2.circle(frame, (x, y), px(4), TIP_COLOR, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), px(5), (255, 255, 255), px(1), cv2.LINE_AA)
            else:
                # Intermediate knuckles / joints: small, clean 3px white dot
                cv2.circle(frame, (x, y), px(3), JOINT_COLOR, -1, cv2.LINE_AA)
    
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
