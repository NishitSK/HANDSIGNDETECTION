"""
Hand Landmark Extraction using MediaPipe
Extracts hand landmarks from images/videos for training data preparation
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import pickle
from tqdm import tqdm
import json


class LandmarkExtractor:
    """Extract hand and face landmarks using MediaPipe"""
    
    def __init__(self, static_mode=False, max_hands=2, 
                 min_detection_confidence=0.7, min_tracking_confidence=0.5,
                 detect_face=True):
        """
        Initialize MediaPipe hands and face mesh
        
        Args:
            static_mode: If True, treats each image independently
            max_hands: Maximum number of hands to detect (2 for both hands)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            detect_face: Whether to detect face landmarks
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize face mesh if needed
        self.detect_face = detect_face
        if detect_face:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_mode,
                max_num_faces=1,
                refine_landmarks=True,  # Enable iris landmarks for full 478 points
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
    
    def extract_from_image(self, image):
        """
        Extract landmarks from a single image (both hands + face)
        
        Args:
            image: BGR image from OpenCV
        
        Returns:
            landmarks: numpy array of combined landmarks or None if no detection
                - Left hand: 21 landmarks (63 features)
                - Right hand: 21 landmarks (63 features)  
                - Face: 20 key landmarks (60 features)
                - Total: 186 features (if both hands + face detected)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hand_results = self.hands.process(image_rgb)
        
        # Initialize landmark arrays with zeros
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))
        face_landmarks = np.zeros((20, 3))
        
        # Extract hand landmarks
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks_data, handedness in zip(
                hand_results.multi_hand_landmarks, 
                hand_results.multi_handedness
            ):
                # Determine if left or right hand
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                
                # Extract coordinates
                landmarks = []
                for landmark in hand_landmarks_data.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                landmarks_array = np.array(landmarks)
                
                if hand_label == "Left":
                    left_hand = landmarks_array
                else:  # Right
                    right_hand = landmarks_array
        
        # Extract face landmarks if enabled
        if self.detect_face:
            face_results = self.face_mesh.process(image_rgb)
            
            if face_results.multi_face_landmarks:
                face_data = face_results.multi_face_landmarks[0]
                
                # Extract key face landmarks (forehead, nose, chin, cheeks, etc.)
                # Using indices: 10, 152, 234, 454, 4, 1, 33, 263, 61, 291, 199, 6, 168, 8, 9, 151, 337, 299, 69, 104
                key_indices = [10, 152, 234, 454, 4, 1, 33, 263, 61, 291, 199, 6, 168, 8, 9, 151, 337, 299, 69, 104]
                
                face_points = []
                for idx in key_indices:
                    landmark = face_data.landmark[idx]
                    face_points.append([landmark.x, landmark.y, landmark.z])
                
                face_landmarks = np.array(face_points)

        # No hand in frame: nothing to recognize, regardless of face detection
        if not np.any(left_hand) and not np.any(right_hand):
            return None

        # Combine all landmarks: left_hand + right_hand + face
        # Shape: (62, 3) = 186 features
        combined_landmarks = np.vstack([
            left_hand,      # 21 x 3 = 63 features
            right_hand,     # 21 x 3 = 63 features
            face_landmarks  # 20 x 3 = 60 features
        ])
        
        return combined_landmarks
    
    def extract_from_video(self, video_path, max_frames=None):
        """
        Extract landmarks from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None = all)
        
        Returns:
            sequence: numpy array of shape (frames, 21, 3)
        """
        cap = cv2.VideoCapture(video_path)
        landmarks_sequence = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.extract_from_image(frame)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        
        return np.array(landmarks_sequence) if landmarks_sequence else None
    
    def extract_from_camera(self, duration_sec=5, fps=30):
        """
        Extract landmarks from live camera feed
        
        Args:
            duration_sec: Duration to record in seconds
            fps: Target frames per second
        
        Returns:
            sequence: numpy array of shape (frames, 21, 3)
        """
        cap = cv2.VideoCapture(0)
        landmarks_sequence = []
        
        total_frames = duration_sec * fps
        frame_count = 0
        
        print(f"Recording for {duration_sec} seconds...")
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.extract_from_image(frame)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)
            
            # Display frame with landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Recording...', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        return np.array(landmarks_sequence) if landmarks_sequence else None
    
    @staticmethod
    def normalize_landmarks(landmarks):
        """
        Normalize landmarks to be translation and scale invariant

        Args:
            landmarks: numpy array of shape (21, 3) or (frames, 21, 3)

        Returns:
            normalized: Normalized landmarks
        """
        # Handle both single frame and sequence
        is_sequence = len(landmarks.shape) == 3

        if is_sequence:
            normalized = []
            for frame in landmarks:
                norm_frame = LandmarkExtractor._normalize_single_frame(frame)
                normalized.append(norm_frame)
            return np.array(normalized)
        else:
            return LandmarkExtractor._normalize_single_frame(landmarks)

    @staticmethod
    def _normalize_single_frame(landmarks):
        """Normalize a single frame of landmarks"""
        # Center at wrist (landmark 0)
        centered = landmarks - landmarks[0]
        
        # Scale based on hand size (distance from wrist to middle finger tip)
        hand_size = np.linalg.norm(centered[12] - centered[0])
        if hand_size > 0:
            normalized = centered / hand_size
        else:
            normalized = centered
        
        return normalized
    
    def flatten_landmarks(self, landmarks):
        """
        Flatten landmarks from (21, 3) to (63,)
        
        Args:
            landmarks: numpy array of shape (..., 21, 3)
        
        Returns:
            flattened: numpy array of shape (..., 63)
        """
        shape = landmarks.shape
        return landmarks.reshape(*shape[:-2], -1)
    
    def process_dataset_folder(self, input_dir, output_dir, file_extension='.mp4'):
        """
        Process entire dataset folder
        
        Args:
            input_dir: Directory containing videos/images organized by class
            output_dir: Directory to save extracted landmarks
            file_extension: File extension to process (.mp4, .avi, .jpg, etc.)
        
        Returns:
            dataset_info: Dictionary with dataset statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            'landmarks': [],
            'labels': [],
            'class_names': []
        }
        
        class_folders = [f for f in input_path.iterdir() if f.is_dir()]
        
        for class_idx, class_folder in enumerate(tqdm(class_folders, desc="Processing classes")):
            class_name = class_folder.name
            dataset['class_names'].append(class_name)
            
            files = list(class_folder.glob(f'*{file_extension}'))
            
            for file in tqdm(files, desc=f"  {class_name}", leave=False):
                if file_extension in ['.mp4', '.avi', '.mov']:
                    sequence = self.extract_from_video(str(file))
                else:  # Image
                    image = cv2.imread(str(file))
                    landmarks = self.extract_from_image(image)
                    sequence = np.array([landmarks]) if landmarks is not None else None
                
                if sequence is not None and len(sequence) > 0:
                    # Normalize and flatten
                    normalized = self.normalize_landmarks(sequence)
                    flattened = self.flatten_landmarks(normalized)
                    
                    dataset['landmarks'].append(flattened)
                    dataset['labels'].append(class_idx)
        
        # Save dataset
        dataset['landmarks'] = dataset['landmarks']
        dataset['labels'] = np.array(dataset['labels'])
        
        output_file = output_path / 'landmarks_dataset.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Save metadata
        metadata = {
            'num_classes': len(dataset['class_names']),
            'class_names': dataset['class_names'],
            'num_samples': len(dataset['labels']),
            'samples_per_class': {
                name: np.sum(dataset['labels'] == idx)
                for idx, name in enumerate(dataset['class_names'])
            }
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset saved to {output_file}")
        print(f"Total samples: {len(dataset['labels'])}")
        print(f"Classes: {len(dataset['class_names'])}")
        
        return metadata
    
    def draw_landmarks_on_image(self, image, landmarks_data):
        """
        Draw hand and face landmarks with professional quality (matching Option 1)
        
        Args:
            image: BGR image from OpenCV
            landmarks_data: Combined landmarks array (62, 3) or MediaPipe results
        
        Returns:
            image: Image with landmarks drawn
        """
        image_copy = image.copy()
        h, w = image_copy.shape[:2]
        
        # Convert BGR to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process hands and face
        hand_results = self.hands.process(image_rgb)
        
        # Hand connections with specific colors for each finger
        HAND_CONNECTIONS = {
            'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
            'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
            'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
            'ring': [(0, 13), (13, 14), (14, 15), (15, 16)],
            'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)],
            'palm': [(5, 9), (9, 13), (13, 17)]
        }
        
        # Color schemes for left and right hands
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
        
        # Draw hands with advanced visualization
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                # Determine hand label and colors
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                colors = LEFT_HAND_COLORS if hand_label == "Left" else RIGHT_HAND_COLORS
                
                # Draw connections
                for finger, finger_connections in HAND_CONNECTIONS.items():
                    color = colors[finger]
                    for connection in finger_connections:
                        start_idx, end_idx = connection
                        start_point = (int(hand_landmarks.landmark[start_idx].x * w), 
                                     int(hand_landmarks.landmark[start_idx].y * h))
                        end_point = (int(hand_landmarks.landmark[end_idx].x * w), 
                                   int(hand_landmarks.landmark[end_idx].y * h))
                        
                        # Outer glow (white)
                        cv2.line(image_copy, start_point, end_point, (255, 255, 255), 6, cv2.LINE_AA)
                        # Main line with color
                        cv2.line(image_copy, start_point, end_point, color, 4, cv2.LINE_AA)
                
                # Draw landmarks
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    
                    if idx == 0:
                        # Wrist - large yellow circle
                        cv2.circle(image_copy, (x, y), 14, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(image_copy, (x, y), 16, (255, 255, 255), 3, cv2.LINE_AA)
                    elif idx in [4, 8, 12, 16, 20]:
                        # Fingertips - bright green
                        cv2.circle(image_copy, (x, y), 12, (0, 255, 0), -1, cv2.LINE_AA)
                        cv2.circle(image_copy, (x, y), 14, (255, 255, 255), 3, cv2.LINE_AA)
                    else:
                        # Joints - cyan
                        cv2.circle(image_copy, (x, y), 8, (255, 200, 100), -1, cv2.LINE_AA)
                        cv2.circle(image_copy, (x, y), 10, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Add hand label
                wrist = hand_landmarks.landmark[0]
                label_x = int(wrist.x * w) - 40
                label_y = int(wrist.y * h) - 30
                
                # Label background
                cv2.rectangle(image_copy, (label_x - 5, label_y - 25), 
                             (label_x + 95, label_y + 5), (30, 30, 30), -1)
                cv2.rectangle(image_copy, (label_x - 5, label_y - 25), 
                             (label_x + 95, label_y + 5), colors['palm'], 2)
                cv2.putText(image_copy, hand_label, (label_x, label_y - 5),
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw FULL MediaPipe face mesh (468 landmarks) with professional quality
        if self.detect_face:
            face_results = self.face_mesh.process(image_rgb)
            if face_results.multi_face_landmarks:
                mp_drawing_styles = self.mp_draw.DrawingSpec
                
                for face_landmarks in face_results.multi_face_landmarks:
                    # Draw tesselation (mesh grid - creates dense pattern)
                    self.mp_draw.draw_landmarks(
                        image=image_copy,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_draw.DrawingSpec(
                            color=(80, 110, 10), thickness=1, circle_radius=1)
                    )
                    
                    # Draw contours (face outline, eyes, lips, eyebrows)
                    self.mp_draw.draw_landmarks(
                        image=image_copy,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_draw.DrawingSpec(
                            color=(80, 256, 121), thickness=1, circle_radius=1)
                    )
                    
                    # Draw irises (eye details)
                    self.mp_draw.draw_landmarks(
                        image=image_copy,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_draw.DrawingSpec(
                            color=(255, 117, 66), thickness=1, circle_radius=1)
                    )
        
        return image_copy
    
    def close(self):
        """Release resources"""
        self.hands.close()
        if self.detect_face:
            self.face_mesh.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract hand landmarks from dataset')
    parser.add_argument('--input', type=str, default='data/ISL_dataset',
                       help='Input directory with videos/images')
    parser.add_argument('--output', type=str, default='data/landmarks',
                       help='Output directory for landmarks')
    parser.add_argument('--extension', type=str, default='.mp4',
                       help='File extension to process')
    
    args = parser.parse_args()
    
    extractor = LandmarkExtractor()
    metadata = extractor.process_dataset_folder(args.input, args.output, args.extension)
    extractor.close()
    
    print("\n" + "="*50)
    print("Landmark extraction complete!")
    print("="*50)
