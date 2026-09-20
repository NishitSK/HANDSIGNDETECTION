"""
Data Augmentation Utilities
Augments hand landmark data for training robustness
"""

import numpy as np
import random


class LandmarkAugmentation:
    """Augmentation class for hand landmark data"""
    
    def __init__(self, rotation_range=15, scale_range=0.2, 
                 noise_std=0.01, shift_range=0.1):
        """
        Args:
            rotation_range: Max rotation in degrees
            scale_range: Scale variation (0.2 = ±20%)
            noise_std: Standard deviation of Gaussian noise
            shift_range: Max translation as fraction of range
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.shift_range = shift_range
    
    def rotate(self, landmarks, angle=None):
        """Rotate landmarks around center"""
        if angle is None:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        angle_rad = np.radians(angle)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        
        # Center landmarks
        center = landmarks.mean(axis=0)
        centered = landmarks - center
        
        # Rotation matrix (2D, ignore z-axis)
        rotated = centered.copy()
        rotated[:, 0] = centered[:, 0] * cos_val - centered[:, 1] * sin_val
        rotated[:, 1] = centered[:, 0] * sin_val + centered[:, 1] * cos_val
        
        return rotated + center
    
    def scale(self, landmarks, factor=None):
        """Scale landmarks"""
        if factor is None:
            factor = random.uniform(1 - self.scale_range, 1 + self.scale_range)
        
        center = landmarks.mean(axis=0)
        return (landmarks - center) * factor + center
    
    def add_noise(self, landmarks, std=None):
        """Add Gaussian noise to landmarks"""
        if std is None:
            std = self.noise_std
        
        noise = np.random.normal(0, std, landmarks.shape)
        return landmarks + noise
    
    def shift(self, landmarks, shift_x=None, shift_y=None):
        """Translate landmarks"""
        if shift_x is None:
            shift_x = random.uniform(-self.shift_range, self.shift_range)
        if shift_y is None:
            shift_y = random.uniform(-self.shift_range, self.shift_range)
        
        shift = np.array([shift_x, shift_y, 0])
        return landmarks + shift
    
    def flip_horizontal(self, landmarks):
        """Flip landmarks horizontally"""
        flipped = landmarks.copy()
        flipped[:, 0] = 1.0 - flipped[:, 0]  # Assuming normalized coordinates
        return flipped
    
    def augment_sequence(self, sequence, augment_prob=0.5):
        """
        Augment a sequence of landmarks
        
        Args:
            sequence: numpy array of shape (frames, 21, 3) or (frames, 63)
            augment_prob: Probability of applying augmentation
        
        Returns:
            Augmented sequence
        """
        if random.random() > augment_prob:
            return sequence
        
        # Reshape if flattened
        original_shape = sequence.shape
        if len(sequence.shape) == 2:
            # Auto-detect number of landmarks (21 for single hand, 62 for both hands+face)
            num_features = sequence.shape[1]
            if num_features == 63:  # Single hand: 21 landmarks * 3
                sequence = sequence.reshape(-1, 21, 3)
            elif num_features == 186:  # Both hands + face: 62 landmarks * 3
                sequence = sequence.reshape(-1, 62, 3)
            else:
                # Default: try to infer
                num_landmarks = num_features // 3
                sequence = sequence.reshape(-1, num_landmarks, 3)
        
        augmented = sequence.copy()
        
        # Apply random augmentations
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            for i in range(len(augmented)):
                augmented[i] = self.rotate(augmented[i], angle)
        
        if random.random() < 0.5:
            factor = random.uniform(1 - self.scale_range, 1 + self.scale_range)
            for i in range(len(augmented)):
                augmented[i] = self.scale(augmented[i], factor)
        
        if random.random() < 0.5:
            for i in range(len(augmented)):
                augmented[i] = self.add_noise(augmented[i])
        
        if random.random() < 0.3:
            shift_x = random.uniform(-self.shift_range, self.shift_range)
            shift_y = random.uniform(-self.shift_range, self.shift_range)
            for i in range(len(augmented)):
                augmented[i] = self.shift(augmented[i], shift_x, shift_y)
        
        # Reshape back to original
        if len(original_shape) == 2:
            augmented = augmented.reshape(original_shape)
        
        return augmented
    
    def temporal_augment(self, sequence, time_stretch_range=0.2):
        """
        Temporal augmentation: speed up or slow down sequence
        
        Args:
            sequence: numpy array of shape (frames, features)
            time_stretch_range: Max time stretch factor (0.2 = ±20% speed)
        
        Returns:
            Time-stretched sequence
        """
        factor = random.uniform(1 - time_stretch_range, 1 + time_stretch_range)
        new_length = int(len(sequence) * factor)
        
        indices = np.linspace(0, len(sequence) - 1, new_length)
        interpolated = np.array([
            np.interp(indices, np.arange(len(sequence)), sequence[:, i])
            for i in range(sequence.shape[1])
        ]).T
        
        return interpolated


def augment_dataset(X, y, augmentation_factor=3):
    """
    Augment entire dataset
    
    Args:
        X: Input sequences (n_samples, seq_len, features)
        y: Labels (n_samples,)
        augmentation_factor: How many augmented samples per original
    
    Returns:
        X_aug, y_aug: Augmented dataset
    """
    augmenter = LandmarkAugmentation()
    
    X_augmented = [X]
    y_augmented = [y]
    
    for _ in range(augmentation_factor):
        X_aug = np.array([
            augmenter.augment_sequence(seq, augment_prob=1.0)
            for seq in X
        ])
        X_augmented.append(X_aug)
        y_augmented.append(y)
    
    X_final = np.concatenate(X_augmented, axis=0)
    y_final = np.concatenate(y_augmented, axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(X_final))
    
    return X_final[indices], y_final[indices]
