"""
Train Deep Sequential GRU Model for Common ISL Gestures:
- HELLO
- THANK YOU
- PLEASE
- YES
- NO
- HELP
- GOOD
- NAMASTE
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

GESTURE_CLASSES = [
    'HELLO',
    'THANK YOU',
    'PLEASE',
    'YES',
    'NO',
    'HELP',
    'GOOD',
    'NAMASTE'
]

NUM_CLASSES = len(GESTURE_CLASSES)
SEQ_LEN = 30
NUM_FEATURES = 126  # 42 hand landmarks * 3 coords (x, y, z)


def generate_kinematic_gesture_samples(samples_per_class=200):
    """
    Generate realistic spatiotemporal landmark sequences for the 8 gestures
    with anatomical kinematic constraints, speed variations, and spatial jitter.
    """
    np.random.seed(42)
    X = []
    y = []

    for class_idx, class_name in enumerate(GESTURE_CLASSES):
        print(f"Generating realistic kinematic sequences for gesture '{class_name}' ({samples_per_class} samples)...")
        for s in range(samples_per_class):
            seq = np.zeros((SEQ_LEN, 42, 3), dtype=np.float32)
            
            # Global speed and timing variation (time warp factor)
            speed_factor = np.random.uniform(0.85, 1.25)
            t_steps = np.linspace(0.0, 1.0 * speed_factor, SEQ_LEN)
            
            # Base spatial offset & scale jitter
            dx = np.random.normal(0.0, 0.03)
            dy = np.random.normal(0.0, 0.03)
            scale = np.random.uniform(0.90, 1.12)
            noise_sigma = 0.008

            for t_idx, t in enumerate(t_steps):
                t_c = min(1.0, max(0.0, t))
                
                # Default resting coordinates
                # Hand 1 (Dominant/Right hand, indices 21..41 in 42-pts)
                r_wrist = np.array([0.60 + dx, 0.65 + dy, 0.0])
                # Hand 0 (Non-dominant/Left hand, indices 0..20 in 42-pts)
                l_wrist = np.array([0.40 + dx, 0.65 + dy, 0.0])
                
                if class_name == 'HELLO':
                    # Right hand near temple (x ~ 0.68, y ~ 0.28), waving laterally
                    # Side to side wave: sin(t * 4 * pi)
                    wave_x = 0.08 * np.sin(t_c * 4 * np.pi)
                    r_wrist = np.array([0.68 + dx + wave_x, 0.32 + dy - 0.04 * t_c, -0.05])
                    # Fingers extended upward in open palm
                    seq[t_idx, 21] = r_wrist
                    for f in range(5):
                        for j in range(4):
                            pt_idx = 22 + f * 4 + j
                            offset = np.array([(f - 2) * 0.02, -(j + 1) * 0.045 * scale, 0.01 * j])
                            seq[t_idx, pt_idx] = r_wrist + offset
                    # Left hand resting low/inactive
                    seq[t_idx, 0] = np.array([0.25, 0.85, 0.1])

                elif class_name == 'THANK YOU':
                    # Right hand starts at chin/mouth (y ~ 0.35, x ~ 0.52), sweeps forward and down (y ~ 0.62, x ~ 0.55, z ~ -0.20)
                    r_x = 0.52 + 0.06 * t_c + dx
                    r_y = 0.35 + 0.28 * t_c + dy
                    r_z = 0.0 - 0.25 * t_c
                    r_wrist = np.array([r_x, r_y, r_z])
                    seq[t_idx, 21] = r_wrist
                    # Flat open hand fingers pointing forward
                    for f in range(5):
                        for j in range(4):
                            pt_idx = 22 + f * 4 + j
                            offset = np.array([(f - 2) * 0.018, -(j + 1) * 0.038 * (1.0 - 0.4 * t_c) * scale, -(j + 1) * 0.04 * t_c])
                            seq[t_idx, pt_idx] = r_wrist + offset
                    # Left hand resting
                    seq[t_idx, 0] = np.array([0.25, 0.85, 0.1])

                elif class_name == 'PLEASE':
                    # Circular rubbing motion over chest (center ~ 0.50, 0.55, radius ~ 0.07)
                    angle = t_c * 3.5 * np.pi
                    r_x = 0.52 + 0.08 * np.cos(angle) + dx
                    r_y = 0.55 + 0.08 * np.sin(angle) + dy
                    r_wrist = np.array([r_x, r_y, -0.02])
                    seq[t_idx, 21] = r_wrist
                    # Flat palm facing chest
                    for f in range(5):
                        for j in range(4):
                            pt_idx = 22 + f * 4 + j
                            offset = np.array([(f - 2) * 0.02, -(j + 1) * 0.04 * scale, 0.01])
                            seq[t_idx, pt_idx] = r_wrist + offset
                    # Left hand resting
                    seq[t_idx, 0] = np.array([0.25, 0.85, 0.1])

                elif class_name == 'YES':
                    # Fist nodding up and down at wrist level (pitch oscillation)
                    nod_y = 0.06 * np.sin(t_c * 4 * np.pi)
                    r_wrist = np.array([0.62 + dx, 0.52 + dy + nod_y, -0.05])
                    seq[t_idx, 21] = r_wrist
                    # Tight fist: fingers curled tightly towards palm
                    for f in range(5):
                        for j in range(4):
                            pt_idx = 22 + f * 4 + j
                            offset = np.array([(f - 2) * 0.015, (j + 1) * 0.012 * scale, (j + 1) * 0.015])
                            seq[t_idx, pt_idx] = r_wrist + offset

                elif class_name == 'NO':
                    # Index and middle finger snapping shut against thumb (beak snap)
                    # Closure parameter: opens and snaps shut repeatedly
                    closure = 0.5 + 0.5 * np.sin(t_c * 3 * np.pi)
                    r_wrist = np.array([0.60 + dx, 0.48 + dy, -0.05])
                    seq[t_idx, 21] = r_wrist
                    # Thumb
                    for j in range(4):
                        seq[t_idx, 22 + j] = r_wrist + np.array([-0.02 * (j + 1), -(j + 1) * 0.025 * scale, 0.0])
                    # Index & Middle fingers (moving toward thumb based on closure)
                    for f in [1, 2]:
                        for j in range(4):
                            pt_idx = 22 + f * 4 + j
                            seq[t_idx, pt_idx] = r_wrist + np.array([0.02 * (1.0 - closure), -(j + 1) * 0.04 * scale * (1.0 - 0.4 * closure), 0.0])
                    # Ring & Pinky curled
                    for f in [3, 4]:
                        for j in range(4):
                            pt_idx = 22 + f * 4 + j
                            seq[t_idx, pt_idx] = r_wrist + np.array([(f - 2) * 0.02, (j + 1) * 0.015, 0.02])

                elif class_name == 'HELP':
                    # Two hands together: Left hand flat palm up at (0.48, 0.65), Right hand thumbs-up fist on top
                    # Both lifting upward from 0.65 to 0.45 in tandem
                    lift_y = -0.18 * t_c
                    l_wrist = np.array([0.48 + dx, 0.65 + dy + lift_y, 0.0])
                    r_wrist = np.array([0.52 + dx, 0.58 + dy + lift_y, -0.03])
                    # Left hand flat open palm
                    seq[t_idx, 0] = l_wrist
                    for f in range(5):
                        for j in range(4):
                            seq[t_idx, 1 + f * 4 + j] = l_wrist + np.array([(f - 2) * 0.02, -(j + 1) * 0.035 * scale, 0.0])
                    # Right hand thumbs-up fist
                    seq[t_idx, 21] = r_wrist
                    # Thumb pointing straight UP
                    for j in range(4):
                        seq[t_idx, 22 + j] = r_wrist + np.array([0.0, -(j + 1) * 0.035 * scale, 0.0])
                    # Other fingers curled
                    for f in range(1, 5):
                        for j in range(4):
                            seq[t_idx, 22 + f * 4 + j] = r_wrist + np.array([(f - 2) * 0.015, (j + 1) * 0.015, 0.02])

                elif class_name == 'GOOD':
                    # Right hand from chin down onto Left flat palm
                    if t_c < 0.35:
                        r_wrist = np.array([0.52 + dx, 0.36 + dy, 0.0])
                    else:
                        u = (t_c - 0.35) / 0.65
                        r_wrist = np.array([0.52 + dx, 0.36 + u * 0.28 + dy, 0.0])
                    l_wrist = np.array([0.48 + dx, 0.66 + dy, 0.0])
                    seq[t_idx, 0] = l_wrist
                    seq[t_idx, 21] = r_wrist
                    for f in range(5):
                        for j in range(4):
                            seq[t_idx, 1 + f * 4 + j] = l_wrist + np.array([(f - 2) * 0.02, -(j + 1) * 0.035 * scale, 0.0])
                            seq[t_idx, 22 + f * 4 + j] = r_wrist + np.array([(f - 2) * 0.02, -(j + 1) * 0.035 * scale, 0.0])

                elif class_name == 'NAMASTE':
                    # Both hands move from chest sides (left ~ 0.38, right ~ 0.62) to press together at centerline (0.50, 0.50)
                    converge = min(1.0, t_c * 1.5)
                    l_x = 0.38 + converge * 0.10 + dx
                    r_x = 0.62 - converge * 0.10 + dx
                    center_y = 0.50 + dy
                    l_wrist = np.array([l_x, center_y, 0.0])
                    r_wrist = np.array([r_x, center_y, 0.0])
                    seq[t_idx, 0] = l_wrist
                    seq[t_idx, 21] = r_wrist
                    # Flat palms pointing upward touching
                    for f in range(5):
                        for j in range(4):
                            seq[t_idx, 1 + f * 4 + j] = l_wrist + np.array([(f - 2) * 0.015, -(j + 1) * 0.045 * scale, 0.0])
                            seq[t_idx, 22 + f * 4 + j] = r_wrist + np.array([-(f - 2) * 0.015, -(j + 1) * 0.045 * scale, 0.0])

            # Add Gaussian jitter across entire sequence
            seq += np.random.normal(0.0, noise_sigma, seq.shape).astype(np.float32)
            
            # Flatten 42*3 to 126
            flat_seq = seq.reshape(SEQ_LEN, NUM_FEATURES)
            X.append(flat_seq)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    return X, y


def build_gesture_gru_model(input_shape=(30, 126), num_classes=8):
    """Deep bidirectional GRU architecture specialized for conversational gestures"""
    inputs = keras.Input(shape=input_shape, name='landmark_sequence')
    
    # Masking for any zeros/padding
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # Bidirectional GRU layer 1
    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Bidirectional GRU layer 2
    x = layers.Bidirectional(
        layers.GRU(96, return_sequences=False, dropout=0.2, recurrent_dropout=0.1)
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Dense classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='gesture_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ISL_Conversational_Gesture_GRU')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    save_dir = Path("results/GESTURES")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("==========================================================")
    print("  TRAINING DEEP SEQUENTIAL GRU MODEL FOR ISL GESTURES")
    print(f"  Classes ({NUM_CLASSES}): {', '.join(GESTURE_CLASSES)}")
    print("==========================================================")
    
    # Generate kinematic dataset
    X, y = generate_kinematic_gesture_samples(samples_per_class=220)
    print(f"Total dataset shape: X={X.shape}, y={y.shape}")
    
    # Split 80/20 train/validation
    split_idx = int(0.80 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    model = build_gesture_gru_model(input_shape=(SEQ_LEN, NUM_FEATURES), num_classes=NUM_CLASSES)
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=35,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print("==========================================================")
    print(f"  FINAL GESTURE MODEL VALIDATION ACCURACY: {val_acc * 100:.2f}%")
    print("==========================================================")
    
    # Save model
    model_path = save_dir / "isl_gesture_model.h5"
    model.save(str(model_path))
    print(f"[OK] Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'GRU_BIDIRECTIONAL',
        'input_shape': [SEQ_LEN, NUM_FEATURES],
        'num_classes': NUM_CLASSES,
        'class_names': GESTURE_CLASSES,
        'accuracy': float(val_acc),
        'timestamp': '2026-09-22'
    }
    with open(save_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata saved to {save_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
