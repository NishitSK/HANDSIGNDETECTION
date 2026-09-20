"""
Model Training Script
Train ISL gesture recognition model with advanced features
"""

import numpy as np
import random
import time
import tensorflow as tf
from tensorflow import keras
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gesture_model import create_model
from utils.config_loader import get_config
from utils.data_augmentation import augment_dataset
from utils.visualization import (plot_training_history, plot_confusion_matrix,
                                 plot_per_class_accuracy, generate_classification_report,
                                 plot_prediction_confidence)
from src.landmark_extraction import LandmarkExtractor


def set_global_seed(seed=42):
    """Seed every source of randomness used in the pipeline (Python, NumPy,
    TensorFlow) so training runs and reported metrics are reproducible."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class TimeBudgetCallback(keras.callbacks.Callback):
    """Hard wall-clock stop, independent of epoch count. Checked every batch
    (not just per-epoch) so a slow architecture can't overshoot the budget by
    an entire epoch — some epochs here take 15+ minutes."""

    def __init__(self, budget_minutes):
        super().__init__()
        self.budget_seconds = budget_minutes * 60
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if time.time() - self.start_time > self.budget_seconds:
            self.model.stop_training = True


class TargetAccuracyCallback(keras.callbacks.Callback):
    """Stop training when target accuracy is reached after minimum epochs"""
    
    def __init__(self, target_accuracy=0.75, min_epochs=10):
        super().__init__()
        self.target_accuracy = target_accuracy
        self.min_epochs = min_epochs
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_accuracy = logs.get('val_accuracy', 0)
        
        # Check if we've reached minimum epochs and target accuracy
        if epoch >= self.min_epochs and current_accuracy >= self.target_accuracy:
            print(f"\n[SUCCESS] Reached target accuracy {current_accuracy:.4f} (>= {self.target_accuracy:.4f}) after {epoch + 1} epochs!")
            print(f"[STOPPING] Early stopping triggered - Target achieved!")
            self.model.stop_training = True


class ISLTrainer:
    """Train ISL gesture recognition model"""
    
    def __init__(self, config_path='config.yaml'):
        self.config = get_config(config_path)
        self.model = None
        self.label_encoder = None
        self.class_names = []
        self.history = None
        self.groups = None
    
    def load_data(self, landmarks_path='data/landmarks/landmarks_dataset.pkl'):
        """Load preprocessed landmark data"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        landmarks_file = Path(landmarks_path)
        if not landmarks_file.exists():
            # Try to generate landmarks from raw dataset paths provided in config
            dataset_paths = self.config.get('data', {}).get('dataset_paths', [])
            if dataset_paths:
                print("Landmarks not found — attempting to extract from raw dataset paths listed in config...")
                extractor = LandmarkExtractor(static_mode=True,
                                              max_hands=2,
                                              detect_face=True)
                out_dir = Path('data/landmarks')
                out_dir.mkdir(parents=True, exist_ok=True)

                for src_path in dataset_paths:
                    src = Path(src_path)
                    if not src.exists():
                        print(f"Warning: dataset path not found: {src}")
                        continue
                    print(f"Processing dataset folder: {src}")
                    try:
                        extractor.process_dataset_folder(str(src), str(out_dir), file_extension='.jpg')
                    except Exception as e:
                        print(f"Error processing {src}: {e}")

                # After attempting extraction, look for the dataset file
                landmarks_file = out_dir / 'landmarks_dataset.pkl'
                if not landmarks_file.exists():
                    raise FileNotFoundError(f"Landmarks file not found after extraction attempt: {landmarks_file}")
            else:
                raise FileNotFoundError(f"Landmarks file not found: {landmarks_path}")
        
        with open(landmarks_file, 'rb') as f:
            dataset = pickle.load(f)
        
        # Get class names from dataset itself (most reliable)
        if 'class_names' in dataset:
            self.class_names = dataset['class_names']
        else:
            # Fallback: try to load from metadata file
            metadata_file = landmarks_file.parent / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.class_names = metadata['class_names']
            else:
                # Last resort: infer from labels
                num_classes = len(np.unique(dataset['labels']))
                self.class_names = [f"Class_{i}" for i in range(num_classes)]
        
        print(f"[OK] Loaded {len(dataset['labels'])} samples")
        print(f"[OK] Classes: {len(self.class_names)}")

        groups = dataset.get('groups')
        if groups is None or len(groups) != len(dataset['labels']):
            print("[WARN] No capture-session groups found in this dataset file.")
            print("       Train/val/test will be split by random frame, which can leak")
            print("       near-duplicate burst-captured frames of the same gesture instance")
            print("       across splits and inflate reported accuracy. Re-run train.py to")
            print("       regenerate landmarks with session groups for a leakage-free split.")
        self.groups = groups

        return dataset
    
    def prepare_sequences(self, landmarks_list, labels, sequence_length=30):
        """
        Prepare fixed-length sequences from variable-length data
        
        Args:
            landmarks_list: List of landmark sequences (variable length)
            labels: Corresponding labels
            sequence_length: Target sequence length
        
        Returns:
            X, y: Numpy arrays of shape (n_samples, seq_len, features)
        """
        print("\nPreparing sequences...")
        
        X_sequences = []
        y_sequences = []
        
        for landmarks, label in zip(landmarks_list, labels):
            # Detect raw (unflattened) landmark data — shape (frames, N, 3) or
            # (N, 3) — and normalize it (translation/scale invariant) BEFORE
            # flattening. The live-inference pipeline (src/inference.py)
            # normalizes every frame before predicting; skipping this at
            # training time would train on a different feature distribution
            # than the model ever sees in real use, and severely hurt
            # real-world accuracy despite looking fine on this pipeline's own
            # (also un-normalized) held-out test split.
            is_raw_sequence = landmarks.ndim == 3
            is_raw_single_frame = landmarks.ndim == 2 and landmarks.shape[1] == 3 and landmarks.shape[0] > 3
            if is_raw_sequence or is_raw_single_frame:
                landmarks = LandmarkExtractor.normalize_landmarks(landmarks)

            # Handle different input formats
            if len(landmarks.shape) == 3:
                # Already in sequence format (frames, landmarks, coords)
                # Flatten landmarks per frame: (frames, landmarks*coords)
                landmarks = landmarks.reshape(landmarks.shape[0], -1)
            elif len(landmarks.shape) == 2:
                # Single frame: (landmarks, coords) or sequence of frames with features
                # Check if this looks like landmarks (62, 3) or features (frames, features)
                if landmarks.shape[1] == 3 and landmarks.shape[0] > 3:
                    # Looks like (landmarks, 3) - single frame, flatten it
                    landmarks = landmarks.reshape(1, -1)
                # else: already in (frames, features) format
            elif len(landmarks.shape) == 1:
                # Already flattened, reshape to single frame
                landmarks = landmarks.reshape(1, -1)
            
            current_len = len(landmarks)

            # Pad or truncate to sequence_length
            if current_len >= sequence_length:
                # Take middle portion or random window
                start_idx = (current_len - sequence_length) // 2
                sequence = landmarks[start_idx:start_idx + sequence_length]
            else:
                # Tile (repeat) the observed frame(s) to fill the window
                # instead of zero-padding. Most training samples are a
                # single static photo of a held gesture; zero-padding would
                # tell the model "nothing is happening" for most of the
                # window, which never occurs at inference time (there every
                # one of the `sequence_length` frames is real webcam data).
                # Repeating the held pose keeps train/inference inputs
                # consistent, matching what a live camera sees for a held sign.
                repeats = int(np.ceil(sequence_length / current_len))
                sequence = np.tile(landmarks, (repeats, 1))[:sequence_length]
            
            X_sequences.append(sequence)
            y_sequences.append(label)
        
        # float32 halves memory vs. the float64 default (landmark coordinates
        # need nowhere near float64 precision, and Keras casts to float32
        # internally anyway) — matters here because augmentation multiplies
        # the array size several times over on top of this.
        X = np.array(X_sequences, dtype=np.float32)
        y = np.array(y_sequences)

        print(f"[OK] Prepared sequences: {X.shape}")
        
        return X, y
    
    def train(self, landmarks_path='data/landmarks/landmarks_dataset.pkl',
              epochs=None, batch_size=None, augment=True, time_budget_minutes=None):
        """
        Train the model

        Args:
            landmarks_path: Path to preprocessed landmarks
            epochs: Number of training epochs
            time_budget_minutes: Optional hard wall-clock cap on training,
                independent of epoch count (see TimeBudgetCallback)
            batch_size: Batch size for training
            augment: Whether to apply data augmentation
        """
        set_global_seed(42)

        # Repeated model building within one long-lived process (e.g. a
        # multi-architecture comparison) accumulates TensorFlow graph/session
        # state across calls even after the old `keras.Model` object is
        # dereferenced — this is a well-known Keras memory leak. Clearing the
        # backend session at the start of every train() call keeps memory use
        # bounded regardless of how many times this is called in one process.
        keras.backend.clear_session()

        # Load configuration
        training_config = self.config['training']
        model_config = self.config['model']

        epochs = epochs or training_config['epochs']
        batch_size = batch_size or training_config['batch_size']
        sequence_length = model_config['sequence_length']

        # Load data
        dataset = self.load_data(landmarks_path)

        # Prepare sequences
        X, y = self.prepare_sequences(
            dataset['landmarks'],
            dataset['labels'],
            sequence_length
        )
        groups = np.array(self.groups) if self.groups else None

        # Fail loudly and clearly here rather than letting sklearn crash later
        # with a cryptic "resulting train set will be empty" traceback — this
        # is usually a landmark-extraction problem (see the per-class
        # detection-rate warnings printed during extraction), not a splitting
        # bug, so say so.
        present_classes, class_counts = np.unique(y, return_counts=True)
        if len(present_classes) < 2:
            found = {self.class_names[c]: int(n) for c, n in zip(present_classes, class_counts)}
            raise ValueError(
                f"Only {len(present_classes)} class(es) have any usable samples after "
                f"landmark extraction: {found}. Need samples from at least 2 classes to "
                "train/evaluate a classifier — check the detection-rate warnings printed "
                "during extraction for why the other classes' samples were dropped."
            )
        min_per_class = int(class_counts.min())
        if min_per_class < 5:
            sparse = {self.class_names[c]: int(n) for c, n in zip(present_classes, class_counts) if n < 5}
            raise ValueError(
                f"These classes have too few usable samples to split into train/val/test: "
                f"{sparse}. Need at least a handful of detected samples per class — check the "
                "detection-rate warnings printed during extraction, and recollect data for "
                "the affected classes if the detection rate was low."
            )

        # Held-out test fraction and validation fraction (of the remaining
        # train pool), both driven by config so they're actually respected.
        test_frac = 1 - self.config.get('data', 'train_test_split', default=0.8)
        val_frac = training_config.get('validation_split', 0.2)

        # Split FIRST, augment AFTER: augmenting before splitting would scatter
        # rotated/scaled/noised near-duplicates of the same original sample
        # across train/val/test, leaking information and inflating reported
        # accuracy. Also split by capture SESSION (group), not by frame, when
        # session ids are available: bursts of near-identical frames from one
        # gesture instance must stay entirely on one side of the split.
        # This needs at least 3 distinct sessions per class (train/val/test);
        # a single continuous burst per class can't be split this way, so
        # fall back to a random split rather than starving a class entirely.
        use_groups = groups is not None and all(
            len(np.unique(groups[y == cls])) >= 3 for cls in np.unique(y)
        )
        if groups is not None and not use_groups:
            print("[WARN] Fewer than 3 separate capture sessions for at least one class —")
            print("       falling back to a random split. Record multiple SEPARATE sessions")
            print("       per class (camera off/on, different pose/lighting each time) so")
            print("       train/val/test can be split by session instead of by frame.")

        if use_groups:
            test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
            trainval_idx, test_idx = next(test_splitter.split(X, y, groups))

            val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
            train_idx, val_idx = next(val_splitter.split(
                X[trainval_idx], y[trainval_idx], groups[trainval_idx]
            ))
            train_idx, val_idx = trainval_idx[train_idx], trainval_idx[val_idx]
        else:
            trainval_idx, test_idx = train_test_split(
                np.arange(len(X)), test_size=test_frac, random_state=42, stratify=y
            )
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=val_frac, random_state=42, stratify=y[trainval_idx]
            )

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Data augmentation — training split only
        if augment and training_config.get('data_augmentation', True):
            print("\nApplying data augmentation to training split only...")
            X_train, y_train = augment_dataset(X_train, y_train, augmentation_factor=3)
            print(f"[OK] Augmented training set size: {len(X_train)} samples")

        print(f"\nTraining samples:   {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples:       {len(X_test)} (held out, never seen during training/selection)")

        # Compute class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Build model
        # Determine model type
        model_type = self.config['model'].get('type', 'LSTM').upper()

        # For sequence-to-sequence models the target vocabulary size is num_classes
        num_classes = len(self.class_names)
        input_shape = (sequence_length, X.shape[2])

        # If seq2seq, ensure targets have shape (n_samples, target_len)
        if model_type == 'SEQ2SEQ':
            target_len = self.config['model'].get('target_seq_length', 1)
            try:
                y_train = y_train.reshape(-1, target_len)
                y_val = y_val.reshape(-1, target_len)
                y_test = y_test.reshape(-1, target_len)
            except Exception:
                raise ValueError('SEQ2SEQ model expects target sequences of length target_seq_length')

        print("\n" + "="*60)
        print("BUILDING MODEL")
        print("="*60)
        
        self.model = create_model(self.config, input_shape, num_classes)
        
        # Callbacks
        callbacks = self._create_callbacks(time_budget_minutes)
        
        # Train model
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print("="*60 + "\n")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict if model_type != 'SEQ2SEQ' else None,
            callbacks=callbacks,
            verbose=1
        )
        
        # Final reported metrics come from the held-out TEST set, not the
        # validation set used for early stopping / checkpoint selection —
        # reusing the validation set here would optimistically bias the
        # numbers reported for the model.
        test_accuracy = self.evaluate(X_test, y_test)

        # Save model and metadata
        self.save_model()

        return self.history, test_accuracy

    def _create_callbacks(self, time_budget_minutes=None):
        """Create training callbacks"""
        training_config = self.config['training']
        paths = self.config['paths']
        model_type = self.config['model'].get('type', 'model')

        checkpoint_dir = Path(paths['checkpoints_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            # Target accuracy callback (stops when target is reached)
            TargetAccuracyCallback(
                target_accuracy=training_config.get('early_stopping_target_accuracy', 0.75),
                min_epochs=training_config.get('early_stopping_min_epochs', 10)
            ),

            # Model checkpoint — filename tagged with architecture so that
            # comparing multiple model types doesn't produce indistinguishable
            # checkpoint files in the same directory.
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / f'{model_type}_model_{{epoch:03d}}_{{val_accuracy:.4f}}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),

            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=training_config.get('early_stopping_patience', 20),
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=training_config.get('reduce_lr_patience', 10),
                min_lr=1e-7,
                verbose=1
            ),

            # TensorBoard — one subdirectory per architecture so runs don't
            # mix in the same event stream.
            keras.callbacks.TensorBoard(
                log_dir=str(Path(paths['logs_dir']) / model_type),
                histogram_freq=1,
                write_graph=True
            ),

            # CSV Logger — tagged filename, same reason as the checkpoint name.
            keras.callbacks.CSVLogger(
                str(Path(paths['logs_dir']) / f'{model_type}_training_log.csv')
            )
        ]

        if time_budget_minutes:
            callbacks.append(TimeBudgetCallback(time_budget_minutes))

        return callbacks
    
    def evaluate(self, X_test, y_test):
        """Evaluate model and generate reports"""
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        model_type = self.config['model'].get('type', 'LSTM').upper()

        if model_type == 'SEQ2SEQ':
            # y_pred_proba shape: (n_samples, target_len, vocab)
            y_pred = np.argmax(y_pred_proba, axis=-1)
            # For single-token targets collapse to shape (n_samples,)
            if y_pred.ndim == 2 and y_pred.shape[1] == 1:
                y_pred = y_pred.reshape(-1)
                confidences = np.max(y_pred_proba, axis=-1).reshape(-1)
            else:
                # multi-token outputs: evaluate only first token for now
                y_pred = y_pred[:, 0]
                confidences = np.max(y_pred_proba[:, 0, :], axis=1)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
            confidences = np.max(y_pred_proba, axis=1)
        
        # Accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"\n[OK] Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Top-K accuracy — only meaningful when K is a real subset of the
        # class space; with num_classes <= 5, "top-5" is close to trivially
        # ~100% and misleading to report alongside top-1 accuracy.
        num_classes = y_pred_proba.shape[-1]
        if num_classes > 5:
            top5_acc = np.mean([
                y_test[i] in np.argsort(y_pred_proba[i])[-5:]
                for i in range(len(y_test))
            ])
            print(f"[OK] Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
        else:
            print(f"[INFO] Skipping Top-5 accuracy: only {num_classes} classes (not meaningful)")
        
        # Mean confidence
        print(f"[OK] Mean Confidence: {confidences.mean():.4f}")
        
        # Generate visualizations
        logs_dir = Path(self.config['paths']['logs_dir'])
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating visualizations...")
        
        # Training history
        if self.history:
            plot_training_history(self.history, save_path=str(logs_dir / 'training_history.png'))
        
        # Confusion matrix
        plot_confusion_matrix(y_test, y_pred, self.class_names,
                            save_path=str(logs_dir / 'confusion_matrix.png'))
        
        # Per-class accuracy
        plot_per_class_accuracy(y_test, y_pred, self.class_names,
                               save_path=str(logs_dir / 'per_class_accuracy.png'))
        
        # Classification report
        generate_classification_report(y_test, y_pred, self.class_names,
                                      save_path=str(logs_dir / 'classification_report.txt'))
        
        # Confidence distribution
        plot_prediction_confidence(confidences, y_pred, y_test,
                                  save_path=str(logs_dir / 'confidence_distribution.png'))
        
        print(f"\n[OK] Visualizations saved to {logs_dir}")
        print("="*60 + "\n")
        
        return accuracy
    
    def save_model(self):
        """Save trained model and metadata"""
        save_dir = Path(self.config['paths']['model_save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_dir / 'isl_model.h5'
        self.model.save(str(model_path))
        print(f"\n[OK] Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'input_shape': self.model.input_shape[1:],
            'model_type': self.config['model']['type'],
            'training_config': self.config['training']
        }
        
        metadata_path = save_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Metadata saved to {metadata_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ISL Recognition Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--data', type=str, default='data/landmarks/landmarks_dataset.pkl',
                       help='Path to landmarks data')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate existing model')
    parser.add_argument('--model', type=str, help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    trainer = ISLTrainer(args.config)
    
    if args.evaluate:
        if args.model:
            trainer.model = keras.models.load_model(args.model)
            dataset = trainer.load_data(args.data)
            X, y = trainer.prepare_sequences(
                dataset['landmarks'],
                dataset['labels'],
                trainer.config['model']['sequence_length']
            )
            trainer.evaluate(X, y)
        else:
            print("Error: --model required for evaluation")
    else:
        trainer.train(
            landmarks_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            augment=not args.no_augment
        )
