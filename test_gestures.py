"""
Test trained conversational gesture model on unseen kinematic sequences
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from tensorflow import keras
from train_gestures import generate_kinematic_gesture_samples, GESTURE_CLASSES

def test_gesture_predictions():
    model_path = Path("results/GESTURES/isl_gesture_model.h5")
    assert model_path.exists(), f"Model path {model_path} does not exist"
    
    model = keras.models.load_model(str(model_path), compile=False)
    print(f"[OK] Loaded gesture model from {model_path}")
    
    # Generate fresh test samples with independent random seed
    np.random.seed(999)
    X_test, y_test = generate_kinematic_gesture_samples(samples_per_class=30)
    
    preds = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    
    correct = np.sum(pred_classes == y_test)
    total = len(y_test)
    accuracy = (correct / total) * 100.0
    
    print("\n=======================================================")
    print(f"  TEST EVALUATION ON UNSEEN KINEMATIC SEQUENCES:")
    print(f"  Total Samples: {total}")
    print(f"  Correct: {correct} / {total}")
    print(f"  Test Accuracy: {accuracy:.2f}%")
    print("=======================================================")
    
    # Per-class breakdown
    print("\nPer-Class Recognition Results:")
    for c_idx, c_name in enumerate(GESTURE_CLASSES):
        mask = (y_test == c_idx)
        c_correct = np.sum(pred_classes[mask] == c_idx)
        c_total = np.sum(mask)
        avg_conf = np.mean(preds[mask, c_idx]) * 100.0
        print(f"  - {c_name:<12}: {c_correct}/{c_total} ({c_correct/c_total*100:.1f}%) | Avg Conf: {avg_conf:.1f}%")
    
    assert accuracy >= 95.0, f"Accuracy {accuracy}% is below required 95%"
    print("\n[PASSED] All conversational gestures successfully recognized with high accuracy!")

if __name__ == "__main__":
    test_gesture_predictions()
