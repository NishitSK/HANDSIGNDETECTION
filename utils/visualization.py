"""
Visualization Utilities
Plots training metrics, confusion matrices, and results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os


def plot_training_history(history, save_path='logs/training_history.png'):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, 
                         save_path='logs/confusion_matrix.png',
                         figsize=(15, 12)):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_accuracy(y_true, y_pred, class_names,
                           save_path='logs/per_class_accuracy.png'):
    """Plot per-class accuracy"""
    # Get unique classes that actually appear in the data
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Use only class names that have samples
    present_class_names = [class_names[i] for i in unique_classes]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(present_class_names)), per_class_acc, color='skyblue', edgecolor='navy')
    
    # Color bars based on accuracy
    for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
        if acc >= 0.95:
            bar.set_color('green')
        elif acc >= 0.85:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xlabel('Sign Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(range(len(present_class_names)), present_class_names, rotation=90)
    plt.ylim([0, 1.0])
    plt.axhline(y=0.99, color='r', linestyle='--', label='99% Target')
    plt.axhline(y=0.95, color='orange', linestyle='--', label='95% Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class accuracy plot saved to {save_path}")


def generate_classification_report(y_true, y_pred, class_names,
                                  save_path='logs/classification_report.txt'):
    """Generate and save classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ISL RECOGNITION SYSTEM - CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Overall Accuracy: {(y_true == y_pred).sum() / len(y_true):.4f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Classification report saved to {save_path}")
    return report


def plot_prediction_confidence(confidences, predictions, true_labels,
                               save_path='logs/confidence_distribution.png'):
    """Plot distribution of prediction confidences"""
    correct = predictions == true_labels
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of confidences
    axes[0].hist(confidences[correct], bins=50, alpha=0.7, label='Correct', color='green')
    axes[0].hist(confidences[~correct], bins=50, alpha=0.7, label='Incorrect', color='red')
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data = [confidences[correct], confidences[~correct]]
    axes[1].boxplot(data, labels=['Correct', 'Incorrect'])
    axes[1].set_ylabel('Confidence', fontsize=12)
    axes[1].set_title('Confidence by Correctness', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confidence distribution plot saved to {save_path}")


def visualize_hand_landmarks(landmarks, save_path='logs/hand_landmarks.png'):
    """Visualize hand landmarks in 3D"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # MediaPipe hand connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], 
              c='red', s=50, alpha=0.8)
    
    # Plot connections
    for connection in HAND_CONNECTIONS:
        start, end = connection
        ax.plot([landmarks[start, 0], landmarks[end, 0]],
               [landmarks[start, 1], landmarks[end, 1]],
               [landmarks[start, 2], landmarks[end, 2]],
               'b-', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hand Landmarks 3D Visualization', fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
