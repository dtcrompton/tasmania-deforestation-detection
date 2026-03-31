#!/usr/bin/env python3
"""
Train binary CNN classifier for clearcut detection in Sentinel-2 imagery.

Inputs:
    - data/training/patch_labels.csv (315 labels, filtered to 263 usable)
    - data/training/patches/*.tif (128×128×5 Sentinel-2 patches)

Outputs:
    - models/clearcut_classifier_final.keras (trained model)
    - models/clearcut_classifier_best.keras (best checkpoint)
    - outputs/figures/training_history.png (loss/accuracy curves)
    - outputs/figures/confusion_matrix.png (test set confusion matrix)
    - outputs/model_evaluation.txt (precision/recall/F1 metrics)

Architecture: Simple 3-layer CNN (32→64→128 filters) to avoid overfitting on small dataset.
Class imbalance handling: Class weighting (10:1 ratio, clearcut minority class).
Key metric: Clearcut recall (target >80%).
"""

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# TensorFlow/Keras for CNN
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Scikit-learn for data splitting and evaluation
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


def load_patch(patch_id, year, patches_dir  =  "data/training/patches"):
    """
    Load a single 128×128×5 Sentinel-2 patch from disk.
    
    Parameters
    ----------
    patch_id : int
        Patch identifier from Hansen Global Forest Change dataset
    year : int
        Year of forest loss detection (2019-2024)
    patches_dir : str
        Directory containing GeoTIFF patch files
    
    Returns
    -------
    numpy.ndarray
        Patch array of shape (128, 128, 5), normalised to [0, 1]
        Bands: [B4 = Red, B3 = Green, B2 = Blue, B8 = NIR, B11 = SWIR1]
    
    Notes
    -----
    - Raw Sentinel-2 DN values (0-10,000) are normalised to [0, 1]
    - Rasterio returns (bands, height, width); transposed to (height, width, bands)
    """
    path  =  Path(patches_dir) / f"patch_{year}_{patch_id:04d}.tif"
    
    with rasterio.open(path) as src:
        # Read all 5 bands: returns shape (5, 128, 128)
        patch  =  src.read()
    
    # Transpose to TensorFlow format: (height, width, channels)
    patch  =  np.transpose(patch, (1, 2, 0))
    
    # Normalise Sentinel-2 DN values (0-10,000) to [0, 1] range
    patch  =  patch / 10000.0
    
    return patch.astype(np.float32)


def load_dataset(labels_path = "data/training/patch_labels.csv"):
    """
    Load all labelled patches and convert to training arrays.
    
    Parameters
    ----------
    labels_path : str
        Path to CSV with columns [patch_id, year, clearing_type]
        clearing_type values: 'clearcut', 'not_clearcut', 'skip'
    
    Returns
    -------
    X : numpy.ndarray
        Patch images of shape (n_samples, 128, 128, 5)
    y : numpy.ndarray
        Binary labels of shape (n_samples,) where 1 = clearcut, 0 = not_clearcut
    df : pandas.DataFrame
        Metadata for each loaded patch (patch_id, year, clearing_type)
    
    Notes
    -----
    - 'skip' labels (ambiguous patches) are excluded from training
    - Expected: 263 usable patches (24 clearcut, 239 not_clearcut)
    """
    # Load labels CSV and filter out ambiguous 'skip' labels
    df  =  pd.read_csv(labels_path)
    df  =  df[df["clearing_type"] !=  "skip"].copy()
    
    print(f"Loading {len(df)} patches (excluding 'skip' labels)...")
    print(f"Class distribution:")
    print(df["clearing_type"].value_counts())
    print()
    
    # Load patch images and convert labels to binary
    X  =  []
    y  =  []
    
    for idx, row in df.iterrows():
        try:
            # Load normalised patch (128, 128, 5)
            patch  =  load_patch(row["patch_id"], row["year"])
            X.append(patch)
            
            # Convert to binary label: 1 for clearcut, 0 for not_clearcut
            label  =  1 if row["clearing_type"]  ==  "clearcut" else 0
            y.append(label)
            
        except Exception as e:
            print(f"Warning: Failed to load patch {row['patch_id']} ({row['year']}): {e}")
            continue
    
    X  =  np.array(X)
    y  =  np.array(y)
    
    print(f"Loaded {len(X)} patches successfully")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Clearcut samples: {y.sum()} ({100*y.mean():.1f}%)")
    print()
    
    return X, y, df


def build_model(input_shape = (128, 128, 5)):
    """
    Build simple CNN for binary clearcut classification.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input patches (height, width, channels)
    
    Returns
    -------
    keras.Model
        Compiled CNN model
    
    Architecture
    ------------
    - Input: (128, 128, 5) Sentinel-2 patch
    - Conv block 1: 32 filters, 3×3 kernel, ReLU → MaxPool 2×2
    - Conv block 2: 64 filters, 3×3 kernel, ReLU → MaxPool 2×2
    - Conv block 3: 128 filters, 3×3 kernel, ReLU → MaxPool 2×2
    - Flatten → Dense(128, ReLU) → Dropout(0.5)
    - Output: Dense(1, sigmoid) for binary probability
    
    Notes
    -----
    - Simple architecture chosen to avoid overfitting on small dataset (263 samples)
    - Dropout(0.5) prevents overfitting by randomly zeroing 50% of activations
    - Compiled with Adam optimiser and binary crossentropy loss
    """
    model  =  keras.Sequential([
        # Input layer
        layers.Input(shape = input_shape),
        
        # Convolutional block 1: 128×128×5 → 64×64×32
        layers.Conv2D(32, (3, 3), activation = "relu", padding = "same"),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional block 2: 64×64×32 → 32×32×64
        layers.Conv2D(64, (3, 3), activation = "relu", padding = "same"),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional block 3: 32×32×64 → 16×16×128
        layers.Conv2D(128, (3, 3), activation = "relu", padding = "same"),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers: 16×16×128 → flatten → 128 → 1
        layers.Flatten(),
        layers.Dense(128, activation = "relu"),
        layers.Dropout(0.5),
        
        # Binary output: sigmoid activation for probability [0, 1]
        layers.Dense(1, activation = "sigmoid"),
    ])
    
    # Compile with binary crossentropy loss and track precision/recall
    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = [
            "accuracy",
            keras.metrics.Precision(name = "precision"),
            keras.metrics.Recall(name = "recall"),
        ],
    )
    
    return model


def plot_training_history(history, output_path = "outputs/figures/training_history.png"):
    """
    Plot training and validation curves for loss, accuracy, precision, recall.
    
    Parameters
    ----------
    history : keras.callbacks.History
        Training history object returned by model.fit()
    output_path : str
        Path to save figure (PNG)
    
    Notes
    -----
    - Creates 2×2 subplot grid showing all tracked metrics
    - Recall curve is most critical for clearcut detection (target >80%)
    """
    fig, axes  =  plt.subplots(2, 2, figsize = (14, 10))
    
    # Loss curves
    axes[0, 0].plot(history.history["loss"], label = "Training Loss")
    axes[0, 0].plot(history.history["val_loss"], label = "Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha = 0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history.history["accuracy"], label = "Training Accuracy")
    axes[0, 1].plot(history.history["val_accuracy"], label = "Validation Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Accuracy Curves")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha = 0.3)
    
    # Precision curves
    axes[1, 0].plot(history.history["precision"], label = "Training Precision")
    axes[1, 0].plot(history.history["val_precision"], label = "Validation Precision")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision Curves")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha = 0.3)
    
    # Recall curves (most important metric)
    axes[1, 1].plot(history.history["recall"], label = "Training Recall")
    axes[1, 1].plot(history.history["val_recall"], label = "Validation Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Recall Curves (Clearcut Detection)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300, bbox_inches = "tight")
    print(f"Saved training history plot to {output_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path = "outputs/figures/confusion_matrix.png"):
    """
    Plot confusion matrix heatmap for test set predictions.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        True binary labels (0 = not_clearcut, 1 = clearcut)
    y_pred : numpy.ndarray
        Predicted binary labels
    output_path : str
        Path to save figure (PNG)
    
    Notes
    -----
    - Rows represent true labels, columns represent predictions
    - Cell values show count of samples in each category
    """
    cm  =  confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize = (8, 6))
    sns.heatmap(
        cm,
        annot = True,
        fmt = "d",
        cmap = "Blues",
        xticklabels = ["Not Clearcut", "Clearcut"],
        yticklabels = ["Not Clearcut", "Clearcut"],
        cbar_kws = {"label": "Count"},
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300, bbox_inches = "tight")
    print(f"Saved confusion matrix to {output_path}")
    plt.close()


def main():
    """
    Main training pipeline: load data, train model, evaluate, save outputs.
    
    Pipeline
    --------
    1. Load 263 labelled patches (24 clearcut, 239 not_clearcut)
    2. Split 70/15/15 (train/val/test) with stratification
    3. Compute class weights to handle 10:1 imbalance
    4. Build and train CNN for up to 50 epochs with early stopping
    5. Evaluate on test set and save metrics/visualisations
    6. Save final model
    
    Outputs
    -------
    - models/clearcut_classifier_best.keras (best checkpoint)
    - models/clearcut_classifier_final.keras (final model)
    - outputs/figures/training_history.png (learning curves)
    - outputs/figures/confusion_matrix.png (test set confusion matrix)
    - outputs/model_evaluation.txt (precision/recall/F1 scores)
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)
    
    # Create output directories if they don't exist
    os.makedirs("models", exist_ok = True)
    os.makedirs("outputs/figures", exist_ok = True)
    
    print("=" * 70)
    print("PHASE 3: CNN TRAINING FOR CLEARCUT DETECTION")
    print("=" * 70)
    print()
    
    #  =====  STEP 1: LOAD DATASET  ===== 
    X, y, df  =  load_dataset()
    
    #  =====  STEP 2: TRAIN/VAL/TEST SPLIT (70/15/15)  ===== 
    # First split: 70% train, 30% temp (val+test)
    X_train, X_temp, y_train, y_temp  =  train_test_split(
        X, y, test_size = 0.30, stratify = y, random_state = 42
    )
    
    # Second split: divide temp into 50/50 (15% val, 15% test)
    X_val, X_test, y_val, y_test  =  train_test_split(
        X_temp, y_temp, test_size = 0.50, stratify = y_temp, random_state = 42
    )
    
    print("Data split (stratified by class):")
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} clearcut, {len(y_train)-y_train.sum()} not-clearcut)")
    print(f"  Val:   {len(X_val)} samples ({y_val.sum()} clearcut, {len(y_val)-y_val.sum()} not-clearcut)")
    print(f"  Test:  {len(X_test)} samples ({y_test.sum()} clearcut, {len(y_test)-y_test.sum()} not-clearcut)")
    print()
    
    #  =====  STEP 3: COMPUTE CLASS WEIGHTS FOR IMBALANCE  ===== 
    # With 10:1 ratio, clearcut class gets ~10× higher loss penalty
    class_weights_array  =  compute_class_weight(
        "balanced", classes = np.unique(y_train), y = y_train
    )
    class_weights  =  {0: class_weights_array[0], 1: class_weights_array[1]}
    
    print(f"Class weights (to handle 10:1 imbalance):")
    print(f"  Not-clearcut (0): {class_weights[0]:.3f}")
    print(f"  Clearcut (1):     {class_weights[1]:.3f}")
    print()
    
    #  =====  STEP 4: BUILD MODEL  ===== 
    print("Building CNN model...")
    model  =  build_model()
    model.summary()
    print()
    
    #  =====  STEP 5: SET UP TRAINING CALLBACKS  ===== 
    callbacks  =  [
        # Stop training if validation loss doesn't improve for 10 epochs
        EarlyStopping(
            monitor = "val_loss",
            patience = 10,
            restore_best_weights = True,
            verbose = 1,
        ),
        # Save best model (lowest validation loss) during training
        ModelCheckpoint(
            "models/clearcut_classifier_best.keras",
            monitor = "val_loss",
            save_best_only = True,
            verbose = 1,
        ),
    ]
    
    #  =====  STEP 6: TRAIN MODEL  ===== 
    print("Training model...")
    print("(Early stopping will halt if validation loss doesn't improve for 10 epochs)")
    print()
    
    history  =  model.fit(
        X_train,
        y_train,
        validation_data = (X_val, y_val),
        epochs = 50,
        batch_size = 16,
        class_weight = class_weights,  # Apply class weights to handle imbalance
        callbacks = callbacks,
        verbose = 1,
    )
    
    print()
    print("Training complete!")
    print()
    
    #  =====  STEP 7: PLOT TRAINING HISTORY  ===== 
    plot_training_history(history)
    
    #  =====  STEP 8: EVALUATE ON TEST SET  ===== 
    print("=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    print()
    
    # Predict on test set (returns probabilities [0, 1])
    y_pred_probs  =  model.predict(X_test, verbose = 0).flatten()
    
    # Threshold at 0.5 to get binary predictions
    y_pred  =  (y_pred_probs > 0.5).astype(int)
    
    # Compute metrics
    precision  =  precision_score(y_test, y_pred, zero_division = 0)
    recall  =  recall_score(y_test, y_pred, zero_division = 0)
    f1  =  f1_score(y_test, y_pred, zero_division = 0)
    
    print("Overall Metrics:")
    print(f"  Precision (clearcut): {precision:.3f}")
    print(f"  Recall (clearcut):    {recall:.3f}")
    print(f"  F1-score:             {f1:.3f}")
    print()
    
    print("Confusion Matrix:")
    cm  =  confusion_matrix(y_test, y_pred)
    print(f"                  Predicted")
    print(f"                  Not-Clear  Clearcut")
    print(f"Actual Not-Clear  {cm[0,0]:9d}  {cm[0,1]:8d}")
    print(f"       Clearcut   {cm[1,0]:9d}  {cm[1,1]:8d}")
    print()
    
    print("Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names = ["Not-Clearcut", "Clearcut"],
        zero_division = 0,
    ))
    
    #  =====  STEP 9: PLOT CONFUSION MATRIX  ===== 
    plot_confusion_matrix(y_test, y_pred)
    
    #  =====  STEP 10: SAVE EVALUATION METRICS TO TEXT FILE  ===== 
    eval_output_path  =  "outputs/model_evaluation.txt"
    with open(eval_output_path, "w") as f:
        f.write("PHASE 3: CNN CLEARCUT CLASSIFIER - TEST SET EVALUATION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Dataset:\n")
        f.write(f"  Total samples: {len(X)}\n")
        f.write(f"  Clearcut: {y.sum()} ({100*y.mean():.1f}%)\n")
        f.write(f"  Not-clearcut: {len(y)-y.sum()} ({100*(1-y.mean()):.1f}%)\n\n")
        
        f.write("Train/Val/Test Split:\n")
        f.write(f"  Train: {len(X_train)} samples\n")
        f.write(f"  Val:   {len(X_val)} samples\n")
        f.write(f"  Test:  {len(X_test)} samples\n\n")
        
        f.write("Test Set Metrics:\n")
        f.write(f"  Precision (clearcut): {precision:.3f}\n")
        f.write(f"  Recall (clearcut):    {recall:.3f}\n")
        f.write(f"  F1-score:             {f1:.3f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"                  Predicted\n")
        f.write(f"                  Not-Clear  Clearcut\n")
        f.write(f"Actual Not-Clear  {cm[0,0]:9d}  {cm[0,1]:8d}\n")
        f.write(f"       Clearcut   {cm[1,0]:9d}  {cm[1,1]:8d}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(
            y_test,
            y_pred,
            target_names = ["Not-Clearcut", "Clearcut"],
            zero_division = 0,
        ))
    
    print(f"Saved evaluation metrics to {eval_output_path}")
    print()
    
    #  =====  STEP 11: SAVE FINAL MODEL  ===== 
    final_model_path  =  "models/clearcut_classifier_final.keras"
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
    print()

    print(f"Key result: Clearcut recall  =  {recall:.1%} (target: >80%)")
    if recall < 0.70:
        print("  ⚠ Recall below 70% — consider oversampling or data augmentation")
    elif recall < 0.80:
        print("  ⚠ Recall below 80% — model performance acceptable but could improve")
    else:
        print("  ✓ Recall meets target (>80%)")


if __name__  ==  "__main__":
    main()