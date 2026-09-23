#!/usr/bin/env python3
"""
Interactive validation tool for model predictions.

Displays patches from validation sample, allows manual review via keyboard,
and calculates validation metrics.

Usage:
    python3 python/validate_predictions.py

Controls:
    1 = Correct prediction
    0 = Incorrect prediction
    s = Skip this patch
    q = Quit and save results
"""

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import os


def save_patch_image(patch_id, year, predicted_prob, predicted_class, patches_dir = "data/training/patches", output_dir = "outputs/temp_validation"):
    """
    Save patch visualization to file for external viewing.
    
    Parameters
    ----------
    patch_id : int
        Patch identifier
    year : int
        Year of forest loss
    predicted_prob : float
        Model's confidence (0-1)
    predicted_class : str
        Predicted class ('clearcut' or 'not_clearcut')
    patches_dir : str
        Directory containing patch GeoTIFFs
    output_dir : str
        Directory to save visualization
    
    Returns
    -------
    Path
        Path to saved image file
    """
    patch_path = Path(patches_dir) / f"patch_{year}_{patch_id:04d}.tif"
    
    with rasterio.open(patch_path) as src:
        # Read bands: [B4=Red, B3=Green, B2=Blue, B8=NIR, B11=SWIR1]
        patch = src.read()
    
    # Transpose to (height, width, bands)
    patch = np.transpose(patch, (1, 2, 0))
    
    # Normalize for display (0-10,000 DN → 0-1 range, then stretch)
    patch_norm = patch / 10000.0
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    
    # True-colour RGB
    rgb = patch_norm[:, :, [2, 1, 0]]  # B4=Red, B3=Green, B2=Blue
    rgb_stretched = np.clip(rgb * 3.0, 0, 1)
    axes[0].imshow(rgb_stretched)
    axes[0].set_title("True-Colour RGB", fontsize = 12)
    axes[0].axis("off")
    
    # False-colour composite (NIR-Red-Green)
    false_colour = patch_norm[:, :, [3, 2, 1]]  # B8=NIR, B4=Red, B3=Green
    false_colour_stretched = np.clip(false_colour * 3.0, 0, 1)
    axes[1].imshow(false_colour_stretched)
    axes[1].set_title("False-Colour (NIR-Red-Green)", fontsize = 12)
    axes[1].axis("off")
    
    # Add prediction information
    confidence_pct = predicted_prob * 100
    fig.suptitle(
        f"Patch {patch_id} ({year}) | Predicted: {predicted_class} ({confidence_pct:.1f}% confidence)",
        fontsize = 14,
        fontweight = "bold"
    )
    
    plt.tight_layout()
    
    # Save to file
    os.makedirs(output_dir, exist_ok = True)
    output_path = Path(output_dir) / f"patch_{patch_id}_{year}.png"
    plt.savefig(output_path, dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    
    return output_path


def validate_sample(sample_path = "outputs/maps/validation_sample.csv", n_samples = 15):
    """
    Interactive validation of prediction sample.
    
    Parameters
    ----------
    sample_path : str
        Path to validation sample CSV
    n_samples : int
        Number of samples to review
    
    Returns
    -------
    pandas.DataFrame
        Validation results
    """
    # Load validation sample
    sample_df = pd.read_csv(sample_path)
    
    print("=" * 70)
    print("PREDICTION VALIDATION TOOL")
    print("=" * 70)
    print()
    print(f"Loaded {len(sample_df)} patches from validation sample")
    print(f"Reviewing first {n_samples} patches...")
    print()
    print("For each patch image that opens:")
    print("  - Left panel = True-colour RGB (natural view)")
    print("  - Right panel = False-colour NIR composite (vegetation emphasis)")
    print("  - Clearcuts appear as geometric shapes with bare soil")
    print("  - Fire appears as irregular boundaries with partial canopy")
    print()
    print("Controls:")
    print("  1 = Prediction is CORRECT (press 1 then Enter)")
    print("  0 = Prediction is INCORRECT (press 0 then Enter)")
    print("  s = Skip this patch (press s then Enter)")
    print("  q = Quit and save results (press q then Enter)")
    print()
    print("=" * 70)
    print()
    
    # Review subset
    review_df = sample_df.head(n_samples).copy()
    
    results = []
    
    for idx, row in review_df.iterrows():
        # Save patch image
        image_path = save_patch_image(
            row["patch_id"],
            row["year"],
            row["predicted_prob"],
            row["predicted_class"]
        )
        
        # Open image in default viewer
        subprocess.run(["open", str(image_path)])
        
        # Get user input
        print(f"\nPatch {row['patch_id']} ({row['year']}) - Predicted: {row['predicted_class']} ({row['predicted_prob']:.2%} confidence)")
        
        while True:
            response = input("Your assessment (1=correct, 0=incorrect, s=skip, q=quit): ").strip().lower()
            
            if response == "1":
                user_label = "correct"
                print("  ✓ Marked as CORRECT")
                break
            elif response == "0":
                user_label = "incorrect"
                print("  ✗ Marked as INCORRECT")
                break
            elif response == "s":
                user_label = "skip"
                print("  → Skipped")
                break
            elif response == "q":
                print("\nQuitting validation...")
                return pd.DataFrame(results)
            else:
                print("  Invalid input. Press 1, 0, s, or q then Enter")
        
        # Record result
        results.append({
            "patch_id": row["patch_id"],
            "year": row["year"],
            "predicted_class": row["predicted_class"],
            "predicted_prob": row["predicted_prob"],
            "user_label": user_label
        })
    
    return pd.DataFrame(results)


def calculate_validation_metrics(results_df):
    """
    Calculate validation metrics from user labels.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        Validation results
    
    Returns
    -------
    dict
        Validation metrics
    """
    # Filter out skipped patches
    reviewed = results_df[results_df["user_label"] != "skip"].copy()
    
    if len(reviewed) == 0:
        print("No patches reviewed (all skipped)")
        return {}
    
    # Overall accuracy
    correct = (reviewed["user_label"] == "correct").sum()
    total = len(reviewed)
    accuracy = correct / total
    
    # Clearcut-specific metrics
    clearcut_reviews = reviewed[reviewed["predicted_class"] == "clearcut"]
    
    if len(clearcut_reviews) > 0:
        clearcut_correct = (clearcut_reviews["user_label"] == "correct").sum()
        clearcut_precision = clearcut_correct / len(clearcut_reviews)
    else:
        clearcut_precision = None
        clearcut_correct = 0
    
    metrics = {
        "total_reviewed": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "clearcut_reviewed": len(clearcut_reviews),
        "clearcut_correct": clearcut_correct,
        "clearcut_precision": clearcut_precision
    }
    
    return metrics


def main():
    """Run interactive validation and save results."""
    
    # Validate sample
    results_df = validate_sample(n_samples = 15)
    
    if len(results_df) == 0:
        print("No validation results to save")
        return
    
    # Save results
    output_path = "outputs/validation_results.csv"
    results_df.to_csv(output_path, index = False)
    print(f"\nSaved validation results to {output_path}")
    
    # Calculate metrics
    metrics = calculate_validation_metrics(results_df)
    
    if metrics:
        print("\n" + "=" * 70)
        print("VALIDATION METRICS")
        print("=" * 70)
        print()
        print(f"Total patches reviewed: {metrics['total_reviewed']}")
        print(f"  Correct predictions:   {metrics['correct']} ({100 * metrics['accuracy']:.1f}%)")
        print(f"  Incorrect predictions: {metrics['incorrect']}")
        print()
        
        if metrics["clearcut_precision"] is not None:
            print(f"Clearcut predictions reviewed: {metrics['clearcut_reviewed']}")
            print(f"  Correct:   {metrics['clearcut_correct']}")
            print(f"  Precision: {100 * metrics['clearcut_precision']:.1f}%")
        
        print("\n" + "=" * 70)
        
        # Save metrics
        metrics_path = "outputs/validation_metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("VALIDATION METRICS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total patches reviewed: {metrics['total_reviewed']}\n")
            f.write(f"Correct predictions: {metrics['correct']} ({100 * metrics['accuracy']:.1f}%)\n")
            f.write(f"Incorrect predictions: {metrics['incorrect']}\n\n")
            
            if metrics["clearcut_precision"] is not None:
                f.write(f"Clearcut predictions reviewed: {metrics['clearcut_reviewed']}\n")
                f.write(f"Clearcut correct: {metrics['clearcut_correct']}\n")
                f.write(f"Clearcut precision: {100 * metrics['clearcut_precision']:.1f}%\n")
        
        print(f"Saved metrics to {metrics_path}")
        
        # Clean up temp images
        print("\nCleaning up temporary images...")
        temp_dir = Path("outputs/temp_validation")
        if temp_dir.exists():
            for img in temp_dir.glob("*.png"):
                img.unlink()
            temp_dir.rmdir()


if __name__ == "__main__":
    main()