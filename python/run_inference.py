#!/usr/bin/env python3
"""
Run inference on all 736 patches and cross-reference with land tenure.

Inputs:
    - models/clearcut_classifier_final.keras (trained CNN)
    - data/training/patches/*.tif (736 Sentinel-2 patches)
    - data/processed/loss_points_study_region.csv (747 Hansen loss points)
    - data/permits/ptpz.geojson (PTPZ boundaries)
    - data/permits/reserve_estate.geojson (reserve boundaries)

Outputs:
    - outputs/maps/predictions.csv (predictions with coordinates)
    - outputs/maps/predictions.geojson (predictions as GeoJSON)
    - outputs/maps/predictions_with_tenure.geojson (with PTPZ/reserve flags)
    - outputs/maps/validation_sample.csv (60 patches for manual review)

Process:
    1. Load trained model
    2. Predict on all 736 patches (clearcut probability)
    3. Merge with loss point coordinates
    4. Spatial join with PTPZ and reserve boundaries
    5. Flag clearcuts outside permitted zones
    6. Generate stratified validation sample (60 patches)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from shapely.geometry import Point

from tensorflow import keras


def load_patch(patch_path):
    """
    Load and normalise a single Sentinel-2 patch.
    
    Parameters
    ----------
    patch_path : Path
        Path to GeoTIFF patch file
    
    Returns
    -------
    numpy.ndarray
        Normalised patch of shape (128, 128, 5)
    """
    with rasterio.open(patch_path) as src:
        patch = src.read()
    
    # Transpose from (bands, height, width) to (height, width, bands)
    patch = np.transpose(patch, (1, 2, 0))
    
    # Normalise Sentinel-2 DN values (0-10,000) to [0, 1]
    patch = patch / 10000.0
    
    return patch.astype(np.float32)


def run_inference(model, patches_dir = "data/training/patches"):
    """
    Run inference on all patches in directory.
    
    Parameters
    ----------
    model : keras.Model
        Trained CNN classifier
    patches_dir : str
        Directory containing patch GeoTIFF files
    
    Returns
    -------
    pandas.DataFrame
        Predictions with columns [patch_id, year, predicted_prob, predicted_class]
    
    Notes
    -----
    Patch filenames format: patch_YYYY_NNNN.tif
    Predicted class threshold: probability >= 0.5 → clearcut
    """
    patch_files = sorted(Path(patches_dir).glob("patch_*.tif"))
    
    print(f"Found {len(patch_files)} patches in {patches_dir}")
    print("Running inference...")
    print()
    
    results = []
    
    for i, patch_path in enumerate(patch_files, 1):
        # Parse filename: patch_YYYY_NNNN.tif
        filename = patch_path.stem
        parts = filename.split("_")
        year = int(parts[1])
        patch_id = int(parts[2])
        
        try:
            # Load and predict
            patch = load_patch(patch_path)
            patch_batch = np.expand_dims(patch, axis = 0)  # Add batch dimension
            
            prob = model.predict(patch_batch, verbose = 0)[0, 0]
            pred_class = "clearcut" if prob >= 0.5 else "not_clearcut"
            
            results.append({
                "patch_id": patch_id,
                "year": year,
                "predicted_prob": prob,
                "predicted_class": pred_class,
            })
            
            if i % 100 == 0:
                print(f"  Processed {i}/{len(patch_files)} patches...")
        
        except Exception as e:
            print(f"  Warning: Failed on {patch_path.name}: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    print(f"Inference complete: {len(df)} predictions")
    print()
    
    return df


def merge_with_coordinates(predictions_df, loss_points_path = "data/processed/loss_points_study_region.csv"):
    """
    Merge predictions with Hansen loss point coordinates.
    
    Parameters
    ----------
    predictions_df : pandas.DataFrame
        Predictions from run_inference()
    loss_points_path : str
        Path to CSV with loss point coordinates
    
    Returns
    -------
    pandas.DataFrame
        Predictions with added longitude/latitude columns
    
    Notes
    -----
    Loss points CSV has columns: id, lossyear, longitude, latitude
    Merge on (id == patch_id) AND (lossyear == year)
    Hansen lossyear format: 19 = 2019, 20 = 2020, etc.
    """
    loss_points = pd.read_csv(loss_points_path)
    
    # Convert Hansen lossyear format (19, 20, 21...) to full year (2019, 2020, 2021...)
    loss_points["year"] = 2000 + loss_points["lossyear"].astype(int)
    
    # Rename 'id' to 'patch_id' for consistent merge
    loss_points = loss_points.rename(columns = {"id": "patch_id"})
    
    # Merge on patch_id and year
    merged = predictions_df.merge(
        loss_points[["patch_id", "year", "longitude", "latitude"]],
        on = ["patch_id", "year"],
        how = "left"
    )
    
    # Check for missing coordinates (shouldn't happen if extraction was clean)
    missing = merged["longitude"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} predictions missing coordinates")
    
    return merged


def predictions_to_geodataframe(predictions_df):
    """
    Convert predictions DataFrame to GeoDataFrame with Point geometries.
    
    Parameters
    ----------
    predictions_df : pandas.DataFrame
        Predictions with longitude/latitude columns
    
    Returns
    -------
    geopandas.GeoDataFrame
        Predictions as GeoDataFrame in EPSG:4326
    """
    # Drop rows with missing coordinates
    df = predictions_df.dropna(subset = ["longitude", "latitude"])
    
    # Create Point geometries
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    
    gdf = gpd.GeoDataFrame(df, geometry = geometry, crs = "EPSG:4326")
    
    return gdf


def cross_reference_tenure(predictions_gdf, ptpz_path = "data/permits/ptpz.geojson", reserves_path = "data/permits/reserve_estate.geojson"):
    """
    Spatial join predictions with PTPZ and reserve boundaries.
    
    Parameters
    ----------
    predictions_gdf : geopandas.GeoDataFrame
        Predictions with Point geometries
    ptpz_path : str
        Path to PTPZ boundaries GeoJSON
    reserves_path : str
        Path to reserve boundaries GeoJSON
    
    Returns
    -------
    geopandas.GeoDataFrame
        Predictions with added columns:
        - in_ptpz (bool): True if point falls within PTPZ boundary
        - in_reserve (bool): True if point falls within reserve boundary
        - flagged (bool): True if clearcut prediction outside permitted zones
    
    Notes
    -----
    PTPZ = Private Timber Reserve Zone (logging permitted)
    Reserves = conservation reserves (logging prohibited)
    Flagged = clearcut predicted AND not in PTPZ AND not in reserve
    """
    print("Loading land tenure boundaries...")
    
    # Load PTPZ boundaries
    ptpz = gpd.read_file(ptpz_path)
    ptpz = ptpz.to_crs("EPSG:4326")
    
    # Load reserve boundaries
    reserves = gpd.read_file(reserves_path)
    reserves = reserves.to_crs("EPSG:4326")
    
    print(f"  PTPZ boundaries: {len(ptpz)} features")
    print(f"  Reserve boundaries: {len(reserves)} features")
    print()
    
    # Spatial join with PTPZ (keep all predictions, mark if inside PTPZ)
    print("Performing spatial joins...")
    ptpz_join = gpd.sjoin(
        predictions_gdf,
        ptpz[["geometry"]],
        how = "left",
        predicate = "within"
    )
    predictions_gdf["in_ptpz"] = ptpz_join["index_right"].notna()
    
    # Spatial join with reserves
    reserves_join = gpd.sjoin(
        predictions_gdf,
        reserves[["geometry"]],
        how = "left",
        predicate = "within"
    )
    predictions_gdf["in_reserve"] = reserves_join["index_right"].notna()
    
    # Flag clearcuts outside permitted zones
    # Flagged = clearcut AND not in PTPZ AND not in reserve
    predictions_gdf["flagged"] = (
        (predictions_gdf["predicted_class"] == "clearcut") &
        (~predictions_gdf["in_ptpz"]) &
        (~predictions_gdf["in_reserve"])
    )
    
    print(f"Spatial join complete:")
    print(f"  Points in PTPZ: {predictions_gdf['in_ptpz'].sum()}")
    print(f"  Points in reserves: {predictions_gdf['in_reserve'].sum()}")
    print(f"  Points outside both: {(~predictions_gdf['in_ptpz'] & ~predictions_gdf['in_reserve']).sum()}")
    print()
    
    return predictions_gdf


def generate_validation_sample(predictions_gdf, n_high_conf_clearcut = 20, n_low_conf_clearcut = 20, n_high_conf_not_clearcut = 20):
    """
    Generate stratified sample of predictions for manual validation.
    
    Parameters
    ----------
    predictions_gdf : geopandas.GeoDataFrame
        All predictions with probabilities
    n_high_conf_clearcut : int
        Number of high-confidence clearcut predictions to sample (prob > 0.8)
    n_low_conf_clearcut : int
        Number of low-confidence clearcut predictions to sample (0.5-0.8)
    n_high_conf_not_clearcut : int
        Number of high-confidence not-clearcut predictions to sample (prob < 0.2)
    
    Returns
    -------
    pandas.DataFrame
        Validation sample with columns [patch_id, year, predicted_prob, predicted_class, longitude, latitude]
    
    Notes
    -----
    This sample allows manual review to verify model performance on unseen data.
    Stratified by confidence to check both certain and uncertain predictions.
    """
    # High-confidence clearcut (prob > 0.8)
    high_clearcut = predictions_gdf[predictions_gdf["predicted_prob"] > 0.8].copy()
    
    # Low-confidence clearcut (0.5 <= prob <= 0.8)
    low_clearcut = predictions_gdf[
        (predictions_gdf["predicted_prob"] >= 0.5) &
        (predictions_gdf["predicted_prob"] <= 0.8)
    ].copy()
    
    # High-confidence not-clearcut (prob < 0.2)
    high_not_clearcut = predictions_gdf[predictions_gdf["predicted_prob"] < 0.2].copy()
    
    # Sample from each stratum
    sample_parts = []
    
    if len(high_clearcut) >= n_high_conf_clearcut:
        sample_parts.append(high_clearcut.sample(n = n_high_conf_clearcut, random_state = 42))
    else:
        sample_parts.append(high_clearcut)
        print(f"Warning: Only {len(high_clearcut)} high-conf clearcut samples (requested {n_high_conf_clearcut})")
    
    if len(low_clearcut) >= n_low_conf_clearcut:
        sample_parts.append(low_clearcut.sample(n = n_low_conf_clearcut, random_state = 42))
    else:
        sample_parts.append(low_clearcut)
        print(f"Warning: Only {len(low_clearcut)} low-conf clearcut samples (requested {n_low_conf_clearcut})")
    
    if len(high_not_clearcut) >= n_high_conf_not_clearcut:
        sample_parts.append(high_not_clearcut.sample(n = n_high_conf_not_clearcut, random_state = 42))
    else:
        sample_parts.append(high_not_clearcut)
        print(f"Warning: Only {len(high_not_clearcut)} high-conf not-clearcut samples (requested {n_high_conf_not_clearcut})")
    
    validation_sample = pd.concat(sample_parts, ignore_index = True)
    
    # Sort by probability (descending) for easier review
    validation_sample = validation_sample.sort_values("predicted_prob", ascending = False)
    
    # Keep only essential columns
    validation_sample = validation_sample[[
        "patch_id",
        "year",
        "predicted_prob",
        "predicted_class",
        "longitude",
        "latitude"
    ]].copy()
    
    return validation_sample


def print_summary(predictions_gdf):
    """Print summary statistics of predictions."""
    total = len(predictions_gdf)
    clearcut_count = (predictions_gdf["predicted_class"] == "clearcut").sum()
    clearcut_pct = 100 * clearcut_count / total
    
    flagged_count = predictions_gdf["flagged"].sum()
    
    print("=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print()
    print(f"Total predictions: {total}")
    print(f"  Clearcut:     {clearcut_count} ({clearcut_pct:.1f}%)")
    print(f"  Not-clearcut: {total - clearcut_count} ({100 - clearcut_pct:.1f}%)")
    print()
    
    print("Predictions by year:")
    year_summary = predictions_gdf.groupby("year")["predicted_class"].value_counts().unstack(fill_value = 0)
    if "clearcut" in year_summary.columns and "not_clearcut" in year_summary.columns:
        year_summary["clearcut_pct"] = 100 * year_summary["clearcut"] / (year_summary["clearcut"] + year_summary["not_clearcut"])
        print(year_summary[["clearcut", "not_clearcut", "clearcut_pct"]])
    else:
        print(year_summary)
    print()
    
    print("Land tenure distribution (clearcut predictions only):")
    if clearcut_count > 0:
        clearcut_gdf = predictions_gdf[predictions_gdf["predicted_class"] == "clearcut"]
        in_ptpz = clearcut_gdf["in_ptpz"].sum()
        in_reserve = clearcut_gdf["in_reserve"].sum()
        outside_both = clearcut_count - in_ptpz - in_reserve
        
        print(f"  In PTPZ:         {in_ptpz} ({100 * in_ptpz / clearcut_count:.1f}%)")
        print(f"  In reserves:     {in_reserve} ({100 * in_reserve / clearcut_count:.1f}%)")
        print(f"  Outside both:    {outside_both} ({100 * outside_both / clearcut_count:.1f}%)")
    else:
        print("  No clearcut predictions")
    print()
    
    print(f"Flagged clearcuts (outside permitted zones): {flagged_count}")
    print()


def main():
    """Main Phase 4 pipeline: inference and tenure cross-reference."""
    
    # Create output directories
    os.makedirs("outputs/maps", exist_ok = True)
    
    print("=" * 70)
    print("PHASE 4: INFERENCE AND TENURE CROSS-REFERENCE")
    print("=" * 70)
    print()
    
    # Load trained model
    print("Loading trained model...")
    model = keras.models.load_model("models/clearcut_classifier_final.keras")
    print("Model loaded successfully")
    print()
    
    # Run inference on all patches
    predictions_df = run_inference(model)
    
    # Merge with coordinates
    print("Merging predictions with loss point coordinates...")
    predictions_df = merge_with_coordinates(predictions_df)
    print(f"Merged: {len(predictions_df)} predictions with coordinates")
    print()
    
    # Save predictions CSV
    predictions_csv_path = "outputs/maps/predictions.csv"
    predictions_df.to_csv(predictions_csv_path, index = False)
    print(f"Saved predictions to {predictions_csv_path}")
    print()
    
    # Convert to GeoDataFrame
    print("Converting to GeoDataFrame...")
    predictions_gdf = predictions_to_geodataframe(predictions_df)
    print(f"GeoDataFrame created: {len(predictions_gdf)} points")
    print()
    
    # Save predictions GeoJSON
    predictions_geojson_path = "outputs/maps/predictions.geojson"
    predictions_gdf.to_file(predictions_geojson_path, driver = "GeoJSON")
    print(f"Saved GeoJSON to {predictions_geojson_path}")
    print()
    
    # Cross-reference with land tenure
    predictions_gdf = cross_reference_tenure(predictions_gdf)
    
    # Save predictions with tenure
    tenure_geojson_path = "outputs/maps/predictions_with_tenure.geojson"
    predictions_gdf.to_file(tenure_geojson_path, driver = "GeoJSON")
    print(f"Saved tenure-referenced GeoJSON to {tenure_geojson_path}")
    print()
    
    # Generate validation sample
    print("Generating validation sample (60 patches)...")
    validation_sample = generate_validation_sample(predictions_gdf)
    validation_csv_path = "outputs/maps/validation_sample.csv"
    validation_sample.to_csv(validation_csv_path, index = False)
    print(f"Saved validation sample to {validation_csv_path}")
    print(f"  {len(validation_sample)} patches selected for manual review")
    print()
    
    # Print summary statistics
    print_summary(predictions_gdf)
    
    print("=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)
    print()
    print("Outputs generated:")
    print(f"  - {predictions_csv_path}")
    print(f"  - {predictions_geojson_path}")
    print(f"  - {tenure_geojson_path}")
    print(f"  - {validation_csv_path}")


if __name__ == "__main__":
    main()