"""
Create a stratified subset of patches for manual labelling.
Selects 100 patches per year (600 total) to ensure temporal balance.
"""

import pandas as pd
import os
from pathlib import Path

def create_labelling_subset():
    """Select 100 patches per year for manual labelling."""
    
    # Load the filtered loss points
    df = pd.read_csv('data/processed/loss_points_filtered.csv')
    
    # Convert lossyear to full year
    df['year'] = 2000 + df['lossyear'].astype(int)
    
    # Sample 100 points per year
    sampled_rows = []
    for year in range(2019, 2025):
        year_data = df[df['year'] == year]
        
        # Random sample of 100 (or all if fewer than 100)
        sample_size = min(100, len(year_data))
        sampled = year_data.sample(n=sample_size, random_state=42)
        sampled_rows.append(sampled)
        
        print(f"{year}: sampled {len(sampled)} patches")
    
    # Combine all samples
    df_subset = pd.concat(sampled_rows, ignore_index=True)
    
    # Check which patches actually exist (some may have failed extraction)
    patch_dir = Path('data/training/patches')
    df_subset['patch_exists'] = df_subset.apply(
        lambda row: (patch_dir / f"patch_{int(row['year'])}_{int(row['id']):04d}.tif").exists(),
        axis=1
    )
    
    df_existing = df_subset[df_subset['patch_exists']].copy()
    df_missing = df_subset[~df_subset['patch_exists']].copy()
    
    print(f"\nTotal sampled: {len(df_subset)}")
    print(f"Patches exist: {len(df_existing)}")
    print(f"Patches missing (edge failures): {len(df_missing)}")
    
    # Save the subset metadata
    output_path = 'data/training/labelling_subset.csv'
    df_existing[['id', 'year', 'longitude', 'latitude']].to_csv(output_path, index=False)
    
    print(f"\nLabelling subset saved to: {output_path}")
    print(f"Ready for manual labelling: {len(df_existing)} patches")

if __name__ == '__main__':
    create_labelling_subset()