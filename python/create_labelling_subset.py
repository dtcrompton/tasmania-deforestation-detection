"""
Create labelling subset for manual annotation.
Targets ~300 patches: enough for CNN training, manageable labelling time.
"""

import pandas as pd
from pathlib import Path

def create_labelling_subset():
    """Select ~300 patches for manual labelling, stratified by year."""
    
    # Load study region points
    df = pd.read_csv('data/processed/loss_points_study_region.csv')
    
    # Check which patches actually exist
    patch_dir = Path('data/training/patches')
    df['year'] = 2000 + df['lossyear'].astype(int)
    df['patch_exists'] = df.apply(
        lambda row: (patch_dir / f"patch_{int(row['year'])}_{int(row['id']):04d}.tif").exists(),
        axis=1
    )
    
    df_existing = df[df['patch_exists']].copy()
    
    print(f"Total patches available: {len(df_existing)}")
    print(f"\nYear distribution:")
    print(df_existing.groupby('year').size())
    print()
    
    # Sampling strategy for ~300 patches:
    # - Take ALL from sparse years (2021, 2022, 2024): 90 total
    # - Sample from dense years: 215 remaining
    #   - 2019: 100 (of 318)
    #   - 2020: 25 (of 67)
    #   - 2023: 100 (of 261)
    # Total: 315 patches
    
    sparse_years = [2021, 2022, 2024]
    dense_samples = {2019: 100, 2020: 25, 2023: 100}
    
    sampled_rows = []
    
    # Take all from sparse years
    for year in sparse_years:
        year_data = df_existing[df_existing['year'] == year]
        sampled_rows.append(year_data)
        print(f"{year}: sampled {len(year_data)} (all available)")
    
    # Sample from dense years
    for year, sample_size in dense_samples.items():
        year_data = df_existing[df_existing['year'] == year]
        available = len(year_data)
        n = min(sample_size, available)
        sampled = year_data.sample(n=n, random_state=42)
        sampled_rows.append(sampled)
        print(f"{year}: sampled {n} of {available}")
    
    # Combine
    df_labelling = pd.concat(sampled_rows, ignore_index=True)
    
    print(f"\nTotal for labelling: {len(df_labelling)}")
    
    # Save
    output_path = 'data/training/labelling_subset_final.csv'
    df_labelling[['id', 'year', 'lossyear', 'longitude', 'latitude']].to_csv(output_path, index=False)
    
    print(f"Saved to: {output_path}")

if __name__ == '__main__':
    create_labelling_subset()