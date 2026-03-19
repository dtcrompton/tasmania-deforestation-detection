"""
Filter loss points to 2019-2024 range (lossyear 19-24).
Creates a clean working dataset aligned with exported Sentinel-2 imagery.
"""

import pandas as pd

def filter_to_recent_years():
    """Load loss points CSV and filter to years 19-24 only."""
    # Load the full dataset
    df = pd.read_csv('data/raw/tasmania_loss_points_2019_2024.csv')
    
    print(f'Original dataset: {len(df)} points')
    print(f'Points with missing lossyear: {df["lossyear"].isna().sum()}')
    
    # Filter to years 19-24 (2019-2024)
    df_filtered = df[df['lossyear'].isin([19, 20, 21, 22, 23, 24])].copy()
    
    print(f'Filtered dataset: {len(df_filtered)} points')
    print(f'\nYear distribution after filtering:')
    print(df_filtered['lossyear'].value_counts().sort_index())
    
    # Save filtered version
    output_path = 'data/processed/loss_points_filtered.csv'
    df_filtered.to_csv(output_path, index=False)
    print(f'\nFiltered dataset saved to: {output_path}')
    
    return df_filtered

if __name__ == '__main__':
    filter_to_recent_years()
