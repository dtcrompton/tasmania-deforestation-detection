"""
Extract 128x128 pixel patches from Sentinel-2 mosaic tiles.
Each patch is centred on a Hansen loss point and saved as an individual GeoTIFF.
"""

import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import glob
import os
from pathlib import Path

# Patch size in pixels
PATCH_SIZE = 128

def find_tile_for_point(lon, lat, year, tiles_dict):
    """
    Find which mosaic tile contains the given coordinate.
    
    Args:
        lon: Longitude of loss point (EPSG:4326)
        lat: Latitude of loss point (EPSG:4326)
        year: Loss year (2019-2024)
        tiles_dict: Dictionary mapping year to list of tile file paths
    
    Returns:
        Path to the tile containing this point, or None if not found
    """
    from pyproj import Transformer
    
    year_tiles = tiles_dict.get(year, [])
    
    # Transform lon/lat (EPSG:4326) to UTM Zone 55S (EPSG:32755)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32755", always_xy=True)
    x, y = transformer.transform(lon, lat)
    
    for tile_path in year_tiles:
        with rasterio.open(tile_path) as src:
            # Check if point falls within this tile's bounds
            bounds = src.bounds
            if (bounds.left <= x <= bounds.right and 
                bounds.bottom <= y <= bounds.top):
                return tile_path
    
    return None
    
def extract_patch(tile_path, lon, lat, point_id, year, output_dir):
    """
    Extract a 128x128 patch centred on the given coordinate.
    
    Args:
        tile_path: Path to the mosaic tile GeoTIFF
        lon: Longitude of patch centre
        lat: Latitude of patch centre
        point_id: Unique identifier for this loss point
        year: Loss year
        output_dir: Directory to save extracted patch
    
    Returns:
        Path to saved patch, or None if extraction failed
    """
    from pyproj import Transformer
    
    with rasterio.open(tile_path) as src:
        # Transform lon/lat to tile CRS (EPSG:32755)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32755", always_xy=True)
        x, y = transformer.transform(lon, lat)
        
        # Convert UTM coordinates to pixel coordinates
        row, col = src.index(x, y)
        
        # Calculate patch bounds (128x128 window centred on point)
        half_size = PATCH_SIZE // 2
        row_start = row - half_size
        col_start = col - half_size
        
        # Check if patch would extend beyond tile bounds
        if (row_start < 0 or col_start < 0 or 
            row_start + PATCH_SIZE > src.height or 
            col_start + PATCH_SIZE > src.width):
            print(f"Warning: Point {point_id} too close to tile edge, skipping")
            return None
        
        # Define the window to read
        window = rasterio.windows.Window(col_start, row_start, PATCH_SIZE, PATCH_SIZE)
        
        # Read all 5 bands for this window
        patch_data = src.read(window=window)
        
        # Create output filename
        output_filename = f"patch_{year}_{int(point_id):04d}.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        # Get transform for this window
        transform = src.window_transform(window)
        
        # Write patch as new GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=PATCH_SIZE,
            width=PATCH_SIZE,
            count=src.count,
            dtype=src.dtypes[0],
            crs=src.crs,
            transform=transform
        ) as dst:
            dst.write(patch_data)
            # Copy band descriptions
            for i in range(src.count):
                dst.set_band_description(i + 1, src.descriptions[i])
        
        return output_path

def main():
    """Extract all patches from annual composite GeoTIFFs."""
    
    # Load study region loss points
    df = pd.read_csv('data/processed/loss_points_study_region.csv')
    print(f"Loaded {len(df)} loss points (study region)")
    
    # Create output directory
    output_dir = 'data/training/patches'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build dictionary of composites by year (handling tiled exports)
    composite_files = {}
    for year in range(2019, 2025):
        pattern = f'data/raw/sentinel2_study_region_{year}*.tif'
        tiles = sorted(glob.glob(pattern))
        if tiles:
            composite_files[year] = tiles
            print(f"Year {year}: found {len(tiles)} tiles")
        else:
            print(f"Warning: No tiles found for year {year}")
    print(f"Found composites for years: {sorted(composite_files.keys())}")
    
    # Extract patches
    successful = 0
    failed = 0
    
    for idx, row in df.iterrows():
        point_id = int(row['id'])
        year = 2000 + int(row['lossyear'])
        lon = row['longitude']
        lat = row['latitude']
        
# Get the tiles for this year
        year_tiles = composite_files.get(year)
        
        if year_tiles is None:
            print(f"Warning: No tiles found for year {year}, skipping point {point_id}")
            failed += 1
            continue
        
        # Find which tile contains this point
        tiles_dict = {year: year_tiles}
        tile_path = find_tile_for_point(lon, lat, year, tiles_dict)
        
        if tile_path is None:
            print(f"Warning: No tile found for point {point_id} (year {year})")
            failed += 1
            continue
        
        # Extract the patch
        output_path = extract_patch(tile_path, lon, lat, point_id, year, output_dir)
        
        if output_path:
            successful += 1
            if successful % 100 == 0:
                print(f"Extracted {successful} patches...")
        else:
            failed += 1
    
    print(f"\nExtraction complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")

if __name__ == '__main__':
    main()