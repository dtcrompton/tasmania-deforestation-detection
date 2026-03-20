"""
Inspect GeoTIFF structure to verify Phase 1 export specifications.
"""

import rasterio
import glob
import sys

def inspect_first_geotiff():
    """Open the first GeoTIFF found and print its metadata."""
    files = sorted(glob.glob('data/raw/*.tif'))
    
    if not files:
        print('No GeoTIFF files found in data/raw/')
        sys.exit(1)
    
    filepath = files[0]
    
    with rasterio.open(filepath) as src:
        print(f'File: {filepath}')
        print(f'Dimensions: {src.width} x {src.height}')
        print(f'Bands: {src.count}')
        print(f'CRS: {src.crs}')
        print(f'Bounds: {src.bounds}')
        print(f'Dtype: {src.dtypes[0]}')
        print(f'Nodata value: {src.nodata}')
        
        # Band descriptions if available
        descriptions = [src.descriptions[i] for i in range(src.count)]
        if any(descriptions):
            print(f'Band descriptions: {descriptions}')

if __name__ == '__main__':
    inspect_first_geotiff()
