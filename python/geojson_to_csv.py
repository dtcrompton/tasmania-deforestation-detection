import geopandas as gpd

gdf = gpd.read_file('/Users/danielcrompton/Documents/Learning/GIS/Projects/tasmania-deforestation-detection/data/raw/tasmania_loss_points_2019_2024.geojson')
gdf['longitude'] = gdf.geometry.x
gdf['latitude'] = gdf.geometry.y
gdf.drop(columns='geometry').to_csv('/Users/danielcrompton/Documents/Learning/GIS/Projects/tasmania-deforestation-detection/data/raw/tasmania_loss_points_2019_2024.csv', index=False)