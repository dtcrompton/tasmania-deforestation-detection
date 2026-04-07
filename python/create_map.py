#!/usr/bin/env python3
"""
Generate interactive Folium map for Tasmania deforestation detection project.

Inputs:
    - outputs/maps/predictions_with_tenure.geojson (predictions with tenure flags)
    - data/permits/ptpz.geojson (PTPZ boundaries)
    - data/permits/reserve_estate.geojson (reserve boundaries)

Output:
    - outputs/maps/tasmania_deforestation_map.html (interactive web map)

Map layers:
    - Loss points (colour-coded by predicted class)
    - PTPZ boundaries (blue outlines)
    - Reserve boundaries (green outlines)
    - Flagged clearcuts (red markers with yellow halo)
"""

import geopandas as gpd
import folium
from folium import plugins
import json


def create_base_map(center = [-42.7, 146.0], zoom = 9):
    """
    Create base Folium map centered on study region.
    """
    m = folium.Map(
        location = center,
        zoom_start = zoom,
        tiles = "CartoDB positron"
    )
    
    return m


def add_tenure_boundaries(m, ptpz_path = "data/permits/ptpz.geojson", reserves_path = "data/permits/reserve_estate.geojson"):
    """
    Add PTPZ and reserve boundary layers to map.
    
    Parameters
    ----------
    m : folium.Map
        Map object
    ptpz_path : str
        Path to PTPZ boundaries GeoJSON
    reserves_path : str
        Path to reserve boundaries GeoJSON
    
    Returns
    -------
    folium.Map
        Map with boundary layers added
    
    Notes
    -----
    Geometries are simplified to reduce file size and improve rendering performance.
    """
    # Load boundaries
    print("  Loading PTPZ boundaries...")
    ptpz = gpd.read_file(ptpz_path)
    print("  Loading reserve boundaries...")
    reserves = gpd.read_file(reserves_path)
    
    # Reproject to WGS84 if needed
    ptpz = ptpz.to_crs("EPSG:4326")
    reserves = reserves.to_crs("EPSG:4326")
    
    # Simplify geometries to reduce file size (tolerance in degrees, ~100m)
    print("  Simplifying geometries...")
    ptpz["geometry"] = ptpz["geometry"].simplify(tolerance = 0.001, preserve_topology = True)
    reserves["geometry"] = reserves["geometry"].simplify(tolerance = 0.001, preserve_topology = True)
    
    # Keep only geometry column
    ptpz = ptpz[["geometry"]].copy()
    reserves = reserves[["geometry"]].copy()
    
    # Dissolve into single multi-polygon to reduce feature count
    print("  Dissolving PTPZ into single feature...")
    ptpz_dissolved = ptpz.dissolve()
    print("  Dissolving reserves into single feature...")
    reserves_dissolved = reserves.dissolve()
    
    # Add PTPZ layer (blue outlines)
    ptpz_layer = folium.FeatureGroup(name = "PTPZ Boundaries (Permitted Logging)", show = True)
    
    folium.GeoJson(
        ptpz_dissolved,
        style_function = lambda x: {
            "fillColor": "blue",
            "color": "blue",
            "weight": 1.5,
            "fillOpacity": 0.1
        },
        tooltip = "PTPZ (Permitted Logging Zone)"
    ).add_to(ptpz_layer)
    
    ptpz_layer.add_to(m)
    
    # Add reserve layer (green outlines)
    reserves_layer = folium.FeatureGroup(name = "Reserve Boundaries (Prohibited Logging)", show = True)
    
    folium.GeoJson(
        reserves_dissolved,
        style_function = lambda x: {
            "fillColor": "green",
            "color": "green",
            "weight": 1.5,
            "fillOpacity": 0.1
        },
        tooltip = "Protected Reserve"
    ).add_to(reserves_layer)
    
    reserves_layer.add_to(m)
    
    print(f"  Added PTPZ boundaries (dissolved from {len(ptpz)} features)")
    print(f"  Added reserve boundaries (dissolved from {len(reserves)} features)")
    
    return m


def add_loss_points(m, predictions_path = "outputs/maps/predictions_with_tenure.geojson"):
    """
    Add forest loss points to map, colour-coded by predicted class.
    
    Parameters
    ----------
    m : folium.Map
        Map object
    predictions_path : str
        Path to predictions GeoJSON with tenure flags
    
    Returns
    -------
    folium.Map
        Map with loss points added
    """
    # Load predictions
    predictions = gpd.read_file(predictions_path)
    
    # Create feature groups for each class
    clearcut_layer = folium.FeatureGroup(name = "Clearcut Predictions", show = True)
    not_clearcut_layer = folium.FeatureGroup(name = "Not-Clearcut Predictions", show = True)
    flagged_layer = folium.FeatureGroup(name = "Flagged Clearcuts (Outside Permitted Zones)", show = True)
    
    # Add markers
    for idx, row in predictions.iterrows():
        # Determine colour and layer
        if row["flagged"]:
            # Flagged clearcuts (yellow halo with red center)
            color = "red"
            layer = flagged_layer
            icon = folium.Icon(color = "red", icon = "exclamation-triangle", prefix = "fa")
        elif row["predicted_class"] == "clearcut":
            color = "red"
            layer = clearcut_layer
            icon = folium.Icon(color = "red", icon = "tree", prefix = "fa")
        else:
            color = "orange"
            layer = not_clearcut_layer
            icon = folium.Icon(color = "orange", icon = "fire", prefix = "fa")
        
        # Create popup content
        tenure_status = []
        if row["in_ptpz"]:
            tenure_status.append("Inside PTPZ (permitted)")
        if row["in_reserve"]:
            tenure_status.append("Inside reserve (prohibited)")
        if not row["in_ptpz"] and not row["in_reserve"]:
            tenure_status.append("Outside permitted zones")
        
        tenure_text = ", ".join(tenure_status)
        
        popup_html = f"""
        <div style="font-family: sans-serif; width: 200px;">
            <h4 style="margin: 0 0 10px 0; color: #2d2d2d;">Patch {row['patch_id']} ({row['year']})</h4>
            <p style="margin: 5px 0;"><strong>Predicted:</strong> {row['predicted_class']}</p>
            <p style="margin: 5px 0;"><strong>Confidence:</strong> {row['predicted_prob']:.1%}</p>
            <p style="margin: 5px 0;"><strong>Tenure:</strong> {tenure_text}</p>
            <p style="margin: 5px 0; font-size: 0.9em; color: #666;">
                Lat: {row['latitude']:.4f}<br>
                Lon: {row['longitude']:.4f}
            </p>
        </div>
        """
        
        # Add marker
        folium.Marker(
            location = [row["latitude"], row["longitude"]],
            popup = folium.Popup(popup_html, max_width = 250),
            icon = icon
        ).add_to(layer)
    
    # Add layers to map
    not_clearcut_layer.add_to(m)
    clearcut_layer.add_to(m)
    flagged_layer.add_to(m)
    
    print(f"Added {len(predictions)} loss points to map")
    print(f"  Clearcut: {(predictions['predicted_class'] == 'clearcut').sum()}")
    print(f"  Not-clearcut: {(predictions['predicted_class'] == 'not_clearcut').sum()}")
    print(f"  Flagged: {predictions['flagged'].sum()}")
    
    return m


def add_map_controls(m, predictions_gdf):
    """
    Add layer control, recentre button, title, and legend to map.
    
    Parameters
    ----------
    m : folium.Map
        Map object
    predictions_gdf : geopandas.GeoDataFrame
        Predictions data for statistics
    
    Returns
    -------
    folium.Map
        Map with controls added
    """
    # Layer control
    folium.LayerControl(position = "topright", collapsed = False).add_to(m)
    
    # Add fullscreen button at top-left (below zoom controls)
    plugins.Fullscreen(position = "topleft").add_to(m)
    
    # Push measure control down to avoid fullscreen button
    measure_offset_css = """
    <style>
    .leaflet-control-measure {
        margin-top: 10px !important;
    }
    </style>
    """
    m.get_root().html.add_child(folium.Element(measure_offset_css))
    
    # Add measure control
    scale = plugins.MeasureControl(
        position = 'topleft',
        primary_length_unit = 'kilometers',
        secondary_length_unit = 'miles',
        primary_area_unit = 'sqkilometers',
        secondary_area_unit = 'sqmiles'
    )
    scale.add_to(m)
    
    # Calculate map center for recentre button
    center_lat = -42.7
    center_lon = 146.0
    zoom_level = 9
    
    # Recentre button (positioned to the right of measure control)
    recentre_script = f"""
    <div id="recentre-btn" style="
        position: fixed;
        top: 10px;
        left: 50px;
        z-index: 1000;
        background-color: white;
        border: 2px solid rgba(0,0,0,0.2);
        border-radius: 4px;
        padding: 8px 12px;
        cursor: pointer;
        font-family: Arial, sans-serif;
        font-size: 14px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.2);
    ">
        Recentre Map
    </div>
    
    <script>
    setTimeout(function() {{
        var recentreBtn = document.getElementById('recentre-btn');
        if (recentreBtn) {{
            recentreBtn.onclick = function() {{
                var mapObj = window[Object.keys(window).filter(key => key.startsWith('map_'))[0]];
                if (mapObj) {{
                    mapObj.setView([{center_lat}, {center_lon}], {zoom_level});
                }}
            }};
        }}
    }}, 1000);
    </script>
    """
    m.get_root().html.add_child(folium.Element(recentre_script))
    
    # Calculate statistics
    total_points = len(predictions_gdf)
    clearcut_count = (predictions_gdf["predicted_class"] == "clearcut").sum()
    not_clearcut_count = (predictions_gdf["predicted_class"] == "not_clearcut").sum()
    flagged_count = predictions_gdf["flagged"].sum()
    
    # Title and info box
    info_box_html = f"""
    <div style="
        position: fixed;
        bottom: 20px;
        right: 10px;
        width: 360px;
        background-color: white;
        border: 2px solid rgba(0,0,0,0.2);
        border-radius: 8px;
        padding: 15px;
        z-index: 1000;
        font-family: Arial, sans-serif;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    ">
        <h3 style="margin: 0 0 10px 0; color: #2C3E50;">Tasmania Deforestation Detection</h3>
        <p style="margin: 0 0 8px 0; font-size: 13px; color: #555;">
            CNN-based classification of forest loss (2019–2024) in central Tasmania using Sentinel-2 imagery.
        </p>
        <div style="font-size: 12px; color: #666; line-height: 1.6;">
            <p style="margin: 10px 0 5px 0;"><strong>Predictions:</strong></p>
            <div style="margin-top: 5px;">
                <p style="margin: 3px 0;">
                    <i class="fa fa-tree" style="color: red;"></i> Clearcut: {clearcut_count} ({100*clearcut_count/total_points:.1f}%)
                </p>
                <p style="margin: 3px 0;">
                    <i class="fa fa-fire" style="color: orange;"></i> Not-clearcut: {not_clearcut_count} ({100*not_clearcut_count/total_points:.1f}%)
                </p>
                <p style="margin: 3px 0;">
                    <i class="fa fa-exclamation-triangle" style="color: red;"></i> Flagged: {flagged_count} (outside permitted zones)
                </p>
            </div>
            <p style="margin: 10px 0 5px 0;"><strong>Land Tenure:</strong></p>
            <p style="margin: 3px 0; font-size: 11px;">
                <span style="color: blue;">▬</span> PTPZ (permitted logging)<br>
                <span style="color: green;">▬</span> Reserves (prohibited)
            </p>
        </div>
        <p style="margin: 12px 0 0 0; font-size: 11px; color: #999; border-top: 1px solid #eee; padding-top: 8px;">
            Click markers for details | Toggle layers to filter | Created by Daniel Crompton, 2026
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_box_html))
    
    return m


def main():
    """Generate interactive Folium map."""
    
    print("=" * 70)
    print("PHASE 5: CREATING INTERACTIVE FOLIUM MAP")
    print("=" * 70)
    print()
    
    # Create base map
    print("Creating base map...")
    m = create_base_map()
    print()
    
    # Add tenure boundaries
    print("Adding land tenure boundaries...")
    m = add_tenure_boundaries(m)
    print()
    
    # Add loss points
    print("Adding forest loss points...")
    predictions_gdf = gpd.read_file("outputs/maps/predictions_with_tenure.geojson")
    m = add_loss_points(m)
    print()
    
    # Add controls
    print("Adding map controls and legend...")
    m = add_map_controls(m, predictions_gdf)
    print()
    
    # Save map
    output_path = "outputs/maps/tasmania_deforestation_map.html"
    m.save(output_path)
    
    print("=" * 70)
    print("MAP GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Saved interactive map to {output_path}")
    print(f"File size: {len(open(output_path).read()) / 1024:.1f} KB")
    print()
    print("Open in browser:")
    print(f"  open {output_path}")


if __name__ == "__main__":
    main()