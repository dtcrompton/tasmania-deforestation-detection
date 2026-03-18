"""
Spatial sanity check for Tasmania deforestation detection.
Loads Hansen loss points and cross-references against PTPZ and reserve
estate boundaries to understand the breakdown of forest loss by land category.
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW = Path("data/raw")
PERMITS = Path("data/permits")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

print("Loading data...")
loss = gpd.read_file(RAW / "tasmania_loss_points_2019_2024.geojson")
ptpz = gpd.read_file(PERMITS / "ptpz.geojson")
reserves = gpd.read_file(PERMITS / "reserve_estate.geojson")

print(f"  Loss points: {len(loss)}")
print(f"  PTPZ features: {len(ptpz)}")
print(f"  Reserve features: {len(reserves)}")
print(f"  Loss CRS: {loss.crs}")
print(f"  PTPZ CRS: {ptpz.crs}")
print(f"  Reserves CRS: {reserves.crs}")

print("\nLoss point lossyear sample values:")
print(loss["lossyear"].value_counts().sort_index())

# Reproject all to WGS84 geographic (EPSG:4326) for joining
# Loss points are already in 4326 from GEE export
target_crs = "EPSG:4326"
loss = loss.to_crs(target_crs)
ptpz = ptpz.to_crs(target_crs)
reserves = reserves.to_crs(target_crs)

# --- Filter PTPZ polygons only ---
ptpz_only = ptpz[ptpz["CATEGORY"] == "Permanent Timber Production Zone Land"].copy()
print(f"\nPTPZ polygons: {len(ptpz_only)}")

# --- Spatial join: which loss points fall inside PTPZ? ---
print("\nJoining loss points to PTPZ...")
loss_in_ptpz = gpd.sjoin(loss, ptpz_only[["CATEGORY", "geometry"]],
                          how="left", predicate="within")
loss["in_ptpz"] = loss_in_ptpz["CATEGORY"].notna()

# --- Spatial join: which loss points fall inside reserves? ---
print("Joining loss points to reserves...")
reserve_cols = ["RES_CLASS", "geometry"]
loss_in_reserves = gpd.sjoin(loss, reserves[reserve_cols],
                              how="left", predicate="within")

# A point may match multiple reserve polygons — keep the first match per point
loss_in_reserves = loss_in_reserves[~loss_in_reserves.index.duplicated(keep="first")]
loss["reserve_class"] = loss_in_reserves["RES_CLASS"]
loss["in_reserve"] = loss["reserve_class"].notna()

# --- Classify each loss point ---
def classify(row):
    if row["in_ptpz"]:
        return "Inside PTPZ (expected logging zone)"
    elif row["in_reserve"]:
        return f"Inside reserve: {row['reserve_class']}"
    else:
        return "Outside PTPZ and reserves (flag for investigation)"

loss["classification"] = loss.apply(classify, axis=1)

# --- Summary ---
print("\n--- Loss point classification summary ---")
summary = loss["classification"].value_counts()
for cls, count in summary.items():
    pct = count / len(loss) * 100
    print(f"  {count:>6} ({pct:4.1f}%)  {cls}")

# --- By year ---
print("\n--- Loss by year ---")
loss["year"] = loss["lossyear"] + 2000
year_summary = loss.groupby(["year", "in_ptpz"]).size().unstack(fill_value=0)
year_summary.columns = ["Outside PTPZ", "Inside PTPZ"]
print(year_summary.to_string())

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plt.style.use("seaborn-v0_8-darkgrid")

# Pie chart of classification
top_classes = summary.head(8)
axes[0].pie(top_classes.values, labels=None, autopct="%1.1f%%", startangle=90)
axes[0].set_title("Loss points by land classification", fontsize=13, fontweight="bold")
axes[0].legend(top_classes.index, loc="lower left", fontsize=7)

# Bar chart by year
year_summary.plot(kind="bar", ax=axes[1], color=["#D9947F", "#7FAD87"])
axes[1].set_title("Loss points by year: inside vs outside PTPZ",
                   fontsize=13, fontweight="bold")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Number of loss points")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
out_path = OUTPUTS / "spatial_sanity_check.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")

# --- Save classified points ---
out_geojson = OUTPUTS / "loss_points_classified.geojson"
loss.to_file(out_geojson, driver="GeoJSON")
print(f"Classified points saved to {out_geojson}")
