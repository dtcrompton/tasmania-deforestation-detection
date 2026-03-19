"""
Spatial sanity check for Tasmania deforestation detection.
Cross-references Hansen loss points against PTPZ and reserve estate boundaries.
Produces classification summary and portfolio-quality chart.
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

RAW     = Path("data/raw")
PERMITS = Path("data/permits")
OUTPUTS_FIGURES = Path("outputs/figures")
OUTPUTS_MAPS    = Path("outputs/maps")
OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUTS_MAPS.mkdir(parents=True, exist_ok=True)

# --- Portfolio colours ---
MAUVE      = "#B794D9"
GREEN      = "#7FAD87"
WARM_RED   = "#D9947F"
TEAL       = "#9BC4CB"
MID_GREY   = "#AAAAAA"
DARK_TEXT  = "#1a1a1a"

print("Loading data...")
loss     = gpd.read_file(RAW / "tasmania_loss_points_2019_2024.geojson")
ptpz     = gpd.read_file(PERMITS / "ptpz.geojson")
reserves = gpd.read_file(PERMITS / "reserve_estate.geojson")

print(f"  Loss points:      {len(loss)}")
print(f"  PTPZ features:    {len(ptpz)}")
print(f"  Reserve features: {len(reserves)}")

# --- Reproject to WGS84 ---
target_crs = "EPSG:4326"
loss     = loss.to_crs(target_crs)
ptpz     = ptpz.to_crs(target_crs)
reserves = reserves.to_crs(target_crs)

# --- Filter to 2019-2024 and decode year ---
loss = loss[loss["lossyear"] >= 19].copy()
loss["year"] = (loss["lossyear"] + 2000).astype(int)

ptpz_only = ptpz[ptpz["CATEGORY"] == "Permanent Timber Production Zone Land"].copy()
print(f"\nPTPZ polygons: {len(ptpz_only)}")

# --- Spatial joins ---
print("\nJoining loss points to PTPZ...")
loss_in_ptpz = gpd.sjoin(loss, ptpz_only[["CATEGORY", "geometry"]],
                          how="left", predicate="within")
loss["in_ptpz"] = loss_in_ptpz["CATEGORY"].notna()

print("Joining loss points to reserves...")
loss_in_reserves = gpd.sjoin(loss, reserves[["RES_CLASS", "geometry"]],
                              how="left", predicate="within")
loss_in_reserves = loss_in_reserves[~loss_in_reserves.index.duplicated(keep="first")]
loss["reserve_class"] = loss_in_reserves["RES_CLASS"]
loss["in_reserve"]    = loss["reserve_class"].notna()

# --- Classify each point ---
def classify(row):
    if row["in_ptpz"]:
        return "Inside PTPZ (expected logging zone)"
    elif row["in_reserve"]:
        return f"Inside reserve: {row['reserve_class']}"
    else:
        return "Outside PTPZ and reserves (flag for investigation)"

loss["classification"] = loss.apply(classify, axis=1)

# --- Console summary ---
print("\n--- Loss point classification summary ---")
summary = loss["classification"].value_counts()
total   = len(loss)
for cls, count in summary.items():
    print(f"  {count:>6} ({count/total*100:4.1f}%)  {cls}")

print("\n--- Loss by year ---")
year_data = loss.groupby(["year", "in_ptpz"]).size().unstack(fill_value=0)
year_data.columns = ["Outside PTPZ", "Inside PTPZ"]
print(year_data.to_string())

# --- Save classified points ---
out_geojson = OUTPUTS_MAPS / "loss_points_classified.geojson"
loss.to_file(out_geojson, driver="GeoJSON")
print(f"\nClassified points saved to {out_geojson}")

# =========================================================
# CHART
# =========================================================

# Consolidate pie categories under 1% into "Other reserve types"
threshold = 0.01 * total
main      = summary[summary >= threshold]
other_sum = summary[summary < threshold].sum()
if other_sum > 0:
    main = pd.concat([main, pd.Series({"Other reserve types": other_sum})])

def shorten(label):
    return (label
            .replace("Inside PTPZ (expected logging zone)", "Inside PTPZ")
            .replace("Inside reserve: ", "")
            .replace("Outside PTPZ and reserves (flag for investigation)",
                     "Outside PTPZ & reserves"))

def assign_colour(label):
    short = shorten(label)
    if short == "Inside PTPZ":
        return GREEN
    elif "Outside" in short:
        return WARM_RED
    elif "Conservation" in short:
        return MAUVE
    elif "Regional" in short:
        return TEAL
    else:
        return MID_GREY

pie_colours = [assign_colour(label) for label in main.index]

fig = plt.figure(figsize=(16, 7), facecolor="white")
fig.patch.set_facecolor("white")
gs = GridSpec(1, 2, figure=fig, wspace=0.35)

ax_pie = fig.add_subplot(gs[0])
ax_bar = fig.add_subplot(gs[1])

# --- Pie ---
wedges, texts, autotexts = ax_pie.pie(
    main.values,
    colors=pie_colours,
    autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
    startangle=90,
    pctdistance=0.78,
    wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_color(DARK_TEXT)
    at.set_fontweight("bold")

legend_patches = [
    mpatches.Patch(color=c, label=f"{shorten(l)}  ({v/total*100:.1f}%)")
    for c, l, v in zip(pie_colours, main.index, main.values)
]
ax_pie.legend(
    handles=legend_patches,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    fontsize=8.5,
    frameon=False,
)
ax_pie.set_title(
    "Forest loss by land classification\n(2019–2024)",
    fontsize=13, fontweight="bold", color=DARK_TEXT, pad=14
)

# --- Bar ---
ax_bar.set_facecolor("white")
bar_width = 0.6
years   = year_data.index.tolist()
outside = year_data["Outside PTPZ"].values
inside  = year_data["Inside PTPZ"].values

ax_bar.bar(years, outside, bar_width, label="Outside PTPZ", color=WARM_RED, zorder=3)
ax_bar.bar(years, inside,  bar_width, label="Inside PTPZ",  color=GREEN,
           bottom=outside, zorder=3)

ax_bar.set_axisbelow(True)
ax_bar.yaxis.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
ax_bar.xaxis.grid(False)

ax_bar.set_xticks(years)
ax_bar.set_xticklabels([str(y) for y in years], fontsize=10, color=DARK_TEXT)
ax_bar.tick_params(axis="x", length=0)
ax_bar.tick_params(axis="y", labelsize=10, colors=DARK_TEXT)

for spine in ["top", "right", "left"]:
    ax_bar.spines[spine].set_visible(False)
ax_bar.spines["bottom"].set_color("#CCCCCC")

ax_bar.set_title(
    "Forest loss points by year\nInside vs outside PTPZ",
    fontsize=13, fontweight="bold", color=DARK_TEXT, pad=14
)
ax_bar.set_ylabel("Loss points sampled", fontsize=10, color=DARK_TEXT)
ax_bar.legend(fontsize=9, frameon=False, loc="upper right")

fig.text(
    0.5, -0.04,
    "Data: Hansen Global Forest Change v1.12 · Tasmania LIST boundary data · "
    "Analysis: D. Crompton — dtcrompton.github.io",
    ha="center", fontsize=8, color="#888888", style="italic"
)

out_chart = OUTPUTS_FIGURES / "spatial_sanity_check.png"
plt.savefig(out_chart, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Chart saved to {out_chart}")