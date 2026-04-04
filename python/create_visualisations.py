#!/usr/bin/env python3
"""
Generate publication-ready visualizations for Phase 4 inference results.

Outputs:
    - outputs/figures/prediction_distribution.png
    - outputs/figures/clearcut_by_tenure.png
    - outputs/figures/predictions_by_year.png

Styling matches portfolio site color palette and typography.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# Portfolio color palette (extracted from spatial_sanity_check.png)
COLORS = {
    "outside_ptpz": "#e8a899",      # Salmon/coral
    "inside_ptpz": "#86b08a",       # Sage green
    "reserve": "#a8d5d5",           # Pale teal
    "conservation": "#b89cc5",      # Muted purple
    "clearcut": "#d9534f",          # Deep red
    "not_clearcut": "#86b08a",      # Sage green (same as inside PTPZ)
    "grid": "#e0e0e0",              # Light grey for grid lines
    "text": "#2d2d2d"               # Dark grey for text
}

# Typography settings
FONT_TITLE = {"family": "sans-serif", "size": 14, "weight": "bold", "color": COLORS["text"]}
FONT_LABEL = {"family": "sans-serif", "size": 12, "color": COLORS["text"]}
FONT_TICK = {"family": "sans-serif", "size": 10, "color": COLORS["text"]}
FONT_ATTRIBUTION = {"family": "sans-serif", "size": 8, "style": "italic", "color": "#888888"}


def add_attribution(fig, text = "Analysis: D. Crompton — dtcrompton.github.io"):
    """Add attribution footer to figure."""
    fig.text(
        0.99, 0.01, text,
        ha = "right", va = "bottom",
        fontdict = FONT_ATTRIBUTION,
        transform = fig.transFigure
    )


def plot_prediction_distribution(predictions_df, output_path = "outputs/figures/prediction_distribution.png"):
    """
    Bar chart showing clearcut vs not-clearcut predictions.
    
    Parameters
    ----------
    predictions_df : pandas.DataFrame
        Predictions with predicted_class column
    output_path : str
        Path to save figure
    """
    # Count predictions by class
    counts = predictions_df["predicted_class"].value_counts()
    
    # Calculate percentages
    total = len(predictions_df)
    percentages = 100 * counts / total
    
    # Create figure
    fig, ax = plt.subplots(figsize = (10, 6))
    fig.patch.set_facecolor("white")
    
    # Map classes to colors
    class_colors = {
        "clearcut": COLORS["clearcut"],
        "not_clearcut": COLORS["not_clearcut"]
    }
    colors_list = [class_colors[cls] for cls in counts.index]
    
    # Create bars
    bars = ax.bar(
        counts.index,
        counts.values,
        color = colors_list,
        alpha = 0.85
    )
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 10,
            f"{count}\n({pct:.1f}%)",
            ha = "center",
            va = "bottom",
            fontsize = 11,
            fontweight = "bold",
            color = COLORS["text"]
        )
    
    # Styling
    ax.set_xlabel("Predicted Class", fontdict = FONT_LABEL)
    ax.set_ylabel("Number of Predictions", fontdict = FONT_LABEL)
    ax.set_title(
        f"Forest Loss Classification\nCentral Tasmania Study Region (2019-2024), n={total}",
        fontdict = FONT_TITLE,
        pad = 15
    )
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.grid(axis = "y", alpha = 0.4, linestyle = "-", linewidth = 0.8, color = COLORS["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(labelsize = FONT_TICK["size"], colors = COLORS["text"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Attribution
    add_attribution(fig)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300, bbox_inches = "tight", facecolor = "white")
    print(f"Saved prediction distribution chart to {output_path}")
    plt.close()


def plot_clearcut_by_tenure(predictions_gdf, output_path = "outputs/figures/clearcut_by_tenure.png"):
    """
    Bar chart showing clearcut predictions by land tenure category.
    
    Parameters
    ----------
    predictions_gdf : geopandas.GeoDataFrame
        Predictions with in_ptpz and in_reserve columns
    output_path : str
        Path to save figure
    """
    # Filter to clearcut predictions only
    clearcuts = predictions_gdf[predictions_gdf["predicted_class"] == "clearcut"].copy()
    
    if len(clearcuts) == 0:
        print("No clearcut predictions to visualize by tenure")
        return
    
    # Categorize by tenure
    clearcuts["tenure_category"] = "Outside PTPZ & reserves"
    clearcuts.loc[clearcuts["in_ptpz"], "tenure_category"] = "Inside PTPZ"
    clearcuts.loc[clearcuts["in_reserve"], "tenure_category"] = "Inside reserves"
    
    # Count by category
    counts = clearcuts["tenure_category"].value_counts()
    
    # Calculate percentages
    total = len(clearcuts)
    percentages = 100 * counts / total
    
    # Create figure
    fig, ax = plt.subplots(figsize = (10, 6))
    fig.patch.set_facecolor("white")
    
    # Map categories to colors
    category_colors = {
        "Inside PTPZ": COLORS["inside_ptpz"],
        "Inside reserves": COLORS["reserve"],
        "Outside PTPZ & reserves": COLORS["outside_ptpz"]
    }
    colors_list = [category_colors.get(cat, COLORS["clearcut"]) for cat in counts.index]
    
    # Create bars
    bars = ax.bar(
        range(len(counts)),
        counts.values,
        color = colors_list,
        alpha = 0.85
    )
    
    # Add value labels
    for i, (bar, count, pct) in enumerate(zip(bars, counts.values, percentages.values)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.4,
            f"{count}\n({pct:.1f}%)",
            ha = "center",
            va = "bottom",
            fontsize = 11,
            fontweight = "bold",
            color = COLORS["text"]
        )
    
    # Styling
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, fontsize = FONT_TICK["size"], color = COLORS["text"])
    ax.set_xlabel("Land Tenure Category", fontdict = FONT_LABEL)
    ax.set_ylabel("Clearcut Predictions", fontdict = FONT_LABEL)
    ax.set_title(
        f"Clearcut Predictions by Land Tenure\nCentral Tasmania Study Region (2019-2024), n={total}",
        fontdict = FONT_TITLE,
        pad = 15
    )
    ax.set_ylim(0, max(counts.values) * 1.25)
    ax.grid(axis = "y", alpha = 0.4, linestyle = "-", linewidth = 0.8, color = COLORS["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(labelsize = FONT_TICK["size"], colors = COLORS["text"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Attribution
    add_attribution(fig)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300, bbox_inches = "tight", facecolor = "white")
    print(f"Saved clearcut-by-tenure chart to {output_path}")
    plt.close()


def plot_predictions_by_year(predictions_df, output_path = "outputs/figures/predictions_by_year.png"):
    """
    Dual chart: stacked bars (total loss) + line (clearcut percentage).
    
    Parameters
    ----------
    predictions_df : pandas.DataFrame
        Predictions with year and predicted_class columns
    output_path : str
        Path to save figure
    """
    # Count predictions by year and class
    year_class_counts = predictions_df.groupby(["year", "predicted_class"]).size().unstack(fill_value = 0)
    
    # Calculate clearcut percentage per year
    if "clearcut" in year_class_counts.columns and "not_clearcut" in year_class_counts.columns:
        year_class_counts["clearcut_pct"] = 100 * year_class_counts["clearcut"] / (
            year_class_counts["clearcut"] + year_class_counts["not_clearcut"]
        )
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize = (14, 6))
    fig.patch.set_facecolor("white")
    
    # Subplot 1: Stacked bar chart (total loss)
    year_class_counts[["not_clearcut", "clearcut"]].plot(
        kind = "bar",
        stacked = True,
        ax = axes[0],
        color = [COLORS["not_clearcut"], COLORS["clearcut"]],
        alpha = 0.85,
        legend = False
    )
    
    axes[0].set_xlabel("Year", fontdict = FONT_LABEL)
    axes[0].set_ylabel("Forest Loss Points Sampled", fontdict = FONT_LABEL)
    axes[0].set_title("Forest Loss by Year and Type", fontdict = FONT_TITLE, pad = 15)
    axes[0].set_xticklabels(year_class_counts.index, rotation = 0, fontsize = FONT_TICK["size"])
    axes[0].tick_params(labelsize = FONT_TICK["size"], colors = COLORS["text"])
    axes[0].grid(axis = "y", alpha = 0.4, linestyle = "-", linewidth = 0.8, color = COLORS["grid"])
    axes[0].set_axisbelow(True)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    
    # Custom legend
    legend_patches = [
        mpatches.Patch(color = COLORS["clearcut"], label = "Clearcut", alpha = 0.85),
        mpatches.Patch(color = COLORS["not_clearcut"], label = "Not clearcut", alpha = 0.85)
    ]
    axes[0].legend(handles = legend_patches, loc = "upper right", frameon = False, fontsize = 10)
    
    # Subplot 2: Clearcut percentage line plot
    if "clearcut_pct" in year_class_counts.columns:
        axes[1].plot(
            year_class_counts.index,
            year_class_counts["clearcut_pct"],
            marker = "o",
            linewidth = 2.5,
            markersize = 8,
            color = COLORS["clearcut"],
            markerfacecolor = COLORS["clearcut"],
            markeredgecolor = COLORS["text"],
            markeredgewidth = 1.5
        )
        
        axes[1].set_xlabel("Year", fontdict = FONT_LABEL)
        axes[1].set_ylabel("Clearcut Percentage (%)", fontdict = FONT_LABEL)
        axes[1].set_title("Clearcut Rate by Year", fontdict = FONT_TITLE, pad = 15)
        axes[1].set_ylim(0, max(year_class_counts["clearcut_pct"]) * 1.25)
        axes[1].tick_params(labelsize = FONT_TICK["size"], colors = COLORS["text"])
        axes[1].grid(True, alpha = 0.4, linestyle = "-", linewidth = 0.8, color = COLORS["grid"])
        axes[1].set_axisbelow(True)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        
        # Add value labels
        for year, pct in year_class_counts["clearcut_pct"].items():
            axes[1].text(
                year, pct + (max(year_class_counts["clearcut_pct"]) * 0.04),
                f"{pct:.1f}%",
                ha = "center",
                fontsize = 9,
                color = COLORS["text"]
            )
    
    # Attribution
    add_attribution(fig)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300, bbox_inches = "tight", facecolor = "white")
    print(f"Saved predictions-by-year chart to {output_path}")
    plt.close()


def main():
    """Generate all Phase 4 visualizations."""
    
    print("=" * 70)
    print("PHASE 4: GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()
    
    # Load predictions
    predictions_csv = "outputs/maps/predictions.csv"
    predictions_gdf_path = "outputs/maps/predictions_with_tenure.geojson"
    
    print(f"Loading predictions from {predictions_csv}...")
    predictions_df = pd.read_csv(predictions_csv)
    
    print(f"Loading tenure-referenced predictions from {predictions_gdf_path}...")
    predictions_gdf = gpd.read_file(predictions_gdf_path)
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    print()
    
    plot_prediction_distribution(predictions_df)
    plot_clearcut_by_tenure(predictions_gdf)
    plot_predictions_by_year(predictions_df)
    
    print()
    print("=" * 70)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print()
    print("Generated files:")
    print("  - outputs/figures/prediction_distribution.png")
    print("  - outputs/figures/clearcut_by_tenure.png")
    print("  - outputs/figures/predictions_by_year.png")


if __name__ == "__main__":
    main()