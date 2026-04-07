# Tasmania Deforestation Detection

CNN-based classification of forest loss in central Tasmania (2019–2024) using Sentinel-2 satellite imagery, with spatial analysis to identify patterns and flag potential violations in protected areas.

**Live Project:** [dtcrompton.github.io/tasmania-deforestation](https://dtcrompton.github.io/tasmania-deforestation/)  
**Interactive Map:** [View Map](https://dtcrompton.github.io/tasmania-deforestation/map.html)

---

## Key Findings

- **3.5% of forest loss was clearcut logging** — the remainder predominantly fire-driven (2019/2023 bushfire seasons)
- **2024 spike:** 37% clearcut rate in 2024 (vs <2% in fire-heavy years), indicating recent logging activity increase
- **96% of predicted clearcuts fall within PTPZ** (Private Timber Reserve Zones where logging is permitted)
- **1 clearcut flagged in protected reserves** (outside permitted zones)

**Implication:** Fire management, not logging enforcement, should be the priority for forest protection in central Tasmania.

---

## Project Overview

### Study Region
Central Tasmania: 145.5–146.5°E, -43.2 to -42.2°S  
Imagery: Sentinel-2 Level-2A (10m resolution, 2019–2024)  
Loss detection: Hansen Global Forest Change v1.12

### Technical Approach

**Phase 1:** Data acquisition via Google Earth Engine  
- Exported 12 Sentinel-2 composite GeoTIFFs (2 per year, 2019–2024)
- Sampled 747 Hansen forest loss points within study region
- Downloaded PTPZ and reserve boundaries (Tasmania LIST)

**Phase 2:** Manual labelling and patch extraction  
- Extracted 736 patches (128×128 pixels, 5 bands: RGB + NIR + SWIR1)
- Manually labelled 315-patch stratified sample via Jupyter notebook
- Binary classification: clearcut vs not-clearcut (fire/natural)
- Final distribution: 24 clearcut, 239 not-clearcut, 52 skip (ambiguous)

**Phase 3:** CNN training  
- Architecture: Simple 3-layer CNN (32→64→128 filters) to avoid overfitting
- Handled 10:1 class imbalance via balanced class weighting
- Train/val/test split: 70/15/15 (stratified)
- Test metrics: 100% recall (4/4 clearcut samples), 100% precision
- Validation: 83% precision on 12 manually-reviewed high-confidence predictions

**Phase 4:** Inference and spatial analysis  
- Applied model to all 736 patches
- Spatial join with PTPZ and reserve boundaries (GeoPandas)
- Flagged clearcuts outside permitted zones
- Generated validation sample (60 patches) for manual review

**Phase 5:** Interactive mapping and portfolio  
- Built Folium map with clickable markers, layer controls, land tenure overlays
- Created project pages with technical write-up and stakeholder explainer
- Published to portfolio site

---

## Repository Structure
```
tasmania-deforestation-detection/
├── data/
│   ├── raw/                              # Sentinel-2 GeoTIFFs (gitignored, ~18GB)
│   ├── processed/
│   │   ├── loss_points_study_region.csv  # 747 Hansen loss points
│   │   └── labelling_subset_final.csv    # 315-patch stratified sample
│   ├── training/
│   │   ├── patches/                      # 736 GeoTIFF patches (gitignored)
│   │   └── patch_labels.csv              # 315 manual labels
│   └── permits/
│       ├── ptpz.geojson                  # PTPZ boundaries (gitignored)
│       └── reserve_estate.geojson        # Reserve boundaries (gitignored)
├── gee/
│   └── export_sentinel2_composites.js    # GEE script for data export
├── models/
│   ├── clearcut_classifier_final.keras   # Trained CNN
│   └── clearcut_classifier_best.keras    # Best checkpoint
├── outputs/
│   ├── figures/
│   │   ├── training_history.png          # Loss/accuracy curves
│   │   ├── confusion_matrix.png          # Test set confusion matrix
│   │   ├── prediction_distribution.png   # Clearcut vs not-clearcut counts
│   │   ├── clearcut_by_tenure.png        # Clearcuts by land category
│   │   └── predictions_by_year.png       # Temporal distribution + 2024 spike
│   ├── maps/
│   │   ├── predictions.csv               # All predictions with coordinates
│   │   ├── predictions.geojson           # Predictions as GeoJSON
│   │   ├── predictions_with_tenure.geojson # With PTPZ/reserve flags
│   │   └── tasmania_deforestation_map.html # Interactive Folium map
│   ├── model_evaluation.txt              # Test set metrics
│   ├── validation_results.csv            # Manual validation (60 patches)
│   └── validation_metrics.txt            # Validation precision/recall
└── python/
├── train_cnn.py                      # Phase 3: CNN training
├── run_inference.py                  # Phase 4: Inference + tenure cross-ref
├── validate_predictions.py           # Phase 4: Interactive validation tool
├── create_visualisations.py          # Phase 4: Charts (styled to portfolio palette)
├── create_map.py                     # Phase 5: Folium map generation
├── extract_patches.py                # Phase 2: Patch extraction
├── filter_loss_points.py             # Phase 1: Hansen point filtering
└── download_boundaries.py            # Phase 1: PTPZ/reserve download
```

---

## Technical Stack

**Languages & Frameworks:**  
Python 3.13, TensorFlow/Keras, GeoPandas, Rasterio, Folium

**Data Sources:**  
- Sentinel-2 Level-2A (ESA Copernicus, via Google Earth Engine)
- Hansen Global Forest Change v1.12 (University of Maryland)
- Tasmania LIST (PTPZ, Reserve Estate boundaries)

**Key Libraries:**  
- `tensorflow` — CNN training and inference
- `geopandas` — spatial joins, CRS transformations
- `rasterio` — GeoTIFF I/O
- `folium` — interactive web mapping
- `scikit-learn` — train/test splitting, class weights, metrics
- `matplotlib`, `seaborn` — visualisations

---

## Reproduction Instructions

### Prerequisites
- Python 3.13+
- Google Earth Engine account (for Phase 1 data export)
- ~20GB disk space for Sentinel-2 imagery

### Setup
```bash
# Clone repository
git clone https://github.com/dtcrompton/tasmania-deforestation-detection.git
cd tasmania-deforestation-detection

# Install dependencies
pip install tensorflow keras geopandas rasterio folium scikit-learn matplotlib seaborn pandas numpy --break-system-packages
```

### Phase 1: Data Acquisition
```bash
# Download PTPZ and reserve boundaries
python3 python/download_boundaries.py

# Export Sentinel-2 composites from Google Earth Engine
# Run gee/export_sentinel2_composites.js in GEE Code Editor
# Download exported GeoTIFFs to data/raw/
```

### Phase 2: Patch Extraction and Labelling
```bash
# Extract 128×128 patches from Sentinel-2 composites
python3 python/extract_patches.py

# Label patches interactively
jupyter notebook notebooks/label_patches.ipynb
```

### Phase 3: CNN Training
```bash
python3 python/train_cnn.py
# Outputs: models/clearcut_classifier_final.keras, training curves, metrics
```

### Phase 4: Inference and Analysis
```bash
# Run inference on all patches + cross-reference with land tenure
python3 python/run_inference.py

# Generate visualisations
python3 python/create_visualisations.py

# (Optional) Validate predictions interactively
python3 python/validate_predictions.py
```

### Phase 5: Interactive Map
```bash
python3 python/create_map.py
# Output: outputs/maps/tasmania_deforestation_map.html
```

---

## Model Performance

**Test Set (n=40, 4 clearcut samples):**
- Precision: 100% (4/4 clearcut predictions correct)
- Recall: 100% (4/4 clearcut samples detected)
- F1-score: 100%

**Manual Validation (n=12 high-confidence clearcut predictions):**
- Precision: 83% (10/12 correct)
- 2 false positives (fire/natural loss misclassified as clearcut)

**Note:** Test set size is small (4 clearcut samples). Manual validation on unseen data provides more robust performance estimate.

---

## Limitations

1. **Study region scope:** Central Tasmania only (145.5–146.5°E, -43.2 to -42.2°S). Findings may not generalise to western or northern Tasmania.

2. **Small training set:** 263 labelled patches (24 clearcut, 239 not-clearcut). Model performance could improve with additional labelled data.

3. **Binary classification:** Does not distinguish clearcut types (salvage logging, plantation harvest, native forest clearfelling).

4. **Fire/clearcut ambiguity:** Model occasionally confuses fire boundaries with clearcut edges (2 false positives in validation sample).

5. **Temporal resolution:** Annual composites (2019–2024) miss intra-annual dynamics. Some clearcuts may occur between composite dates.

---

## Future Work

- **Expand study region:** Apply model to all Tasmania (4,051 Hansen loss points)
- **Multi-class classification:** Distinguish clearcut subtypes (salvage, plantation, native)
- **Temporal analysis:** Track individual sites across multiple years to detect regeneration
- **Ground-truthing:** Field validation of flagged clearcuts in protected areas
- **Real-time monitoring:** Automate monthly inference on new Sentinel-2 imagery

---

## Citation

If using this work, please cite:
```
Crompton, D. (2026). Tasmania Deforestation Detection: CNN-based classification
of forest loss from Sentinel-2 imagery. GitHub repository.
https://github.com/dtcrompton/tasmania-deforestation-detection
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Data sources retain their original licenses:
- Sentinel-2: ESA Copernicus (free and open access)
- Hansen GFC: CC BY 4.0
- Tasmania LIST: Creative Commons Attribution 4.0 International

---

## Contact

**Daniel Crompton**  
Portfolio: [dtcrompton.github.io](https://dtcrompton.github.io)  
LinkedIn: [linkedin.com/in/dtcrompton](https://linkedin.com/in/dtcrompton/)  
GitHub: [@dtcrompton](https://github.com/dtcrompton)