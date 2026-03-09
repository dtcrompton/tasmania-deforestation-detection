# Tasmania Deforestation Detection

**Deep learning computer vision system for detecting undocumented forest clearing in Tasmania's old-growth forests using Sentinel-2 satellite imagery (2019-2025)**

## Project Overview

This project uses convolutional neural networks (CNNs) to identify potential illegal logging in Tasmania's protected forests by comparing satellite imagery over time and cross-referencing with official forest clearing permits.

**Research Questions:**
1. Can deep learning detect forest clearing events in satellite imagery with >90% accuracy?
2. Which areas show forest loss without corresponding government clearing permits?
3. What is the spatial distribution of suspected illegal deforestation (2019-2025)?
4. How much undocumented forest clearing has occurred in protected areas?

**Why Tasmania:**
- High-value old-growth forests (global conservation significance)
- Well-documented illegal logging controversies
- Clear distinction between permitted (plantations, sustainable forestry) and illegal clearing
- Manageable geographic scale for proof-of-concept
- Strong validation data (government permits, NGO reports, investigative journalism)

## Study Area

**Tasmania, Australia** — focusing on:
- **Tasmanian Wilderness World Heritage Area** (1.58 million hectares)
- **Regional Forest Agreement areas** (where logging permits are issued)
- **Known illegal logging hotspots** (from NGO reports: Bob Brown Foundation, Wilderness Society)

## Data Sources

### **Satellite Imagery:**
- **Sentinel-2** (10m resolution, free, 2019-2025 coverage)
- Bands: RGB + NIR for vegetation indices
- Cloud-free composites (annual or seasonal)

### **Validation Data:**
- **Forest clearing permits:** Tasmanian Government (Sustainable Timber Tasmania)
- **Protected area boundaries:** CAPAD (Collaborative Australian Protected Area Database)
- **Ground truth:** Investigative reports, media coverage, NGO documentation
- **Historical logging:** Known illegal clearing sites for training data

### **Forest Baselines:**
- **Hansen Global Forest Change** (2000-2024 forest cover)
- **Tasmania vegetation mapping** (old-growth vs. plantation vs. regrowth)

## Methodology

### Phase 1: Data Collection & Preprocessing
- [ ] Download Sentinel-2 imagery (2019-2025) for Tasmania
- [ ] Obtain forest clearing permit records
- [ ] Map protected area boundaries
- [ ] Create cloud-free annual composites

### Phase 2: Training Data Creation
- [ ] Identify known legal clearing sites (permitted logging)
- [ ] Identify known illegal clearing sites (documented cases)
- [ ] Extract image patches (before/after pairs)
- [ ] Label dataset: legal, illegal, no change, natural (fire/storm)

### Phase 3: Model Development
- [ ] Build CNN for binary classification (forest loss vs. no change)
- [ ] Train change detection model (U-Net or similar)
- [ ] Evaluate performance (accuracy, precision, recall)
- [ ] Fine-tune on Tasmania-specific features

### Phase 4: Inference & Analysis
- [ ] Run model on entire Tasmania (2019-2025)
- [ ] Identify forest loss pixels
- [ ] Cross-reference with permit database
- [ ] Flag areas: loss detected + no permit = suspected illegal

### Phase 5: Validation & Spatial Analysis
- [ ] Ground-truth with Google Earth historical imagery
- [ ] Compare to NGO reports and media investigations
- [ ] Calculate area of suspected illegal clearing
- [ ] Map spatial distribution and hotspots

### Phase 6: Outputs & Reporting
- [ ] Interactive map of suspected illegal clearing sites
- [ ] Time-lapse animation (2019-2025 forest change)
- [ ] Technical report with model performance metrics
- [ ] Policy brief for environmental enforcement agencies
- [ ] Portfolio write-up

## Expected Outputs

**Model:**
- Trained CNN for forest change detection
- Performance metrics (accuracy, F1-score, confusion matrix)
- Model weights and architecture documentation

**Geospatial Analysis:**
- Hectares of forest loss (2019-2025)
- Hectares with permits vs. without
- Spatial hotspots of suspected illegal clearing

**Visualisations:**
- Interactive map (Folium): suspected sites with satellite imagery overlays
- Time-series animation (GIF/video): forest change over time
- Before/after image comparisons for key sites

**Reports:**
- Technical write-up (methodology, results, limitations)
- Policy brief (1-page summary for enforcement agencies)
- Portfolio page with embedded map and findings

## Technologies

**Satellite Data:** Google Earth Engine, Sentinel Hub  
**Deep Learning:** TensorFlow/Keras or PyTorch  
**Image Processing:** Rasterio, GDAL, scikit-image  
**Geospatial:** GeoPandas, Folium, QGIS  
**Visualisation:** matplotlib, seaborn, Plotly  

## Key Literature

- Hansen, M. C. et al. (2013). "High-Resolution Global Maps of 21st-Century Forest Cover Change"
- Tracking Tasmania's threatened forests (Bob Brown Foundation reports)
- Sentinel-2 for forest monitoring (ESA documentation)
- Deep learning for deforestation detection (recent ML papers)

## Timeline

**Month 5 (March-April):**
- Phase 1-2: Data collection + training data creation
- Phase 3: Model development

**Month 6 (April-May):**
- Phase 4-5: Inference + validation
- Phase 6: Outputs + portfolio write-up

## Ethical Considerations

This project aims to support environmental protection and law enforcement. Findings may be:
- Shared with relevant authorities and NGOs
- Presented as "suspected" illegal clearing (requires ground verification)
- Used to demonstrate technical capability, not to accuse specific entities

## License

MIT License

## Contact

Daniel Crompton | [LinkedIn](https://www.linkedin.com/in/dtcrompton/) | [Portfolio](https://dtcrompton.github.io)