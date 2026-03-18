// sentinel2_patches.js
// For each Hansen loss point, extract a 128x128 Sentinel-2 composite
// centred on that point from the year loss was detected.
// Exports a single image stack (all patches as a mosaic) per year batch.

// --- Load loss points exported from hansen_loss_points.js ---
var loss_points = ee.FeatureCollection(
  'projects/change-in-forest-cover/assets/tasmania_loss_points_2019_2024'
);
// Replace YOUR_GEE_PROJECT with your GEE cloud project ID.

// --- Sentinel-2 surface reflectance collection ---
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(ee.Geometry.Rectangle([144.5, -43.7, 148.5, -40.5]))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

// Bands: B4=Red, B3=Green, B2=Blue, B8=NIR, B11=SWIR1
// SWIR helps distinguish fire scars from clearcuts
var BANDS = ['B4', 'B3', 'B2', 'B8', 'B11'];

// --- Cloud masking using Sentinel-2 scene classification layer ---
function maskS2clouds(image) {
  var scl = image.select('SCL');
  // SCL classes 4 (vegetation) and 5 (non-veg) are clear; exclude cloud, shadow, snow
  var mask = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(11));
  return image.updateMask(mask).select(BANDS).divide(10000); // scale to 0-1 reflectance
}

// --- Build a cloud-free annual composite for a given year ---
function annualComposite(year) {
  return s2
    .filter(ee.Filter.calendarRange(year, year, 'year'))
    .map(maskS2clouds)
    .median(); // median composite suppresses remaining cloud artefacts
}

// --- Patch size: 128 pixels at 10m = 1280m ---
var PATCH_SIZE = 1280; // metres

// --- Export a mosaic of patches for a subset of points ---
// Sample 100 points per loss year (2019-2024) = 600 patches total.
// This is sufficient for initial labelling and model development.
// Scale up after architecture is validated.

var years = [2019, 2020, 2021, 2022, 2023, 2024];

years.forEach(function(year) {
  // lossyear in GeoJSON is stored as an integer offset: 2019=19, etc.
  var year_code = year - 2000;

  var year_points = loss_points
    .filter(ee.Filter.eq('lossyear', year_code))
    .limit(100); // 100 points per year

  var composite = annualComposite(year);

  // Build a mosaic of 128x128 patches centred on each point
  // by painting each patch region into a single image
  var patches = composite.clipToCollection(
    year_points.map(function(pt) {
      return pt.buffer(PATCH_SIZE / 2).bounds();
    })
  );

  Export.image.toDrive({
    image: patches,
    description: 'sentinel2_patches_' + year,
    folder: 'tasmania_deforestation',
    region: ee.Geometry.Rectangle([144.5, -43.7, 148.5, -40.5]),
    scale: 10,
    crs: 'EPSG:32755', // WGS84 UTM Zone 55S — appropriate for Tasmania
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });
});

// --- Visual check: 2022 composite over Tasmania ---
var vis_params = {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3};
Map.centerObject(ee.Geometry.Rectangle([144.5, -43.7, 148.5, -40.5]), 7);
Map.addLayer(annualComposite(2022), vis_params, 'S2 composite 2022');
Map.addLayer(loss_points, {color: 'red'}, 'Loss points');
