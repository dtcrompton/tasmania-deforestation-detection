// sentinel2_patches.js
// Export Sentinel-2 composites for central Tasmania study region only

var loss_points = ee.FeatureCollection(
  'projects/change-in-forest-cover/assets/tasmania_loss_points_filtered'
);

// Define study region bounding box
var STUDY_REGION = ee.Geometry.Rectangle([145.5, -43.2, 146.5, -42.2]);

// Filter loss points to study region
var study_points = loss_points.filterBounds(STUDY_REGION);

print('Study region points:', study_points.size());

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(STUDY_REGION)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30));

var BANDS = ['B4', 'B3', 'B2', 'B8', 'B11'];

function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).select(BANDS);
}

function annualComposite(year) {
  return s2
    .filter(ee.Filter.calendarRange(year, year, 'year'))
    .map(maskS2clouds)
    .median();
}

var years = [2019, 2020, 2021, 2022, 2023, 2024];

years.forEach(function(year) {
  var year_code = year - 2000;
  
  var year_points = study_points
    .filter(ee.Filter.eq('lossyear', year_code));
  
  print(year + ':', year_points.size(), 'points');
  
  var composite = annualComposite(year);
  
  Export.image.toDrive({
    image: composite,
    description: 'sentinel2_study_region_' + year,
    folder: 'tasmania_deforestation',
    region: STUDY_REGION,
    scale: 10,
    crs: 'EPSG:32755',
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });
});

// Visual check
var vis_params = {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000};
Map.centerObject(STUDY_REGION, 10);
Map.addLayer(annualComposite(2022), vis_params, 'S2 composite 2022');
Map.addLayer(STUDY_REGION, {color: 'yellow'}, 'Study region');
Map.addLayer(study_points, {color: 'red'}, 'Loss points');