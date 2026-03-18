// hansen_loss_points.js
// Find all pixels where Hansen records forest loss 2019-2024
// over Tasmania, sample them statewide, export as a point FeatureCollection

var tasmania_bbox = ee.Geometry.Rectangle([144.5, -43.7, 148.5, -40.5]);

var hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');

var loss_year = hansen.select('lossyear');

// lossyear values 19=2019 ... 24=2024
var loss_mask = loss_year.gte(19).and(loss_year.lte(24));

var tree_cover = hansen.select('treecover2000');
var forest_mask = tree_cover.gt(30);

var target_pixels = loss_mask.and(forest_mask);

var loss_points = loss_year
  .updateMask(target_pixels)
  .stratifiedSample({
    numPoints: 834,  // ~834 per year × 6 years = ~5000 total
    classBand: 'lossyear',
    region: tasmania_bbox,
    scale: 100,
    seed: 42,
    geometries: true
  });

// Explicitly sample lossyear band value at each point and set as named property
var loss_points_with_year = loss_points.map(function(feature) {
  var year_val = loss_year.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: feature.geometry(),
    scale: 30
  }).get('lossyear');
  return feature.set('lossyear', year_val);
});

Export.table.toDrive({
  collection: loss_points_with_year,
  description: 'tasmania_loss_points_2019_2024',
  fileFormat: 'GeoJSON',
  folder: 'tasmania_deforestation'
});

Map.centerObject(tasmania_bbox, 7);
Map.addLayer(
  loss_year.updateMask(target_pixels),
  {min: 19, max: 24, palette: ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026']},
  'Forest loss 2019-2024'
);
