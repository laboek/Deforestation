
// COURSEWORK 2 - DEFORESTATION IN THE STATE OF RONDONIA, BRAZIL, ASSESSED WITH LANDSAT 8 (2013-2022)


// Import Amazon Rainforest biome shapefile in assets and add to script
// Import State of Amazon shapefile in assets and add to script
// print(Rainforest)
// print(Rondonia)
// Map.addLayer(Rondonia)

// Center view to area of interest
Map.centerObject(Rondonia, 6.6);




///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////Pre-processing of data//////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


// Create cloud and cirrus mask function, set to 0 to only display clear skies
function mask(image) {
  var cloudBitmask = (1 << 3);
  var cloudshadowBitmask = (1 << 4);
  var cirrusBitmask = (1 << 2);
  var qa = image.select('QA_PIXEL');
  var maskFinal = qa.bitwiseAnd(cloudBitmask).eq(0).clip(Rondonia)
                    .and(qa.bitwiseAnd(cloudshadowBitmask).eq(0).clip(Rondonia))
                    .and(qa.bitwiseAnd(cirrusBitmask).eq(0).clip(Rondonia));
  return image.updateMask(maskFinal);
}

// Apply scaling factors
function scaling(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  // var thermalBands = image.select('ST_B.').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              // .addBands(thermalBands, null, true);
}

// Set visualisation parameters
var params = {
  bands: ['SR_B4', 'SR_B3', 'SR_B2'],
  min: 0,
  max: 2.1,
  gamma: 3.5,
}

// Import Landsat 8 dataset, set filters and display on map

var startDate = '2013-03-18';
var endDate = '2022-03-18';

var dataset = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

var dataFilt = dataset
    .filterDate(startDate,endDate)
    .filter(ee.Filter.calendarRange(6,9,'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', 30))
    .filter(ee.Filter.bounds(Rondonia))
    // .select('SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL')
    .map(mask)
    .map(scaling)

var composite = dataFilt.median().clip(Rondonia)

Map.addLayer(dataFilt, params, 'Composite 2013-2022')


// // Export image for processing in QGIS 
// Export.image.toDrive({
//   image: composite,
//   description: '2013-2022_Rondonia',
//   folder: 'CW2_Images',
//   fileNamePrefix: '2013-2022_Rondonia',
//   region: Rondonia,
//   scale: 30,
//   maxPixels: 1e13,
//   fileFormat: 'GeoTIFF'})




///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////Supervised classification////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


// Create feature collection for each land cover class (manually) and combine into one training collection
var LCC = Wasser.merge(Wald)
                    .merge(Feld)

// print(LCC, 'Sample points');

// // Export feature collection to assets
// Export.table.toAsset({
//   collection: LCC,
//   description: 'Rondonia_samples',
//   assetId: 'Rondonia_samples'
// });

// Select the bands for training (VIS, IR, NIR, SWIR), define input data
var bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'];
var input = composite;
// var landcover_labels = 'landcover'

// Spatial overlay of sample points on imagery to assign spectral profiles to each LC type
var training_data = input.select(bands).sampleRegions({
  collection: LCC,
  properties: ['landcover'],
  scale: 30
});
// print(training_data, 'Training Data');

// Add random column  
var ran = training_data.randomColumn({columnName:'x',seed:2});

// Split training data into training and testing data (ratio: 70 / 30)
var split = 0.7;
var training = ran.filter(ee.Filter.lt('x', split));
var testing = ran.filter(ee.Filter.gte('x', split));

// Train Random Forest classifier
var classifier = ee.Classifier.smileRandomForest({numberOfTrees:250})
  .train({
  features: training_data,
  classProperty: 'landcover',
  inputProperties: bands,
  });

// Run classification
var classified = input.select(bands).classify(classifier);

// Define land cover type colours
var palette = [
  '0096FF', // Wasser  (0)  // blue
  'FFFF00', //  Wald (1) // green
  'E4D00A' // Feld (2) // yellow
  ];

// Display classification
Map.addLayer(classified, {min: 0, max: 3, palette: palette}, 'RF Classification');




///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////Quantitative Assessment//////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


// Conduct quantitative assessment of performance
var test_prediction = testing.classify(classifier);

// Query performance metrics
var errorMatrix = test_prediction.errorMatrix('landcover', 'classification');
var OAV = errorMatrix.accuracy();
var UAV = errorMatrix.consumersAccuracy();
var PAV = errorMatrix.producersAccuracy();
var Kappa = errorMatrix.kappa();

print('Performance  metrics - RF classifier'); print('Error Matrix:', errorMatrix);
print('Overall Accuracy:', OAV);
print('User Accuracy:', UAV);
print('Producer Accuracy:', PAV);
print('Kappa Coefficient: ', Kappa);




///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////Post-classification comparison//////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


// Comparison of 2013 and 2021 images of the State of Rondonia
var Rondonia13 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

var composite13 = Rondonia13
    .filter(ee.Filter.calendarRange(2013, 2013, 'year'))
    .filter(ee.Filter.calendarRange(6, 9, 'month'))
    .filter(ee.Filter.bounds(Rondonia))
    .select('SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL')
    .map(mask)
    .map(scaling)


var Rondonia21 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

var composite21 = Rondonia21
    .filter(ee.Filter.calendarRange(2021, 2021, 'year'))
    .filter(ee.Filter.calendarRange(6, 9, 'month'))
    .filter(ee.Filter.bounds(Rondonia))
    .select('SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL')
    .map(mask)
    .map(scaling)


// var size2013 = composite_2013.size();
// print('Number of 2013 images: ', size2013);
// var size2021 = composite_2021.size();
// print('Number of 2021 images: ', size2021);


// Create median for 2013 and 2021
var median13 = composite13.median() 
var median21 = composite21.median()

// Classify 2013 and 2021 data using the trained algorithm
var classified13 = median13.select(bands).classify(classifier);
var classified21 = median21.select(bands).classify(classifier);

// // Define land cover type colours
// var palette = [
//   '0096FF', // Wasser  (0)  // blue
//   'FFFF00', //  Wald (1) // green
//   'E4D00A' // Feld (2) // yellow
//   ];

// Map.addLayer(median13.clip(Rondonia), params, '2013 VIS');
// Map.addLayer(median21.clip(Rondonia), params, '2021 VIS');

Map.addLayer(classified13.clip(Rondonia), {min: 0, max: 3, palette: palette}, '2013 LC');
Map.addLayer(classified21.clip(Rondonia), {min: 0, max: 3, palette: palette}, '2021 LC');


// Calculate change detection (visually & histogram to see changes in pixel count per land cover class)
var before = classified13.remap([0, 1, 2], [1, 2, 3])
var after = classified21.remap([0, 1, 2], [1, 2, 3])

var changed = after.subtract(before).neq(0)
Map.addLayer(changed, {min:0, max:1, palette: ['white', 'red']}, 'Change')


var merged = before.multiply(100).add(after).rename('Pixel Count Change')

var transition = merged.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(), 
  geometry: Rondonia,
  maxPixels: 1e10,
  scale: 35,
  tileScale: 16
});
// print(transition.get('Pixel Count Change'))


// // Calculate area for each land cover type in 2013 and 2022, calculate forest loss

// var names = ['Wasser', 'Wald', 'Feld']
// var count = classified13.eq([0,1,2]).rename(names);
// var total = count.multiply(ee.Image.pixelArea()).divide(1000*1000);
// var area = total.reduceRegion({
// reducer:ee.Reducer.sum(),
//   geometry:Rondonia,
//   scale:35,
// maxPixels: 1e11,
// bestEffort:true
// });
// var area_pxa13 = ee.Number(area)
// print ('2013 area in km2:', area_pxa13)


// var names = ['Wasser', 'Wald', 'Feld']
// var count = classified21.eq([0,1,2]).rename(names);
// var total = count.multiply(ee.Image.pixelArea()).divide(1000*1000);
// var area = total.reduceRegion({
// reducer:ee.Reducer.sum(),
//   geometry:Rondonia,
//   scale:35,
// maxPixels: 1e11,
// bestEffort:true
// });
// var area_pxa21 = ee.Number(area)
// print ('2021 area in km2:', area_pxa21)


///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////Time series////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


// i) Time series chart 2013-2022 based on EVI and NDVI
// Calculate EVI: EVI = G * ((NIR - R) / (NIR + C1 * R – C2 * B + L)) and add EVI to bands
var addEVI = function(image) {
  var EVI = image.expression(
  '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': image.select('SR_B5'),
      'RED': image.select('SR_B4'),
      'BLUE': image.select('SR_B2')})
      .rename('EVI');
return image.addBands(EVI).copyProperties(image).set('system:time_start', image.get('system:time_start'));
}

// Calculate NDVI: NDVI = (NIR - R) / (NIR + R) and add NDVI to bands
var addNDVI = function(image) {
  var NDVI = image.expression(
  '((NIR - RED) / (NIR + RED))', {
      'NIR': image.select('SR_B5'),
      'RED': image.select('SR_B4')})
      .rename('NDVI');
return image.addBands(NDVI).copyProperties(image).set('system:time_start', image.get('system:time_start'));
}

// Add EVI band to datasets (2013-2022, 2013, 2021)
var VIdata = dataset
    .filterDate(startDate,endDate)
    .filter(ee.Filter.calendarRange(6,9,'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', 30))
    .filter(ee.Filter.bounds(Rondonia))
    .map(mask)
    .map(scaling)
    .map(addEVI)
    .map(addNDVI)

// 2013
var VIdata13 = dataset
    .filter(ee.Filter.calendarRange(2013, 2013, 'year'))
    .filter(ee.Filter.calendarRange(6,9,'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', 30))
    .filter(ee.Filter.bounds(Rondonia))
    .map(mask)
    .map(scaling)
    .map(addEVI)
    .map(addNDVI)

// 2021
var VIdata21 = dataset
    .filter(ee.Filter.calendarRange(2021, 2021, 'year'))
    .filter(ee.Filter.calendarRange(6,9,'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', 30))
    .filter(ee.Filter.bounds(Rondonia))
    .map(mask)
    .map(scaling)
    .map(addEVI)
    .map(addNDVI)

// Create Charts for 2013-2022, 2013 and 2021
var chart = ui.Chart.image.series({
          imageCollection: VIdata.select('NDVI', 'EVI'),
          region: Rondonia,
          reducer: ee.Reducer.mean(),
          scale: 250,
          xProperty: 'system:time_start'
        })
        .setSeriesNames(['NDVI', 'EVI'])
        .setOptions({
          title: 'Median Normalized Difference Vegetation Index (NDVI) and Enhanced Vegetation Index (EVI) Values Over Time',
          hAxis: {title: 'Date', titleTextStyle: {italic: false, bold: true}},
          vAxis: {
            title: 'Vegetation Index',
            titleTextStyle: {italic: false, bold: true}
          },
          trendlines: { 0: {
          type: 'linear',
          color: 'lightblue',
          lineWidth: 2,
          opacity: 0.7,
          showR2: true,
          visibleInLegend: true
          }, 1: {
          type: 'linear',
          color: 'lightblue',
          lineWidth: 2,
          opacity: 0.7,
          showR2: true,
          visibleInLegend: true
          }},
          lineWidth: 1,
          colors: ['e37d05', '2f7a04'],
          curveType: 'function'
        });
print(chart);

// 2013
var chart13 = ui.Chart.image.series({
          imageCollection: VIdata13.select('NDVI', 'EVI'),
          region: Rondonia,
          reducer: ee.Reducer.mean(),
          scale: 250,
          xProperty: 'system:time_start'
        })
        .setSeriesNames(['NDVI', 'EVI'])
        .setOptions({
          title: 'Median NDVI and EVI Values in 2013',
          hAxis: {title: 'Date', titleTextStyle: {italic: false, bold: true}},
          vAxis: {
            title: 'Vegetation Index',
            titleTextStyle: {italic: false, bold: true},
            minValue: 0,
            maxValue: 1
          },
          lineWidth: 1,
          colors: ['e37d05', '2f7a04'],
          curveType: 'function'
        });
print(chart13);

// 2021
var chart21 = ui.Chart.image.series({
          imageCollection: VIdata21.select('NDVI', 'EVI'),
          region: Rondonia,
          reducer: ee.Reducer.mean(),
          scale: 250,
          xProperty: 'system:time_start'
        })
        .setSeriesNames(['NDVI', 'EVI'])
        .setOptions({
          title: 'Median NDVI and EVI Values in 2021',
          hAxis: {title: 'Date', titleTextStyle: {italic: false, bold: true}},
          vAxis: {
            title: 'Vegetation Index',
            titleTextStyle: {italic: false, bold: true},
            minValue: 0,
            maxValue: 1
          },
          lineWidth: 1,
          colors: ['e37d05', '2f7a04'],
          curveType: 'function'
        });
print(chart21);