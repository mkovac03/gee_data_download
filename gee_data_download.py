import ee
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
import rasterio
import multiprocessing
from tqdm import tqdm
import time
import logging
from math import floor
from pyproj import CRS
import json

# Configure logging to show errors in red
class RedFormatter(logging.Formatter):
    RED = '\033[31m'
    RESET = '\033[0m'

    def format(self, record):
        if record.levelno == logging.ERROR:
            record.msg = f"{self.RED}{record.msg}{self.RESET}"
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(RedFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

# Load configuration from file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Initialize the Earth Engine API
try:
    ee.Initialize()
except ee.EEException:
    ee.Authenticate()
    ee.Initialize()

# Load parameters from config file
START_DATE = config['START_DATE']
END_DATE = config['END_DATE']
YEAR = START_DATE[:4]  # Extract year for folder naming
CLOUD_THRESH = config['CLOUD_THRESH']
RES = config['RES']
GRID_SIZE = config['GRID_SIZE']
COUNTRY_NAME = config['COUNTRY_NAME']
SATELLITE = config['SATELLITE']
INCREMENT = config['INCREMENT']
INTERVAL = config['INTERVAL']
try:
    ASSET_FOLDER = config['ASSET_FOLDER']
except KeyError:
    logging.error(f"'ASSET_FOLDER' is missing in config.json. Please make sure to set it to your GEE-enabled cloud project asset path.")
    raise SystemExit
NO_DATA_VALUE = config['NO_DATA_VALUE']
BANDS = config['BANDS']
OUTPUT_DIR = config['OUTPUT_DIR']

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get the country geometry from GAUL dataset
country_fc = ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.eq('ADM0_NAME', COUNTRY_NAME))
ASSET_ID = f'{ASSET_FOLDER}{COUNTRY_NAME}_utm_grid_{GRID_SIZE // 1000}km'

# Cloud masking function using Sentinel-2 QA band
def maskS2clouds(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)

# Add NDVI band to image
def addS2Variables(image):
    return image.addBands(
        image.normalizedDifference(['B8', 'B4']).rename('NDVI').multiply(10000).toInt32()
    )

# Fetch filtered and processed Sentinel-2 image collection
def getS2(feature, index):
    return ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterBounds(feature) \
        .filterDate(START_DATE, END_DATE) \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', CLOUD_THRESH) \
        .map(maskS2clouds) \
        .map(addS2Variables) \
        .select(BANDS)

# Create a monthly median composite image from the collection
def get_monthly_imgs(feature, month, index):
    start_date = ee.Date(f'{YEAR}-{month:02d}-01')
    end_date = start_date.advance(INTERVAL, INCREMENT)
    image_collection = getS2(feature, index)
    monthly_ndvi = image_collection.filterDate(start_date, end_date).median().unmask(NO_DATA_VALUE)
    return monthly_ndvi

# Determine UTM zone based on feature centroid
def get_utm_zone(feature):
    centroid = feature.geometry().centroid().coordinates().getInfo()
    lon = centroid[0]
    utm_zone = floor((lon + 180) / 6) + 1
    return utm_zone

# Check if a GEE asset already exists
def asset_exists(asset_id):
    try:
        ee.data.getAsset(asset_id)
        logging.info(f"Asset {asset_id} exists.")
        return True
    except ee.EEException:
        logging.info(f"Asset {asset_id} does not exist.")
        return False

# Create and export a spatial grid to a GEE asset if it doesn't exist
def export_grid_to_asset():
    if not asset_exists(ASSET_ID):
        logging.info("Exporting grid to GEE asset...")
        crs_code = CRS.from_dict({'proj': 'utm', 'zone': get_utm_zone(country_fc.first()), 'south': False}).to_authority()[1]
        crs = f"EPSG:{crs_code}"
        grid = country_fc.geometry().coveringGrid(crs, GRID_SIZE)
        grid_size = grid.size().getInfo()

        if grid_size == 0:
            logging.error("Generated grid is empty. Check the input parameters.")
            return

        logging.info(f"Grid generated with {grid_size} cells.")

        try:
            task = ee.batch.Export.table.toAsset(
                collection=grid,
                description=f'{COUNTRY_NAME}_utm_grid_{GRID_SIZE // 1000}km',
                assetId=ASSET_ID
            )
        except ee.EEException as e:
            logging.error(f"[31mProject ID is invalid. Make sure to update 'ASSET_FOLDER' in your config.json with a valid GEE-enabled cloud project ID.\nDetails: {e}[0m")
            raise SystemExit
        task.start()
        logging.info('Grid export started.')

        while task.active():
            logging.info("Waiting for grid export to complete...")
            time.sleep(30)

        status = task.status()
        if status['state'] != 'COMPLETED':
            logging.error(
                f"Error exporting grid. Task state: {status['state']}, Error message: {status.get('error_message', 'No error message provided')}")
        else:
            logging.info('Grid export completed.')
    else:
        logging.info(f"Asset {ASSET_ID} already exists. Skipping export.")

# Download the processed image as a multiband GeoTIFF
def download_images(param):
    feature, month, index = param
    utm_zone = get_utm_zone(feature)
    crs_code = CRS.from_dict({'proj': 'utm', 'zone': utm_zone, 'south': False}).to_authority()[1]
    crs = f"EPSG:{crs_code}"

    # Create a subfolder per country/year/month
    subfolder = os.path.join(OUTPUT_DIR, COUNTRY_NAME, YEAR, f"{month:02d}")
    os.makedirs(subfolder, exist_ok=True)
    output_path = os.path.join(subfolder, f'{SATELLITE}_{COUNTRY_NAME}_{YEAR}_{month:02d}_{INCREMENT}ly_median_{RES}m_{index}.tif')

    if not os.path.exists(output_path):
        try:
            monthly_ndvi = get_monthly_imgs(feature.geometry(), month, index)
            url = monthly_ndvi.getDownloadURL({
                'bands': BANDS,
                'region': feature.geometry(),
                'scale': RES,
                'crs': crs,
                'format': 'GEO_TIFF'
            })
            response = requests.get(url)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
        except Exception as e:
            logging.error(f"Error downloading feature index {index} month {month}: {e}")
            return None
    else:
        return output_path

# Main entry point for script execution
def main():
    export_grid_to_asset()
    logging.info("Reading grid from GEE asset...")
    grid = ee.FeatureCollection(ASSET_ID)

    logging.info("Preparing parameters for multiprocessing...")
    months = list(range(1, 13))  # Loop through all months
    grid_list = grid.toList(grid.size()).getInfo()
    params = [(ee.Feature(grid_list[i]), month, i) for i in range(len(grid_list)) for month in months]
    logging.info(f"Total tasks to process: {len(params)}")

    logging.info("Starting multiprocessing download...")
    num_cores = multiprocessing.cpu_count() - 1  # Use all but one core
    with multiprocessing.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(download_images, params), total=len(params)))
    logging.info(f'Download completed. Total files: {len([r for r in results if r is not None])}')

# Required for Windows multiprocessing compatibility
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
