import ee
import numpy as np
import os
import sys
import requests
import rasterio
import multiprocessing
from tqdm import tqdm
import time
import logging
from math import floor
from pyproj import CRS
import json
import itertools

class RedFormatter(logging.Formatter):
    RED = '\033[31m'
    RESET = '\033[0m'
    def format(self, record):
        if record.levelno == logging.ERROR:
            record.msg = f"{self.RED}{record.msg}{self.RESET}"
        return super().format(record)

class WarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING

handler = logging.StreamHandler()
handler.setFormatter(RedFormatter('%(asctime)s - %(levelname)s - %(message)s'))
handler.addFilter(WarningFilter())
root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

# Load configuration
try:
    with open('config.json', 'r', encoding='utf-8') as cfg:
        config = json.load(cfg)
except (json.JSONDecodeError, FileNotFoundError) as e:
    logging.error(f"Failed to load config.json: {e}")
    sys.exit(1)

# Parameters from config
COUNTRY_NAME    = config.get('COUNTRY_NAME') or sys.exit("COUNTRY_NAME missing in config.json")
UTM_GRID_ASSET  = config.get('UTM_GRID_ASSET') or sys.exit("UTM_GRID_ASSET missing in config.json")
SOUTH           = config.get('SOUTH', False)
MAX_RETRIES     = config.get('MAX_RETRIES', 5)
BASE_WAIT       = config.get('BASE_WAIT', 2.0)

START_DATE      = config['START_DATE']
END_DATE        = config['END_DATE']
YEAR            = START_DATE[:4]
CLOUD_THRESH    = config['CLOUD_THRESH']
RES             = config['RES']
GRID_SIZE       = config['GRID_SIZE']
SATELLITE       = config['SATELLITE']
INCREMENT       = config['INCREMENT']
INTERVAL        = config['INTERVAL']
ASSET_FOLDER    = config['ASSET_FOLDER']
NO_DATA_VALUE   = config['NO_DATA_VALUE']
BANDS           = config['BANDS']
OUTPUT_DIR      = config['OUTPUT_DIR']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Earth Engine
try:
    ee.Initialize()
except ee.EEException:
    ee.Authenticate()
    ee.Initialize()

# Country geometry from GAUL
country_fc = ee.FeatureCollection('FAO/GAUL/2015/level0') \
    .filter(ee.Filter.eq('ADM0_NAME', COUNTRY_NAME))
utm_grid  = ee.FeatureCollection(UTM_GRID_ASSET)

# Check asset existence
def asset_exists(aid):
    try:
        ee.data.getAsset(aid)
        return True
    except ee.EEException:
        return False

# Sentinel-2 cloud mask + NDVI
def maskS2clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
    return image.updateMask(mask)

def addS2Variables(image):
    return image.addBands(
        image.normalizedDifference(['B8','B4'])
             .rename('NDVI')
             .multiply(10000)
             .toInt32()
    )

def getS2(feature, idx):
    return ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterBounds(feature) \
        .filterDate(START_DATE, END_DATE) \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',CLOUD_THRESH) \
        .map(maskS2clouds) \
        .map(addS2Variables) \
        .select(BANDS)

def get_monthly_imgs(feature, month, idx):
    start = ee.Date(f'{YEAR}-{month:02d}-01')
    end   = start.advance(INTERVAL, INCREMENT)
    col   = getS2(feature, idx)
    return col.filterDate(start, end).median().unmask(NO_DATA_VALUE)

# Export per-zone grids
def export_zone_grids():
    logging.info("Starting export of per-zone grids…")
    utm_zone = utm_grid.filterBounds(country_fc)
    zones = list(utm_zone.aggregate_histogram('ZONE').getInfo().keys())
    zone_info = []
    for z in zones:
        zone = int(z)
        zone_poly = utm_grid.filter(ee.Filter.eq('ZONE', zone)).geometry()
        clipped = country_fc.geometry().intersection(zone_poly, 1)
        try:
            area_sq_m = clipped.area().getInfo()
        except Exception as e:
            logging.error(f"Error computing area for zone {zone}: {e}")
            continue
        if area_sq_m == 0:
            continue
        epsg = int(CRS.from_dict({'proj':'utm','zone':zone,'south':SOUTH})
                   .to_authority()[1])
        crs  = f"EPSG:{epsg}"
        grid = clipped.coveringGrid(crs, GRID_SIZE).map(lambda f: f.set('ZONE', zone))
        asset_id = f"{ASSET_FOLDER}{COUNTRY_NAME}_utm_grid_{GRID_SIZE//1000}km_zone{zone}"
        if not asset_exists(asset_id):
            logging.info(f"  Exporting grid asset {asset_id}")
            task = ee.batch.Export.table.toAsset(
                collection=grid,
                description=f"{COUNTRY_NAME}_grid_{GRID_SIZE}_zone{zone}",
                assetId=asset_id
            )
            task.start()
            while task.active():
                logging.info(f"    Waiting for export of zone {zone}…")
                time.sleep(10)
            logging.info(f"  Export of zone {zone} completed.")
        else:
            logging.info(f"  Asset {asset_id} already exists, skipping export.")
        zone_info.append((zone, epsg, crs, asset_id))
    if not zone_info:
        logging.error("No UTM zones found; exiting.")
        sys.exit(1)
    logging.info("Finished exporting all zone grids.")
    return zone_info

# Download images per feature
def download_images(params):
    feature, month, idx, crs, epsg = params
    out_dir = os.path.join(OUTPUT_DIR, COUNTRY_NAME, YEAR, f"{month:02d}")
    os.makedirs(out_dir, exist_ok=True)
    filename = (f"{SATELLITE}_{COUNTRY_NAME}_{YEAR}_{month:02d}_"
                f"{INCREMENT}ly_{RES}m_z{epsg}_{idx}.tif")
    outp = os.path.join(out_dir, filename)
    if os.path.exists(outp):
        return outp
    for i in range(MAX_RETRIES):
        try:
            img = get_monthly_imgs(feature.geometry(), month, idx)
            img_utm = img.reproject(crs, None, RES)
            url = img_utm.getDownloadURL({
                'bands': BANDS,
                'region': feature.geometry(),
                'scale': RES,
                'format':'GEO_TIFF'
            })
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            with open(outp, 'wb') as f:
                f.write(r.content)
            with rasterio.open(outp, 'r+') as dst:
                for bi, bn in enumerate(BANDS, start=1):
                    dst.set_band_description(bi, bn)
            return outp
        except Exception as e:
            logging.warning(f"    Retry {i+1}/{MAX_RETRIES} failed: {e}")
            time.sleep(BASE_WAIT * (2**i))
    logging.error(f"Failed to download after retries: {filename}")
    return None

# Main
def main():
    zone_info = export_zone_grids()
    months = list(range(1, 13))
    cores = max(1, multiprocessing.cpu_count() - 1)
    for zone, epsg, crs, aid in zone_info:
        fc = ee.FeatureCollection(aid)
        lst = fc.toList(fc.size()).getInfo()
        params = [(ee.Feature(lst[i]), m, i, crs, epsg)
                  for i in range(len(lst)) for m in months]
        logging.info(f"  Total tasks for this zone: {len(params)}")
        with multiprocessing.Pool(cores) as pool:
            results = list(tqdm(pool.imap(download_images, params), total=len(params)))
        succeeded = sum(r is not None for r in results)
        logging.info(f"Zone {zone}: {succeeded}/{len(params)} succeeded.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
