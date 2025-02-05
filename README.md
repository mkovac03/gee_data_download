# GEE Data Download Script

This project automates the downloading of Sentinel-2 NDVI data using Google Earth Engine (GEE). The script processes monthly NDVI composites over a specified country and time period, leveraging multiprocessing for faster downloads.

## Features

- **Automated Grid Export:** Exports a UTM grid to GEE assets.
- **Cloud Masking:** Uses Sentinel-2 QA bands to mask clouds and cirrus.
- **Monthly NDVI Composites:** Generates monthly median NDVI composites.
- **Multiprocessing:** Utilizes multiple CPU cores for parallel downloads.
- **Customizable Configurations:** Adjustable parameters via a `config.json` file.
- **Environment Setup:** Easily set up a Conda environment with all required dependencies.

## Prerequisites

- **Python 3.8+**
- [Google Earth Engine Python API](https://developers.google.com/earth-engine/python_install)
- **Conda** (Recommended for environment management)

## Environment Setup

To ensure all dependencies are correctly installed, use the provided `environment.yml` file to set up a Conda environment.

### 1. Create the Conda Environment

Run the following command in your terminal:

```
conda env create -f environment.yml
```

### 2. Activate the Environment

Activate the environment using:

```
conda activate gee_ndvi_download
```

### 3. Authenticate Google Earth Engine (First-Time Setup)

The script will prompt for authentication the first time it is run. Alternatively, you can manually authenticate by running:

```
earthengine authenticate
```

---

## Configuration

Create a `config.json` file in the project root directory with the following content:

```json
{
  "START_DATE": "2018-01-01",
  "END_DATE": "2019-01-01",
  "CLOUD_THRESH": 60,
  "RES": 10,
  "GRID_SIZE": 5000,
  "COUNTRY_NAME": "Denmark",
  "SATELLITE": "S2",
  "INCREMENT": "month",
  "INTERVAL": 1,
  "ASSET_FOLDER": "projects/ee-gmkovacs/assets/grids/",
  "BANDS": ["NDVI"],
  "OUTPUT_DIR": "./output"
}
```

### Parameter Descriptions:

- **START_DATE**: Start date for image collection.
- **END_DATE**: End date for image collection.
- **CLOUD_THRESH**: Maximum allowed cloud cover percentage.
- **RES**: Resolution of output images in meters.
- **GRID_SIZE**: Size of the grid cells (in meters).
- **COUNTRY_NAME**: Country for which data will be downloaded.
- **SATELLITE**: Satellite source (e.g., `S2` for Sentinel-2).
- **INCREMENT**: Time increment for composites (`month`).
- **INTERVAL**: Interval length (e.g., 1 month).
- **ASSET_FOLDER**: Google Earth Engine asset folder for grids.
- **BANDS**: List of bands to download (e.g., `NDVI`).
- **OUTPUT_DIR**: Root directory where output files will be saved.

---

## Usage

### 1. Run the Script

Execute the script to start downloading NDVI data:

```
python gee_ndvi_download.py
```

### 2. Output

The downloaded NDVI images will be saved in the directory specified by the `OUTPUT_DIR` parameter in `config.json`. The output structure will look like this:

```
./output/S2_Denmark_2018/Month_01/S2_Denmark_2018_01_monthly_median_10m_NDVI_0.tif
./output/S2_Denmark_2018/Month_02/S2_Denmark_2018_02_monthly_median_10m_NDVI_0.tif
...
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Google Earth Engine](https://earthengine.google.com/)
- [Sentinel-2 Data](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)

---

Feel free to modify the `config.json` file to suit different countries, time periods, or bands. Happy mapping! 🌍🌿

