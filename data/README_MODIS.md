# MODIS Data Fetcher

This directory contains tools for downloading MODIS satellite data for the BloomWatch project.

## Files

- `fetch_modis.py` - Main script for downloading MODIS MOD13Q1 vegetation indices data
- `raw/MODIS/` - Directory where downloaded .hdf files are stored

## Setup

1. Install the required dependency:
   ```bash
   pip install earthaccess
   ```

2. Get a NASA Earthdata account:
   - Visit https://urs.earthdata.nasa.gov/
   - Create an account and log in
   - Accept the terms for MODIS data access

## Usage

### Basic Usage

Download MODIS data for a specific region and time period:

```bash
python data/fetch_modis.py --start 2022-01-01 --end 2022-12-31 --bbox "70,8,90,37"
```

### List Available Data

See what data is available without downloading:

```bash
python data/fetch_modis.py --start 2022-01-01 --end 2022-01-31 --list-only
```

### Advanced Options

```bash
python data/fetch_modis.py \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --bbox "70,8,90,37" \
  --cloud-cover 20 \
  --output-dir "custom/path" \
  --max-workers 8
```

## Parameters

- `--start` - Start date in YYYY-MM-DD format (required)
- `--end` - End date in YYYY-MM-DD format (required)
- `--bbox` - Bounding box as "min_lon,min_lat,max_lon,max_lat" (optional)
- `--cloud-cover` - Maximum cloud cover percentage 0-100 (optional)
- `--output-dir` - Output directory for files (default: data/raw/MODIS)
- `--list-only` - Only list available granules, don't download
- `--max-workers` - Number of concurrent downloads (default: 4)
- `--username` - NASA Earthdata username (will prompt if not provided)
- `--password` - NASA Earthdata password (will prompt if not provided)

## Data Format

Downloaded files are in HDF4 format (.hdf) containing:
- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- Quality flags
- Pixel reliability information

## Authentication

The script will prompt for NASA Earthdata credentials if not provided via command line. You can also set them as environment variables:

```bash
export EARTHDATA_USERNAME="your_username"
export EARTHDATA_PASSWORD="your_password"
```

## Examples

### Download data for India (approximate bounding box)
```bash
python data/fetch_modis.py --start 2022-01-01 --end 2022-12-31 --bbox "68,6,97,37"
```

### Download data for a small region with low cloud cover
```bash
python data/fetch_modis.py --start 2022-06-01 --end 2022-06-30 --bbox "77,12,78,13" --cloud-cover 10
```

### Check what data is available for a month
```bash
python data/fetch_modis.py --start 2022-01-01 --end 2022-01-31 --bbox "70,8,90,37" --list-only
```
