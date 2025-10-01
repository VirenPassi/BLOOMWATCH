# NASA MODIS Data Fetching Functionality

This document describes the new MODIS data fetching capabilities added to the BloomWatch project.

## Overview

The `data/fetch_modis.py` script enables automated downloading of NASA MODIS MOD13Q1 (Vegetation Indices 16-Day, 250m) satellite data using the `earthaccess` library. This functionality allows researchers to easily obtain vegetation index data for plant bloom analysis and monitoring.

## Features

1. **NASA Earthdata Authentication**: Secure authentication with NASA Earthdata credentials
2. **MODIS MOD13Q1 Data Access**: Download of Vegetation Indices 16-Day, 250m data
3. **Flexible Filtering**: Date range and bounding box (lat/lon) filtering
4. **Bulk Downloading**: Efficient bulk download with progress logging
5. **Smart File Management**: Automatic skipping of already downloaded files
6. **CLI Interface**: Command-line interface for easy execution

## Installation Requirements

The required dependencies are already included in `requirements.txt`:
- `earthaccess>=0.12.0`

## Usage

### Command-Line Interface

```bash
# Basic usage
python data/fetch_modis.py --start 2022-01-01 --end 2022-12-31 --bbox "70,8,90,37"

# List available granules without downloading
python data/fetch_modis.py --start 2022-01-01 --end 2022-12-31 --bbox "70,8,90,37" --list-only

# Force re-download even if files exist
python data/fetch_modis.py --start 2022-01-01 --end 2022-12-31 --bbox "70,8,90,37" --force

# Custom output directory
python data/fetch_modis.py --start 2022-01-01 --end 2022-12-31 --bbox "70,8,90,37" --output data/my_modis_data
```

### Programmatic Usage

```python
from data.fetch_modis import authenticate_earthdata, list_modis_granules, download_granules

# Authenticate with NASA Earthdata
if authenticate_earthdata():
    # List available granules
    granules = list_modis_granules(
        start_date="2022-01-01",
        end_date="2022-12-31",
        bbox=(70, 8, 90, 37)  # (west, south, east, north)
    )
    
    # Download granules
    download_granules(
        granules=granules,
        output_dir="data/raw/MODIS"
    )
```

## Parameters

### Required Parameters
- `--start`: Start date in YYYY-MM-DD format
- `--end`: End date in YYYY-MM-DD format
- `--bbox`: Bounding box in format "west,south,east,north"

### Optional Parameters
- `--output`: Output directory for downloaded files (default: data/raw/MODIS)
- `--list-only`: Only list available granules without downloading
- `--force`: Force download even if files already exist

## Output

Downloaded MODIS HDF files are saved to the specified output directory (default: `data/raw/MODIS/`). The script automatically creates the directory if it doesn't exist.

## Authentication

Before using the script, you need NASA Earthdata credentials:
1. Register at https://urs.earthdata.nasa.gov/
2. The `earthaccess` library will prompt for credentials on first use
3. Credentials are securely stored for future use

## Error Handling

The script includes comprehensive error handling for:
- Authentication failures
- Network issues
- Invalid parameters
- File system errors
- Granule download failures

## Logging

All operations are logged to both console and `modis_download.log` file with timestamps and severity levels.

## Integration with BloomWatch

The MODIS data can be integrated with the BloomWatch pipeline for:
- Vegetation index analysis
- Correlation studies between satellite data and ground observations
- Environmental context for bloom predictions
- Large-scale bloom pattern analysis

## Example Use Cases

1. **Seasonal Analysis**: Download data for multiple years to analyze seasonal bloom patterns
2. **Regional Studies**: Focus on specific geographic regions using bounding boxes
3. **Climate Impact Research**: Correlate vegetation indices with bloom timing
4. **Validation Studies**: Compare satellite observations with ground truth data

## Data Format

The script downloads MOD13Q1 HDF files which contain:
- 16-day composite vegetation indices
- 250m spatial resolution
- Multiple spectral bands
- Quality assurance information

## Future Enhancements

Potential improvements for future versions:
- Support for additional MODIS products
- Integration with data preprocessing pipelines
- Automated extraction of specific data bands
- Cloud detection and filtering
- Parallel downloading for improved performance