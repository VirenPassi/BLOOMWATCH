# MODIS Data Preprocessing

This directory contains tools for preprocessing NASA MODIS MOD13Q1 HDF files for the BloomWatch project.

## Files

- `preprocess_modis.py` - Main script for preprocessing MODIS HDF files
- `raw/MODIS/` - Directory containing raw .hdf files (from fetch_modis.py)
- `processed/MODIS/` - Directory containing processed .npy files

## Setup

1. Install the required dependencies:
   ```bash
   pip install h5py gdal
   ```

2. Ensure you have raw MODIS data:
   - Use `fetch_modis.py` to download .hdf files first
   - Raw files should be in `data/raw/MODIS/`

## Usage

### Process All Files

Preprocess all .hdf files in the raw directory:

```bash
python data/preprocess_modis.py --all
```

### Process Single File

Process a specific .hdf file:

```bash
python data/preprocess_modis.py --file "MOD13Q1.A2022013.h31v08.061.2022025132825.hdf"
```

### Custom Directories

Specify custom input and output directories:

```bash
python data/preprocess_modis.py --all --raw-dir "custom/raw" --processed-dir "custom/processed"
```

## What It Does

The preprocessor extracts and processes the following data from MODIS MOD13Q1 HDF files:

### Input: MODIS MOD13Q1 HDF Files
- **NDVI (Normalized Difference Vegetation Index)** - Band 1
- **EVI (Enhanced Vegetation Index)** - Band 2
- 250m spatial resolution
- 16-day temporal resolution

### Output: Processed NumPy Arrays
- `{filename}_ndvi.npy` - NDVI array as float32
- `{filename}_evi.npy` - EVI array as float32  
- `{filename}_metadata.json` - Processing metadata

### Data Processing
1. **Scale Factor Application**: Raw values multiplied by 0.0001
2. **Fill Value Handling**: Invalid pixels set to NaN
3. **Data Type Conversion**: Converted to float32 for efficiency
4. **Quality Control**: Tracks valid pixel counts and value ranges

## File Structure

```
data/
├── raw/
│   └── MODIS/
│       ├── MOD13Q1.A2022013.h31v08.061.2022025132825.hdf
│       └── MOD13Q1.A2022029.h31v08.061.2022031132825.hdf
└── processed/
    └── MODIS/
        ├── MOD13Q1.A2022013.h31v08.061.2022025132825_ndvi.npy
        ├── MOD13Q1.A2022013.h31v08.061.2022025132825_evi.npy
        ├── MOD13Q1.A2022013.h31v08.061.2022025132825_metadata.json
        ├── MOD13Q1.A2022029.h31v08.061.2022031132825_ndvi.npy
        ├── MOD13Q1.A2022029.h31v08.061.2022031132825_evi.npy
        └── MOD13Q1.A2022029.h31v08.061.2022031132825_metadata.json
```

## Metadata

Each processed file includes a JSON metadata file with:

```json
{
  "filename": "MOD13Q1.A2022013.h31v08.061.2022025132825.hdf",
  "ndvi_shape": [2400, 2400],
  "evi_shape": [2400, 2400],
  "ndvi_min": -0.2,
  "ndvi_max": 0.9,
  "evi_min": -0.2,
  "evi_max": 0.8,
  "ndvi_valid_pixels": 5760000,
  "evi_valid_pixels": 5760000
}
```

## Features

### Smart Processing
- **Skip Already Processed**: Automatically skips files that have been processed
- **Progress Logging**: Detailed progress information and statistics
- **Error Handling**: Robust error handling with detailed error messages
- **Memory Efficient**: Processes files one at a time to manage memory usage

### Quality Control
- **Fill Value Detection**: Identifies and handles invalid pixels
- **Data Validation**: Checks array shapes and data ranges
- **Statistics Tracking**: Monitors processing success/failure rates

### Flexible Input/Output
- **Multiple HDF Formats**: Handles different MODIS HDF file structures
- **Custom Directories**: Specify custom input and output locations
- **Batch Processing**: Process all files or individual files

## Dependencies

- `h5py>=3.8.0` - HDF5 file reading
- `gdal>=3.6.0` - Alternative HDF reading method
- `numpy>=1.24.0` - Array processing

## Troubleshooting

### Common Issues

1. **"Could not find MODIS data in HDF file"**
   - The HDF file structure may be different than expected
   - Try using GDAL as an alternative reading method

2. **Memory errors with large files**
   - MODIS files can be large (2400x2400 pixels)
   - Ensure sufficient RAM (recommended: 8GB+)

3. **Permission errors**
   - Ensure write permissions to the processed directory
   - Check that the directory exists and is accessible

### Performance Tips

- Process files during off-peak hours for better performance
- Use SSD storage for faster I/O operations
- Monitor disk space as .npy files can be large

## Integration with BloomWatch

The processed .npy files can be directly loaded in your BloomWatch analysis:

```python
import numpy as np

# Load processed arrays
ndvi = np.load('data/processed/MODIS/MOD13Q1.A2022013.h31v08.061.2022025132825_ndvi.npy')
evi = np.load('data/processed/MODIS/MOD13Q1.A2022013.h31v08.061.2022025132825_evi.npy')

# Arrays are ready for analysis
print(f"NDVI shape: {ndvi.shape}")
print(f"NDVI range: {np.nanmin(ndvi):.3f} to {np.nanmax(ndvi):.3f}")
```
