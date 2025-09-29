#!/usr/bin/env python3
"""
MODIS Data Preprocessor for BloomWatch Project

This script preprocesses NASA MODIS MOD13Q1 HDF files by extracting NDVI and EVI bands
and converting them to NumPy arrays for further analysis.

Features:
- Reads .hdf files from data/raw/MODIS/
- Extracts NDVI (band 1) and EVI (band 2) arrays
- Saves processed arrays as .npy files in data/processed/MODIS/
- Skips already processed files
- Provides CLI interface for batch and single file processing

Usage:
    python data/preprocess_modis.py --all
    python data/preprocess_modis.py --file "MOD13Q1.A2022013.h31v08.061.2022025132825.hdf"
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json

import numpy as np

try:
    import h5py
except ImportError:
    print("Error: h5py library not found. Please install it with: pip install h5py")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('modis_preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MODISPreprocessor:
    """Handles preprocessing of MODIS MOD13Q1 HDF files."""
    
    def __init__(self, raw_dir: str = "data/raw/MODIS", processed_dir: str = "data/processed/MODIS"):
        """
        Initialize the MODIS preprocessor.
        
        Args:
            raw_dir: Directory containing raw .hdf files
            processed_dir: Directory to save processed .npy files
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # MODIS MOD13Q1 band information
        self.band_info = {
            'ndvi': {
                'band_name': '250m 16 days NDVI',
                'band_index': 0,  # First band in the dataset
                'scale_factor': 0.0001,
                'fill_value': -3000
            },
            'evi': {
                'band_name': '250m 16 days EVI', 
                'band_index': 1,  # Second band in the dataset
                'scale_factor': 0.0001,
                'fill_value': -3000
            }
        }
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_size_mb': 0.0
        }
    
    def list_hdf_files(self) -> List[Path]:
        """
        List all .hdf files in the raw directory.
        
        Returns:
            List of Path objects for .hdf files
        """
        if not self.raw_dir.exists():
            logger.warning(f"Raw directory {self.raw_dir} does not exist")
            return []
        
        hdf_files = list(self.raw_dir.glob("*.hdf"))
        logger.info(f"Found {len(hdf_files)} .hdf files in {self.raw_dir}")
        
        return hdf_files
    
    def is_file_processed(self, hdf_file: Path) -> bool:
        """
        Check if a file has already been processed.
        
        Args:
            hdf_file: Path to the .hdf file
            
        Returns:
            True if both NDVI and EVI .npy files exist, False otherwise
        """
        base_name = hdf_file.stem
        ndvi_file = self.processed_dir / f"{base_name}_ndvi.npy"
        evi_file = self.processed_dir / f"{base_name}_evi.npy"
        
        return ndvi_file.exists() and evi_file.exists()
    
    def extract_ndvi_evi(self, hdf_file: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Extract NDVI and EVI arrays from a single .hdf file.
        
        Args:
            hdf_file: Path to the .hdf file
            
        Returns:
            Tuple of (ndvi_array, evi_array, metadata)
        """
        try:
            logger.info(f"Processing {hdf_file.name}")
            
            with h5py.File(hdf_file, 'r') as f:
                # Navigate to the MODIS data structure
                # MOD13Q1 structure: HDF4_EOS:EOS_GRID:"MODIS_Grid_16DAY_250m_500m_VI":250m 16 days VI
                try:
                    # Try different possible paths in the HDF structure
                    data_paths = [
                        'HDF4_EOS/EOS_GRID/MOD13Q1/Data Fields/250m 16 days NDVI',
                        'HDF4_EOS/EOS_GRID/MOD13Q1/Data Fields/250m 16 days EVI',
                        'MODIS_Grid_16DAY_250m_500m_VI/Data Fields/250m 16 days NDVI',
                        'MODIS_Grid_16DAY_250m_500m_VI/Data Fields/250m 16 days EVI'
                    ]
                    
                    # Find the correct data group
                    data_group = None
                    for path in data_paths:
                        if path in f:
                            data_group = f[path]
                            break
                    
                    if data_group is None:
                        # Try to find any group containing MODIS data
                        def find_modis_data(name, obj):
                            if isinstance(obj, h5py.Dataset) and 'NDVI' in str(name):
                                return obj
                            return None
                        
                        modis_data = f.visititems(find_modis_data)
                        if modis_data is None:
                            raise ValueError("Could not find MODIS data in HDF file")
                        data_group = modis_data.parent
                    
                    # Extract NDVI and EVI data
                    ndvi_data = None
                    evi_data = None
                    
                    for key in data_group.keys():
                        if 'NDVI' in key:
                            ndvi_data = data_group[key][:]
                        elif 'EVI' in key:
                            evi_data = data_group[key][:]
                    
                    if ndvi_data is None or evi_data is None:
                        raise ValueError("Could not find NDVI or EVI data in HDF file")
                    
                    # Apply scale factor and handle fill values
                    ndvi_scaled = ndvi_data.astype(np.float32) * self.band_info['ndvi']['scale_factor']
                    evi_scaled = evi_data.astype(np.float32) * self.band_info['evi']['scale_factor']
                    
                    # Set fill values to NaN
                    ndvi_scaled[ndvi_data == self.band_info['ndvi']['fill_value']] = np.nan
                    evi_scaled[evi_data == self.band_info['evi']['fill_value']] = np.nan
                    
                    # Extract metadata
                    metadata = {
                        'filename': hdf_file.name,
                        'ndvi_shape': ndvi_scaled.shape,
                        'evi_shape': evi_scaled.shape,
                        'ndvi_min': np.nanmin(ndvi_scaled),
                        'ndvi_max': np.nanmax(ndvi_scaled),
                        'evi_min': np.nanmin(evi_scaled),
                        'evi_max': np.nanmax(evi_scaled),
                        'ndvi_valid_pixels': np.sum(~np.isnan(ndvi_scaled)),
                        'evi_valid_pixels': np.sum(~np.isnan(evi_scaled))
                    }
                    
                    logger.info(f"Successfully extracted NDVI {ndvi_scaled.shape} and EVI {evi_scaled.shape}")
                    return ndvi_scaled, evi_scaled, metadata
                    
                except Exception as e:
                    logger.error(f"Error reading HDF structure: {e}")
                    # Try alternative approach - direct dataset access
                    return self._extract_alternative(hdf_file)
                    
        except Exception as e:
            logger.error(f"Error processing {hdf_file.name}: {e}")
            return None, None, {}
    
    def _extract_alternative(self, hdf_file: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Alternative extraction method using different HDF reading approach.
        
        Args:
            hdf_file: Path to the .hdf file
            
        Returns:
            Tuple of (ndvi_array, evi_array, metadata)
        """
        try:
            import gdal
            from osgeo import gdal
            
            # Use GDAL to read the HDF file
            dataset = gdal.Open(str(hdf_file))
            if dataset is None:
                raise ValueError("Could not open HDF file with GDAL")
            
            # Get subdatasets (MODIS HDF files contain multiple datasets)
            subdatasets = dataset.GetMetadata('SUBDATASETS')
            
            ndvi_dataset = None
            evi_dataset = None
            
            for key, value in subdatasets.items():
                if 'NDVI' in value and '250m' in value:
                    ndvi_dataset = gdal.Open(value)
                elif 'EVI' in value and '250m' in value:
                    evi_dataset = gdal.Open(value)
            
            if ndvi_dataset is None or evi_dataset is None:
                raise ValueError("Could not find NDVI or EVI subdatasets")
            
            # Read the data
            ndvi_data = ndvi_dataset.ReadAsArray().astype(np.float32)
            evi_data = evi_dataset.ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            ndvi_scaled = ndvi_data * self.band_info['ndvi']['scale_factor']
            evi_scaled = evi_data * self.band_info['evi']['scale_factor']
            
            # Set fill values to NaN
            ndvi_scaled[ndvi_data == self.band_info['ndvi']['fill_value']] = np.nan
            evi_scaled[evi_data == self.band_info['evi']['fill_value']] = np.nan
            
            metadata = {
                'filename': hdf_file.name,
                'ndvi_shape': ndvi_scaled.shape,
                'evi_shape': evi_scaled.shape,
                'ndvi_min': np.nanmin(ndvi_scaled),
                'ndvi_max': np.nanmax(ndvi_scaled),
                'evi_min': np.nanmin(evi_scaled),
                'evi_max': np.nanmax(evi_scaled),
                'ndvi_valid_pixels': np.sum(~np.isnan(ndvi_scaled)),
                'evi_valid_pixels': np.sum(~np.isnan(evi_scaled))
            }
            
            return ndvi_scaled, evi_scaled, metadata
            
        except ImportError:
            logger.error("GDAL not available for alternative extraction method")
            return None, None, {}
        except Exception as e:
            logger.error(f"Alternative extraction failed: {e}")
            return None, None, {}
    
    def save_processed_arrays(self, ndvi: np.ndarray, evi: np.ndarray, filename: str, metadata: Dict[str, Any]) -> bool:
        """
        Save processed NDVI and EVI arrays to .npy files.
        
        Args:
            ndvi: NDVI array
            evi: EVI array
            filename: Base filename (without extension)
            metadata: Metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            base_name = Path(filename).stem
            
            # Save NDVI array
            ndvi_file = self.processed_dir / f"{base_name}_ndvi.npy"
            np.save(ndvi_file, ndvi)
            
            # Save EVI array
            evi_file = self.processed_dir / f"{base_name}_evi.npy"
            np.save(evi_file, evi)
            
            # Save metadata
            metadata_file = self.processed_dir / f"{base_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update statistics
            file_size_mb = (ndvi_file.stat().st_size + evi_file.stat().st_size) / (1024 * 1024)
            self.stats['total_size_mb'] += file_size_mb
            
            logger.info(f"Saved processed arrays: {base_name}_ndvi.npy, {base_name}_evi.npy")
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed arrays for {filename}: {e}")
            return False
    
    def preprocess_all(self) -> Dict[str, Any]:
        """
        Process all .hdf files in the raw directory.
        
        Returns:
            Dictionary with processing statistics
        """
        hdf_files = self.list_hdf_files()
        
        if not hdf_files:
            logger.warning("No .hdf files found to process")
            return self.stats
        
        self.stats['total_files'] = len(hdf_files)
        
        logger.info(f"Starting preprocessing of {len(hdf_files)} files...")
        logger.info(f"Output directory: {self.processed_dir}")
        
        for i, hdf_file in enumerate(hdf_files, 1):
            logger.info(f"Processing file {i}/{len(hdf_files)}: {hdf_file.name}")
            
            # Check if already processed
            if self.is_file_processed(hdf_file):
                logger.info(f"Skipping {hdf_file.name} (already processed)")
                self.stats['skipped_files'] += 1
                continue
            
            # Extract NDVI and EVI
            ndvi, evi, metadata = self.extract_ndvi_evi(hdf_file)
            
            if ndvi is not None and evi is not None:
                # Save processed arrays
                if self.save_processed_arrays(ndvi, evi, hdf_file.name, metadata):
                    self.stats['processed_files'] += 1
                    logger.info(f"✓ Successfully processed {hdf_file.name}")
                else:
                    self.stats['failed_files'] += 1
                    logger.error(f"✗ Failed to save {hdf_file.name}")
            else:
                self.stats['failed_files'] += 1
                logger.error(f"✗ Failed to extract data from {hdf_file.name}")
        
        # Log final statistics
        logger.info(f"\nProcessing Summary:")
        logger.info(f"  Total files: {self.stats['total_files']}")
        logger.info(f"  Successfully processed: {self.stats['processed_files']}")
        logger.info(f"  Skipped (already processed): {self.stats['skipped_files']}")
        logger.info(f"  Failed: {self.stats['failed_files']}")
        logger.info(f"  Total output size: {self.stats['total_size_mb']:.2f} MB")
        
        return self.stats
    
    def preprocess_single_file(self, filename: str) -> bool:
        """
        Process a single .hdf file.
        
        Args:
            filename: Name of the .hdf file to process
            
        Returns:
            True if successful, False otherwise
        """
        hdf_file = self.raw_dir / filename
        
        if not hdf_file.exists():
            logger.error(f"File {filename} not found in {self.raw_dir}")
            return False
        
        logger.info(f"Processing single file: {filename}")
        
        # Check if already processed
        if self.is_file_processed(hdf_file):
            logger.info(f"File {filename} already processed")
            return True
        
        # Extract NDVI and EVI
        ndvi, evi, metadata = self.extract_ndvi_evi(hdf_file)
        
        if ndvi is not None and evi is not None:
            # Save processed arrays
            if self.save_processed_arrays(ndvi, evi, filename, metadata):
                logger.info(f"✓ Successfully processed {filename}")
                return True
            else:
                logger.error(f"✗ Failed to save {filename}")
                return False
        else:
            logger.error(f"✗ Failed to extract data from {filename}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return self.stats.copy()


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Preprocess MODIS MOD13Q1 HDF files to extract NDVI and EVI arrays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all .hdf files in the raw directory
  python data/preprocess_modis.py --all
  
  # Process a specific file
  python data/preprocess_modis.py --file "MOD13Q1.A2022013.h31v08.061.2022025132825.hdf"
  
  # Process with custom directories
  python data/preprocess_modis.py --all --raw-dir "custom/raw" --processed-dir "custom/processed"
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all .hdf files in the raw directory"
    )
    group.add_argument(
        "--file",
        help="Process a specific .hdf file"
    )
    
    parser.add_argument(
        "--raw-dir",
        default="data/raw/MODIS",
        help="Directory containing raw .hdf files (default: data/raw/MODIS)"
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed/MODIS",
        help="Directory to save processed .npy files (default: data/processed/MODIS)"
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = MODISPreprocessor(args.raw_dir, args.processed_dir)
    
    try:
        if args.all:
            # Process all files
            stats = preprocessor.preprocess_all()
            return 0 if stats['failed_files'] == 0 else 1
        else:
            # Process single file
            success = preprocessor.preprocess_single_file(args.file)
            return 0 if success else 1
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
