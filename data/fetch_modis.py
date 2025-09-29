#!/usr/bin/env python3
"""
MODIS Data Fetcher for BloomWatch Project

This script automates the download of MODIS MOD13Q1 (Vegetation Indices 16-Day, 250m) 
granules from NASA Earthdata using the earthaccess library.

Features:
- Authenticates with NASA Earthdata
- Downloads MODIS MOD13Q1 granules for specified date ranges and bounding boxes
- Automatically skips already-downloaded files
- Provides progress logging and bulk download capabilities
- Includes CLI interface for easy usage

Usage:
    python data/fetch_modis.py --start 2022-01-01 --end 2022-12-31 --bbox "70,8,90,37"
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json

try:
    import earthaccess
except ImportError:
    print("Error: earthaccess library not found. Please install it with: pip install earthaccess")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('modis_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MODISFetcher:
    """Handles MODIS data discovery and downloading from NASA Earthdata."""
    
    def __init__(self, output_dir: str = "data/raw/MODIS"):
        """
        Initialize the MODIS fetcher.
        
        Args:
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MODIS MOD13Q1 collection details
        self.collection = "MODIS/061/MOD13Q1"
        self.granule_pattern = "*.hdf"
        
        # Initialize earthaccess
        self.auth = None
        
    def authenticate(self, username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        Authenticate with NASA Earthdata.
        
        Args:
            username: NASA Earthdata username (optional, will prompt if not provided)
            password: NASA Earthdata password (optional, will prompt if not provided)
            
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            if username and password:
                self.auth = earthaccess.login(username, password)
            else:
                self.auth = earthaccess.login()
            
            if self.auth:
                logger.info("Successfully authenticated with NASA Earthdata")
                return True
            else:
                logger.error("Failed to authenticate with NASA Earthdata")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def search_granules(
        self, 
        start_date: str, 
        end_date: str, 
        bbox: Optional[Tuple[float, float, float, float]] = None,
        cloud_cover: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for MODIS MOD13Q1 granules matching the specified criteria.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            cloud_cover: Maximum cloud cover percentage (0-100)
            
        Returns:
            List of granule metadata dictionaries
        """
        if not self.auth:
            logger.error("Not authenticated. Please call authenticate() first.")
            return []
        
        try:
            # Convert dates to datetime objects
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Build search parameters
            search_params = {
                "short_name": "MOD13Q1",
                "version": "061",
                "temporal": (start_dt, end_dt),
                "count": 10000  # Large count to get all results
            }
            
            if bbox:
                search_params["bounding_box"] = bbox
                
            if cloud_cover is not None:
                search_params["cloud_cover"] = (0, cloud_cover)
            
            logger.info(f"Searching for MODIS granules from {start_date} to {end_date}")
            if bbox:
                logger.info(f"Bounding box: {bbox}")
            if cloud_cover is not None:
                logger.info(f"Max cloud cover: {cloud_cover}%")
            
            # Search for granules
            results = earthaccess.search_data(**search_params)
            
            logger.info(f"Found {len(results)} granules")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for granules: {e}")
            return []
    
    def list_granules(self, granules: List[Dict[str, Any]]) -> None:
        """
        Print information about available granules.
        
        Args:
            granules: List of granule metadata dictionaries
        """
        if not granules:
            logger.info("No granules found")
            return
        
        logger.info(f"\nFound {len(granules)} granules:")
        logger.info("-" * 80)
        
        for i, granule in enumerate(granules[:10]):  # Show first 10
            try:
                # Extract granule information
                granule_id = granule.get('umm', {}).get('Granule', {}).get('GranuleUR', 'Unknown')
                temporal = granule.get('umm', {}).get('TemporalExtent', {})
                start_time = temporal.get('SingleDateTime', 'Unknown')
                
                # Get file size
                size_info = granule.get('umm', {}).get('DataGranule', {}).get('ArchiveAndDistributionInformation', [])
                size = "Unknown"
                if size_info and len(size_info) > 0:
                    size = size_info[0].get('Size', 'Unknown')
                
                logger.info(f"{i+1:3d}. {granule_id}")
                logger.info(f"     Time: {start_time}")
                logger.info(f"     Size: {size}")
                
            except Exception as e:
                logger.warning(f"Error parsing granule {i+1}: {e}")
        
        if len(granules) > 10:
            logger.info(f"... and {len(granules) - 10} more granules")
    
    def get_granule_filename(self, granule: Dict[str, Any]) -> str:
        """
        Extract filename from granule metadata.
        
        Args:
            granule: Granule metadata dictionary
            
        Returns:
            Filename for the granule
        """
        try:
            # Try to get filename from various possible locations
            granule_ur = granule.get('umm', {}).get('Granule', {}).get('GranuleUR', '')
            if granule_ur and granule_ur.endswith('.hdf'):
                return granule_ur
            
            # Alternative: construct filename from temporal info
            temporal = granule.get('umm', {}).get('TemporalExtent', {})
            start_time = temporal.get('SingleDateTime', '')
            
            if start_time:
                # Parse date and construct MODIS filename
                dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                date_str = dt.strftime('%Y%j')  # YYYYDDD format
                return f"MOD13Q1.A{date_str}.h{dt.hour:02d}v{dt.minute:02d}.061.hdf"
            
            return "unknown_granule.hdf"
            
        except Exception as e:
            logger.warning(f"Error extracting filename: {e}")
            return "unknown_granule.hdf"
    
    def is_file_downloaded(self, filename: str) -> bool:
        """
        Check if a file has already been downloaded.
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if file exists and has content, False otherwise
        """
        file_path = self.output_dir / filename
        return file_path.exists() and file_path.stat().st_size > 0
    
    def download_granules(
        self, 
        granules: List[Dict[str, Any]], 
        skip_existing: bool = True,
        max_workers: int = 4
    ) -> List[str]:
        """
        Download MODIS granules to the output directory.
        
        Args:
            granules: List of granule metadata dictionaries
            skip_existing: Whether to skip already downloaded files
            max_workers: Maximum number of concurrent downloads
            
        Returns:
            List of successfully downloaded filenames
        """
        if not granules:
            logger.info("No granules to download")
            return []
        
        if not self.auth:
            logger.error("Not authenticated. Please call authenticate() first.")
            return []
        
        downloaded_files = []
        skipped_files = []
        
        logger.info(f"Starting download of {len(granules)} granules...")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Skip existing files: {skip_existing}")
        
        try:
            # Download files
            results = earthaccess.download(
                granules, 
                local_path=str(self.output_dir),
                max_workers=max_workers
            )
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, str) and result.endswith('.hdf'):
                    # Successful download
                    filename = os.path.basename(result)
                    downloaded_files.append(filename)
                    logger.info(f"✓ Downloaded: {filename}")
                else:
                    # Failed download
                    logger.warning(f"✗ Failed to download granule {i+1}")
            
            logger.info(f"\nDownload Summary:")
            logger.info(f"  Successfully downloaded: {len(downloaded_files)} files")
            logger.info(f"  Failed downloads: {len(granules) - len(downloaded_files)} files")
            
        except Exception as e:
            logger.error(f"Error during download: {e}")
        
        return downloaded_files
    
    def get_download_stats(self) -> Dict[str, Any]:
        """
        Get statistics about downloaded files.
        
        Returns:
            Dictionary with download statistics
        """
        hdf_files = list(self.output_dir.glob("*.hdf"))
        
        total_size = sum(f.stat().st_size for f in hdf_files)
        total_size_mb = total_size / (1024 * 1024)
        
        return {
            "total_files": len(hdf_files),
            "total_size_mb": round(total_size_mb, 2),
            "files": [f.name for f in hdf_files]
        }


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """
    Parse bounding box string into tuple of floats.
    
    Args:
        bbox_str: Bounding box as "min_lon,min_lat,max_lon,max_lat"
        
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        return tuple(coords)
    except Exception as e:
        raise ValueError(f"Invalid bounding box format: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Download MODIS MOD13Q1 vegetation indices data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data for a specific region and time period
  python data/fetch_modis.py --start 2022-01-01 --end 2022-12-31 --bbox "70,8,90,37"
  
  # List available granules without downloading
  python data/fetch_modis.py --start 2022-01-01 --end 2022-01-31 --list-only
  
  # Download with cloud cover filter
  python data/fetch_modis.py --start 2022-01-01 --end 2022-01-31 --cloud-cover 20
        """
    )
    
    parser.add_argument(
        "--start", 
        required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end", 
        required=True,
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--bbox",
        help="Bounding box as 'min_lon,min_lat,max_lon,max_lat' (e.g., '70,8,90,37')"
    )
    parser.add_argument(
        "--cloud-cover",
        type=float,
        help="Maximum cloud cover percentage (0-100)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/MODIS",
        help="Output directory for downloaded files (default: data/raw/MODIS)"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available granules, don't download"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent downloads (default: 4)"
    )
    parser.add_argument(
        "--username",
        help="NASA Earthdata username (will prompt if not provided)"
    )
    parser.add_argument(
        "--password",
        help="NASA Earthdata password (will prompt if not provided)"
    )
    
    args = parser.parse_args()
    
    # Parse bounding box if provided
    bbox = None
    if args.bbox:
        try:
            bbox = parse_bbox(args.bbox)
            logger.info(f"Using bounding box: {bbox}")
        except ValueError as e:
            logger.error(f"Error parsing bounding box: {e}")
            return 1
    
    # Initialize fetcher
    fetcher = MODISFetcher(args.output_dir)
    
    # Authenticate
    logger.info("Authenticating with NASA Earthdata...")
    if not fetcher.authenticate(args.username, args.password):
        logger.error("Authentication failed. Exiting.")
        return 1
    
    # Search for granules
    granules = fetcher.search_granules(
        start_date=args.start,
        end_date=args.end,
        bbox=bbox,
        cloud_cover=args.cloud_cover
    )
    
    if not granules:
        logger.warning("No granules found matching the specified criteria")
        return 0
    
    # List granules
    fetcher.list_granules(granules)
    
    if args.list_only:
        logger.info("List-only mode: skipping download")
        return 0
    
    # Download granules
    downloaded = fetcher.download_granules(
        granules, 
        skip_existing=True,
        max_workers=args.max_workers
    )
    
    # Show final statistics
    stats = fetcher.get_download_stats()
    logger.info(f"\nFinal Statistics:")
    logger.info(f"  Total files: {stats['total_files']}")
    logger.info(f"  Total size: {stats['total_size_mb']} MB")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())