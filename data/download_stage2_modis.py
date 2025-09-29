"""
Stage-2 MODIS Data Download for BloomWatch Pipeline

Downloads MODIS MOD13Q1 data for 4-5 AOIs covering diverse climates:
- AOI 1: California Central Valley (Mediterranean)
- AOI 2: Iowa Corn Belt (Continental) 
- AOI 3: Amazon Basin (Tropical)
- AOI 4: Great Plains (Semi-arid)
- AOI 5: Southeast US (Humid subtropical)

Date range: 2023-01-01 to 2024-12-31 (extended temporal coverage)
Target: 7-8GB additional MODIS data
"""

import os
import sys
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import List, Dict, Tuple
import warnings

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Configuration
EARTHDATA_BASE_URL = "https://e4ftl01.cr.usgs.gov/MOLT/MOD13Q1.061/"
STAGE2_RAW_DIR = ROOT / "data" / "raw" / "MODIS" / "stage2"

# Enhanced AOI Definitions (diverse climates)
AOIS = {
    "california_central_valley": {
        "name": "California Central Valley",
        "climate": "Mediterranean",
        "tiles": ["h08v05", "h09v05", "h08v06"],
        "description": "Agricultural region with Mediterranean climate"
    },
    "iowa_corn_belt": {
        "name": "Iowa Corn Belt", 
        "climate": "Continental",
        "tiles": ["h10v04", "h11v04", "h10v05"],
        "description": "Major agricultural region with continental climate"
    },
    "amazon_basin": {
        "name": "Amazon Basin",
        "climate": "Tropical",
        "tiles": ["h10v08", "h11v08", "h10v09"],
        "description": "Tropical rainforest with high vegetation density"
    },
    "great_plains": {
        "name": "Great Plains",
        "climate": "Semi-arid",
        "tiles": ["h09v05", "h10v05", "h09v06"],
        "description": "Grassland region with semi-arid climate"
    },
    "southeast_us": {
        "name": "Southeast US",
        "climate": "Humid subtropical",
        "tiles": ["h12v04", "h12v05", "h13v04"],
        "description": "Humid subtropical region with diverse vegetation"
    }
}

# Extended date range for Stage-2
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Suppress warnings
warnings.filterwarnings("ignore")


def ensure_dirs():
    """Create necessary directories."""
    STAGE2_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for aoi_name in AOIS.keys():
        (STAGE2_RAW_DIR / aoi_name).mkdir(exist_ok=True)


def get_modis_dates(start_date: datetime, end_date: datetime) -> List[str]:
    """Get MODIS 16-day composite dates for the given range."""
    dates = []
    current = start_date
    
    while current <= end_date:
        # MODIS 16-day composites start on specific dates
        year_day = current.timetuple().tm_yday
        composite_day = ((year_day - 1) // 16) * 16 + 1
        
        if composite_day <= 365:
            date_str = current.strftime("%Y-%m-%d")
            dates.append(date_str)
        
        current += timedelta(days=16)
    
    return dates


def download_modis_granule(aoi_name: str, tile: str, date: str, session: requests.Session) -> bool:
    """Download a single MODIS granule."""
    try:
        # Construct filename
        year = date[:4]
        year_day = datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday
        year_day_str = f"{year_day:03d}"
        
        filename = f"MOD13Q1.A{year}{year_day_str}.{tile}.061.hdf"
        url = f"{EARTHDATA_BASE_URL}{year}.{year_day_str:03d}.{date[5:7]}.{date[8:10]}/{filename}"
        
        # Download file
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save to appropriate directory
        save_path = STAGE2_RAW_DIR / aoi_name / filename
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {filename} ({len(response.content) / 1024 / 1024:.1f} MB)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename}: {e}")
        return False
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False


def download_aoi_data(aoi_name: str, aoi_config: Dict, session: requests.Session) -> Dict:
    """Download all MODIS data for a specific AOI."""
    print(f"\nDownloading data for {aoi_config['name']} ({aoi_config['climate']} climate)")
    print(f"   Tiles: {aoi_config['tiles']}")
    print(f"   Description: {aoi_config['description']}")
    
    dates = get_modis_dates(START_DATE, END_DATE)
    print(f"   Date range: {dates[0]} to {dates[-1]} ({len(dates)} composites)")
    
    results = {
        "aoi_name": aoi_name,
        "climate": aoi_config["climate"],
        "tiles": aoi_config["tiles"],
        "total_attempts": 0,
        "successful_downloads": 0,
        "failed_downloads": 0,
        "downloaded_files": []
    }
    
    for tile in aoi_config["tiles"]:
        print(f"\nProcessing tile: {tile}")
        
        for date in dates:
            results["total_attempts"] += 1
            
            if download_modis_granule(aoi_name, tile, date, session):
                results["successful_downloads"] += 1
                results["downloaded_files"].append(f"{tile}_{date}")
            else:
                results["failed_downloads"] += 1
            
            # Rate limiting
            time.sleep(0.3)
    
    return results


def main():
    """Main download function."""
    print("Starting Stage-2 MODIS Data Download")
    print(f"Date range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Save directory: {STAGE2_RAW_DIR}")
    print(f"Target AOIs: {len(AOIS)} covering diverse climates")
    
    ensure_dirs()
    
    # Create session for connection reuse
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    all_results = []
    total_downloaded = 0
    
    # Download data for each AOI
    for aoi_name, aoi_config in AOIS.items():
        results = download_aoi_data(aoi_name, aoi_config, session)
        all_results.append(results)
        total_downloaded += results["successful_downloads"]
        
        print(f"\n{aoi_config['name']} Summary:")
        print(f"   Successful: {results['successful_downloads']}")
        print(f"   Failed: {results['failed_downloads']}")
        print(f"   Success rate: {results['successful_downloads']/results['total_attempts']*100:.1f}%")
    
    # Save download report
    report = {
        "download_timestamp": datetime.now().isoformat(),
        "stage": "stage2",
        "date_range": {
            "start": START_DATE.isoformat(),
            "end": END_DATE.isoformat()
        },
        "total_downloaded": total_downloaded,
        "aoi_results": all_results,
        "download_directory": str(STAGE2_RAW_DIR),
        "target_size_gb": "7-8"
    }
    
    report_path = ROOT / "outputs" / "stage2_download_report.json"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDownload Complete!")
    print(f"Total files downloaded: {total_downloaded}")
    print(f"Data saved to: {STAGE2_RAW_DIR}")
    print(f"Report saved to: {report_path}")
    
    # Calculate total size
    total_size = 0
    for aoi_dir in STAGE2_RAW_DIR.iterdir():
        if aoi_dir.is_dir():
            for file in aoi_dir.glob("*.hdf"):
                total_size += file.stat().st_size
    
    print(f"Total data size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    
    return report


if __name__ == "__main__":
    main()
