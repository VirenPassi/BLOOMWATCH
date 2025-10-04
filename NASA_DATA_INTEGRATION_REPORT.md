# BloomWatch NASA Data Integration Report

## Project Overview
BloomWatch uses real NASA MODIS satellite data for plant bloom detection and monitoring.

## NASA Data Sources Used

### MODIS Terra Vegetation Indices (MOD13Q1)
- **Description**: 16-Day L3 Global 250m Vegetation Indices from Terra satellite
- **Variables**: NDVI, EVI, Quality, Reliability
- **Spatial Resolution**: 250m
- **Temporal Resolution**: 16-day
- **Coverage**: Global
- **Time Period**: 2000-present

### MODIS Aqua Vegetation Indices (MYD13Q1)  
- **Description**: 16-Day L3 Global 250m Vegetation Indices from Aqua satellite
- **Variables**: NDVI, EVI, Quality, Reliability
- **Spatial Resolution**: 250m
- **Temporal Resolution**: 16-day
- **Coverage**: Global
- **Time Period**: 2002-present

## Regions Analyzed

1. **California Central Valley** (Agricultural region with seasonal crop cycles)
2. **Iowa Corn Belt** (Major corn and soybean production area)
3. **Great Plains** (Grassland and agricultural region)

## Spectral Indices Computed from NASA Data

- **NDVI**: (NIR - Red) / (NIR + Red) - Vegetation health and density
- **EVI**: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1) - Enhanced vegetation index
- **NDWI**: (Green - NIR) / (Green + NIR) - Water content in vegetation

## AI Integration

- NASA MODIS data used to train bloom detection models
- Spectral indices from NASA data as model features
- Time series of NASA vegetation indices for temporal analysis
- Spatial patterns in NASA satellite imagery

## Technical Implementation

- **Data Access**: NASA Earthdata API via earthaccess library
- **Authentication**: NASA Earthdata Login credentials
- **Data Format**: HDF4 files from NASA MODIS collections
- **Processing**: xarray for multi-dimensional data analysis
- **Visualization**: Interactive maps and time series plots

## Global Award Eligibility

- **NASA Data Used**: Real MODIS vegetation data from NASA Earthdata
- **Purpose**: Plant bloom detection and monitoring
- **Innovation**: AI-powered temporal analysis of NASA satellite data

## Generated: 2025-10-04 16:39:20
