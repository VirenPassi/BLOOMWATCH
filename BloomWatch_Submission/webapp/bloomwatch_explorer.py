#!/usr/bin/env python3
"""
Streamlit app for exploring BloomWatch temporal workflow results.

This app provides an interactive interface to explore AOIs, dates, 
indices overlays, and bloom probability maps.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional

# Set page config
st.set_page_config(
 page_title="BloomWatch Explorer",
 page_icon="",
 layout="wide"
)

# Title and description
st.title(" BloomWatch Explorer")
st.markdown("""
Explore plant bloom detection results using satellite data and AI models.
Upload your analysis results or use sample data to visualize bloom patterns over time.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# File uploaders
report_file = st.sidebar.file_uploader("Upload Report JSON", type=["json"])
timeseries_file = st.sidebar.file_uploader("Upload Timeseries HTML", type=["html"])
map_file = st.sidebar.file_uploader("Upload Map HTML", type=["html"])

# Sample data option
use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)

# If we have data, show exploration interface
if report_file or use_sample_data:
 # Load report data
 if report_file:
 report_data = json.load(report_file)
 else:
 # Sample data
 report_data = {
 "aoi": [-122.7, 37.7, -121.8, 38.4],
 "date_range": ["2023-05-01", "2023-09-30"],
 "sensor": "MODIS",
 "items": 15,
 "checkpoint": "stage2_transfer_learning_bloomwatch.pt",
 "indices": ["ndvi", "evi", "ndwi", "mndwi", "fai", "mci", "ndci", "ci_cy"],
 "notes": "Bloom probability is model output; indices normalized per-pixel across time."
 }
 
 # Display report summary
 st.header("Analysis Summary")
 col1, col2, col3 = st.columns(3)
 
 with col1:
 st.metric("Sensor", report_data["sensor"])
 st.metric("Time Period", f"{report_data['date_range'][0]} to {report_data['date_range'][1]}")
 
 with col2:
 st.metric("Data Items", report_data["items"])
 st.metric("Indices Computed", len(report_data["indices"]))
 
 with col3:
 st.metric("Model Checkpoint", Path(report_data["checkpoint"]).name)
 if "aoi" in report_data:
 aoi = report_data["aoi"]
 st.metric("AOI Bounds", f"{aoi[0]:.2f}, {aoi[1]:.2f} to {aoi[2]:.2f}, {aoi[3]:.2f}")
 
 # Display computed indices
 st.subheader("Computed Spectral Indices")
 indices_df = pd.DataFrame({
 "Index": report_data["indices"],
 "Description": [
 "Normalized Difference Vegetation Index",
 "Enhanced Vegetation Index", 
 "Normalized Difference Water Index",
 "Modified Normalized Difference Water Index",
 "Floating Algae Index",
 "Maximum Chlorophyll Index",
 "Normalized Difference Chlorophyll Index",
 "Cyano Index"
 ]
 })
 st.table(indices_df)
 
 # Visualization section
 st.header("Visualizations")
 
 # Create tabs for different visualizations
 tab1, tab2, tab3 = st.tabs(["Map View", "Time Series", "Index Comparison"])
 
 with tab1:
 st.subheader("Interactive Map")
 
 # Create a simple folium map with AOI
 if "aoi" in report_data:
 aoi = report_data["aoi"]
 center = [(aoi[1] + aoi[3]) / 2, (aoi[0] + aoi[2]) / 2]
 
 # Create folium map
 m = folium.Map(location=center, zoom_start=10)
 
 # Add AOI rectangle
 folium.Rectangle(
 bounds=[[aoi[1], aoi[0]], [aoi[3], aoi[2]]],
 color="blue",
 fill=True,
 fill_opacity=0.1
 ).add_to(m)
 
 # Display map
 st_folium(m, width=800, height=600)
 
 st.info("In a full implementation, this would show the actual bloom probability maps overlaid on satellite imagery.")
 
 with tab2:
 st.subheader("Bloom Probability Over Time")
 
 # Create sample time series data
 dates = pd.date_range(start=report_data["date_range"][0], 
 end=report_data["date_range"][1], 
 periods=20)
 probabilities = np.random.beta(2, 5, len(dates)) # Sample bloom probabilities
 
 # Create Plotly figure
 fig = px.line(
 x=dates, 
 y=probabilities,
 labels={"x": "Date", "y": "Bloom Probability"},
 title="Bloom Probability Time Series"
 )
 fig.update_layout(hovermode="x unified")
 
 st.plotly_chart(fig, use_container_width=True)
 
 # Statistics
 st.subheader("Time Series Statistics")
 col1, col2, col3, col4 = st.columns(4)
 with col1:
 st.metric("Mean Probability", f"{np.mean(probabilities):.3f}")
 with col2:
 st.metric("Max Probability", f"{np.max(probabilities):.3f}")
 with col3:
 st.metric("Min Probability", f"{np.min(probabilities):.3f}")
 with col4:
 st.metric("Std Deviation", f"{np.std(probabilities):.3f}")
 
 with tab3:
 st.subheader("Index Comparison")
 
 # Create sample index comparison data
 indices = report_data["indices"]
 values = np.random.rand(len(indices))
 
 # Bar chart
 fig = px.bar(
 x=indices,
 y=values,
 labels={"x": "Spectral Index", "y": "Average Value"},
 title="Average Spectral Index Values"
 )
 st.plotly_chart(fig, use_container_width=True)
 
 # Correlation heatmap (sample)
 st.subheader("Index Correlations")
 corr_data = np.random.rand(len(indices), len(indices))
 corr_data = (corr_data + corr_data.T) / 2 # Make symmetric
 np.fill_diagonal(corr_data, 1) # Fill diagonal with 1s
 
 fig = go.Figure(data=go.Heatmap(
 z=corr_data,
 x=indices,
 y=indices,
 colorscale='RdBu',
 zmid=0
 ))
 fig.update_layout(title="Spectral Index Correlations")
 st.plotly_chart(fig, use_container_width=True)
 
 # Notes section
 st.header("Analysis Notes")
 st.info(report_data["notes"])
 
 # Download section
 st.header("Export Results")
 st.download_button(
 label="Download Report JSON",
 data=json.dumps(report_data, indent=2),
 file_name="bloomwatch_report.json",
 mime="application/json"
 )

else:
 st.info("Please upload a report file or enable sample data to explore the results.")
 st.markdown("""
 ## How to Use This Explorer
 
 1. Run the BloomWatch temporal workflow:
 ```bash
 python pipelines/bloomwatch_temporal_workflow.py \\
 --aoi "[-122.7,37.7,-121.8,38.4]" \\
 --start 2023-05-01 --end 2023-09-30 \\
 --sensor MODIS \\
 --checkpoint outputs/models/stage2_transfer_learning_bloomwatch.pt
 ```
 
 2. Upload the generated `report.json` file using the uploader in the sidebar.
 
 3. Explore the results using the different visualization tabs.
 """)

# Footer
st.markdown("---")
st.markdown(" BloomWatch Explorer - Plant Bloom Detection Using Satellite Data and AI")