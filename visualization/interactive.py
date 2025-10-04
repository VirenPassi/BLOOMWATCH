"""
Interactive visualization utilities using Plotly for BloomWatch.

This module provides interactive dashboards and visualizations
for exploring bloom detection results and model performance.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import base64
from PIL import Image
import io

def create_interactive_dashboard(
 metrics_history: Dict[str, List[float]],
 confusion_matrix: np.ndarray,
 class_names: List[str],
 save_path: Optional[str] = None
) -> go.Figure:
 """
 Create an interactive dashboard with training metrics and model performance.
 
 Args:
 metrics_history: Dictionary containing metric histories
 confusion_matrix: Confusion matrix
 class_names: List of class names
 save_path: Optional path to save HTML file
 
 Returns:
 Plotly Figure object
 """
 # Create subplots
 fig = make_subplots(
 rows=2, cols=2,
 subplot_titles=('Training Metrics', 'Confusion Matrix', 'Accuracy Trends', 'Loss Trends'),
 specs=[[{"secondary_y": True}, {"type": "heatmap"}],
 [{"secondary_y": False}, {"secondary_y": False}]],
 vertical_spacing=0.12,
 horizontal_spacing=0.1
 )
 
 epochs = metrics_history.get('epoch', list(range(len(metrics_history.get('train_loss', [])))))
 
 # 1. Training metrics with dual y-axis
 if 'train_loss' in metrics_history:
 fig.add_trace(
 go.Scatter(
 x=epochs, y=metrics_history['train_loss'],
 mode='lines+markers', name='Train Loss',
 line=dict(color='#1f77b4', width=2),
 marker=dict(size=4)
 ),
 row=1, col=1, secondary_y=False
 )
 
 if 'val_loss' in metrics_history:
 fig.add_trace(
 go.Scatter(
 x=epochs, y=metrics_history['val_loss'],
 mode='lines+markers', name='Val Loss',
 line=dict(color='#ff7f0e', width=2),
 marker=dict(size=4)
 ),
 row=1, col=1, secondary_y=False
 )
 
 if 'train_acc' in metrics_history:
 fig.add_trace(
 go.Scatter(
 x=epochs, y=metrics_history['train_acc'],
 mode='lines+markers', name='Train Acc',
 line=dict(color='#2ca02c', width=2),
 marker=dict(size=4)
 ),
 row=1, col=1, secondary_y=True
 )
 
 if 'val_acc' in metrics_history:
 fig.add_trace(
 go.Scatter(
 x=epochs, y=metrics_history['val_acc'],
 mode='lines+markers', name='Val Acc',
 line=dict(color='#d62728', width=2),
 marker=dict(size=4)
 ),
 row=1, col=1, secondary_y=True
 )
 
 # 2. Confusion Matrix Heatmap
 fig.add_trace(
 go.Heatmap(
 z=confusion_matrix,
 x=class_names,
 y=class_names,
 colorscale='Blues',
 showscale=True,
 text=confusion_matrix,
 texttemplate="%{text}",
 textfont={"size": 12},
 hoverongaps=False
 ),
 row=1, col=2
 )
 
 # 3. Accuracy trends comparison
 if 'train_acc' in metrics_history and 'val_acc' in metrics_history:
 fig.add_trace(
 go.Scatter(
 x=epochs, y=metrics_history['train_acc'],
 mode='lines+markers', name='Train Accuracy',
 line=dict(color='#17becf', width=3),
 marker=dict(size=6)
 ),
 row=2, col=1
 )
 
 fig.add_trace(
 go.Scatter(
 x=epochs, y=metrics_history['val_acc'],
 mode='lines+markers', name='Val Accuracy',
 line=dict(color='#e377c2', width=3),
 marker=dict(size=6)
 ),
 row=2, col=1
 )
 
 # 4. Loss trends comparison
 if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
 fig.add_trace(
 go.Scatter(
 x=epochs, y=metrics_history['train_loss'],
 mode='lines+markers', name='Train Loss',
 line=dict(color='#bcbd22', width=3),
 marker=dict(size=6)
 ),
 row=2, col=2
 )
 
 fig.add_trace(
 go.Scatter(
 x=epochs, y=metrics_history['val_loss'],
 mode='lines+markers', name='Val Loss',
 line=dict(color='#ff9896', width=3),
 marker=dict(size=6)
 ),
 row=2, col=2
 )
 
 # Update layout
 fig.update_layout(
 title_text="BloomWatch Training Dashboard",
 title_x=0.5,
 title_font_size=20,
 showlegend=True,
 height=800,
 template="plotly_white"
 )
 
 # Update axis labels
 fig.update_xaxes(title_text="Epoch", row=1, col=1)
 fig.update_yaxes(title_text="Loss", row=1, col=1, secondary_y=False)
 fig.update_yaxes(title_text="Accuracy", row=1, col=1, secondary_y=True)
 
 fig.update_xaxes(title_text="Predicted", row=1, col=2)
 fig.update_yaxes(title_text="Actual", row=1, col=2)
 
 fig.update_xaxes(title_text="Epoch", row=2, col=1)
 fig.update_yaxes(title_text="Accuracy", row=2, col=1)
 
 fig.update_xaxes(title_text="Epoch", row=2, col=2)
 fig.update_yaxes(title_text="Loss", row=2, col=2)
 
 if save_path:
 fig.write_html(save_path)
 
 return fig

def plot_model_comparison(
 model_results: Dict[str, Dict[str, float]],
 save_path: Optional[str] = None
) -> go.Figure:
 """
 Create interactive comparison of multiple models.
 
 Args:
 model_results: Dictionary with model names and their metrics
 save_path: Optional path to save HTML file
 
 Returns:
 Plotly Figure object
 """
 # Prepare data
 models = list(model_results.keys())
 metrics = list(model_results[models[0]].keys()) if models else []
 
 # Create radar chart
 fig = go.Figure()
 
 colors = px.colors.qualitative.Set1[:len(models)]
 
 for i, model_name in enumerate(models):
 values = [model_results[model_name].get(metric, 0) for metric in metrics]
 
 fig.add_trace(go.Scatterpolar(
 r=values + [values[0]], # Close the polygon
 theta=metrics + [metrics[0]],
 fill='toself',
 name=model_name,
 line_color=colors[i],
 fillcolor=colors[i],
 opacity=0.6
 ))
 
 fig.update_layout(
 polar=dict(
 radialaxis=dict(
 visible=True,
 range=[0, 1]
 )),
 showlegend=True,
 title="Model Performance Comparison",
 title_x=0.5,
 title_font_size=18,
 template="plotly_white"
 )
 
 if save_path:
 fig.write_html(save_path)
 
 return fig

def create_prediction_viewer(
 images: List[np.ndarray],
 predictions: List[int],
 true_labels: List[int],
 class_names: List[str],
 confidences: Optional[List[float]] = None,
 save_path: Optional[str] = None
) -> go.Figure:
 """
 Create an interactive viewer for model predictions.
 
 Args:
 images: List of image arrays
 predictions: List of predicted class indices
 true_labels: List of true class indices
 class_names: List of class names
 confidences: Optional list of prediction confidences
 save_path: Optional path to save HTML file
 
 Returns:
 Plotly Figure object
 """
 # Convert images to base64 for embedding
 def image_to_base64(img_array):
 """Convert numpy array to base64 string."""
 if img_array.dtype != np.uint8:
 img_array = (img_array * 255).astype(np.uint8)
 
 img = Image.fromarray(img_array)
 buffer = io.BytesIO()
 img.save(buffer, format='PNG')
 img_str = base64.b64encode(buffer.getvalue()).decode()
 return f"data:image/png;base64,{img_str}"
 
 # Prepare data
 num_samples = min(len(images), 20) # Limit for performance
 sample_indices = np.linspace(0, len(images) - 1, num_samples, dtype=int)
 
 sample_data = []
 for i, idx in enumerate(sample_indices):
 img_b64 = image_to_base64(images[idx])
 
 pred_name = class_names[predictions[idx]] if predictions[idx] < len(class_names) else f"Class {predictions[idx]}"
 true_name = class_names[true_labels[idx]] if true_labels[idx] < len(class_names) else f"Class {true_labels[idx]}"
 
 confidence = confidences[idx] if confidences else 0.0
 correct = predictions[idx] == true_labels[idx]
 
 sample_data.append({
 'index': idx,
 'image': img_b64,
 'prediction': pred_name,
 'true_label': true_name,
 'confidence': confidence,
 'correct': correct,
 'x': i % 5, # Grid layout
 'y': i // 5
 })
 
 # Create scatter plot with images as markers
 fig = go.Figure()
 
 # Separate correct and incorrect predictions
 correct_samples = [s for s in sample_data if s['correct']]
 incorrect_samples = [s for s in sample_data if not s['correct']]
 
 # Plot correct predictions
 if correct_samples:
 fig.add_trace(go.Scatter(
 x=[s['x'] for s in correct_samples],
 y=[s['y'] for s in correct_samples],
 mode='markers',
 marker=dict(
 size=20,
 color='green',
 symbol='circle'
 ),
 name='Correct Predictions',
 customdata=[(s['prediction'], s['true_label'], s['confidence']) for s in correct_samples],
 hovertemplate="<b>Prediction:</b> %{customdata[0]}<br>" +
 "<b>True Label:</b> %{customdata[1]}<br>" +
 "<b>Confidence:</b> %{customdata[2]:.3f}<br>" +
 "<extra></extra>"
 ))
 
 # Plot incorrect predictions
 if incorrect_samples:
 fig.add_trace(go.Scatter(
 x=[s['x'] for s in incorrect_samples],
 y=[s['y'] for s in incorrect_samples],
 mode='markers',
 marker=dict(
 size=20,
 color='red',
 symbol='x'
 ),
 name='Incorrect Predictions',
 customdata=[(s['prediction'], s['true_label'], s['confidence']) for s in incorrect_samples],
 hovertemplate="<b>Prediction:</b> %{customdata[0]}<br>" +
 "<b>True Label:</b> %{customdata[1]}<br>" +
 "<b>Confidence:</b> %{customdata[2]:.3f}<br>" +
 "<extra></extra>"
 ))
 
 fig.update_layout(
 title="Model Predictions Viewer",
 title_x=0.5,
 title_font_size=18,
 xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
 yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
 showlegend=True,
 template="plotly_white",
 height=600
 )
 
 if save_path:
 fig.write_html(save_path)
 
 return fig

def plot_attention_maps(
 attention_weights: np.ndarray,
 input_image: np.ndarray,
 save_path: Optional[str] = None
) -> go.Figure:
 """
 Create interactive visualization of attention maps.
 
 Args:
 attention_weights: Attention weight matrix (H, W)
 input_image: Input image array (H, W, 3)
 save_path: Optional path to save HTML file
 
 Returns:
 Plotly Figure object
 """
 # Create subplots
 fig = make_subplots(
 rows=1, cols=3,
 subplot_titles=('Original Image', 'Attention Map', 'Overlaid Attention'),
 specs=[[{"type": "image"}, {"type": "heatmap"}, {"type": "image"}]]
 )
 
 # 1. Original image
 fig.add_trace(
 go.Image(z=input_image),
 row=1, col=1
 )
 
 # 2. Attention heatmap
 fig.add_trace(
 go.Heatmap(
 z=attention_weights,
 colorscale='Reds',
 showscale=True,
 hoverongaps=False
 ),
 row=1, col=2
 )
 
 # 3. Overlaid attention (blend image with attention)
 # Normalize attention weights
 attention_norm = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
 
 # Create overlay
 overlay = input_image.copy()
 for c in range(3):
 overlay[:, :, c] = overlay[:, :, c] * (1 - attention_norm * 0.5) + attention_norm * 255 * (c == 0) # Red overlay
 
 fig.add_trace(
 go.Image(z=overlay.astype(np.uint8)),
 row=1, col=3
 )
 
 fig.update_layout(
 title="Attention Visualization",
 title_x=0.5,
 title_font_size=18,
 template="plotly_white",
 height=400
 )
 
 if save_path:
 fig.write_html(save_path)
 
 return fig

def create_bloom_progression_timeline(
 plant_data: Dict[str, Any],
 save_path: Optional[str] = None
) -> go.Figure:
 """
 Create interactive timeline showing bloom progression for a plant.
 
 Args:
 plant_data: Dictionary with 'timestamps', 'stages', 'images', 'plant_id'
 save_path: Optional path to save HTML file
 
 Returns:
 Plotly Figure object
 """
 timestamps = plant_data['timestamps']
 stages = plant_data['stages']
 plant_id = plant_data.get('plant_id', 'Unknown Plant')
 
 # Convert to datetime if needed
 if isinstance(timestamps[0], str):
 timestamps = pd.to_datetime(timestamps)
 
 stage_names = ['Bud', 'Early Bloom', 'Full Bloom', 'Late Bloom', 'Dormant']
 stage_colors = ['#8B4513', '#90EE90', '#FFB6C1', '#FFA500', '#808080']
 
 # Create timeline
 fig = go.Figure()
 
 # Add line trace
 fig.add_trace(go.Scatter(
 x=timestamps,
 y=stages,
 mode='lines+markers',
 line=dict(width=4, color='lightblue'),
 marker=dict(
 size=12,
 color=[stage_colors[stage] for stage in stages],
 line=dict(width=2, color='black')
 ),
 name='Bloom Progression',
 hovertemplate="<b>Date:</b> %{x}<br>" +
 "<b>Stage:</b> %{customdata}<br>" +
 "<extra></extra>",
 customdata=[stage_names[stage] for stage in stages]
 ))
 
 # Highlight peak bloom periods
 peak_bloom_indices = [i for i, stage in enumerate(stages) if stage == 2] # Full bloom
 if peak_bloom_indices:
 fig.add_trace(go.Scatter(
 x=[timestamps[i] for i in peak_bloom_indices],
 y=[stages[i] for i in peak_bloom_indices],
 mode='markers',
 marker=dict(
 size=20,
 color='red',
 symbol='star',
 line=dict(width=2, color='darkred')
 ),
 name='Peak Bloom',
 hovertemplate="<b>Peak Bloom!</b><br>" +
 "<b>Date:</b> %{x}<br>" +
 "<extra></extra>"
 ))
 
 # Update layout
 fig.update_layout(
 title=f"Bloom Progression Timeline - {plant_id}",
 title_x=0.5,
 title_font_size=18,
 xaxis_title="Date",
 yaxis_title="Bloom Stage",
 yaxis=dict(
 tickmode='array',
 tickvals=list(range(len(stage_names))),
 ticktext=stage_names
 ),
 template="plotly_white",
 height=500,
 showlegend=True
 )
 
 # Add range slider
 fig.update_layout(
 xaxis=dict(
 rangeselector=dict(
 buttons=list([
 dict(count=7, label="7d", step="day", stepmode="backward"),
 dict(count=30, label="30d", step="day", stepmode="backward"),
 dict(count=90, label="3m", step="day", stepmode="backward"),
 dict(step="all")
 ])
 ),
 rangeslider=dict(visible=True),
 type="date"
 )
 )
 
 if save_path:
 fig.write_html(save_path)
 
 return fig

def create_dataset_explorer_dashboard(
 dataset_stats: Dict[str, Any],
 save_path: Optional[str] = None
) -> go.Figure:
 """
 Create comprehensive dataset exploration dashboard.
 
 Args:
 dataset_stats: Dictionary with dataset statistics
 save_path: Optional path to save HTML file
 
 Returns:
 Plotly Figure object
 """
 # Create subplots
 fig = make_subplots(
 rows=2, cols=2,
 subplot_titles=('Class Distribution', 'Temporal Distribution', 
 'Plant Coverage', 'Data Quality Metrics'),
 specs=[[{"type": "bar"}, {"type": "scatter"}],
 [{"type": "pie"}, {"type": "table"}]]
 )
 
 # 1. Class distribution
 class_names = dataset_stats.get('class_names', [])
 class_counts = dataset_stats.get('class_counts', [])
 
 if class_names and class_counts:
 fig.add_trace(
 go.Bar(
 x=class_names,
 y=class_counts,
 marker_color='lightblue',
 name='Class Distribution'
 ),
 row=1, col=1
 )
 
 # 2. Temporal distribution (if available)
 if 'temporal_data' in dataset_stats:
 temporal_data = dataset_stats['temporal_data']
 fig.add_trace(
 go.Scatter(
 x=temporal_data.get('dates', []),
 y=temporal_data.get('counts', []),
 mode='lines+markers',
 name='Samples Over Time'
 ),
 row=1, col=2
 )
 
 # 3. Plant coverage pie chart
 if 'plant_distribution' in dataset_stats:
 plant_data = dataset_stats['plant_distribution']
 fig.add_trace(
 go.Pie(
 labels=plant_data.get('plant_ids', []),
 values=plant_data.get('sample_counts', []),
 name="Plant Coverage"
 ),
 row=2, col=1
 )
 
 # 4. Data quality metrics table
 quality_metrics = dataset_stats.get('quality_metrics', {})
 
 if quality_metrics:
 fig.add_trace(
 go.Table(
 header=dict(values=['Metric', 'Value'],
 fill_color='lightblue',
 align='left'),
 cells=dict(values=[list(quality_metrics.keys()), 
 list(quality_metrics.values())],
 fill_color='white',
 align='left')
 ),
 row=2, col=2
 )
 
 fig.update_layout(
 title="Dataset Explorer Dashboard",
 title_x=0.5,
 title_font_size=20,
 showlegend=False,
 height=800,
 template="plotly_white"
 )
 
 if save_path:
 fig.write_html(save_path)
 
 return fig