"""
Time-lapse and temporal visualization utilities for plant bloom tracking.

This module provides functions to create animations and visualizations
for tracking plant growth and blooming over time.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from PIL import Image
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import seaborn as sns
from datetime import datetime, timedelta

def create_bloom_timelapse(
 image_paths: List[str],
 timestamps: List[str],
 save_path: str,
 plant_id: str = "Plant",
 fps: int = 2,
 figsize: Tuple[int, int] = (10, 8)
) -> str:
 """
 Create a time-lapse animation of plant bloom progression.
 
 Args:
 image_paths: List of paths to images in temporal order
 timestamps: List of timestamps corresponding to images
 save_path: Path to save the animation (should end with .mp4 or .gif)
 plant_id: Identifier for the plant
 fps: Frames per second for the animation
 figsize: Figure size tuple
 
 Returns:
 str: Path to the saved animation
 """
 fig, ax = plt.subplots(figsize=figsize)
 
 # Load all images
 images = []
 for img_path in image_paths:
 try:
 img = Image.open(img_path)
 images.append(np.array(img))
 except Exception as e:
 print(f"Error loading {img_path}: {e}")
 continue
 
 if not images:
 raise ValueError("No valid images loaded")
 
 # Initialize plot
 im = ax.imshow(images[0])
 ax.axis('off')
 title = ax.set_title(f'{plant_id} - {timestamps[0]}', fontsize=16, fontweight='bold')
 
 def animate(frame):
 """Animation function."""
 if frame < len(images):
 im.set_array(images[frame])
 title.set_text(f'{plant_id} - {timestamps[frame]}')
 return [im, title]
 
 # Create animation
 anim = animation.FuncAnimation(
 fig, animate, frames=len(images),
 interval=1000//fps, blit=True, repeat=True
 )
 
 # Save animation
 if save_path.endswith('.gif'):
 anim.save(save_path, writer='pillow', fps=fps)
 elif save_path.endswith('.mp4'):
 anim.save(save_path, writer='ffmpeg', fps=fps, extra_args=['-vcodec', 'libx264'])
 else:
 raise ValueError("save_path must end with .gif or .mp4")
 
 plt.close(fig)
 return save_path

def create_growth_animation(
 plant_data: Dict[str, Any],
 save_path: str,
 fps: int = 2,
 figsize: Tuple[int, int] = (15, 10)
) -> str:
 """
 Create an animation showing plant growth with metrics overlay.
 
 Args:
 plant_data: Dictionary containing 'images', 'timestamps', 'stages', 'metrics'
 save_path: Path to save the animation
 fps: Frames per second
 figsize: Figure size tuple
 
 Returns:
 str: Path to the saved animation
 """
 images = plant_data['images']
 timestamps = plant_data['timestamps']
 stages = plant_data.get('stages', [])
 metrics = plant_data.get('metrics', {})
 
 fig = plt.figure(figsize=figsize)
 gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[2, 1, 1])
 
 # Main image subplot
 ax_img = fig.add_subplot(gs[0, :2])
 ax_img.axis('off')
 
 # Stage progression subplot
 ax_stage = fig.add_subplot(gs[0, 2])
 
 # Metrics subplot
 ax_metrics = fig.add_subplot(gs[1, :])
 
 # Initialize plots
 im = ax_img.imshow(images[0] if images else np.zeros((224, 224, 3)))
 title = ax_img.set_title(timestamps[0] if timestamps else "Time 0", fontsize=16, fontweight='bold')
 
 # Stage progression setup
 stage_names = ['Bud', 'Early Bloom', 'Full Bloom', 'Late Bloom', 'Dormant']
 stage_colors = ['brown', 'lightgreen', 'pink', 'orange', 'gray']
 
 if stages:
 ax_stage.bar(stage_names, [0]*len(stage_names), color=stage_colors, alpha=0.3)
 ax_stage.set_title('Bloom Stage', fontweight='bold')
 ax_stage.set_ylabel('Confidence')
 ax_stage.tick_params(axis='x', rotation=45)
 
 # Metrics setup
 if metrics:
 metric_lines = {}
 metric_names = list(metrics.keys())
 colors = plt.cm.tab10(np.linspace(0, 1, len(metric_names)))
 
 for i, (metric_name, color) in enumerate(zip(metric_names, colors)):
 line, = ax_metrics.plot([], [], label=metric_name, color=color, linewidth=2)
 metric_lines[metric_name] = line
 
 ax_metrics.set_title('Metrics Over Time', fontweight='bold')
 ax_metrics.set_xlabel('Time')
 ax_metrics.legend()
 ax_metrics.grid(True, alpha=0.3)
 
 def animate(frame):
 """Animation function."""
 artists = []
 
 # Update main image
 if frame < len(images):
 im.set_array(images[frame])
 artists.append(im)
 
 # Update title with timestamp
 if frame < len(timestamps):
 title.set_text(f'Frame {frame} - {timestamps[frame]}')
 artists.append(title)
 
 # Update stage progression
 if stages and frame < len(stages):
 ax_stage.clear()
 current_stage = stages[frame]
 
 # Create bar heights (assuming current_stage is class index)
 heights = [0.2] * len(stage_names) # Base height
 if current_stage < len(stage_names):
 heights[current_stage] = 1.0 # Highlight current stage
 
 bars = ax_stage.bar(stage_names, heights, color=stage_colors, alpha=0.7)
 ax_stage.set_title('Current Bloom Stage', fontweight='bold')
 ax_stage.set_ylabel('Confidence')
 ax_stage.tick_params(axis='x', rotation=45)
 ax_stage.set_ylim(0, 1.2)
 
 artists.extend(bars)
 
 # Update metrics
 if metrics:
 for metric_name, line in metric_lines.items():
 if metric_name in metrics and frame < len(metrics[metric_name]):
 x_data = list(range(frame + 1))
 y_data = metrics[metric_name][:frame + 1]
 line.set_data(x_data, y_data)
 artists.append(line)
 
 # Adjust axis limits
 if frame > 0:
 ax_metrics.set_xlim(0, max(len(timestamps) - 1, frame))
 
 all_values = []
 for metric_name in metrics:
 if frame < len(metrics[metric_name]):
 all_values.extend(metrics[metric_name][:frame + 1])
 
 if all_values:
 margin = (max(all_values) - min(all_values)) * 0.1
 ax_metrics.set_ylim(min(all_values) - margin, max(all_values) + margin)
 
 return artists
 
 # Create animation
 num_frames = max(len(images), len(timestamps), len(stages) if stages else 0)
 anim = animation.FuncAnimation(
 fig, animate, frames=num_frames,
 interval=1000//fps, blit=False, repeat=True
 )
 
 # Save animation
 if save_path.endswith('.gif'):
 anim.save(save_path, writer='pillow', fps=fps)
 elif save_path.endswith('.mp4'):
 anim.save(save_path, writer='ffmpeg', fps=fps, extra_args=['-vcodec', 'libx264'])
 else:
 raise ValueError("save_path must end with .gif or .mp4")
 
 plt.close(fig)
 return save_path

def plot_growth_curve(
 timestamps: List[str],
 bloom_stages: List[int],
 plant_id: str = "Plant",
 save_path: Optional[str] = None,
 figsize: Tuple[int, int] = (12, 6),
 title: Optional[str] = None
) -> plt.Figure:
 """
 Plot plant growth curve showing bloom stage progression over time.
 
 Args:
 timestamps: List of timestamp strings
 bloom_stages: List of bloom stage indices
 plant_id: Plant identifier
 save_path: Optional path to save the plot
 figsize: Figure size tuple
 title: Optional custom title
 
 Returns:
 matplotlib Figure object
 """
 # Convert timestamps to datetime if they're strings
 if isinstance(timestamps[0], str):
 try:
 dates = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
 except:
 # If parsing fails, use indices
 dates = list(range(len(timestamps)))
 else:
 dates = timestamps
 
 fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
 
 if title is None:
 title = f'Growth Curve - {plant_id}'
 fig.suptitle(title, fontsize=16, fontweight='bold')
 
 # Main growth curve
 stage_names = ['Bud', 'Early Bloom', 'Full Bloom', 'Late Bloom', 'Dormant']
 stage_colors = ['#8B4513', '#90EE90', '#FFB6C1', '#FFA500', '#808080']
 
 # Plot line with color coding
 for i in range(len(bloom_stages) - 1):
 x = [dates[i], dates[i + 1]] if isinstance(dates[0], datetime) else [i, i + 1]
 y = [bloom_stages[i], bloom_stages[i + 1]]
 color = stage_colors[bloom_stages[i]] if bloom_stages[i] < len(stage_colors) else 'black'
 ax1.plot(x, y, color=color, linewidth=3, alpha=0.8)
 
 # Add markers
 for i, (date, stage) in enumerate(zip(dates, bloom_stages)):
 color = stage_colors[stage] if stage < len(stage_colors) else 'black'
 marker_size = 100 if stage == 2 else 60 # Larger marker for full bloom
 ax1.scatter(date, stage, color=color, s=marker_size, edgecolor='black', 
 linewidth=2, zorder=5)
 
 ax1.set_ylabel('Bloom Stage', fontweight='bold')
 ax1.set_yticks(range(len(stage_names)))
 ax1.set_yticklabels(stage_names)
 ax1.grid(True, alpha=0.3)
 ax1.set_ylim(-0.5, len(stage_names) - 0.5)
 
 # Handle x-axis formatting
 if isinstance(dates[0], datetime):
 ax1.tick_params(axis='x', rotation=45)
 fig.autofmt_xdate()
 else:
 ax1.set_xlabel('Time Point')
 
 # Stage duration histogram
 stage_durations = {}
 for stage in bloom_stages:
 stage_name = stage_names[stage] if stage < len(stage_names) else f'Stage {stage}'
 stage_durations[stage_name] = stage_durations.get(stage_name, 0) + 1
 
 bars = ax2.bar(stage_durations.keys(), stage_durations.values(), 
 color=[stage_colors[stage_names.index(name)] if name in stage_names else 'gray' 
 for name in stage_durations.keys()],
 alpha=0.7, edgecolor='black')
 
 ax2.set_title('Stage Duration Distribution', fontweight='bold')
 ax2.set_ylabel('Time Points')
 ax2.tick_params(axis='x', rotation=45)
 
 # Add count labels on bars
 for bar, count in zip(bars, stage_durations.values()):
 height = bar.get_height()
 ax2.text(bar.get_x() + bar.get_width()/2., height + max(stage_durations.values()) * 0.01,
 f'{count}', ha='center', va='bottom', fontweight='bold')
 
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
 
 return fig

def visualize_temporal_patterns(
 dataset_df: pd.DataFrame,
 save_path: Optional[str] = None,
 figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
 """
 Visualize temporal patterns in the bloom dataset.
 
 Args:
 dataset_df: DataFrame with columns ['timestamp', 'bloom_stage', 'plant_id']
 save_path: Optional path to save the plot
 figsize: Figure size tuple
 
 Returns:
 matplotlib Figure object
 """
 fig, axes = plt.subplots(2, 2, figsize=figsize)
 fig.suptitle('Temporal Bloom Patterns Analysis', fontsize=16, fontweight='bold')
 
 # Convert timestamp to datetime
 dataset_df['datetime'] = pd.to_datetime(dataset_df['timestamp'])
 dataset_df['month'] = dataset_df['datetime'].dt.month
 dataset_df['day_of_year'] = dataset_df['datetime'].dt.dayofyear
 
 stage_names = ['Bud', 'Early Bloom', 'Full Bloom', 'Late Bloom', 'Dormant']
 stage_colors = ['#8B4513', '#90EE90', '#FFB6C1', '#FFA500', '#808080']
 
 # 1. Bloom stage distribution by month
 monthly_stages = dataset_df.groupby(['month', 'bloom_stage']).size().unstack(fill_value=0)
 
 monthly_stages.plot(kind='bar', stacked=True, ax=axes[0, 0], 
 color=stage_colors[:len(monthly_stages.columns)])
 axes[0, 0].set_title('Bloom Stages by Month', fontweight='bold')
 axes[0, 0].set_xlabel('Month')
 axes[0, 0].set_ylabel('Count')
 axes[0, 0].legend(stage_names[:len(monthly_stages.columns)], loc='upper right')
 axes[0, 0].tick_params(axis='x', rotation=45)
 
 # 2. Peak bloom timing
 full_bloom_data = dataset_df[dataset_df['bloom_stage'] == 2] # Assuming full_bloom = 2
 
 if not full_bloom_data.empty:
 axes[0, 1].hist(full_bloom_data['day_of_year'], bins=20, 
 color=stage_colors[2], alpha=0.7, edgecolor='black')
 axes[0, 1].set_title('Peak Bloom Timing Distribution', fontweight='bold')
 axes[0, 1].set_xlabel('Day of Year')
 axes[0, 1].set_ylabel('Frequency')
 axes[0, 1].grid(True, alpha=0.3)
 else:
 axes[0, 1].text(0.5, 0.5, 'No Full Bloom Data', ha='center', va='center',
 transform=axes[0, 1].transAxes, fontsize=14, style='italic')
 axes[0, 1].set_title('Peak Bloom Timing Distribution', fontweight='bold')
 
 # 3. Plant-wise bloom progression heatmap
 if 'plant_id' in dataset_df.columns:
 # Create pivot table for heatmap
 plant_timeline = dataset_df.pivot_table(
 values='bloom_stage', 
 index='plant_id', 
 columns='day_of_year', 
 aggfunc='mean'
 )
 
 if not plant_timeline.empty:
 # Limit to reasonable number of plants for visualization
 if len(plant_timeline) > 20:
 plant_timeline = plant_timeline.head(20)
 
 sns.heatmap(plant_timeline, cmap='RdYlGn', ax=axes[1, 0], 
 cbar_kws={'label': 'Bloom Stage'})
 axes[1, 0].set_title('Plant Bloom Progression Heatmap', fontweight='bold')
 axes[1, 0].set_xlabel('Day of Year')
 axes[1, 0].set_ylabel('Plant ID')
 else:
 axes[1, 0].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
 transform=axes[1, 0].transAxes, fontsize=14, style='italic')
 axes[1, 0].set_title('Plant Bloom Progression Heatmap', fontweight='bold')
 else:
 axes[1, 0].text(0.5, 0.5, 'No Plant ID Data', ha='center', va='center',
 transform=axes[1, 0].transAxes, fontsize=14, style='italic')
 axes[1, 0].set_title('Plant Bloom Progression Heatmap', fontweight='bold')
 
 # 4. Bloom duration analysis
 if 'plant_id' in dataset_df.columns:
 bloom_durations = []
 
 for plant_id in dataset_df['plant_id'].unique():
 plant_data = dataset_df[dataset_df['plant_id'] == plant_id].sort_values('datetime')
 
 # Find bloom start and end
 bloom_stages = plant_data['bloom_stage'].values
 
 # Find first and last occurrence of bloom stages (1, 2, 3)
 bloom_indices = np.where(np.isin(bloom_stages, [1, 2, 3]))[0]
 
 if len(bloom_indices) > 1:
 start_date = plant_data.iloc[bloom_indices[0]]['datetime']
 end_date = plant_data.iloc[bloom_indices[-1]]['datetime']
 duration = (end_date - start_date).days
 bloom_durations.append(duration)
 
 if bloom_durations:
 axes[1, 1].hist(bloom_durations, bins=15, color='skyblue', 
 alpha=0.7, edgecolor='black')
 axes[1, 1].set_title('Bloom Duration Distribution', fontweight='bold')
 axes[1, 1].set_xlabel('Duration (days)')
 axes[1, 1].set_ylabel('Frequency')
 axes[1, 1].grid(True, alpha=0.3)
 
 # Add statistics
 mean_duration = np.mean(bloom_durations)
 axes[1, 1].axvline(mean_duration, color='red', linestyle='--', linewidth=2,
 label=f'Mean: {mean_duration:.1f} days')
 axes[1, 1].legend()
 else:
 axes[1, 1].text(0.5, 0.5, 'Insufficient Data for Duration Analysis', 
 ha='center', va='center', transform=axes[1, 1].transAxes, 
 fontsize=12, style='italic')
 axes[1, 1].set_title('Bloom Duration Distribution', fontweight='bold')
 else:
 axes[1, 1].text(0.5, 0.5, 'No Plant ID Data', ha='center', va='center',
 transform=axes[1, 1].transAxes, fontsize=14, style='italic')
 axes[1, 1].set_title('Bloom Duration Distribution', fontweight='bold')
 
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
 
 return fig

def create_bloom_calendar(
 dataset_df: pd.DataFrame,
 year: int,
 save_path: Optional[str] = None,
 figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
 """
 Create a calendar visualization showing bloom patterns throughout the year.
 
 Args:
 dataset_df: DataFrame with timestamp and bloom_stage columns
 year: Year to visualize
 save_path: Optional path to save the plot
 figsize: Figure size tuple
 
 Returns:
 matplotlib Figure object
 """
 # Filter data for the specified year
 dataset_df['datetime'] = pd.to_datetime(dataset_df['timestamp'])
 year_data = dataset_df[dataset_df['datetime'].dt.year == year].copy()
 
 if year_data.empty:
 raise ValueError(f"No data available for year {year}")
 
 # Create calendar grid
 fig, axes = plt.subplots(3, 4, figsize=figsize)
 fig.suptitle(f'Bloom Calendar - {year}', fontsize=20, fontweight='bold')
 
 axes = axes.flatten()
 month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
 
 stage_colors = {0: '#8B4513', 1: '#90EE90', 2: '#FFB6C1', 3: '#FFA500', 4: '#808080'}
 
 for month in range(1, 13):
 ax = axes[month - 1]
 
 # Get data for this month
 month_data = year_data[year_data['datetime'].dt.month == month]
 
 if not month_data.empty:
 # Group by day and get most common bloom stage
 daily_stages = month_data.groupby(month_data['datetime'].dt.day)['bloom_stage'].agg(
 lambda x: x.mode().iloc[0] if not x.empty else 0
 )
 
 # Create calendar grid for the month
 import calendar
 cal = calendar.monthcalendar(year, month)
 
 # Plot calendar
 for week_num, week in enumerate(cal):
 for day_num, day in enumerate(week):
 if day == 0:
 continue
 
 # Get bloom stage for this day
 stage = daily_stages.get(day, -1) # -1 for no data
 
 if stage >= 0:
 color = stage_colors.get(stage, 'white')
 alpha = 0.8
 else:
 color = 'lightgray'
 alpha = 0.3
 
 # Draw day square
 rect = plt.Rectangle((day_num, 5-week_num), 1, 1, 
 facecolor=color, alpha=alpha, edgecolor='black')
 ax.add_patch(rect)
 
 # Add day number
 ax.text(day_num + 0.5, 5-week_num + 0.5, str(day), 
 ha='center', va='center', fontweight='bold')
 
 ax.set_xlim(0, 7)
 ax.set_ylim(0, 6)
 ax.set_title(month_names[month-1], fontweight='bold')
 ax.set_xticks([])
 ax.set_yticks([])
 
 # Add day labels
 day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
 for i, label in enumerate(day_labels):
 ax.text(i + 0.5, -0.2, label, ha='center', va='center', fontsize=8)
 
 # Add legend
 legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='black')
 for color in stage_colors.values()]
 stage_names = ['Bud', 'Early Bloom', 'Full Bloom', 'Late Bloom', 'Dormant']
 
 fig.legend(legend_elements, stage_names, loc='lower center', 
 bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=12)
 
 plt.tight_layout()
 plt.subplots_adjust(bottom=0.1)
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
 
 return fig