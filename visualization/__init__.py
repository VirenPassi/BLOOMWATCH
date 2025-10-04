"""
Visualization module for BloomWatch project.

This module provides comprehensive visualization tools for plant bloom detection,
including training plots, data exploration, and time-lapse animations.
"""

from .plots import (
 plot_training_metrics, plot_confusion_matrix,
 plot_class_distribution, plot_feature_maps
)
from .data_viz import (
 plot_dataset_samples, plot_bloom_progression,
 create_data_explorer, visualize_augmentations
)
from .timelapse import (
 create_bloom_timelapse, create_growth_animation,
 plot_growth_curve, visualize_temporal_patterns
)
from .interactive import (
 create_interactive_dashboard, plot_model_comparison,
 create_prediction_viewer, plot_attention_maps
)

__all__ = [
 "plot_training_metrics",
 "plot_confusion_matrix", 
 "plot_class_distribution",
 "plot_feature_maps",
 "plot_dataset_samples",
 "plot_bloom_progression",
 "create_data_explorer", 
 "visualize_augmentations",
 "create_bloom_timelapse",
 "create_growth_animation",
 "plot_growth_curve",
 "visualize_temporal_patterns",
 "create_interactive_dashboard",
 "plot_model_comparison",
 "create_prediction_viewer",
 "plot_attention_maps"
]