"""
Hyperparameter Tuning for BloomWatch Stage-2
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import existing pipeline components
from pipelines.mini_bloomwatch import (
 MODISPlantBloomDataset, EnhancedDataAugmentation, TransferLearningCNN,
 check_dataset_leakage, plot_learning_curves, compute_confusion_matrix,
 check_suspicious_accuracy, resplit_dataset_by_plant_id
)
from pipelines.stage2_training import (
 FineTunedTransferLearningCNN, train_stage2_enhanced
)

# Configuration
STAGE2_PLANT_DIR = ROOT / "data" / "expanded_dataset" / "plant_images"
STAGE2_METADATA = ROOT / "data" / "expanded_dataset" / "metadata.csv"
STAGE2_PROCESSED_DIR = ROOT / "data" / "processed" / "MODIS" / "stage2"
OUTPUTS_DIR = ROOT / "outputs"

CLASSES = ["bud", "early_bloom", "full_bloom", "late_bloom", "dormant"]

def run_hyperparameter_experiment(experiment_name: str, params: Dict) -> Dict:
 """Run a single hyperparameter experiment."""
 print(f"\n{'='*50}")
 print(f"Running experiment: {experiment_name}")
 print(f"Parameters: {params}")
 print(f"{'='*50}")
 
 # Run training with specified parameters
 training_results = train_stage2_enhanced(
 model_type=params.get("model_type", "transfer_learning"),
 fine_tune=params.get("fine_tune", False),
 epochs=params.get("epochs", 10),
 batch_size=params.get("batch_size", 8)
 )
 
 # Save experiment results
 experiment_results = {
 "experiment_name": experiment_name,
 "parameters": params,
 "results": training_results,
 "timestamp": time.time()
 }
 
 # Save to file
 experiment_file = OUTPUTS_DIR / f"hyperparameter_experiment_{experiment_name}.json"
 with open(experiment_file, 'w') as f:
 json.dump(experiment_results, f, indent=2)
 
 print(f"Experiment results saved to: {experiment_file}")
 return experiment_results

def run_hyperparameter_tuning():
 """Run systematic hyperparameter tuning experiments."""
 print("Starting hyperparameter tuning for BloomWatch Stage-2")
 
 # Define experiments
 experiments = [
 {
 "name": "baseline_transfer_learning",
 "params": {
 "model_type": "transfer_learning",
 "fine_tune": False,
 "epochs": 10,
 "batch_size": 8
 }
 },
 {
 "name": "fine_tuned_low_lr",
 "params": {
 "model_type": "fine_tuned",
 "fine_tune": True,
 "epochs": 10,
 "batch_size": 8
 }
 },
 {
 "name": "fine_tuned_higher_batch",
 "params": {
 "model_type": "fine_tuned",
 "fine_tune": True,
 "epochs": 10,
 "batch_size": 16
 }
 },
 {
 "name": "transfer_learning_higher_batch",
 "params": {
 "model_type": "transfer_learning",
 "fine_tune": False,
 "epochs": 10,
 "batch_size": 16
 }
 }
 ]
 
 # Run experiments
 results = []
 for experiment in experiments:
 result = run_hyperparameter_experiment(
 experiment["name"], 
 experiment["params"]
 )
 results.append(result)
 
 # Find best experiment
 best_result = max(results, key=lambda x: x["results"]["best_val_acc"])
 print(f"\n{'='*50}")
 print(f"Best experiment: {best_result['experiment_name']}")
 print(f"Best validation accuracy: {best_result['results']['best_val_acc']:.4f}")
 print(f"{'='*50}")
 
 # Save summary
 summary = {
 "experiments": results,
 "best_experiment": best_result,
 "timestamp": time.time()
 }
 
 summary_file = OUTPUTS_DIR / "hyperparameter_tuning_summary.json"
 with open(summary_file, 'w') as f:
 json.dump(summary, f, indent=2)
 
 print(f"Tuning summary saved to: {summary_file}")
 return summary

def main():
 """Main hyperparameter tuning function."""
 run_hyperparameter_tuning()

if __name__ == "__main__":
 main()