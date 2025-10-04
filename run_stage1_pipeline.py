"""
Stage-1 BloomWatch Pipeline Runner

Automates the complete Stage-1 dataset expansion and validation pipeline:
1. Download MODIS data for 2 AOIs
2. Preprocess MODIS granules
3. Synthesize plant images
4. Run training with quality assurance
5. Generate comprehensive report
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import json
from datetime import datetime
from typing import Dict

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

# Configuration
OUTPUTS_DIR = ROOT / "outputs"
STAGE1_RAW_DIR = ROOT / "data" / "raw" / "MODIS" / "stage1"
STAGE1_PROCESSED_DIR = ROOT / "data" / "processed" / "MODIS" / "stage1"
STAGE1_PLANT_DIR = ROOT / "data" / "expanded_dataset" / "plant_images"
STAGE1_METADATA = ROOT / "data" / "expanded_dataset" / "metadata.csv"

def run_command(command: str, description: str) -> bool:
 """Run a command and return success status."""
 print(f"\n {description}")
 print(f" Command: {command}")
 
 try:
 result = subprocess.run(command, shell=True, check=True, 
 capture_output=True, text=True)
 print(f" {description} completed successfully")
 if result.stdout:
 print(f" Output: {result.stdout[:500]}...")
 return True
 except subprocess.CalledProcessError as e:
 print(f" {description} failed with exit code {e.returncode}")
 if e.stderr:
 print(f" Error: {e.stderr}")
 return False

def check_prerequisites() -> bool:
 """Check if prerequisites are met."""
 print(" Checking prerequisites...")
 
 # Check Python
 try:
 result = subprocess.run("python --version", shell=True, capture_output=True, text=True)
 print(f" Python: {result.stdout.strip()}")
 except:
 print(" Python not found")
 return False
 
 # Check if we're in the right directory
 if not (ROOT / "pipelines" / "mini_bloomwatch.py").exists():
 print(" BloomWatch pipeline not found")
 print(f" Looking for: {ROOT / 'pipelines' / 'mini_bloomwatch.py'}")
 print(f" Current directory: {ROOT}")
 return False
 
 print(" Prerequisites check passed")
 return True

def download_modis_data() -> bool:
 """Download MODIS data for Stage-1."""
 print("\n Step 1: Downloading MODIS MOD13Q1 data...")
 
 # Check if data already exists
 if STAGE1_RAW_DIR.exists() and any(STAGE1_RAW_DIR.iterdir()):
 print(f" MODIS data already exists in {STAGE1_RAW_DIR}")
 response = input("Continue with existing data? (y/n): ").lower()
 if response != 'y':
 return False
 
 command = "python data/download_stage1_modis.py"
 return run_command(command, "MODIS data download")

def preprocess_modis_data() -> bool:
 """Preprocess MODIS granules."""
 print("\n Step 2: Preprocessing MODIS granules...")
 
 command = "python data/preprocess_stage1_modis.py"
 return run_command(command, "MODIS preprocessing")

def synthesize_plant_images() -> bool:
 """Synthesize plant images."""
 print("\n Step 3: Synthesizing plant images...")
 
 command = "python data/synthesize_stage1_plants.py"
 return run_command(command, "Plant image synthesis")

def run_training_pipeline() -> bool:
 """Run the training pipeline with quality assurance."""
 print("\n Step 4: Running training pipeline...")
 
 command = "python pipelines/mini_bloomwatch.py"
 return run_command(command, "Training pipeline")

def generate_stage1_report() -> Dict:
 """Generate comprehensive Stage-1 report."""
 print("\n Step 5: Generating Stage-1 report...")
 
 report = {
 "stage1_timestamp": datetime.now().isoformat(),
 "pipeline_version": "Stage-1 Enhanced",
 "dataset_overview": {},
 "modis_data": {},
 "plant_images": {},
 "training_results": {},
 "quality_assurance": {},
 "recommendations": {}
 }
 
 # Dataset overview
 if STAGE1_METADATA.exists():
 import pandas as pd
 df = pd.read_csv(STAGE1_METADATA)
 
 report["dataset_overview"] = {
 "total_images": len(df),
 "classes": df['bloom_stage'].value_counts().to_dict(),
 "train_samples": len(df[df['stage'] == 'train']),
 "val_samples": len(df[df['stage'] == 'val']),
 "test_samples": len(df[df['stage'] == 'test']),
 "is_synthetic": df['is_synthetic'].iloc[0] if 'is_synthetic' in df.columns else True
 }
 
 # MODIS data info
 if STAGE1_PROCESSED_DIR.exists():
 ndvi_files = list(STAGE1_PROCESSED_DIR.glob("*_ndvi.npy"))
 evi_files = list(STAGE1_PROCESSED_DIR.glob("*_evi.npy"))
 
 report["modis_data"] = {
 "ndvi_files": len(ndvi_files),
 "evi_files": len(evi_files),
 "processed_directory": str(STAGE1_PROCESSED_DIR),
 "total_size_gb": sum(f.stat().st_size for f in ndvi_files + evi_files) / (1024**3)
 }
 
 # Plant images info
 if STAGE1_PLANT_DIR.exists():
 total_images = 0
 for class_dir in STAGE1_PLANT_DIR.iterdir():
 if class_dir.is_dir():
 total_images += len(list(class_dir.glob("*.png")))
 
 report["plant_images"] = {
 "total_images": total_images,
 "images_directory": str(STAGE1_PLANT_DIR),
 "total_size_gb": sum(f.stat().st_size for f in STAGE1_PLANT_DIR.rglob("*.png")) / (1024**3)
 }
 
 # Training results
 prediction_files = list(OUTPUTS_DIR.glob("*_prediction.json"))
 if prediction_files:
 latest_prediction = max(prediction_files, key=lambda x: x.stat().st_mtime)
 with open(latest_prediction, 'r') as f:
 pred_data = json.load(f)
 
 report["training_results"] = {
 "model_type": pred_data.get("model_type", "unknown"),
 "final_train_acc": pred_data.get("quality_assurance", {}).get("final_train_acc", 0),
 "final_val_acc": pred_data.get("quality_assurance", {}).get("final_val_acc", 0),
 "best_val_acc": pred_data.get("quality_assurance", {}).get("best_val_acc", 0),
 "confusion_matrix_accuracy": pred_data.get("quality_assurance", {}).get("confusion_matrix_accuracy", 0),
 "training_epochs": pred_data.get("training_epochs", 0),
 "batch_size": pred_data.get("batch_size", 0)
 }
 
 # Quality assurance
 dataset_check_file = OUTPUTS_DIR / "dataset_check.json"
 if dataset_check_file.exists():
 with open(dataset_check_file, 'r') as f:
 qa_data = json.load(f)
 
 report["quality_assurance"] = {
 "leakage_detected": qa_data.get("leakage_detected", False),
 "train_val_overlap": qa_data.get("overlap_counts", {}).get("train_val", 0),
 "suspicious_accuracy": pred_data.get("quality_assurance", {}).get("suspicious_accuracy", False)
 }
 
 # Recommendations
 val_acc = report["training_results"].get("final_val_acc", 0)
 total_images = report["dataset_overview"].get("total_images", 0)
 
 if val_acc > 0.8 and total_images > 2000:
 report["recommendations"] = {
 "scale_to_10gb": "YES - High accuracy achieved with good dataset size",
 "next_steps": [
 "Scale to 10GB dataset with more AOIs",
 "Add more diverse plant species",
 "Implement real plant image collection",
 "Add temporal sequences for better temporal modeling"
 ],
 "confidence": "High"
 }
 elif val_acc > 0.6:
 report["recommendations"] = {
 "scale_to_10gb": "MAYBE - Good accuracy but consider improvements",
 "next_steps": [
 "Increase dataset diversity",
 "Improve data augmentation",
 "Add more AOIs for geographic diversity",
 "Consider transfer learning improvements"
 ],
 "confidence": "Medium"
 }
 else:
 report["recommendations"] = {
 "scale_to_10gb": "NO - Accuracy needs improvement first",
 "next_steps": [
 "Improve data quality and diversity",
 "Fix any data leakage issues",
 "Optimize model architecture",
 "Increase training data size"
 ],
 "confidence": "Low"
 }
 
 return report

def save_stage1_report(report: Dict):
 """Save Stage-1 report to markdown file."""
 report_path = OUTPUTS_DIR / "stage1_report.md"
 
 with open(report_path, 'w') as f:
 f.write("# BloomWatch Stage-1 Dataset Expansion Report\n\n")
 f.write(f"**Generated:** {report['stage1_timestamp']}\n")
 f.write(f"**Pipeline Version:** {report['pipeline_version']}\n\n")
 
 # Dataset Overview
 f.write("## Dataset Overview\n\n")
 overview = report.get("dataset_overview", {})
 f.write(f"- **Total Images:** {overview.get('total_images', 0):,}\n")
 f.write(f"- **Train Samples:** {overview.get('train_samples', 0):,}\n")
 f.write(f"- **Validation Samples:** {overview.get('val_samples', 0):,}\n")
 f.write(f"- **Test Samples:** {overview.get('test_samples', 0):,}\n")
 f.write(f"- **Synthetic Data:** {overview.get('is_synthetic', True)}\n\n")
 
 # Class Distribution
 f.write("### Class Distribution\n\n")
 classes = overview.get('classes', {})
 for class_name, count in classes.items():
 f.write(f"- **{class_name}:** {count:,} images\n")
 f.write("\n")
 
 # MODIS Data
 f.write("## MODIS Data\n\n")
 modis = report.get("modis_data", {})
 f.write(f"- **NDVI Files:** {modis.get('ndvi_files', 0)}\n")
 f.write(f"- **EVI Files:** {modis.get('evi_files', 0)}\n")
 f.write(f"- **Total Size:** {modis.get('total_size_gb', 0):.2f} GB\n\n")
 
 # Training Results
 f.write("## Training Results\n\n")
 training = report.get("training_results", {})
 f.write(f"- **Model Type:** {training.get('model_type', 'Unknown')}\n")
 f.write(f"- **Final Training Accuracy:** {training.get('final_train_acc', 0):.3f}\n")
 f.write(f"- **Final Validation Accuracy:** {training.get('final_val_acc', 0):.3f}\n")
 f.write(f"- **Best Validation Accuracy:** {training.get('best_val_acc', 0):.3f}\n")
 f.write(f"- **Confusion Matrix Accuracy:** {training.get('confusion_matrix_accuracy', 0):.3f}\n")
 f.write(f"- **Training Epochs:** {training.get('training_epochs', 0)}\n")
 f.write(f"- **Batch Size:** {training.get('batch_size', 0)}\n\n")
 
 # Quality Assurance
 f.write("## Quality Assurance\n\n")
 qa = report.get("quality_assurance", {})
 f.write(f"- **Data Leakage Detected:** {qa.get('leakage_detected', False)}\n")
 f.write(f"- **Train-Val Overlap:** {qa.get('train_val_overlap', 0)} plant_ids\n")
 f.write(f"- **Suspicious Accuracy:** {qa.get('suspicious_accuracy', False)}\n\n")
 
 # Recommendations
 f.write("## Recommendations\n\n")
 rec = report.get("recommendations", {})
 f.write(f"### Scale to 10GB Dataset: **{rec.get('scale_to_10gb', 'UNKNOWN')}**\n\n")
 f.write(f"**Confidence:** {rec.get('confidence', 'Unknown')}\n\n")
 
 f.write("### Next Steps:\n\n")
 for step in rec.get('next_steps', []):
 f.write(f"- {step}\n")
 f.write("\n")
 
 f.write("---\n")
 f.write("*Report generated by BloomWatch Stage-1 Pipeline*\n")
 
 print(f" Stage-1 report saved to: {report_path}")
 return report_path

def main():
 """Main Stage-1 pipeline runner."""
 print(" Starting BloomWatch Stage-1 Dataset Expansion Pipeline")
 print("=" * 60)
 
 start_time = time.time()
 
 # Check prerequisites
 if not check_prerequisites():
 print(" Prerequisites check failed")
 return False
 
 # Step 1: Download MODIS data
 if not download_modis_data():
 print(" MODIS data download failed")
 return False
 
 # Step 2: Preprocess MODIS data
 if not preprocess_modis_data():
 print(" MODIS preprocessing failed")
 return False
 
 # Step 3: Synthesize plant images
 if not synthesize_plant_images():
 print(" Plant image synthesis failed")
 return False
 
 # Step 4: Run training pipeline
 if not run_training_pipeline():
 print(" Training pipeline failed")
 return False
 
 # Step 5: Generate report
 report = generate_stage1_report()
 report_path = save_stage1_report(report)
 
 # Final summary
 total_time = time.time() - start_time
 print("\n" + "=" * 60)
 print(" Stage-1 Pipeline Complete!")
 print(f"⏱ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
 print(f" Report: {report_path}")
 
 # Print key results
 val_acc = report["training_results"].get("final_val_acc", 0)
 total_images = report["dataset_overview"].get("total_images", 0)
 scale_recommendation = report["recommendations"].get("scale_to_10gb", "UNKNOWN")
 
 print(f"\n Key Results:")
 print(f" Validation Accuracy: {val_acc:.3f}")
 print(f" Total Images: {total_images:,}")
 print(f" Scale to 10GB: {scale_recommendation}")
 
 return True

if __name__ == "__main__":
 main()
