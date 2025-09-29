# BloomWatch Stage-2 - Submission Summary

## Model Architecture

- Backbone: MobileNetV2 (ImageNet weights)
- Input: 5 channels (RGB + NDVI + EVI) via 1x1 Conv adapter
- Classifier: Dropout(0.2) -> Linear(512) -> ReLU -> Dropout(0.3) -> Linear(5)

## Dataset

- Plant images (5 bloom classes) + MODIS NDVI/EVI (stage-2)
- Train/Val/Test split: 70/15/15 (by random split with fixed seed)

## Training Results (from prior run)

- Best Validation Accuracy: 91.2%
- Final Train Accuracy: 89.6%
- Final Validation Accuracy: 91.1%
- Curves: see `outputs/learning_curves.png`

## Final Evaluation (CPU)

- Checkpoint: `D:\NASA(0)\BloomWatch\outputs\models\stage2_fine_tuned_bloomwatch.pt`
- Validation Accuracy: 0.7980
- Test Accuracy: 0.7972
- Confusion Matrices: `outputs/final_confusion_val.png`, `outputs/final_confusion_test.png`
- Full reports: `outputs/final_metrics.json`, `outputs/final_classification_report_val.json`, `outputs/final_classification_report_test.json`

## Reproduce (CPU-only)

```powershell
# Create venv and install
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Evaluate-only with existing checkpoint
python pipelines\finalize_submission.py
```

## Notes

- All steps are CPU-friendly.
- To retrain, use `pipelines/stage2_training.py` with `--fine-tune`.
