"""
BloomWatch Fine-Tuned Flowers – Streamlit Inference App

Features:
- Loads fine-tuned ResNet50 model for flower classification
- Supports both original BloomWatch classes and flower classes
- CPU-optimized inference with real-time predictions

Usage:
 streamlit run webapp/app.py
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

import streamlit as st

# -----------------------------
# Configuration
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"

# Model paths
FLOWERS_EFFICIENTNET_MODEL = MODELS_DIR / "flowers_efficientnet_best.pt"
FLOWERS_BEST_MODEL = MODELS_DIR / "flowers_resnet50_best.pt"
FLOWERS_FINETUNED_MODEL = MODELS_DIR / "stage2_real_finetuned.pt"
BLOOMWATCH_MODEL = MODELS_DIR / "stage2_transfer_learning_bloomwatch.pt"

# Flower classes (new fine-tuned model)
FLOWER_CLASSES: List[str] = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
# Original BloomWatch classes (fallback)
BLOOMWATCH_CLASSES: List[str] = ["bud", "early_bloom", "full_bloom", "late_bloom", "dormant"]

# ImageNet normalization for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------
# Model Definitions
# -----------------------------
class EfficientNetFlowerClassifier(nn.Module):
 """EfficientNet-based flower classifier (matches advanced training script)."""
 
 def __init__(self, num_classes: int, pretrained: bool = False):
 super().__init__()
 # Load EfficientNetB0 backbone
 self.backbone = models.efficientnet_b0(weights=None) # No pretrained weights in inference
 
 # Adapt 5-channel input to 3-channel EfficientNet
 self.input_adaptation = nn.Conv2d(5, 3, kernel_size=1, bias=False)
 
 # Replace classifier with custom head
 in_features = self.backbone.classifier[1].in_features
 self.backbone.classifier = nn.Sequential(
 nn.Dropout(0.3),
 nn.Linear(in_features, 1024),
 nn.ReLU(inplace=True),
 nn.BatchNorm1d(1024),
 nn.Dropout(0.4),
 nn.Linear(1024, 512),
 nn.ReLU(inplace=True),
 nn.BatchNorm1d(512),
 nn.Dropout(0.3),
 nn.Linear(512, num_classes)
 )
 
 def forward(self, x):
 x = self.input_adaptation(x)
 return self.backbone(x)

class ResNet50FlowerClassifier(nn.Module):
 """ResNet50-based flower classifier (matches fine-tuning script)."""
 
 def __init__(self, num_classes: int, pretrained: bool = False):
 super().__init__()
 # Load ResNet50 backbone
 self.backbone = models.resnet50(weights=None) # No pretrained weights in inference
 
 # Adapt 5-channel input to 3-channel ResNet50
 self.input_adaptation = nn.Conv2d(5, 3, kernel_size=1, bias=False)
 
 # Replace final classifier
 self.backbone.fc = nn.Sequential(
 nn.Dropout(0.5),
 nn.Linear(self.backbone.fc.in_features, 512),
 nn.ReLU(inplace=True),
 nn.Dropout(0.3),
 nn.Linear(512, num_classes)
 )
 
 def forward(self, x):
 x = self.input_adaptation(x)
 return self.backbone(x)

class FineTunedTransferLearningCNN(nn.Module):
 """Legacy MobileNetV2 model for backward compatibility."""
 
 def __init__(self, num_classes: int = 5):
 super().__init__()
 self.backbone = models.mobilenet_v2(weights=None)
 self.input_adaptation = nn.Conv2d(5, 3, kernel_size=1, bias=False)
 self.backbone.classifier = nn.Sequential(
 nn.Dropout(0.2),
 nn.Linear(self.backbone.last_channel, 512),
 nn.ReLU(inplace=True),
 nn.Dropout(0.3),
 nn.Linear(512, num_classes),
 )

 def forward(self, x: torch.Tensor) -> torch.Tensor:
 x = self.input_adaptation(x)
 return self.backbone(x)

# -----------------------------
# Utilities
# -----------------------------
def preprocess_image_to_5ch(img: Image.Image) -> torch.Tensor:
 """Convert a PIL image to a normalized 5-channel tensor (RGB + synthetic NDVI/EVI)."""
 img = img.convert("RGB")
 img = img.resize((224, 224))
 arr = np.asarray(img, dtype=np.float32) / 255.0 # HWC, [0,1]
 rgb = torch.from_numpy(arr).permute(2, 0, 1) # 3x224x224
 
 # Normalize with ImageNet stats
 mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
 std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
 rgb = (rgb - mean) / std
 
 # Synthetic NDVI/EVI channels for single-image inference
 ndvi = torch.zeros(1, 224, 224, dtype=torch.float32)
 evi = torch.zeros(1, 224, 224, dtype=torch.float32)
 x = torch.cat([rgb, ndvi, evi], dim=0).unsqueeze(0) # 1x5x224x224
 return x

@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: Path) -> tuple[nn.Module, List[str]]:
 """Load model and return (model, class_names)."""
 device = torch.device("cpu")
 
 if not checkpoint_path.exists():
 raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
 
 # Determine model type based on checkpoint name
 checkpoint_str = str(checkpoint_path)
 
 if "flowers_efficientnet_best" in checkpoint_str:
 # Load EfficientNet flower classifier
 model = EfficientNetFlowerClassifier(num_classes=len(FLOWER_CLASSES), pretrained=False)
 class_names = FLOWER_CLASSES
 model_type = "EfficientNetB0 Flowers (Advanced)"
 elif "flowers_resnet50_best" in checkpoint_str or "real_finetuned" in checkpoint_str:
 # Load ResNet50 flower classifier
 model = ResNet50FlowerClassifier(num_classes=len(FLOWER_CLASSES), pretrained=False)
 class_names = FLOWER_CLASSES
 model_type = "ResNet50 Flowers"
 else:
 # Load legacy MobileNetV2 model
 model = FineTunedTransferLearningCNN(num_classes=len(BLOOMWATCH_CLASSES))
 class_names = BLOOMWATCH_CLASSES
 model_type = "MobileNetV2 BloomWatch"
 
 model.to(device)
 
 try:
 state = torch.load(checkpoint_path, map_location=device)
 # Load non-strict to tolerate minor head/backbone key diffs
 missing, unexpected = model.load_state_dict(state, strict=False)
 if missing or unexpected:
 st.warning(f"Model keys mismatched. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
 model.eval()
 return model, class_names, model_type
 except Exception as e:
 st.error(f"Failed to load model weights: {e}")
 raise

def infer(model: nn.Module, img: Image.Image, class_names: List[str]):
 """Run inference and return predictions."""
 device = torch.device("cpu")
 x = preprocess_image_to_5ch(img).to(device)
 with torch.no_grad():
 logits = model(x)
 probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
 
 # Get top 3 predictions
 top_indices = np.argsort(probs)[::-1][:3]
 top_predictions = [
 {"class": class_names[i], "confidence": float(probs[i])}
 for i in top_indices
 ]
 
 return top_predictions, probs

# -----------------------------
# UI
# -----------------------------
st.set_page_config(
 page_title="BloomWatch Fine-Tuned Flowers", 
 page_icon="", 
 layout="centered"
)

st.title(" BloomWatch AI - Multi-Model Classifier")
st.markdown("""
**Upload an image to get real-time predictions!**

Choose between different trained models:
- ** Flowers Model**: Classifies flower types (daisy, dandelion, rose, sunflower, tulip)
- ** Bloom Stages Model**: Classifies plant bloom stages (bud, early_bloom, full_bloom, late_bloom, dormant)

*CPU-optimized inference with ResNet50 and MobileNetV2 backbones*
""")

# Sidebar for model selection
with st.sidebar:
 st.header(" Model Settings")
 
 # Model selection
 model_choice = st.selectbox(
 "Select Model Type",
 [" Advanced Flowers Model (EfficientNetB0)", " Flowers Model (ResNet50)", " Bloom Stages Model (MobileNetV2)"],
 help="Choose between advanced flower classification, standard flower classification, or plant bloom stage classification"
 )
 
 # Set default checkpoint based on selection
 if "Advanced Flowers Model" in model_choice:
 default_checkpoint = str(FLOWERS_EFFICIENTNET_MODEL)
 model_description = "EfficientNetB0 trained on real-world flower datasets (90-95% accuracy target)"
 elif "Flowers Model" in model_choice:
 # Try best model first, fallback to fine-tuned
 if FLOWERS_BEST_MODEL.exists():
 default_checkpoint = str(FLOWERS_BEST_MODEL)
 model_description = "ResNet50 trained on full Kaggle Flowers dataset"
 else:
 default_checkpoint = str(FLOWERS_FINETUNED_MODEL)
 model_description = "ResNet50 fine-tuned on flowers dataset"
 else:
 default_checkpoint = str(BLOOMWATCH_MODEL)
 model_description = "MobileNetV2 trained on plant bloom stages"
 
 st.info(f"**{model_description}**")
 
 # Show available models
 st.write("**Available Models:**")
 if FLOWERS_EFFICIENTNET_MODEL.exists():
 st.write(" Advanced Flowers Model (EfficientNetB0)")
 if FLOWERS_BEST_MODEL.exists():
 st.write(" Full Flowers Model (ResNet50)")
 if FLOWERS_FINETUNED_MODEL.exists():
 st.write(" Fine-tuned Flowers Model")
 if BLOOMWATCH_MODEL.exists():
 st.write(" BloomWatch Model")
 
 # Custom checkpoint path
 ckpt_str = st.text_input(
 "Custom Checkpoint Path (Optional)", 
 value=default_checkpoint,
 help="Path to a custom model checkpoint file"
 )
 
 if st.button(" Load / Reload Model", type="primary"):
 st.cache_resource.clear()
 
 # Model loading status
 try:
 model, class_names, model_type = load_model(Path(ckpt_str))
 st.success(f" {model_type} loaded successfully")
 st.write(f"**Classes:** {', '.join(class_names)}")
 
 # Show model info
 if "Flowers" in model_type:
 st.write("**Flower Classes:**")
 for i, cls in enumerate(class_names):
 st.write(f"{i+1}. {cls.capitalize()}")
 else:
 st.write("**Bloom Stage Classes:**")
 for i, cls in enumerate(class_names):
 st.write(f"{i+1}. {cls.replace('_', ' ').title()}")
 
 except Exception as e:
 model, class_names, model_type = None, None, None
 st.error(f" Failed to load model: {e}")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
 st.subheader(" Upload Image")
 uploaded = st.file_uploader(
 "Choose an image", 
 type=["png", "jpg", "jpeg"], 
 accept_multiple_files=False,
 help="Upload an image for classification"
 )
 
 if uploaded is not None:
 try:
 image = Image.open(uploaded)
 st.image(image, caption="Uploaded Image", use_column_width=True)
 
 # Image info
 st.info(f"**Image Info:** {image.size[0]}×{image.size[1]} pixels, {image.mode} mode")
 except Exception as e:
 st.error(f"Failed to load image: {e}")
 image = None
 else:
 image = None
 st.info(" Please upload an image to get started")

with col2:
 st.subheader(" Predictions")
 
 if model is not None and image is not None:
 try:
 # Run inference
 with st.spinner("Running inference..."):
 predictions, all_probs = infer(model, image, class_names)
 
 # Display top prediction
 top_pred = predictions[0]
 st.success(f"**Predicted Class:** {top_pred['class']}")
 st.metric("Confidence", f"{top_pred['confidence']:.1%}")
 
 # Top 3 predictions
 st.write("**Top 3 Predictions:**")
 for i, pred in enumerate(predictions):
 st.write(f"{i+1}. **{pred['class']}**: {pred['confidence']:.1%}")
 
 # Confidence bar chart
 st.write("**Confidence Distribution:**")
 chart_data = pd.DataFrame({
 "Class": [p["class"] for p in predictions],
 "Confidence": [p["confidence"] for p in predictions]
 })
 st.bar_chart(chart_data.set_index("Class"))
 
 # Detailed probabilities
 with st.expander(" All Class Probabilities"):
 all_data = pd.DataFrame({
 "Class": class_names,
 "Probability": all_probs
 }).sort_values("Probability", ascending=False)
 st.dataframe(all_data, use_container_width=True)
 
 except Exception as e:
 st.error(f"Inference failed: {e}")
 st.exception(e)
 
 elif model is None:
 st.warning(" Please load a model first using the sidebar")
 elif image is None:
 st.info(" Please upload an image to see predictions")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
 <small>BloomWatch AI Multi-Model Classifier | CPU-Optimized | ResNet50 & MobileNetV2</small>
</div>
""", unsafe_allow_html=True)

