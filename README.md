# 🌸 BloomWatch

> **AI-Powered Plant Bloom Detection and Tracking System**

BloomWatch is a comprehensive, modular Python project template for detecting and tracking plant blooming stages from image datasets using deep learning. Perfect for researchers, botanists, and AI enthusiasts working with time-lapse plant growth data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **🔬 Research-Ready**: Modular architecture perfect for experimentation and deployment
- **🏗️ Production-Ready**: FastAPI web service for model serving and predictions
- **📊 Comprehensive Analytics**: Built-in visualization and metrics tracking
- **☁️ Cloud Integration**: AWS S3 support for large-scale dataset management
- **🛰️ NASA Data Support**: Automated MODIS satellite data downloading
- **🧪 Fully Tested**: Complete pytest suite with unit test coverage
- **📝 Interactive Notebooks**: Jupyter notebooks for data exploration and experiments
- **⚙️ Configurable**: YAML-based configuration with Hydra/OmegaConf support

## 📁 Project Structure

```
BloomWatch/
├── 📂 app/                     # FastAPI web application
│   ├── __init__.py
│   ├── main.py                 # FastAPI app setup with lifespan management
│   ├── endpoints.py            # API endpoints for predictions
│   └── models.py              # Pydantic models for API
├── 📂 configs/                 # Configuration files
│   └── config.yaml            # Training configuration
├── 📂 data/                    # Data handling and preprocessing
│   ├── __init__.py
│   ├── dataset.py             # Dataset classes
│   ├── preprocessing.py        # Image processing utilities
│   ├── augmentations.py       # Data augmentation
│   ├── downloaders.py         # AWS S3 integration
│   └── fetch_modis.py         # NASA MODIS data fetching
├── 📂 models/                  # Model architectures and utilities
│   ├── __init__.py
│   ├── baseline.py            # SimpleCNN and ResNet models
│   ├── advanced.py            # Advanced architectures (EfficientNet, Vision Transformer)
│   ├── losses.py              # Custom loss functions
│   └── utils.py               # Model utilities (save/load, etc.)
├── 📂 notebooks/               # Jupyter notebooks for experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_experiments.ipynb
│   └── 03_model_evaluation.ipynb
├── 📂 tests/                   # Test suite
│   ├── conftest.py            # Test configuration and fixtures
│   ├── test_data.py           # Data module tests
│   ├── test_models.py         # Model tests
│   └── test_utils.py          # Utility tests
├── 📂 utils/                   # Utility functions
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── logging_utils.py       # Logging setup
│   ├── metrics.py             # Metrics tracking
│   └── helpers.py             # General utilities
├── 📂 visualization/           # Visualization and plotting
│   ├── __init__.py
│   ├── plots.py               # Training plots and confusion matrices
│   ├── plot_growth_curve.py   # Plant growth visualizations
│   ├── interactive.py         # Interactive dashboards
│   └── timelapse.py           # Time-lapse animations
├── main.py                     # Main training script
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/BloomWatch.git
   cd BloomWatch
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### 1. Run Training Pipeline

Start with the dummy training loop to test the complete pipeline:

```bash
# Basic training with default config
python main.py

# Custom training with specific parameters
python main.py --model resnet_baseline --epochs 20 --batch_size 32 --lr 0.001

# Training with custom config file
python main.py --config configs/config.yaml --device cuda
```

### 2. Fetch NASA MODIS Data

Download MODIS vegetation index data for your region of interest:

```bash
# Download MODIS MOD13Q1 data for a bounding box
python data/fetch_modis.py --start 2022-01-01 --end 2022-12-31 --bbox "70,8,90,37"

# List available granules without downloading
python data/fetch_modis.py --start 2022-01-01 --end 2022-12-31 --bbox "70,8,90,37" --list-only

# Force re-download even if files exist
python data/fetch_modis.py --start 2022-01-01 --end 2022-12-31 --bbox "70,8,90,37" --force
```

### 3. Start Web API

Launch the FastAPI web service for model predictions:

```bash
# Start the FastAPI server
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### 4. Explore with Notebooks

Open Jupyter notebooks for interactive experimentation:

```bash
jupyter notebook notebooks/
```

## 📖 Usage Examples

### Training a Model

```python
from data.dataset import PlantBloomDataset
from models.baseline import SimpleCNN
from utils.config import ConfigManager

# Load configuration
config = ConfigManager('configs/config.yaml').get_config()

# Create dataset
dataset = PlantBloomDataset(
    data_dir="path/to/data",
    annotations_file="path/to/annotations.csv"
)

# Initialize model
model = SimpleCNN(num_classes=5)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Making Predictions via API

```python
import requests

# Single image prediction
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    files={"file": open("plant_image.jpg", "rb")}
)
result = response.json()
print(f"Predicted stage: {result['predicted_stage']}")
```

### Visualizing Growth Curves

```python
from visualization.plot_growth_curve import plot_growth_curve

# Plot plant growth over time
plot_growth_curve(
    time_points=[0, 7, 14, 21, 28],
    bloom_scores=[0.1, 0.3, 0.7, 0.9, 0.8],
    save_path="growth_curve.png"
)
```

### Fetching NASA MODIS Data

```python
from data.fetch_modis import authenticate_earthdata, list_modis_granules, download_granules

# Authenticate with NASA Earthdata
if authenticate_earthdata():
    # List available granules
    granules = list_modis_granules(
        start_date="2022-01-01",
        end_date="2022-12-31",
        bbox=(70, 8, 90, 37)  # (west, south, east, north)
    )
    
    # Download granules
    download_granules(
        granules=granules,
        output_dir="data/raw/MODIS"
    )
```

## 🧪 Testing

Run the complete test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test modules
pytest tests/test_models.py -v
```

## ⚙️ Configuration

The project uses YAML configuration files with OmegaConf. Modify `configs/config.yaml`:

```yaml
# Model configuration
model:
  num_classes: 5
  name: "SimpleCNN"

# Training parameters
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

# Data configuration
data:
  image_size: [224, 224]
  batch_size: 32
  num_workers: 4

# AWS S3 configuration (optional)
aws:
  bucket_name: "your-bloom-dataset"
  region: "us-west-2"
```

## 🌐 API Endpoints

The FastAPI application provides several endpoints:

- **POST `/api/v1/predict`** - Single image prediction
- **POST `/api/v1/predict/batch`** - Batch image predictions
- **POST `/api/v1/predict/url`** - Predict from image URL
- **GET `/api/v1/models`** - List available models
- **GET `/health`** - Health check

## 📊 Supported Models

### Baseline Models
- **SimpleCNN**: Lightweight CNN for quick experimentation
- **ResNetBaseline**: ResNet-based architecture for better performance

### Advanced Models (Placeholders)
- **EfficientNet**: Efficient convolutional networks
- **Vision Transformer**: Transformer-based image classification
- **Attention Models**: Custom attention mechanisms

## 🔧 Extending the Project

### Adding New Models

1. Create your model in `models/advanced.py`:
```python
class YourCustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Your model implementation
        
    def forward(self, x):
        # Forward pass
        return x
```

2. Update the model factory in `models/baseline.py`

### Adding New Data Sources

1. Create a new dataset class in `data/dataset.py`
2. Implement required methods: `__len__`, `__getitem__`
3. Add preprocessing in `data/preprocessing.py`

### Custom Loss Functions

Add new loss functions to `models/losses.py`:

```python
class YourCustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        # Your loss computation
        return loss
```

## 📈 Monitoring and Visualization

### Training Metrics
- Loss curves (training/validation)
- Accuracy plots
- Confusion matrices
- Learning rate schedules

### Growth Analysis
- Time-series bloom progression
- Interactive dashboards
- Time-lapse animations
- Statistical analysis

## 🐳 Docker Support (Optional)

Create a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔍 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in config
python main.py --batch_size 16
```

**Import Errors**
```bash
# Ensure you're in the project root and have activated the virtual environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**AWS Permissions**
```bash
# Configure AWS credentials
aws configure
```

**NASA Earthdata Authentication**
```bash
# Set up Earthdata credentials
# Visit https://urs.earthdata.nasa.gov/ to register and get credentials
```

## 📚 Research Applications

BloomWatch is designed for:

- **Botanical Research**: Track flowering patterns across seasons
- **Agricultural Monitoring**: Optimize crop timing and yield prediction
- **Climate Studies**: Analyze blooming responses to environmental changes
- **Phenology Research**: Study plant life cycle timing
- **Conservation**: Monitor endangered plant species

## 🎯 Future Enhancements

- [ ] Multi-modal learning (images + environmental data)
- [ ] Real-time streaming from IoT cameras
- [ ] 3D plant reconstruction
- [ ] Mobile app for field data collection
- [ ] Integration with weather APIs
- [ ] Automated report generation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- PyTorch team for the deep learning framework
- FastAPI developers for the excellent web framework
- The open-source community for inspiration and tools
- NASA Earthdata for providing open access to MODIS satellite data

## 📧 Contact

- **Project Maintainer**: Viren Passi
- **Email**: virenpassi79@gmail.com
- **GitHub**: [@VirenPassi](https://github.com/VirenPassi)

---

**Happy Blooming! 🌸**

*BloomWatch - Bringing AI to the Garden*
