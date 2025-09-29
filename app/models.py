"""
Pydantic models for API request/response schemas.

This module defines the data models used for API requests and responses
with proper validation and documentation.
"""

from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import datetime


class BloomStage(str, Enum):
    """Enumeration of plant bloom stages."""
    BUD = "bud"
    EARLY_BLOOM = "early_bloom"
    FULL_BLOOM = "full_bloom"
    LATE_BLOOM = "late_bloom"
    DORMANT = "dormant"


class PredictionRequest(BaseModel):
    """Request model for image prediction."""
    
    image_url: HttpUrl = Field(
        ...,
        description="URL of the image to analyze",
        example="https://example.com/plant_image.jpg"
    )
    
    model_name: str = Field(
        default="default",
        description="Name of the model to use for prediction",
        example="default"
    )
    
    include_confidence: bool = Field(
        default=True,
        description="Whether to include confidence scores for all classes"
    )
    
    include_features: bool = Field(
        default=False,
        description="Whether to include extracted features in the response"
    )

    @validator('model_name')
    def validate_model_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Model name cannot be empty')
        return v.strip()


class ImageInfo(BaseModel):
    """Information about the processed image."""
    
    filename: Optional[str] = Field(
        None,
        description="Original filename of the image"
    )
    
    url: Optional[str] = Field(
        None,
        description="URL of the image (if applicable)"
    )
    
    size: Optional[List[int]] = Field(
        None,
        description="Image dimensions [width, height]",
        example=[1024, 768]
    )
    
    format: Optional[str] = Field(
        None,
        description="Image format (JPEG, PNG, etc.)",
        example="JPEG"
    )
    
    index: Optional[int] = Field(
        None,
        description="Index in batch processing"
    )


class PredictionResponse(BaseModel):
    """Response model for bloom stage prediction."""
    
    predicted_class: int = Field(
        ...,
        description="Predicted class index (0-4)",
        ge=0,
        le=4,
        example=2
    )
    
    predicted_stage: BloomStage = Field(
        ...,
        description="Predicted bloom stage name",
        example=BloomStage.FULL_BLOOM
    )
    
    confidence: float = Field(
        ...,
        description="Confidence score for the prediction (0-1)",
        ge=0,
        le=1,
        example=0.87
    )
    
    class_confidences: Optional[Dict[str, float]] = Field(
        None,
        description="Confidence scores for all classes",
        example={
            "bud": 0.05,
            "early_bloom": 0.10,
            "full_bloom": 0.87,
            "late_bloom": 0.15,
            "dormant": 0.03
        }
    )
    
    features: Optional[List[float]] = Field(
        None,
        description="Extracted features from the model (if requested)"
    )
    
    processing_time: float = Field(
        ...,
        description="Processing time in seconds",
        ge=0,
        example=0.234
    )
    
    model_name: str = Field(
        ...,
        description="Name of the model used for prediction",
        example="default"
    )
    
    image_info: ImageInfo = Field(
        ...,
        description="Information about the processed image"
    )
    
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp of the prediction"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of successful predictions"
    )
    
    total_images: int = Field(
        ...,
        description="Total number of images in the batch",
        ge=0,
        example=10
    )
    
    successful_predictions: int = Field(
        ...,
        description="Number of successful predictions",
        ge=0,
        example=8
    )
    
    failed_predictions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of failed predictions with error details"
    )
    
    model_name: str = Field(
        ...,
        description="Name of the model used for predictions",
        example="default"
    )
    
    total_processing_time: float = Field(
        default=0.0,
        description="Total processing time for the batch",
        ge=0
    )
    
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp of the batch prediction"
    )

    @validator('failed_predictions')
    def validate_failed_predictions(cls, v, values):
        # Ensure consistency between counts and lists
        if 'successful_predictions' in values and 'total_images' in values:
            expected_failures = values['total_images'] - values['successful_predictions']
            if len(v) != expected_failures:
                raise ValueError('Inconsistent failure count')
        return v


class ModelInfo(BaseModel):
    """Information about a loaded model."""
    
    name: str = Field(
        ...,
        description="Model name/identifier",
        example="resnet18_bloom_v1"
    )
    
    architecture: str = Field(
        ...,
        description="Model architecture type",
        example="ResNet18"
    )
    
    version: str = Field(
        default="unknown",
        description="Model version",
        example="1.0.0"
    )
    
    num_classes: int = Field(
        ...,
        description="Number of output classes",
        example=5
    )
    
    input_size: List[int] = Field(
        ...,
        description="Expected input image size [height, width]",
        example=[224, 224]
    )
    
    num_parameters: Optional[int] = Field(
        None,
        description="Total number of model parameters",
        example=11689512
    )
    
    model_size_mb: Optional[float] = Field(
        None,
        description="Model size in megabytes",
        example=44.7
    )
    
    training_accuracy: Optional[float] = Field(
        None,
        description="Training accuracy (if available)",
        ge=0,
        le=1,
        example=0.95
    )
    
    validation_accuracy: Optional[float] = Field(
        None,
        description="Validation accuracy (if available)",
        ge=0,
        le=1,
        example=0.91
    )
    
    created_at: Optional[datetime.datetime] = Field(
        None,
        description="Model creation timestamp"
    )
    
    loaded_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp when model was loaded"
    )
    
    device: str = Field(
        default="cpu",
        description="Device where model is loaded (cpu, cuda, mps)",
        example="cuda"
    )


class ModelStatus(BaseModel):
    """Current status of a model."""
    
    name: str = Field(
        ...,
        description="Model name",
        example="default"
    )
    
    status: str = Field(
        ...,
        description="Current model status",
        example="loaded"
    )
    
    device: str = Field(
        ...,
        description="Device where model is running",
        example="cuda"
    )
    
    memory_usage_mb: Optional[float] = Field(
        None,
        description="Current memory usage in MB",
        example=512.3
    )
    
    total_predictions: int = Field(
        default=0,
        description="Total number of predictions made",
        ge=0,
        example=1347
    )
    
    avg_processing_time: Optional[float] = Field(
        None,
        description="Average processing time per prediction",
        example=0.156
    )
    
    last_prediction_time: Optional[datetime.datetime] = Field(
        None,
        description="Timestamp of last prediction"
    )
    
    load_time: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When the model was loaded"
    )
    
    health_status: str = Field(
        default="healthy",
        description="Health status of the model",
        example="healthy"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(
        ...,
        description="Error type",
        example="ValidationError"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message",
        example="Invalid input image format"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Error timestamp"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request identifier for debugging"
    )


class HealthResponse(BaseModel):
    """API health check response."""
    
    status: str = Field(
        ...,
        description="Overall API status",
        example="healthy"
    )
    
    timestamp: float = Field(
        ...,
        description="Current timestamp",
        example=1634567890.123
    )
    
    api_version: str = Field(
        ...,
        description="API version",
        example="0.1.0"
    )
    
    python_version: str = Field(
        ...,
        description="Python/PyTorch version",
        example="3.9.7"
    )
    
    models: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of loaded models",
        example={"default": "loaded", "resnet50": "loading"}
    )
    
    system_info: Optional[Dict[str, Any]] = Field(
        None,
        description="System information (optional)"
    )


class BloomStageInfo(BaseModel):
    """Information about a bloom stage."""
    
    name: str = Field(
        ...,
        description="Stage name",
        example="full_bloom"
    )
    
    description: str = Field(
        ...,
        description="Stage description",
        example="Plant is at peak flowering"
    )
    
    index: int = Field(
        ...,
        description="Stage index",
        example=2
    )


class BloomStagesResponse(BaseModel):
    """Response containing information about all bloom stages."""
    
    bloom_stages: Dict[int, BloomStageInfo] = Field(
        ...,
        description="Dictionary of bloom stages indexed by their numeric ID"
    )
    
    total_stages: int = Field(
        ...,
        description="Total number of bloom stages",
        example=5
    )