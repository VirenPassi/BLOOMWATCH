"""
API endpoints for BloomWatch plant bloom detection.

This module defines all the API routes and their handlers
for plant bloom classification and model management.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import asyncio
import logging
from PIL import Image
import io
import base64
import numpy as np

from .models import (
    PredictionRequest, PredictionResponse, BatchPredictionResponse,
    ModelInfo, ModelStatus, BloomStage
)
from .utils import ModelManager, preprocess_image, encode_image_to_base64

# Setup logging
logger = logging.getLogger(__name__)

# Global model manager (will be set by main.py)
model_manager = None

def get_model_manager() -> ModelManager:
    """Dependency to get the global model manager."""
    global model_manager
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return model_manager

# Create router
router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_bloom_stage(
    file: UploadFile = File(...),
    model_name: str = Form("default"),
    include_confidence: bool = Form(True),
    include_features: bool = Form(False),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict the bloom stage of a plant from an uploaded image.
    
    Args:
        file: Image file to analyze
        model_name: Name of the model to use for prediction
        include_confidence: Whether to include confidence scores for all classes
        include_features: Whether to include extracted features
        
    Returns:
        PredictionResponse with predicted bloom stage and metadata
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"File must be an image, got {file.content_type}"
            )
        
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Get model
        model_info = await model_manager.get_model(model_name)
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        # Preprocess image
        processed_image = preprocess_image(image, model_info.get('input_size', (224, 224)))
        
        # Make prediction
        prediction_result = await model_manager.predict(
            model_name, 
            processed_image,
            include_confidence=include_confidence,
            include_features=include_features
        )
        
        # Prepare response
        response = PredictionResponse(
            predicted_class=prediction_result['predicted_class'],
            predicted_stage=BloomStage(prediction_result['predicted_class']),
            confidence=prediction_result['confidence'],
            processing_time=prediction_result['processing_time'],
            model_name=model_name,
            image_info={
                'filename': file.filename,
                'size': image.size,
                'format': image.format or 'Unknown'
            }
        )
        
        if include_confidence:
            response.class_confidences = prediction_result.get('class_confidences', {})
        
        if include_features:
            response.features = prediction_result.get('features', [])
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_name: str = Form("default"),
    include_confidence: bool = Form(True),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict bloom stages for multiple images in a batch.
    
    Args:
        files: List of image files to analyze
        model_name: Name of the model to use for prediction
        include_confidence: Whether to include confidence scores
        
    Returns:
        BatchPredictionResponse with predictions for all images
    """
    try:
        if len(files) > 50:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Batch size cannot exceed 50 images"
            )
        
        # Get model
        model_info = await model_manager.get_model(model_name)
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        # Process all images
        predictions = []
        failed_predictions = []
        
        for i, file in enumerate(files):
            try:
                # Validate and read image
                if not file.content_type.startswith('image/'):
                    failed_predictions.append({
                        'index': i,
                        'filename': file.filename,
                        'error': f"Invalid content type: {file.content_type}"
                    })
                    continue
                
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert('RGB')
                
                # Preprocess and predict
                processed_image = preprocess_image(image, model_info.get('input_size', (224, 224)))
                result = await model_manager.predict(
                    model_name, 
                    processed_image,
                    include_confidence=include_confidence
                )
                
                # Create individual prediction response
                pred_response = PredictionResponse(
                    predicted_class=result['predicted_class'],
                    predicted_stage=BloomStage(result['predicted_class']),
                    confidence=result['confidence'],
                    processing_time=result['processing_time'],
                    model_name=model_name,
                    image_info={
                        'filename': file.filename,
                        'size': image.size,
                        'format': image.format or 'Unknown',
                        'index': i
                    }
                )
                
                if include_confidence:
                    pred_response.class_confidences = result.get('class_confidences', {})
                
                predictions.append(pred_response)
                
            except Exception as e:
                failed_predictions.append({
                    'index': i,
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_images=len(files),
            successful_predictions=len(predictions),
            failed_predictions=failed_predictions,
            model_name=model_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post("/predict/url")
async def predict_from_url(
    request: PredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict bloom stage from an image URL.
    
    Args:
        request: Prediction request with image URL and options
        
    Returns:
        PredictionResponse with prediction results
    """
    try:
        import requests
        
        # Download image from URL
        response = requests.get(request.image_url, timeout=30)
        response.raise_for_status()
        
        # Validate and process image
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Get model
        model_info = await model_manager.get_model(request.model_name)
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found"
            )
        
        # Preprocess and predict
        processed_image = preprocess_image(image, model_info.get('input_size', (224, 224)))
        result = await model_manager.predict(
            request.model_name, 
            processed_image,
            include_confidence=request.include_confidence,
            include_features=request.include_features
        )
        
        # Prepare response
        pred_response = PredictionResponse(
            predicted_class=result['predicted_class'],
            predicted_stage=BloomStage(result['predicted_class']),
            confidence=result['confidence'],
            processing_time=result['processing_time'],
            model_name=request.model_name,
            image_info={
                'url': request.image_url,
                'size': image.size,
                'format': image.format or 'Unknown'
            }
        )
        
        if request.include_confidence:
            pred_response.class_confidences = result.get('class_confidences', {})
        
        if request.include_features:
            pred_response.features = result.get('features', [])
        
        return pred_response
        
    except requests.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image from URL: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"URL prediction failed: {str(e)}"
        )


@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    List all available models.
    
    Returns:
        List of ModelInfo objects with model details
    """
    try:
        models = await model_manager.list_models()
        return [ModelInfo(**model) for model in models]
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelInfo with model details
    """
    try:
        model_info = await model_manager.get_model(model_name)
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        return ModelInfo(**model_info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/models/{model_name}/status", response_model=ModelStatus)
async def get_model_status(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Get the current status of a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelStatus with current model status
    """
    try:
        status = await model_manager.get_model_status_detailed(model_name)
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        return ModelStatus(**status)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )


@router.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    model_path: str = Form(...),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Load a model from a file path.
    
    Args:
        model_name: Name to assign to the loaded model
        model_path: Path to the model file
        
    Returns:
        Success message and model info
    """
    try:
        success = await model_manager.load_model(model_name, model_path)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model from {model_path}"
            )
        
        model_info = await model_manager.get_model(model_name)
        return {
            "message": f"Model '{model_name}' loaded successfully",
            "model_info": model_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@router.delete("/models/{model_name}")
async def unload_model(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Unload a model from memory.
    
    Args:
        model_name: Name of the model to unload
        
    Returns:
        Success message
    """
    try:
        if model_name == "default":
            raise HTTPException(
                status_code=400,
                detail="Cannot unload default model"
            )
        
        success = await model_manager.unload_model(model_name)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        return {"message": f"Model '{model_name}' unloaded successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}"
        )


@router.get("/bloom-stages")
async def get_bloom_stages():
    """
    Get information about available bloom stages.
    
    Returns:
        Dictionary with bloom stage information
    """
    return {
        "bloom_stages": {
            0: {
                "name": "bud",
                "description": "Plant has formed buds but no flowers are visible"
            },
            1: {
                "name": "early_bloom", 
                "description": "First flowers are starting to open"
            },
            2: {
                "name": "full_bloom",
                "description": "Plant is at peak flowering"
            },
            3: {
                "name": "late_bloom",
                "description": "Flowers are beginning to fade"
            },
            4: {
                "name": "dormant",
                "description": "Plant is not actively flowering"
            }
        },
        "total_stages": 5
    }