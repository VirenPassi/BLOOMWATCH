"""
Main FastAPI application for BloomWatch.

This module sets up the FastAPI application with middleware,
error handling, and API documentation.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
import torch

# Import application components
from .endpoints import router
from .config import AppConfig
from .utils import ModelManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model manager
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown events.
    """
    # Startup
    logger.info("Starting BloomWatch API...")
    
    # Initialize model manager
    global model_manager
    config = AppConfig()
    model_manager = ModelManager(config)
    
    # Set the model manager in endpoints module
    from . import endpoints
    endpoints.model_manager = model_manager
    
    # Load default model
    try:
        if config.default_model_path:
            await model_manager.load_model("default", config.default_model_path)
            logger.info(f"Loaded default model from {config.default_model_path}")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")
    
    logger.info("BloomWatch API startup complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down BloomWatch API...")
    if model_manager:
        model_manager.cleanup()
    logger.info("BloomWatch API shutdown complete!")


# Create FastAPI application
app = FastAPI(
    title="BloomWatch API",
    description="""
    BloomWatch Plant Bloom Detection API
    
    This API provides endpoints for detecting and tracking plant blooming stages
    from images using deep learning models.
    
    ## Features
    
    * **Plant Bloom Classification**: Classify images into bloom stages (bud, early_bloom, full_bloom, late_bloom, dormant)
    * **Batch Processing**: Process multiple images at once
    * **Model Management**: Support for multiple models and hot-swapping
    * **Health Monitoring**: API health and model status endpoints
    * **Interactive Documentation**: Swagger UI and ReDoc available
    
    ## Bloom Stages
    
    - **Bud**: Plant has formed buds but no flowers are visible
    - **Early Bloom**: First flowers are starting to open
    - **Full Bloom**: Plant is at peak flowering
    - **Late Bloom**: Flowers are beginning to fade
    - **Dormant**: Plant is not actively flowering
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url.path)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An internal server error occurred"
        }
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the API and loaded models.
    """
    global model_manager
    
    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "api_version": "0.1.0",
        "python_version": f"{torch.__version__}",
        "models": {}
    }
    
    if model_manager:
        status["models"] = model_manager.get_model_status()
    
    return status


@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Welcome to BloomWatch API",
        "description": "Plant bloom detection and tracking API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


# Include API routes
app.include_router(router, prefix="/api/v1", tags=["BloomWatch"])