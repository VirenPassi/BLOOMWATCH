"""
Image preprocessing utilities for plant bloom detection.

This module provides classes and functions for preprocessing plant images
including resizing, normalization, and format conversion.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
from typing import Tuple, Optional, Union, List
import cv2


class ImageProcessor:
    """
    Comprehensive image processor for plant bloom datasets.
    
    Handles standard preprocessing tasks like resizing, normalization,
    and format conversion with sensible defaults for plant imagery.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize the image processor.
        
        Args:
            image_size: Target size for images (height, width)
            normalize: Whether to apply ImageNet normalization
            mean: Mean values for normalization (RGB)
            std: Standard deviation values for normalization (RGB)
        """
        self.image_size = image_size
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        self.transform = self._build_transform()
    
    def _build_transform(self) -> transforms.Compose:
        """Build the preprocessing transform pipeline."""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Process an image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            torch.Tensor: Processed image tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return self.transform(image)
    
    def preprocess_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            torch.Tensor: Batch of processed image tensors
        """
        processed_images = [self(img) for img in images]
        return torch.stack(processed_images)
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor for visualization.
        
        Args:
            tensor: Normalized image tensor
            
        Returns:
            torch.Tensor: Denormalized tensor
        """
        if not self.normalize:
            return tensor
        
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        
        if tensor.device != mean.device:
            mean = mean.to(tensor.device)
            std = std.to(tensor.device)
        
        return tensor * std + mean


class PlantImageEnhancer:
    """
    Specialized image enhancement for plant photography.
    
    Provides methods to enhance plant images specifically,
    focusing on vegetation characteristics.
    """
    
    def __init__(self):
        pass
    
    def enhance_vegetation(self, image: Image.Image, enhancement_factor: float = 1.2) -> Image.Image:
        """
        Enhance vegetation in the image by boosting green channel.
        
        Args:
            image: Input PIL image
            enhancement_factor: Factor to enhance vegetation
            
        Returns:
            Image.Image: Enhanced image
        """
        # Convert to numpy for channel manipulation
        img_array = np.array(image)
        
        # Enhance green channel (vegetation)
        img_array[:, :, 1] = np.clip(
            img_array[:, :, 1] * enhancement_factor, 0, 255
        ).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def adjust_lighting(self, image: Image.Image, brightness: float = 1.1, contrast: float = 1.1) -> Image.Image:
        """
        Adjust lighting conditions for better plant visibility.
        
        Args:
            image: Input PIL image
            brightness: Brightness adjustment factor
            contrast: Contrast adjustment factor
            
        Returns:
            Image.Image: Adjusted image
        """
        # Adjust brightness
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(brightness)
        
        # Adjust contrast
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(contrast)
        
        return image
    
    def remove_background(self, image: Image.Image, threshold: int = 100) -> Image.Image:
        """
        Simple background removal for plant images.
        
        This is a basic implementation that removes low-intensity backgrounds.
        For production use, consider more sophisticated segmentation methods.
        
        Args:
            image: Input PIL image
            threshold: Threshold for background removal
            
        Returns:
            Image.Image: Image with background removed
        """
        img_array = np.array(image)
        
        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Create mask
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply mask
        img_array[mask == 0] = [255, 255, 255]  # White background
        
        return Image.fromarray(img_array)


class ColorSpaceConverter:
    """
    Convert images between different color spaces for analysis.
    
    Different color spaces can be useful for different aspects
    of plant analysis (e.g., HSV for vegetation analysis).
    """
    
    @staticmethod
    def rgb_to_hsv(image: Image.Image) -> np.ndarray:
        """Convert RGB image to HSV color space."""
        img_array = np.array(image)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    @staticmethod
    def rgb_to_lab(image: Image.Image) -> np.ndarray:
        """Convert RGB image to LAB color space."""
        img_array = np.array(image)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    @staticmethod
    def extract_green_channel(image: Image.Image) -> np.ndarray:
        """Extract and enhance green channel for vegetation analysis."""
        img_array = np.array(image)
        green_channel = img_array[:, :, 1]
        
        # Normalize to 0-255 range
        green_normalized = ((green_channel - green_channel.min()) / 
                           (green_channel.max() - green_channel.min()) * 255).astype(np.uint8)
        
        return green_normalized


def create_standard_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True
) -> transforms.Compose:
    """
    Create standard transform pipelines for training and validation.
    
    Args:
        image_size: Target image size
        is_training: Whether this is for training (includes augmentations)
        
    Returns:
        transforms.Compose: Transform pipeline
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor using ImageNet statistics.
    
    Args:
        tensor: Normalized tensor
        
    Returns:
        torch.Tensor: Denormalized tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    return tensor * std + mean