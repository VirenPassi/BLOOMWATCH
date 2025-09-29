"""
Data augmentation strategies for plant bloom detection.

This module provides specialized augmentations for plant imagery
that preserve botanical characteristics while increasing dataset diversity.
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional, Tuple, Union


class BloomAugmentations:
    """
    Specialized augmentations for plant bloom detection.
    
    Provides botanically-aware augmentations that preserve important
    plant characteristics while adding useful variations.
    """
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        saturation_range: float = 0.3,
        hue_range: float = 0.1,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.1,
        gaussian_blur_prob: float = 0.3,
        noise_prob: float = 0.2
    ):
        """
        Initialize bloom-specific augmentations.
        
        Args:
            rotation_range: Maximum rotation in degrees
            brightness_range: Brightness variation range
            contrast_range: Contrast variation range  
            saturation_range: Saturation variation range
            hue_range: Hue variation range
            horizontal_flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip
            gaussian_blur_prob: Probability of gaussian blur
            noise_prob: Probability of adding noise
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.gaussian_blur_prob = gaussian_blur_prob
        self.noise_prob = noise_prob
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply random augmentations to an image.
        
        Args:
            image: Input PIL image
            
        Returns:
            Image.Image: Augmented image
        """
        # Geometric augmentations
        image = self._apply_rotation(image)
        image = self._apply_flips(image)
        
        # Color augmentations
        image = self._apply_color_jitter(image)
        
        # Quality augmentations
        image = self._apply_blur(image)
        image = self._apply_noise(image)
        
        return image
    
    def _apply_rotation(self, image: Image.Image) -> Image.Image:
        """Apply random rotation within specified range."""
        if random.random() < 0.7:  # 70% chance of rotation
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = F.rotate(image, angle, fill=255)  # White fill for background
        return image
    
    def _apply_flips(self, image: Image.Image) -> Image.Image:
        """Apply random horizontal and vertical flips."""
        # Horizontal flip (common for plants)
        if random.random() < self.horizontal_flip_prob:
            image = F.hflip(image)
        
        # Vertical flip (less common, plants usually have clear orientation)
        if random.random() < self.vertical_flip_prob:
            image = F.vflip(image)
        
        return image
    
    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        """Apply color jittering to simulate lighting variations."""
        color_jitter = transforms.ColorJitter(
            brightness=self.brightness_range,
            contrast=self.contrast_range,
            saturation=self.saturation_range,
            hue=self.hue_range
        )
        return color_jitter(image)
    
    def _apply_blur(self, image: Image.Image) -> Image.Image:
        """Apply gaussian blur to simulate camera focus variations."""
        if random.random() < self.gaussian_blur_prob:
            radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image
    
    def _apply_noise(self, image: Image.Image) -> Image.Image:
        """Add random noise to simulate sensor noise."""
        if random.random() < self.noise_prob:
            img_array = np.array(image)
            noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
            noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy_img)
        return image


class SeasonalAugmentations:
    """
    Augmentations that simulate seasonal variations in plant imagery.
    
    These augmentations help the model generalize across different
    seasons and environmental conditions.
    """
    
    def __init__(self):
        pass
    
    def simulate_spring(self, image: Image.Image) -> Image.Image:
        """Simulate spring conditions - enhanced greens, higher brightness."""
        # Enhance green channel
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.2)
        
        # Increase brightness slightly
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(1.1)
        
        return image
    
    def simulate_summer(self, image: Image.Image) -> Image.Image:
        """Simulate summer conditions - high saturation, bright lighting."""
        # Increase saturation
        color_enhancer = ImageEnhance.Color(image)
        image = color_enhancer.enhance(1.3)
        
        # Increase brightness
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(1.15)
        
        return image
    
    def simulate_autumn(self, image: Image.Image) -> Image.Image:
        """Simulate autumn conditions - warmer tones, reduced greens."""
        # Shift towards warmer tones
        img_array = np.array(image)
        
        # Reduce green channel slightly
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 0.9, 0, 255)
        
        # Enhance red channel slightly
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.1, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def simulate_winter(self, image: Image.Image) -> Image.Image:
        """Simulate winter conditions - reduced saturation, cooler tones."""
        # Reduce saturation
        color_enhancer = ImageEnhance.Color(image)
        image = color_enhancer.enhance(0.8)
        
        # Reduce brightness slightly
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(0.9)
        
        return image


class LightingAugmentations:
    """
    Augmentations that simulate different lighting conditions.
    
    Important for robust plant detection across various
    environmental conditions and times of day.
    """
    
    def __init__(self):
        pass
    
    def simulate_overcast(self, image: Image.Image) -> Image.Image:
        """Simulate overcast lighting conditions."""
        # Reduce contrast
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(0.8)
        
        # Slightly reduce brightness
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(0.9)
        
        return image
    
    def simulate_direct_sunlight(self, image: Image.Image) -> Image.Image:
        """Simulate direct sunlight with harsh shadows."""
        # Increase contrast
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(1.3)
        
        # Increase brightness
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(1.2)
        
        return image
    
    def simulate_golden_hour(self, image: Image.Image) -> Image.Image:
        """Simulate golden hour lighting."""
        img_array = np.array(image)
        
        # Add warm tones
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.15, 0, 255)  # Red
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.05, 0, 255)  # Green
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.95, 0, 255)  # Blue
        
        return Image.fromarray(img_array.astype(np.uint8))


def create_training_augmentations(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Create comprehensive training augmentations for plant bloom detection.
    
    Args:
        image_size: Target image size
        
    Returns:
        transforms.Compose: Complete augmentation pipeline
    """
    bloom_augs = BloomAugmentations()
    
    return transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
        transforms.RandomCrop(image_size),
        transforms.Lambda(bloom_augs),  # Apply custom bloom augmentations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_validation_augmentations(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Create minimal augmentations for validation.
    
    Args:
        image_size: Target image size
        
    Returns:
        transforms.Compose: Validation pipeline
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class MixUpAugmentation:
    """
    MixUp augmentation for plant bloom detection.
    
    Combines pairs of images and their labels to create
    synthetic training examples that improve generalization.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp augmentation.
        
        Args:
            alpha: Beta distribution parameter for mixing coefficient
        """
        self.alpha = alpha
    
    def __call__(self, batch_images: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to a batch of images and labels.
        
        Args:
            batch_images: Batch of images [B, C, H, W]
            batch_labels: Batch of labels [B]
            
        Returns:
            Tuple of:
                - Mixed images
                - Original labels
                - Shuffled labels  
                - Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index, :]
        labels_a, labels_b = batch_labels, batch_labels[index]
        
        return mixed_images, labels_a, labels_b, lam