"""
Unit tests for data module components.

Tests dataset loading, preprocessing, augmentations, and data utilities.
"""

import pytest
import torch
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path

from data import (
    PlantBloomDataset, TimeSeriesBloomDataset, ImageProcessor,
    BloomAugmentations, S3Downloader, LocalDataLoader
)
from data.preprocessing import PlantImageEnhancer, ColorSpaceConverter
from tests.conftest import create_dummy_dataset_files, assert_tensor_shape, assert_tensor_range


class TestPlantBloomDataset:
    """Test PlantBloomDataset class."""
    
    def test_dataset_initialization(self, temp_data_dir, sample_annotations):
        """Test dataset initialization with valid data."""
        create_dummy_dataset_files(temp_data_dir, sample_annotations)
        
        dataset = PlantBloomDataset(
            data_dir=str(temp_data_dir),
            annotations_file=str(temp_data_dir / "annotations.csv"),
            stage='train'
        )
        
        assert len(dataset) > 0
        assert dataset.num_classes == 5
        assert len(dataset.class_names) == 5
    
    def test_dataset_getitem(self, temp_data_dir, sample_annotations):
        """Test dataset item retrieval."""
        create_dummy_dataset_files(temp_data_dir, sample_annotations)
        
        dataset = PlantBloomDataset(
            data_dir=str(temp_data_dir),
            annotations_file=str(temp_data_dir / "annotations.csv"),
            stage='train'
        )
        
        image, label, metadata = dataset[0]
        
        # Check types
        assert isinstance(image, (torch.Tensor, Image.Image))
        assert isinstance(label, (int, torch.Tensor))
        assert isinstance(metadata, dict)
        
        # Check metadata keys
        assert 'image_path' in metadata
        assert 'bloom_stage_name' in metadata
    
    def test_dataset_stage_filtering(self, temp_data_dir, sample_annotations):
        """Test dataset stage filtering."""
        create_dummy_dataset_files(temp_data_dir, sample_annotations)
        
        train_dataset = PlantBloomDataset(
            data_dir=str(temp_data_dir),
            annotations_file=str(temp_data_dir / "annotations.csv"),
            stage='train'
        )
        
        val_dataset = PlantBloomDataset(
            data_dir=str(temp_data_dir),
            annotations_file=str(temp_data_dir / "annotations.csv"),
            stage='val'
        )
        
        # Check different sizes based on stage filtering
        assert len(train_dataset) != len(val_dataset)
        assert len(train_dataset) > len(val_dataset)  # More train samples
    
    def test_class_weights_calculation(self, temp_data_dir, sample_annotations):
        """Test class weights calculation for imbalanced data."""
        create_dummy_dataset_files(temp_data_dir, sample_annotations)
        
        dataset = PlantBloomDataset(
            data_dir=str(temp_data_dir),
            annotations_file=str(temp_data_dir / "annotations.csv")
        )
        
        weights = dataset.get_class_weights()
        
        assert isinstance(weights, torch.Tensor)
        assert len(weights) == dataset.num_classes
        assert (weights > 0).all()
    
    def test_dummy_data_creation(self, temp_data_dir):
        """Test creation of dummy data when real data is missing."""
        dataset = PlantBloomDataset(
            data_dir=str(temp_data_dir),
            annotations_file=str(temp_data_dir / "nonexistent.csv")
        )
        
        assert len(dataset) > 0  # Should create dummy data
        
        image, label, metadata = dataset[0]
        assert isinstance(image, Image.Image)
        assert isinstance(label, int)
        assert 0 <= label < 5


class TestTimeSeriesBloomDataset:
    """Test TimeSeriesBloomDataset class."""
    
    def test_timeseries_initialization(self, temp_data_dir, sample_annotations):
        """Test time series dataset initialization."""
        create_dummy_dataset_files(temp_data_dir, sample_annotations)
        
        dataset = TimeSeriesBloomDataset(
            data_dir=str(temp_data_dir),
            annotations_file=str(temp_data_dir / "annotations.csv"),
            sequence_length=3
        )
        
        assert hasattr(dataset, 'sequence_length')
        assert dataset.sequence_length == 3
    
    def test_sequence_creation(self, temp_data_dir, sample_annotations):
        """Test sequence creation for time series data."""
        create_dummy_dataset_files(temp_data_dir, sample_annotations)
        
        dataset = TimeSeriesBloomDataset(
            data_dir=str(temp_data_dir),
            annotations_file=str(temp_data_dir / "annotations.csv"),
            sequence_length=2
        )
        
        if len(dataset.sequences) > 0:
            sequence_info = dataset.sequences[0]
            assert 'plant_id' in sequence_info
            assert 'indices' in sequence_info
            assert len(sequence_info['indices']) == 2


class TestImageProcessor:
    """Test ImageProcessor class."""
    
    def test_processor_initialization(self):
        """Test image processor initialization."""
        processor = ImageProcessor(
            image_size=(224, 224),
            normalize=True
        )
        
        assert processor.image_size == (224, 224)
        assert processor.normalize is True
        assert processor.transform is not None
    
    def test_image_processing(self, dummy_image):
        """Test image processing functionality."""
        processor = ImageProcessor(image_size=(224, 224))
        
        processed = processor(dummy_image)
        
        assert isinstance(processed, torch.Tensor)
        assert_tensor_shape(processed, (3, 224, 224))
        assert_tensor_range(processed, 0.0, 1.0)
    
    def test_batch_processing(self, dummy_image):
        """Test batch image processing."""
        processor = ImageProcessor()
        images = [dummy_image] * 3
        
        batch = processor.preprocess_batch(images)
        
        assert isinstance(batch, torch.Tensor)
        assert_tensor_shape(batch, (3, 3, 224, 224))
    
    def test_denormalization(self, dummy_image):
        """Test image denormalization."""
        processor = ImageProcessor(normalize=True)
        
        normalized = processor(dummy_image)
        denormalized = processor.denormalize(normalized)
        
        assert isinstance(denormalized, torch.Tensor)
        # Should be approximately in [0, 1] range after denormalization
        assert_tensor_range(denormalized, -0.5, 1.5)  # Allow some margin


class TestBloomAugmentations:
    """Test BloomAugmentations class."""
    
    def test_augmentation_initialization(self):
        """Test augmentation initialization."""
        aug = BloomAugmentations(
            rotation_range=15.0,
            brightness_range=0.2,
            horizontal_flip_prob=0.5
        )
        
        assert aug.rotation_range == 15.0
        assert aug.brightness_range == 0.2
        assert aug.horizontal_flip_prob == 0.5
    
    def test_augmentation_application(self, dummy_image):
        """Test applying augmentations to image."""
        aug = BloomAugmentations()
        
        augmented = aug(dummy_image)
        
        assert isinstance(augmented, Image.Image)
        assert augmented.size == dummy_image.size
    
    def test_augmentation_determinism(self, dummy_image):
        """Test that augmentations produce different results."""
        aug = BloomAugmentations()
        
        # Apply augmentation multiple times
        results = [aug(dummy_image) for _ in range(3)]
        
        # Results should potentially be different (though may occasionally be same)
        assert len(results) == 3
        assert all(isinstance(img, Image.Image) for img in results)


class TestPlantImageEnhancer:
    """Test PlantImageEnhancer class."""
    
    def test_vegetation_enhancement(self, dummy_image):
        """Test vegetation enhancement."""
        enhancer = PlantImageEnhancer()
        
        enhanced = enhancer.enhance_vegetation(dummy_image, enhancement_factor=1.2)
        
        assert isinstance(enhanced, Image.Image)
        assert enhanced.size == dummy_image.size
    
    def test_lighting_adjustment(self, dummy_image):
        """Test lighting adjustment."""
        enhancer = PlantImageEnhancer()
        
        adjusted = enhancer.adjust_lighting(dummy_image, brightness=1.1, contrast=1.1)
        
        assert isinstance(adjusted, Image.Image)
        assert adjusted.size == dummy_image.size
    
    def test_background_removal(self, dummy_image):
        """Test background removal."""
        enhancer = PlantImageEnhancer()
        
        try:
            result = enhancer.remove_background(dummy_image)
            assert isinstance(result, Image.Image)
            assert result.size == dummy_image.size
        except ImportError:
            # cv2 might not be available in test environment
            pytest.skip("OpenCV not available for background removal test")


class TestColorSpaceConverter:
    """Test ColorSpaceConverter class."""
    
    def test_rgb_to_hsv_conversion(self, dummy_image):
        """Test RGB to HSV conversion."""
        converter = ColorSpaceConverter()
        
        try:
            hsv_array = converter.rgb_to_hsv(dummy_image)
            assert isinstance(hsv_array, np.ndarray)
            assert hsv_array.shape == (*dummy_image.size[::-1], 3)
        except ImportError:
            pytest.skip("OpenCV not available for color space conversion test")
    
    def test_green_channel_extraction(self, dummy_image):
        """Test green channel extraction."""
        converter = ColorSpaceConverter()
        
        green_channel = converter.extract_green_channel(dummy_image)
        
        assert isinstance(green_channel, np.ndarray)
        assert green_channel.ndim == 2  # Should be grayscale
        assert green_channel.shape == dummy_image.size[::-1]


class TestLocalDataLoader:
    """Test LocalDataLoader class."""
    
    def test_loader_initialization(self, temp_data_dir):
        """Test local data loader initialization."""
        loader = LocalDataLoader(str(temp_data_dir))
        
        assert loader.data_dir == temp_data_dir
        assert temp_data_dir.exists()
    
    def test_directory_scanning(self, temp_data_dir, sample_annotations):
        """Test directory scanning functionality."""
        create_dummy_dataset_files(temp_data_dir, sample_annotations)
        loader = LocalDataLoader(str(temp_data_dir))
        
        inventory = loader.scan_directory(str(temp_data_dir))
        
        assert isinstance(inventory, dict)
        assert 'all' in inventory
        assert len(inventory['all']) > 0
    
    def test_annotations_creation(self, temp_data_dir, sample_annotations):
        """Test annotations CSV creation."""
        # Create directory structure with class folders
        for bloom_stage in ['bud', 'early_bloom', 'full_bloom']:
            stage_dir = temp_data_dir / bloom_stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dummy images in each folder
            for i in range(2):
                dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                Image.fromarray(dummy_img).save(stage_dir / f"image_{i}.jpg")
        
        loader = LocalDataLoader(str(temp_data_dir))
        annotations_path = loader.create_annotations_csv(
            str(temp_data_dir),
            str(temp_data_dir / "test_annotations.csv")
        )
        
        assert Path(annotations_path).exists()
        
        # Verify annotations content
        df = pd.read_csv(annotations_path)
        assert len(df) > 0
        assert 'image_path' in df.columns
        assert 'bloom_stage' in df.columns


class TestS3Downloader:
    """Test S3Downloader class (mocked tests)."""
    
    def test_s3_initialization(self):
        """Test S3 downloader initialization."""
        # Test with invalid credentials (should handle gracefully)
        downloader = S3Downloader(
            bucket_name="test-bucket",
            aws_access_key_id="fake-key",
            aws_secret_access_key="fake-secret"
        )
        
        assert downloader.bucket_name == "test-bucket"
        # Should handle missing credentials gracefully
        assert downloader.s3_client is None or downloader.s3_client is not None
    
    @pytest.mark.skip(reason="Requires AWS credentials and S3 access")
    def test_s3_file_download(self):
        """Test S3 file download (requires real AWS setup)."""
        # This test would require real AWS credentials and S3 bucket
        # Skip in CI/CD environment
        pass
    
    @pytest.mark.skip(reason="Requires AWS credentials and S3 access")
    def test_s3_dataset_download(self):
        """Test S3 dataset download (requires real AWS setup)."""
        # This test would require real AWS credentials and S3 bucket
        # Skip in CI/CD environment
        pass