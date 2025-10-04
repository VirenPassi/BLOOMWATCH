#!/usr/bin/env python3
"""
Train NASA MODIS Model with Manual Data
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configuration
MODIS_DIR = Path("data/processed/MODIS/manual")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

class ManualMODISDataset(Dataset):
    """Dataset for manually downloaded MODIS data"""
    
    def __init__(self, modis_dir, num_samples=1000):
        self.modis_dir = Path(modis_dir)
        self.num_samples = num_samples
        
        # Find NDVI and EVI files
        self.ndvi_files = sorted(list(self.modis_dir.glob("ndvi_*.npy")))
        self.evi_files = sorted(list(self.modis_dir.glob("evi_*.npy")))
        
        print(f"Found {len(self.ndvi_files)} NDVI files")
        print(f"Found {len(self.evi_files)} EVI files")
        
        if not self.ndvi_files or not self.evi_files:
            raise RuntimeError("No MODIS files found!")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load NDVI/EVI pair
        ndvi_idx = idx % len(self.ndvi_files)
        evi_idx = idx % len(self.evi_files)
        
        ndvi = np.load(self.ndvi_files[ndvi_idx])
        evi = np.load(self.evi_files[evi_idx])
        
        # Convert to tensors
        ndvi_tensor = torch.from_numpy(ndvi).float()
        evi_tensor = torch.from_numpy(evi).float()
        
        # Stack into 2-channel tensor
        modis_data = torch.stack([ndvi_tensor, evi_tensor], dim=0)
        
        # Create realistic labels based on vegetation patterns
        avg_ndvi = ndvi.mean()
        avg_evi = evi.mean()
        combined_vi = (avg_ndvi + avg_evi) / 2
        
        # Map vegetation index to bloom stages
        if combined_vi < 0.1:
            label = 4  # dormant
        elif combined_vi < 0.3:
            label = 0  # bud
        elif combined_vi < 0.5:
            label = 1  # early_bloom
        elif combined_vi < 0.7:
            label = 2  # full_bloom
        else:
            label = 3  # late_bloom
        
        return modis_data, label

class ManualMODISModel(nn.Module):
    """Advanced CNN for manual MODIS data"""
    
    def __init__(self, num_classes=5):
        super().__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(2, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def train_manual_modis():
    """Train model with manual MODIS data"""
    
    print("Training with Manual MODIS Data")
    print("=" * 50)
    
    # Create dataset
    dataset = ManualMODISDataset(MODIS_DIR, num_samples=1000)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = ManualMODISModel(num_classes=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nStarting training on {device}...")
    
    # Training loop
    for epoch in range(15):
        model.train()
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/15: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    test_acc = 100. * test_correct / test_total
    
    # Save model
    model_path = OUTPUTS_DIR / "manual_modis_model.pt"
    torch.save(model.state_dict(), model_path)
    
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print(f"Model saved to: {model_path}")
    print(f"\nTraining completed successfully!")

if __name__ == "__main__":
    train_manual_modis()
