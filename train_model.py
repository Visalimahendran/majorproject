#!/usr/bin/env python3
"""
Simple training script for demonstration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class SyntheticHandwritingDataset(Dataset):
    """Synthetic dataset for demonstration"""
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic image-like data
        # In a real system, you would load actual handwriting images
        image = torch.randn(3, 224, 224)  # Random noise
        label = torch.randint(0, 3, (1,)).item()  # Random label
        
        return image, label

class SimpleCNN(nn.Module):
    """Simple CNN for handwriting classification"""
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model():
    """Train a simple model for demonstration"""
    print("Training demonstration model...")
    
    # Create directories
    os.makedirs("saved_models", exist_ok=True)
    
    # Create dataset and dataloader
    dataset = SyntheticHandwritingDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # Save model
    save_path = "saved_models/best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': num_epochs,
        'accuracy': epoch_acc
    }, save_path)
    
    print(f"\n✅ Model saved to {save_path}")
    print("Note: This is a demonstration model trained on synthetic data.")
    print("For production use, train on real handwriting datasets.")

if __name__ == "__main__":
    train_model()