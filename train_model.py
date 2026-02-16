#!/usr/bin/env python3
"""
Complete training pipeline for Lung Cancer Segmentation Model.
This script handles data preparation, training, and model management.
"""

import os
import sys
import argparse
from pathlib import Path
import shutil
import json

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Import model and dataset
from src.segmentation.unetr import get_unetr_model
from src.preprocessing.dataset import create_data_loaders
from src.utils.config import cfg, load_config
from src.utils.visualization import LungVisualizer

# Loss functions
class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice

class DiceCELoss(nn.Module):
    """Dice + CrossEntropy loss for segmentation."""
    
    def __init__(self, weight_ce=1.0, weight_dice=1.0, smooth=1e-5):
        super(DiceCELoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.dice = DiceLoss(smooth=smooth)
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, inputs, targets):
        # For binary segmentation, we can use same target for both channels
        if targets.dim() == 4:  # If target is (B, 1, H, W)
            targets = targets.squeeze(1).long()  # Convert to (B, H, W) with class indices
            
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(torch.sigmoid(inputs[:, 1]), (targets == 1).float())
        
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss

def validate_model(model, val_loader, criterion, device, epoch):
    """Validate the model on validation set."""
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                # Handle different batch formats
                if isinstance(batch, dict):
                    if 'image' in batch and 'mask' in batch:
                        inputs = batch['image'].to(device)
                        targets = batch['mask'].to(device)
                    else:
                        continue
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                else:
                    continue
                
                # Add channel dimension if needed
                if inputs.dim() == 3:  # [B, H, W]
                    inputs = inputs.unsqueeze(1)  # [B, 1, H, W]
                
                # Convert targets to class indices
                if targets.dim() == 4 and targets.size(1) == 1:
                    targets = targets.squeeze(1).long()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
    
    avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    return avg_val_loss

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(checkpoint, best_path)
        print(f"New best model saved with loss: {loss:.4f}")
    
    # Keep only last 5 checkpoints to save space
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    for checkpoint in checkpoints[:-5]:
        os.remove(os.path.join(checkpoint_dir, checkpoint))

def train_model(args):
    """Main training function."""
    print("🚀 Starting Lung Cancer Segmentation Training")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load configuration
    if args.config:
        load_config(args.config)
    
    # Create data loaders
    print("📊 Creating data loaders...")
    try:
        data_loaders = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            input_size=(args.img_size, args.img_size),
            use_augmentation=True
        )
        
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        print("Please ensure your data directory has the following structure:")
        print("data/")
        print("├── train/")
        print("│   ├── images/")
        print("│   └── masks/")
        print("├── val/")
        print("│   ├── images/")
        print("│   └── masks/")
        print("└── test/")
        print("    ├── images/")
        print("    └── masks/")
        return
    
    # Create model
    print("🧠 Creating model...")
    model = get_unetr_model()
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training components
    criterion = DiceCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Training loop
    print("🏋️ Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch in pbar:
            try:
                # Handle different batch formats
                if isinstance(batch, dict):
                    if 'image' in batch and 'mask' in batch:
                        inputs = batch['image'].to(device)
                        targets = batch['mask'].to(device)
                    else:
                        continue
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                else:
                    continue
                
                # Add channel dimension if needed
                if inputs.dim() == 3:  # [B, H, W]
                    inputs = inputs.unsqueeze(1)  # [B, 1, H, W]
                
                # Convert targets to class indices
                if targets.dim() == 4 and targets.size(1) == 1:
                    targets = targets.squeeze(1).long()
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
                # Log to TensorBoard
                step = epoch * len(train_loader) + num_batches
                if step % 10 == 0:
                    writer.add_scalar('train/loss', loss.item(), step)
                    
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        # Calculate average training loss
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        print(f"Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        val_loss = validate_model(model, val_loader, criterion, device, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)
        writer.add_scalar('val/epoch_loss', val_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint_dir, is_best)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Close TensorBoard writer
    writer.close()
    print("\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train Lung Cancer Segmentation Model')
    parser.add_argument('--data-dir', type=str, default='data/', 
                       help='Path to data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/',
                       help='Path to save checkpoints')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--img-size', type=int, default=256,
                       help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Start training
    train_model(args)

if __name__ == '__main__':
    main()
