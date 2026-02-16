"""Training script for the UNETR segmentation model."""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm

# Import model and dataset
from src.segmentation.unetr import get_unetr_model
from src.preprocessing.dataset import LungNoduleDataset, create_data_loaders
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
        # For binary segmentation, we can use the same target for both channels
        if targets.dim() == 4:  # If target is (B, 1, H, W)
            targets = targets.squeeze(1).long()  # Convert to (B, H, W) with class indices
            
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(torch.sigmoid(inputs[:, 1]), (targets == 1).float())
        
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    print("\n=== Starting training epoch ===")
    print(f"Dataloader length: {len(dataloader)}")
    
    # Convert dataloader to list to debug
    dataloader_list = list(dataloader)
    print(f"Dataloader converted to list with length: {len(dataloader_list)}")
    
    pbar = tqdm(enumerate(dataloader_list), total=len(dataloader_list), desc=f"Epoch {epoch + 1}")
    for i, batch in pbar:
        try:
            print(f"\n--- Processing batch {i} ---")
            print(f"Batch type: {type(batch)}")
            
            # Debug: Print the content of the batch
            if isinstance(batch, (list, tuple)):
                print(f"Batch is a sequence with {len(batch)} items")
                for j, item in enumerate(batch):
                    print(f"  Item {j} type: {type(item)}")
                    if hasattr(item, 'shape'):
                        print(f"  Item {j} shape: {item.shape}")
                    elif isinstance(item, (list, tuple)):
                        print(f"  Item {j} is a sequence with {len(item)} elements")
                        for k, subitem in enumerate(item):
                            print(f"    Subitem {k} type: {type(subitem)}")
                            if hasattr(subitem, 'shape'):
                                print(f"    Subitem {k} shape: {subitem.shape}")
            elif isinstance(batch, dict):
                print("Batch is a dictionary with keys:", list(batch.keys()))
                for key, value in batch.items():
                    print(f"  {key} type: {type(value)}")
                    if hasattr(value, 'shape'):
                        print(f"  {key} shape: {value.shape}")
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        print(f"  {key} is a sequence with {len(value)} elements")
                        if hasattr(value[0], 'shape'):
                            print(f"    First element shape: {value[0].shape}")
            else:
                print(f"Batch is of type: {type(batch)}")
                if isinstance(batch, str):
                    print(f"Batch content: {batch}")
            
            # Skip if batch is a string or empty
            if isinstance(batch, str):
                print(f"Skipping batch {i} - batch is a string: {batch}")
                continue
                
            if not batch:
                print(f"Skipping batch {i} - batch is empty")
                continue
            
            # Try to extract inputs and targets from batch
            try:
                if isinstance(batch, dict):
                    # Case 1: Batch is a dictionary with 'image' and 'mask' keys
                    if 'image' in batch and 'mask' in batch:
                        if isinstance(batch['image'], (list, tuple)) and len(batch['image']) > 0:
                            # If values are lists, take the first element
                            inputs = batch['image'][0].to(device) if isinstance(batch['image'], (list, tuple)) else batch['image'].to(device)
                            targets = batch['mask'][0].to(device) if isinstance(batch['mask'], (list, tuple)) else batch['mask'].to(device)
                        else:
                            inputs = batch['image'].to(device)
                            targets = batch['mask'].to(device)
                    else:
                        print(f"Skipping batch {i} - missing 'image' or 'mask' keys in batch")
                        print("Available keys:", list(batch.keys()))
                        continue
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # Case 2: Batch is a tuple of (inputs, targets)
                    inputs, targets = batch
                    inputs = inputs.to(device) if hasattr(inputs, 'to') else torch.tensor(inputs, device=device)
                    targets = targets.to(device) if hasattr(targets, 'to') else torch.tensor(targets, device)
                else:
                    print(f"Skipping batch {i} - unexpected batch format")
                    continue
                    
                print(f"Inputs shape: {inputs.shape if hasattr(inputs, 'shape') else 'N/A'}")
                print(f"Targets shape: {targets.shape if hasattr(targets, 'shape') else 'N/A'}")
                
                # Ensure inputs and targets are tensors
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
                if not isinstance(targets, torch.Tensor):
                    targets = torch.tensor(targets, device=device, dtype=torch.long)
                
                # Add channel dimension if needed (for grayscale images)
                if inputs.dim() == 3:  # [B, H, W]
                    inputs = inputs.unsqueeze(1)  # [B, 1, H, W]
                
                # Convert targets to class indices (B, 1, H, W) -> (B, H, W) if needed
                if targets.dim() == 4 and targets.size(1) == 1:
                    targets = targets.squeeze(1).long()
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
                # Log to TensorBoard
                if writer is not None and i % 10 == 0:
                    step = epoch * len(dataloader) + i
                    writer.add_scalar('train/loss', loss.item(), step)
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return epoch_loss

def validate(model, dataloader, criterion, device, epoch, writer=None):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    # Convert dataloader to list to debug
    dataloader_list = list(dataloader)
    print(f"\n=== Starting validation ===")
    print(f"Validation dataloader length: {len(dataloader_list)}")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader_list, desc="Validating")):
            try:
                print(f"\n--- Processing validation batch {i} ---")
                print(f"Batch type: {type(batch)}")
                
                # Skip if batch is a string or empty
                if isinstance(batch, str):
                    print(f"Skipping validation batch {i} - batch is a string: {batch}")
                    continue
                    
                if not batch:
                    print(f"Skipping validation batch {i} - batch is empty")
                    continue
                
                # Try to extract inputs and targets from batch
                if isinstance(batch, dict):
                    # Case 1: Batch is a dictionary with 'image' and 'mask' keys
                    if 'image' in batch and 'mask' in batch:
                        if isinstance(batch['image'], (list, tuple)) and len(batch['image']) > 0:
                            # If values are lists, take the first element
                            inputs = batch['image'][0].to(device) if isinstance(batch['image'], (list, tuple)) else batch['image'].to(device)
                            targets = batch['mask'][0].to(device) if isinstance(batch['mask'], (list, tuple)) else batch['mask'].to(device)
                        else:
                            inputs = batch['image'].to(device)
                            targets = batch['mask'].to(device)
                    else:
                        print(f"Skipping validation batch {i} - missing 'image' or 'mask' keys in batch")
                        print("Available keys:", list(batch.keys()))
                        continue
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # Case 2: Batch is a tuple of (inputs, targets)
                    inputs, targets = batch
                    inputs = inputs.to(device) if hasattr(inputs, 'to') else torch.tensor(inputs, device=device)
                    targets = targets.to(device) if hasattr(targets, 'to') else torch.tensor(targets, device)
                else:
                    print(f"Skipping validation batch {i} - unexpected batch format")
                    continue
                
                print(f"Validation inputs shape: {inputs.shape if hasattr(inputs, 'shape') else 'N/A'}")
                print(f"Validation targets shape: {targets.shape if hasattr(targets, 'shape') else 'N/A'}")
                
                # Ensure inputs and targets are tensors
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
                if not isinstance(targets, torch.Tensor):
                    targets = torch.tensor(targets, device=device, dtype=torch.long)
                
                # Add channel dimension if needed (for grayscale images)
                if inputs.dim() == 3:  # [B, H, W]
                    inputs = inputs.unsqueeze(1)  # [B, 1, H, W]
                
                # Convert targets to class indices (B, 1, H, W) -> (B, H, W) if needed
                if targets.dim() == 4 and targets.size(1) == 1:
                    targets = targets.squeeze(1).long()
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                num_batches += 1
                print(f"Validation batch {i} loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in validation batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', epoch_loss, epoch)
    
    return epoch_loss

def save_checkpoint(model, optimizer, epoch, loss, is_best, checkpoint_dir):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_path)
    
    # Keep only the 3 most recent checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if len(checkpoints) > 3:
        for f in checkpoints[:-3]:  # Keep the 3 most recent checkpoints
            os.remove(os.path.join(checkpoint_dir, f))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train UNETR for lung nodule segmentation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/segmentation',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/segmentation',
                        help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    args = parser.parse_args()
    
    # Load configuration
    load_config(args.config)
    
    # Update config with command line arguments
    cfg['training.batch_size'] = args.batch_size
    cfg['training.learning_rate'] = args.lr
    cfg['training.weight_decay'] = args.weight_decay
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    print("\n=== Creating data loaders ===")
    loaders = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=cfg['training.batch_size'],
        num_workers=0,  # Set to 0 for easier debugging
        input_size=cfg['data.input_size'],
        use_augmentation=True,
        volume_depth=cfg['data.volume_depth']
    )
    
    # Get the data loaders
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    test_loader = loaders.get('test')
    
    if train_loader is None or val_loader is None:
        raise ValueError("Failed to create data loaders")
    
    # Create model
    model = get_unetr_model().to(device)
    
    # Loss function
    criterion = DiceCELoss(weight_ce=1.0, weight_dice=1.0)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['training.learning_rate'],
        weight_decay=cfg['training.weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('loss', best_loss)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch}, loss: {checkpoint['loss']:.4f})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up TensorBoard
    writer = SummaryWriter(
        log_dir=os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    )

    print("\n=== Starting training ===")
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{args.num_epochs} ===")
        
        # Training phase
        model.train()
        train_losses = []
        
        # Debug: Print batch info before training
        print("\n=== Before training loop ===")
        print(f"Train loader length: {len(train_loader)}")
        
        # Process training batches
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            try:
                # Debug: Print batch info
                print(f"\n--- Processing batch {batch_idx} ---")
                print(f"Batch type: {type(batch)}")
                if isinstance(batch, dict):
                    print("Batch keys:", batch.keys())
                    for key, value in batch.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: {value.shape} ({value.dtype})")
                        else:
                            print(f"  {key}: {type(value).__name__}")
                
                # Get inputs and targets
                if isinstance(batch, dict):
                    inputs = batch['image'].to(device)  # [B, 1, H, W]
                    targets = batch['mask'].to(device)  # [B, 1, H, W]
                else:
                    print(f"Unexpected batch type: {type(batch)}")
                    continue
                
                # Ensure inputs and targets are 4D [B, C, H, W]
                if inputs.dim() == 3:  # [B, H, W]
                    inputs = inputs.unsqueeze(1)  # [B, 1, H, W]
                if targets.dim() == 3:  # [B, H, W]
                    targets = targets.unsqueeze(1)  # [B, 1, H, W]
                
                print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_losses.append(loss.item())
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate epoch loss
        train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        print(f"Epoch {epoch+1} - Average Train Loss: {train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            print("\n=== Starting validation ===")
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
                try:
                    # Debug: Print batch info
                    print(f"\n--- Processing validation batch {batch_idx} ---")
                    print(f"Batch type: {type(batch)}")
                    if isinstance(batch, dict):
                        print("Batch keys:", batch.keys())
                        for key, value in batch.items():
                            if hasattr(value, 'shape'):
                                print(f"  {key}: {value.shape} ({value.dtype})")
                            else:
                                print(f"  {key}: {type(value).__name__}")
                    
                    # Get inputs and targets
                    if isinstance(batch, dict):
                        inputs = batch['image'].to(device)  # [B, 1, H, W]
                        targets = batch['mask'].to(device)  # [B, 1, H, W]
                    else:
                        print(f"Unexpected batch type in validation: {type(batch)}")
                        continue
                    
                    # Ensure inputs and targets are 4D [B, C, H, W]
                    if inputs.dim() == 3:
                        inputs = inputs.unsqueeze(1)
                    if targets.dim() == 3:
                        targets = targets.unsqueeze(1)
                    
                    print(f"Validation input shape: {inputs.shape}, Target shape: {targets.shape}")
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Calculate loss
                    loss = criterion(outputs, targets)
                    val_losses.append(loss.item())
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Calculate validation loss
        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")
        
        # Log to TensorBoard if writer is available
        if writer is not None:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_loss,
            is_best=is_best,
            checkpoint_dir=args.checkpoint_dir
        )
    
    print("Training finished!")

if __name__ == '__main__':
    main()
