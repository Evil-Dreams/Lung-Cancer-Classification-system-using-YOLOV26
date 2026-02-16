"""
Script to evaluate the trained UNETR model on the test set.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.dataset import create_data_loaders
from src.segmentation.unetr import get_unetr_model
from src.utils.metrics import calculate_metrics

def evaluate_model(model, test_loader, device='cuda', save_dir='results'):
    """
    Evaluate the model on the test set and calculate metrics.
    
    Args:
        model: Trained model
        test_loader: DataLoader for the test set
        device: Device to run evaluation on
        save_dir: Directory to save results
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize metrics
    metrics = {
        'dice': [],
        'iou': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'specificity': []
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            try:
                # The batch should be a dictionary with 'image' and 'mask' keys
                if not isinstance(batch, dict) or 'image' not in batch or 'mask' not in batch:
                    print(f"Unexpected batch format: {type(batch)}")
                    if isinstance(batch, (list, tuple)) and len(batch) > 0:
                        print(f"Batch content type: {type(batch[0])}")
                    continue
                
                # Get images and masks from the batch
                images = batch['image']
                masks = batch['mask']
                
                # Ensure we have tensors
                if not isinstance(images, torch.Tensor):
                    print(f"Images is not a tensor: {type(images)}")
                    continue
                if not isinstance(masks, torch.Tensor):
                    print(f"Masks is not a tensor: {type(masks)}")
                    continue
                
                # Move to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Ensure masks are float32 for loss calculation
                masks = masks.float()
                
                # Add channel dimension if needed (for grayscale images)
                if len(images.shape) == 3:  # N, H, W
                    images = images.unsqueeze(1)  # N, 1, H, W
                
                if len(masks.shape) == 3:  # N, H, W
                    masks = masks.unsqueeze(1)  # N, 1, H, W
                
                # Ensure masks are binary (0 or 1)
                masks = (masks > 0.5).float()
                
                # Forward pass
                outputs = model(images)
                
                # Handle model output (2 channels for background + foreground)
                if outputs.shape[1] == 2:  # If model outputs 2 channels
                    preds = torch.softmax(outputs, dim=1)[:, 1:2]  # Take only foreground channel
                else:  # If single channel output
                    preds = torch.sigmoid(outputs)
                
                # Ensure masks have the right shape (B, 1, H, W)
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)  # Add channel dimension if missing
                    
                # Ensure predictions and masks have the same shape
                if preds.shape[2:] != masks.shape[2:]:
                    print(f"Warning: Shape mismatch - preds: {preds.shape}, masks: {masks.shape}")
                    # Resize predictions to match mask dimensions
                    preds = torch.nn.functional.interpolate(
                        preds, size=masks.shape[2:], mode='bilinear', align_corners=False)
                
                # Calculate metrics
                batch_metrics = calculate_metrics(preds, masks)
                
                # Update metrics
                for key in metrics:
                    if key in batch_metrics:
                        metrics[key].append(batch_metrics[key])
                
                # Save some sample predictions
                if batch_idx < 5:  # Save first 5 batches
                    save_predictions(images, masks, preds, batch_idx, save_dir)
                    
            except Exception as e:
                import traceback
                print(f"Error processing batch {batch_idx}: {str(e)}")
                print(traceback.format_exc())
                continue
    
    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
    
    # Print metrics
    print("\nTest Set Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    return avg_metrics

def save_predictions(images, true_masks, pred_masks, batch_idx, save_dir):
    """Save sample predictions for visualization."""
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
    
    # Ensure we're working with numpy arrays
    if torch.is_tensor(images):
        images = images.cpu().numpy()
    if torch.is_tensor(true_masks):
        true_masks = true_masks.cpu().numpy()
    if torch.is_tensor(pred_masks):
        pred_masks = pred_masks.cpu().numpy()
    
    # Handle different input shapes
    if len(images.shape) == 4:  # NCHW format
        if images.shape[1] == 1:  # Single channel
            images = images[:, 0]  # Remove channel dim for display
        elif images.shape[1] == 3:  # RGB
            images = np.transpose(images, (0, 2, 3, 1))  # Convert to NHWC for display
    
    if len(true_masks.shape) == 4:  # NCHW format
        true_masks = true_masks[:, 0]  # Take first channel for display
    
    if len(pred_masks.shape) == 4:  # NCHW format
        pred_masks = pred_masks[:, 0]  # Take first channel for display
    
    # Save first few samples from the batch
    num_samples = min(4, images.shape[0])
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(images[i].shape) == 2:  # Grayscale
            axes[0].imshow(images[i], cmap='gray')
        else:  # RGB
            axes[0].imshow(images[i])
        axes[0].set_title('Input Image')
        
        # True mask
        axes[1].imshow(true_masks[i], cmap='gray')
        axes[1].set_title('Ground Truth')
        
        # Predicted mask
        axes[2].imshow(pred_masks[i], cmap='gray')
        axes[2].set_title('Prediction')
        
        # Remove axis ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'samples', f'batch{batch_idx}_sample{i}.png'))
        plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the best model
    model = get_unetr_model().to(device)
    checkpoint_path = 'checkpoints/model_best.pth'
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded model from checkpoint (epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f})")
    model.eval()
    
    # Create test dataset directly (bypassing create_data_loaders)
    print("Creating test dataset...")
    from src.preprocessing.dataset import LungNoduleDataset
    from torch.utils.data import DataLoader
    
    # Create test dataset
    test_dataset = LungNoduleDataset(
        data_dir='data',
        mode='test',
        input_size=(256, 256),
        use_augmentation=False
    )
    
    # Create a simple data loader with batch_size=1 and num_workers=0
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing to avoid serialization issues
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Evaluate model
    print("\nStarting evaluation...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Save metrics
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', 'metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print("\nEvaluation complete! Results saved to 'results/' directory.")

if __name__ == "__main__":
    main()
