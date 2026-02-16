"""Script to visualize model predictions and training results."""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.config import cfg, load_config
from src.segmentation.unetr import get_unetr_model
from src.preprocessing.dataset import LungNoduleDataset
from src.utils.visualization import LungVisualizer

def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    model = get_unetr_model().to(device)
    
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    
    model.eval()
    return model

def visualize_predictions(model, dataset, num_samples=5, output_dir='outputs/predictions'):
    """Visualize model predictions on sample images."""
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    
    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        # Get sample
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            output = model(image)
            if isinstance(output, tuple):  # Handle multiple outputs
                output = output[0]
            pred = torch.softmax(output, dim=1).argmax(dim=1).squeeze().cpu().numpy()
        
        # Get ground truth if available
        gt_mask = sample.get('mask')
        if gt_mask is not None:
            gt_mask = gt_mask.squeeze().numpy()
        
        # Visualize
        image_np = image.squeeze().cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(15, 5))
        
        # Plot input image
        axes[0].imshow(image_np[0] if image_np.shape[0] == 3 else image_np, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Plot prediction
        axes[1].imshow(pred, cmap='jet', alpha=0.7)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # Plot ground truth if available
        if gt_mask is not None:
            axes[2].imshow(gt_mask, cmap='jet', alpha=0.7)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'), bbox_inches='tight', dpi=150)
        plt.close()

def plot_training_curves(log_dir, output_dir='outputs/plots'):
    """Plot training and validation loss curves from TensorBoard logs."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import os
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the latest log directory
    log_dirs = sorted(glob.glob(os.path.join(log_dir, '*')))
    if not log_dirs:
        print(f"No log directories found in {log_dir}")
        return
    
    latest_log_dir = log_dirs[-1]
    print(f"Loading logs from {latest_log_dir}")
    
    # Load TensorBoard data
    event_acc = EventAccumulator(latest_log_dir)
    event_acc.Reload()
    
    # Get scalar data
    scalars = {}
    for tag in event_acc.Tags()['scalars']:
        scalars[tag] = event_acc.Scalars(tag)
    
    # Plot training and validation loss
    if 'train/loss' in scalars and 'val/loss' in scalars:
        train_loss = [s.value for s in scalars['train/loss']]
        val_loss = [s.value for s in scalars['val/loss']]
        steps = [s.step for s in scalars['train/loss']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_loss, label='Training Loss')
        plt.plot(steps, val_loss, label='Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'), bbox_inches='tight', dpi=150)
        plt.close()
    
    # Plot learning rate if available
    if 'lr' in scalars:
        lr_data = scalars['lr']
        steps = [s.step for s in lr_data]
        lr_values = [s.value for s in lr_data]
        
        plt.figure(figsize=(10, 4))
        plt.plot(steps, lr_values)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), bbox_inches='tight', dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions and training results')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs/segmentation',
                        help='Directory containing TensorBoard logs')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['predictions', 'curves', 'all'],
                        help='Visualization mode')
    
    args = parser.parse_args()
    
    # Load configuration
    load_config(args.config)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize training curves
    if args.mode in ['curves', 'all']:
        print('Plotting training curves...')
        plot_training_curves(
            log_dir=args.log_dir,
            output_dir=os.path.join(args.output_dir, 'plots')
        )
    
    # Visualize predictions
    if args.mode in ['predictions', 'all']:
        print('Loading model...')
        model = load_model(args.checkpoint, device)
        
        print('Loading dataset...')
        dataset = LungNoduleDataset(
            data_dir=args.data_dir,
            mode='val',  # Use validation set for visualization
            input_size=cfg['data.input_size'],
            use_augmentation=False
        )
        
        print('Generating visualizations...')
        visualize_predictions(
            model=model,
            dataset=dataset,
            num_samples=args.num_samples,
            output_dir=os.path.join(args.output_dir, 'predictions')
        )
    
    print('Visualization complete!')

if __name__ == '__main__':
    main()
