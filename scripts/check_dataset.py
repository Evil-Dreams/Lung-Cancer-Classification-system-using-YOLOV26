"""Script to verify dataset structure and data loading."""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.preprocessing.dataset import LungNoduleDataset
    from src.utils.config import cfg, load_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current Python path: {sys.path}")
    raise

def check_directory_structure(data_dir):
    """Verify the dataset directory structure."""
    data_dir = Path(data_dir)
    required_dirs = ['train', 'val', 'test']
    required_subdirs = ['images', 'masks']
    
    print("\n=== Checking directory structure ===")
    
    # Check main directories
    for split in required_dirs:
        split_dir = data_dir / split
        print(f"\nChecking {split} directory: {split_dir}")
        
        if not split_dir.exists():
            print(f"❌ {split} directory not found at {split_dir}")
            continue
            
        # Check subdirectories
        for subdir in required_subdirs:
            subdir_path = split_dir / subdir
            if not subdir_path.exists():
                print(f"❌ {subdir} directory not found in {split}")
            else:
                # Count files
                file_count = len(list(subdir_path.glob('*.*')))
                print(f"✅ {subdir}: {file_count} files")
    
    print("\n=== Directory structure check complete ===\n")

def check_sample_loading(data_dir, mode='train', num_samples=3):
    """Check if samples can be loaded correctly."""
    print(f"\n=== Checking sample loading for {mode} set ===")
    
    # Create dataset
    try:
        dataset = LungNoduleDataset(
            data_dir=data_dir,
            mode=mode,
            input_size=cfg['data.input_size'],
            use_augmentation=False  # Don't use augmentation for checking
        )
        print(f"✅ Successfully created dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return
    
    if len(dataset) == 0:
        print("❌ No samples found in the dataset")
        return
    
    # Check first few samples
    print("\nSample information:")
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            print(f"\nSample {i+1}:")
            print(f"- Image shape: {sample['image'].shape}")
            print(f"- Image type: {sample['image'].dtype}")
            print(f"- Image range: {sample['image'].min():.2f} to {sample['image'].max():.2f}")
            
            if 'mask' in sample:
                print(f"- Mask shape: {sample['mask'].shape}")
                print(f"- Mask unique values: {np.unique(sample['mask'].numpy())}")
                
                # Check if mask contains valid values (0 and 1 for binary segmentation)
                mask_values = np.unique(sample['mask'].numpy())
                if set(mask_values) - {0, 1}:
                    print(f"⚠️  Warning: Mask contains unexpected values: {mask_values}")
            else:
                print("⚠️  No mask found in sample")
            
            # Visualize sample
            plt.figure(figsize=(12, 4))
            
            # Plot image
            plt.subplot(1, 2, 1)
            img = sample['image'][0].numpy()  # Get first channel if multi-channel
            plt.imshow(img, cmap='gray')
            plt.title('Image')
            plt.axis('off')
            
            # Plot mask if available
            if 'mask' in sample:
                plt.subplot(1, 2, 2)
                mask = sample['mask'][0].numpy()  # Get first channel if multi-channel
                plt.imshow(mask, cmap='jet')
                plt.title('Mask')
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error loading sample {i+1}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Check dataset structure and loading')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to check')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    load_config(args.config)
    
    # Check directory structure
    check_directory_structure(args.data_dir)
    
    # Check sample loading
    check_sample_loading(args.data_dir, mode=args.mode)
    
    print("\n=== Dataset check complete ===")

if __name__ == '__main__':
    main()
