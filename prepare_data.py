#!/usr/bin/env python3
"""
Data preparation script for Lung Cancer Segmentation.
This script helps organize and prepare your training data.
"""

import os
import sys
import shutil
from pathlib import Path
import random
import json
from PIL import Image
import numpy as np

def create_sample_data(output_dir, num_samples=50):
    """Create sample training data for testing purposes."""
    print("🎨 Creating sample training data...")
    
    # Create directories
    train_dir = Path(output_dir) / 'train'
    val_dir = Path(output_dir) / 'val'
    test_dir = Path(output_dir) / 'test'
    
    for split_dir in [train_dir, val_dir, test_dir]:
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Generate sample images and masks
    samples_per_split = {
        'train': int(num_samples * 0.7),
        'val': int(num_samples * 0.15),
        'test': int(num_samples * 0.15)
    }
    
    for split, count in samples_per_split.items():
        split_dir = locals()[f'{split}_dir']
        print(f"Creating {count} samples for {split}...")
        
        for i in range(count):
            # Create sample CT-like image (grayscale)
            img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            
            # Add some lung-like structure
            center_x, center_y = 128, 128
            y, x = np.ogrid[:256, :256]
            lung_mask = ((x - center_x)**2 + (y - center_y)**2) < 80**2
            img[lung_mask] = np.random.randint(100, 200, lung_mask.sum())
            
            # Create corresponding mask (simulated nodules)
            mask = np.zeros((256, 256), dtype=np.uint8)
            
            # Add random nodules
            num_nodules = random.randint(0, 3)
            for _ in range(num_nodules):
                nodule_x = random.randint(50, 200)
                nodule_y = random.randint(50, 200)
                nodule_size = random.randint(5, 15)
                
                y, x = np.ogrid[:256, :256]
                nodule_mask = ((x - nodule_x)**2 + (y - nodule_y)**2) < nodule_size**2
                mask[nodule_mask] = 1
            
            # Save image and mask
            img_path = split_dir / 'images' / f'sample_{i:04d}.png'
            mask_path = split_dir / 'masks' / f'sample_{i:04d}.png'
            
            Image.fromarray(img).save(img_path)
            Image.fromarray(mask * 255).save(mask_path)
    
    print(f"✅ Sample data created in {output_dir}")
    print(f"Train: {samples_per_split['train']} samples")
    print(f"Val: {samples_per_split['val']} samples") 
    print(f"Test: {samples_per_split['test']} samples")

def check_data_structure(data_dir):
    """Check if data directory has the correct structure."""
    print("🔍 Checking data structure...")
    
    required_dirs = [
        'train/images', 'train/masks',
        'val/images', 'val/masks',
        'test/images', 'test/masks'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = Path(data_dir) / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
        else:
            # Count files
            files = list(full_path.glob('*.png')) + list(full_path.glob('*.jpg')) + list(full_path.glob('*.jpeg'))
            print(f"✓ {dir_path}: {len(files)} files")
    
    if missing_dirs:
        print("❌ Missing directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    
    # Check image-mask pairs
    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = Path(data_dir) / split / 'images'
        mask_dir = Path(data_dir) / split / 'masks'
        
        if img_dir.exists() and mask_dir.exists():
            img_files = {f.stem for f in img_dir.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']}
            mask_files = {f.stem for f in mask_dir.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']}
            
            missing_masks = img_files - mask_files
            missing_images = mask_files - img_files
            
            if missing_masks:
                print(f"❌ {split}: Missing masks for {len(missing_masks)} images")
            if missing_images:
                print(f"❌ {split}: Missing images for {len(missing_images)} masks")
            
            if not missing_masks and not missing_images:
                print(f"✓ {split}: All image-mask pairs match")
    
    return len(missing_dirs) == 0

def organize_data(source_dir, target_dir):
    """Organize scattered data into proper structure."""
    print("📁 Organizing data...")
    
    target_path = Path(target_dir)
    source_path = Path(source_dir)
    
    # Create target directories
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'masks']:
            (target_path / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    all_files = []
    
    for ext in image_extensions:
        all_files.extend(source_path.glob(f'*{ext}'))
        all_files.extend(source_path.glob(f'*{ext.upper()}'))
    
    # Separate images and masks
    images = []
    masks = []
    
    for file in all_files:
        filename = file.name.lower()
        if any(keyword in filename for keyword in ['mask', 'label', 'seg', 'annotation']):
            masks.append(file)
        else:
            images.append(file)
    
    print(f"Found {len(images)} images and {len(masks)} masks")
    
    # Match images with masks
    matched_pairs = []
    unmatched_images = []
    unmatched_masks = []
    
    for img in images:
        img_stem = img.stem.lower()
        matching_mask = None
        
        for mask in masks:
            mask_stem = mask.stem.lower()
            # Try different naming conventions
            if (mask_stem == img_stem or 
                mask_stem == f"{img_stem}_mask" or 
                mask_stem == f"{img_stem}_label" or
                img_stem == f"{mask_stem}_mask" or
                img_stem == f"{mask_stem}_label"):
                matching_mask = mask
                break
        
        if matching_mask:
            matched_pairs.append((img, matching_mask))
        else:
            unmatched_images.append(img)
    
    # Find unmatched masks
    used_masks = {mask for _, mask in matched_pairs}
    unmatched_masks = [mask for mask in masks if mask not in used_masks]
    
    print(f"Matched {len(matched_pairs)} image-mask pairs")
    if unmatched_images:
        print(f"Unmatched images: {len(unmatched_images)}")
    if unmatched_masks:
        print(f"Unmatched masks: {len(unmatched_masks)}")
    
    # Split data (80% train, 10% val, 10% test)
    random.shuffle(matched_pairs)
    
    train_split = int(len(matched_pairs) * 0.8)
    val_split = int(len(matched_pairs) * 0.9)
    
    train_pairs = matched_pairs[:train_split]
    val_pairs = matched_pairs[train_split:val_split]
    test_pairs = matched_pairs[val_split:]
    
    # Copy files to appropriate directories
    def copy_pairs(pairs, split_name):
        for i, (img, mask) in enumerate(pairs):
            # Generate new filename
            new_img_name = f"{split_name}_{i:04d}{img.suffix}"
            new_mask_name = f"{split_name}_{i:04d}{mask.suffix}"
            
            # Copy files
            shutil.copy2(img, target_path / split_name / 'images' / new_img_name)
            shutil.copy2(mask, target_path / split_name / 'masks' / new_mask_name)
    
    print("Copying files...")
    copy_pairs(train_pairs, 'train')
    copy_pairs(val_pairs, 'val')
    copy_pairs(test_pairs, 'test')
    
    print(f"✅ Data organized and saved to {target_dir}")
    print(f"Train: {len(train_pairs)} pairs")
    print(f"Val: {len(val_pairs)} pairs")
    print(f"Test: {len(test_pairs)} pairs")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare training data for lung cancer segmentation')
    parser.add_argument('--action', type=str, required=True,
                       choices=['check', 'organize', 'create-sample'],
                       help='Action to perform')
    parser.add_argument('--source-dir', type=str, default='data/',
                       help='Source data directory')
    parser.add_argument('--target-dir', type=str, default='data/',
                       help='Target data directory')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to create (for create-sample)')
    
    args = parser.parse_args()
    
    if args.action == 'check':
        check_data_structure(args.source_dir)
    elif args.action == 'organize':
        organize_data(args.source_dir, args.target_dir)
    elif args.action == 'create-sample':
        create_sample_data(args.target_dir, args.num_samples)

if __name__ == '__main__':
    main()
