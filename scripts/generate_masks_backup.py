"""Script to generate masks using Segment Anything Model (SAM)."""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class MaskGenerator:
    def __init__(self, model_type="vit_b", device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the mask generator.
        
        Args:
            model_type (str): Type of SAM model ('vit_b', 'vit_l', or 'vit_h')
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_type = model_type
        self.model = None
        self.mask_generator = None
        
        # Define model checkpoints (will be downloaded automatically)
        self.checkpoints = {
            "vit_b": "sam_vit_b_01ec64.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_h": "sam_vit_h_4b8939.pth"
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the SAM model and mask generator."""
        print(f"Loading {self.model_type} model...")
        
        # Initialize SAM
        sam = sam_model_registry[self.model_type](checkpoint=None)
        sam.to(device=self.device)
        
        # Initialize mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,  # Reduced for faster processing
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Filter out small regions
        )
    
    def process_image(self, image_path, output_path=None):
        """Process a single image and generate masks.
        
        Args:
            image_path (str): Path to input image
            output_path (str, optional): Path to save output mask
            
        Returns:
            numpy.ndarray: Generated mask or None if processing failed
        """
        try:
            # Handle path with quotes to deal with spaces
            image_path = str(Path(image_path).resolve())
            
            # Read and preprocess image
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                return None
                
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Skip if image is too small
            if min(image.shape[:2]) < 50:
                print(f"Skipping {image_path}: Image too small {image.shape}")
                return None
            
            # Generate masks
            try:
                masks = self.mask_generator.generate(image_rgb)
                
                if not masks:
                    print(f"No masks generated for {image_path}")
                    return None
                    
                # Sort by area (largest first)
                masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
                
                # Take the largest mask (usually the lung area)
                combined_mask = masks[0]['segmentation'].astype(np.uint8) * 255
                
                # Post-processing
                kernel = np.ones((5, 5), np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                
                # Save mask if output path is provided
                if output_path:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    # Use cv2.imencode to handle paths with non-ASCII characters
                    success, buffer = cv2.imencode(output_path.suffix, combined_mask)
                    if success:
                        with open(output_path, 'wb') as f:
                            buffer.tofile(f)
                    else:
                        print(f"Failed to save mask to {output_path}")
                        return None
                        
                return combined_mask
                
            except Exception as e:
                print(f"Error in mask generation for {image_path}: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

def process_directory(input_dir, output_dir, model_type="vit_b"):
    """Process all images in a directory.
    
    Args:
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save masks
        model_type (str): SAM model type
    """
    # Initialize mask generator
    generator = MaskGenerator(model_type=model_type)
    
    # Get all image files (case-insensitive)
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.PNG', '.JPG', '.JPEG', '.TIF', '.TIFF', '.BMP']
    image_files = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
        
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    if not image_files:
        print("No image files found. Supported formats:", ", ".join(ext for ext in image_extensions if not ext.isupper()))
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    processed = 0
    for img_path in tqdm(image_files, desc="Generating masks"):
        try:
            # Create output filename (same name as input but in masks directory)
            mask_filename = f"{img_path.stem}.png"  # Always save as PNG
            mask_path = output_path / mask_filename
            
            # Skip if mask already exists and has content
            if mask_path.exists() and mask_path.stat().st_size > 0:
                continue
                
            # Generate and save mask
            if generator.process_image(str(img_path), str(mask_path)) is not None:
                processed += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\nSuccessfully processed {processed} out of {len(image_files)} images")
    if processed < len(image_files):
        print(f"Skipped {len(image_files) - processed} images due to errors or existing masks")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate masks using SAM')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/val/test splits')
    parser.add_argument('--model_type', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM model type')
    
    args = parser.parse_args()
    
    # Process each split
    for split in ['train', 'val', 'test']:
        input_dir = Path(args.data_dir) / split / 'images'
        output_dir = Path(args.data_dir) / split / 'masks'
        
        if input_dir.exists() and any(input_dir.iterdir()):
            print(f"\nProcessing {split} set...")
            process_directory(input_dir, output_dir, args.model_type)
        else:
            print(f"Skipping {split} - no images found")
    
    print("\nMask generation complete!")

if __name__ == '__main__':
    main()
