import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2

from .dicom_loader import DICOMLoader, create_lung_mask
from .augmentations import LungAugmentor

class LungNoduleDataset(Dataset):
    """
    PyTorch Dataset for loading and augmenting lung CT scans and corresponding masks.
    Handles both 2D slices and 3D volumes.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 mode: str = 'train',
                 input_size: Tuple[int, int] = (512, 512),
                 use_augmentation: bool = True,
                 volume_depth: int = 32):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Root directory containing the dataset
            mode (str): One of 'train', 'val', or 'test'
            input_size (tuple): Target size for resizing (height, width)
            use_augmentation (bool): Whether to apply data augmentation
            volume_depth (int): Number of slices to use for 3D volumes
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.input_size = input_size
        self.use_augmentation = use_augmentation
        self.volume_depth = volume_depth
        self.dicom_loader = DICOMLoader()
        
        # Set up augmentation
        self.augmentor = LungAugmentor(input_size=input_size, mode=mode if use_augmentation else 'val')
        
        # Load data samples
        self.samples = self._load_samples()
        
        # Print dataset statistics
        print(f"Initialized {mode} dataset with {len(self.samples)} samples")
        if self.samples:
            print(f"Sample keys: {list(self.samples[0].keys())}")
            
        # Ensure we have valid samples
        if not self.samples:
            raise ValueError(f"No valid samples found in {mode} dataset")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Load dataset samples from the data directory.
        
        Returns:
            List[Dict]: List of sample dictionaries
        """
        samples = []
        
        # Try different possible directory structures
        possible_structures = [
            (self.data_dir / self.mode / 'images', self.data_dir / self.mode / 'masks'),  # data/train/images, data/train/masks
            (self.data_dir / 'images' / self.mode, self.data_dir / 'masks' / self.mode),  # data/images/train, data/masks/train
            (self.data_dir / self.mode, self.data_dir / self.mode)  # data/train (images and masks in same directory)
        ]
        
        image_dir, mask_dir = None, None
        
        # Find the first valid directory structure
        for img_dir, msk_dir in possible_structures:
            if img_dir.exists() and (not msk_dir.exists() or msk_dir == img_dir):
                # If mask dir doesn't exist or is same as image dir, we'll handle it specially
                image_dir = img_dir
                mask_dir = img_dir  # Will look for masks in the same directory
                print(f"Using directory structure: {image_dir} (images and masks in same directory)")
                break
            elif img_dir.exists() and msk_dir.exists():
                image_dir = img_dir
                mask_dir = msk_dir
                print(f"Using directory structure: {image_dir} (images) and {mask_dir} (masks)")
                break
        
        if not image_dir or not image_dir.exists():
            print(f"Error: No valid image directory found in {self.data_dir} for mode {self.mode}")
            return samples
            
        print(f"Image directory: {image_dir}")
        print(f"Mask directory: {mask_dir}")
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.PNG', '.JPG', '.JPEG', '.TIF', '.TIFF', '.BMP']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        # For mask files, we'll look for both exact matches and pattern matches
        mask_files = {}
        if mask_dir.exists():
            for ext in image_extensions:
                for mask_file in mask_dir.glob(f'*{ext}'):
                    # Use the exact filename as key
                    mask_files[mask_file.name] = str(mask_file)
        
        print(f"Found {len(mask_files)} masks in {mask_dir}")
        
        # Match images with masks
        for img_path in image_files:
            img_name = img_path.name
            mask_path = None
            
            # Try exact match first
            if img_name in mask_files:
                mask_path = mask_files[img_name]
            else:
                # Try to find a matching mask with a different extension
                for ext in image_extensions:
                    if img_name.lower().endswith(ext):
                        # Try with different case
                        for mask_ext in image_extensions:
                            mask_name = img_name[:-len(ext)] + mask_ext
                            if mask_name in mask_files:
                                mask_path = mask_files[mask_name]
                                break
                        break
                
                # If still not found, try to find a mask with a similar name pattern
                if mask_path is None and mask_dir != image_dir:
                    # Remove any file extension and try to find a matching mask
                    base_name = img_path.stem
                    for mask_file in mask_dir.glob(f'*'):
                        if mask_file.stem == base_name:
                            mask_path = str(mask_file)
                            break
            
            # If we have a mask, add the sample
            if mask_path is not None:
                samples.append({
                    'image_path': str(img_path),
                    'mask_path': mask_path,
                    'patient_id': '0',  # Default patient ID
                    'slice_idx': 0      # Default slice index
                })
            else:
                print(f"Warning: No matching mask found for {img_name}")
        
        print(f"Created {len(samples)} valid image-mask pairs")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the image and target
        """
        if not self.samples:
            print("No samples found in dataset")
            return self._get_dummy_sample()
        
        print(f"\n--- Processing sample {idx} ---")
        sample = self.samples[idx]
        print(f"Sample: {sample}")
        
        try:
            print(f"Loading image from: {sample['image_path']}")
            image = self._load_image(sample['image_path'])
            print(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")
            
            mask = None
            if 'mask_path' in sample and sample['mask_path'] is not None:
                print(f"Loading mask from: {sample['mask_path']}")
                mask = self._load_mask(sample['mask_path'])
                print(f"Loaded mask shape: {mask.shape}, dtype: {mask.dtype}")
            
            # Apply augmentations
            if mask is not None:
                print("Applying augmentations to image and mask")
                augmented = self.augmentor(image, mask)
                if isinstance(augmented, tuple) and len(augmented) == 2:
                    image, mask = augmented
                    print(f"Augmented image shape: {image.shape}, mask shape: {mask.shape}")
                else:
                    print(f"Unexpected augmentation output: {type(augmented)}")
            else:
                print("Applying augmentations to image only")
                image = self.augmentor(image)
                print(f"Augmented image shape: {image.shape}")
            
            # Convert to PyTorch tensors
            print("Converting to PyTorch tensors")
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
            print(f"Image tensor shape: {image.shape}, dtype: {image.dtype}")
            
            result = {
                'image': image,
                'patient_id': sample.get('patient_id', ''),
                'slice_idx': sample.get('slice_idx', 0)
            }
            
            if mask is not None:
                mask = torch.from_numpy(mask).permute(2, 0, 1).float()  # (C, H, W)
                print(f"Mask tensor shape: {mask.shape}, dtype: {mask.dtype}")
                result['mask'] = mask
            
            print("Returning sample:", {k: v.shape if hasattr(v, 'shape') else v for k, v in result.items()})
            return result
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            return self._get_dummy_sample()
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load an image from disk."""
        if str(image_path).lower().endswith(('.dcm', '.dicom')):
            # Load DICOM file
            return self.dicom_loader.load_dicom_series(image_path)[0]  # Return first slice for now
        else:
            # Load regular image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            return np.expand_dims(image, axis=-1)  # Add channel dimension
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load a mask from disk."""
        if str(mask_path).lower().endswith(('.dcm', '.dicom')):
            # Load DICOM mask
            mask = self.dicom_loader.load_dicom_series(mask_path)[0]  # Return first slice for now
        else:
            # Load regular mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Ensure binary mask
        if mask.max() > 1:
            mask = (mask > 0).astype(np.float32)
            
        return np.expand_dims(mask, axis=-1)  # Add channel dimension
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample (for testing or when no data is available)."""
        dummy_image = torch.zeros((1, *self.input_size), dtype=torch.float32)
        dummy_mask = torch.zeros((1, *self.input_size), dtype=torch.float32)
        
        return {
            'image': dummy_image,
            'mask': dummy_mask,
            'patient_id': 'dummy',
            'slice_idx': 0
        }


def dict_collate_fn(batch):
    """Collate function for dictionary outputs."""
    print("\n=== Inside dict_collate_fn ===")
    print(f"Batch type: {type(batch)}")
    print(f"Batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
    
    if not batch:
        print("Empty batch, returning empty dict")
        return {}
    
    # Print the type of the first item in the batch
    first_item = batch[0] if hasattr(batch, '__getitem__') else None
    print(f"First item type: {type(first_item) if first_item is not None else 'None'}")
    
    try:
        # Handle case where batch is a list of dictionaries
        if isinstance(first_item, dict):
            print("Processing batch as list of dictionaries")
            keys = first_item.keys()
            print(f"Dictionary keys: {list(keys)}")
            
            collated = {}
            for key in keys:
                if key in ['image', 'mask']:
                    # Stack tensors for image and mask
                    print(f"Stacking tensor for key: {key}")
                    items = [item[key] for item in batch if key in item]
                    print(f"  Number of items: {len(items)}")
                    if items:
                        print(f"  First item shape: {items[0].shape if hasattr(items[0], 'shape') else 'N/A'}")
                        collated[key] = torch.stack(items)
                        print(f"  Stacked tensor shape: {collated[key].shape}")
                else:
                    # Keep as list for other fields
                    print(f"Collecting list for key: {key}")
                    collated[key] = [item.get(key, None) for item in batch]
            
            print("Collated keys:", collated.keys())
            return collated
            
        # Handle case where batch is a list of tuples (image, mask)
        elif isinstance(first_item, (tuple, list)) and len(first_item) == 2:
            print("Processing batch as list of (image, mask) tuples")
            images, masks = zip(*batch)
            return {
                'image': torch.stack(images),
                'mask': torch.stack(masks)
            }
            
        # Handle case where batch is already a string (shouldn't happen)
        elif isinstance(batch, str):
            print(f"Warning: Batch is a string: {batch}")
            return {'dummy': torch.zeros(1, 1, 256, 256)}  # Return a dummy tensor
            
        else:
            print(f"Unexpected batch format. First item type: {type(first_item)}")
            print("Default collation will be used")
            return torch.utils.data.dataloader.default_collate(batch)
            
    except Exception as e:
        print(f"Error in dict_collate_fn: {e}")
        import traceback
        traceback.print_exc()
        # Return a dummy batch to prevent crashing
        return {
            'image': torch.zeros(1, 1, 256, 256),
            'mask': torch.zeros(1, 1, 256, 256),
            'error': str(e)
        }

def create_data_loaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,  # Temporarily set to 0 for debugging
    input_size: Tuple[int, int] = (512, 512),
    use_augmentation: bool = True,
    volume_depth: int = 32
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir (str): Root directory containing the dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker processes for data loading
        input_size (tuple): Target size for resizing (height, width)
        use_augmentation (bool): Whether to use data augmentation
        volume_depth (int): Number of slices to use for 3D volumes
        
    Returns:
        Dict[str, DataLoader]: Dictionary containing data loaders for each split
    """
    print("\n=== Creating data loaders ===")
    
    # Create datasets with consistent input size (256x256 for UNETR)
    unetr_input_size = (256, 256)
    print(f"Creating datasets with input size: {unetr_input_size}")
    
    print("Creating train dataset...")
    train_dataset = LungNoduleDataset(
        data_dir=data_dir,
        mode='train',
        input_size=unetr_input_size,
        use_augmentation=use_augmentation,
        volume_depth=volume_depth
    )
    
    print("Creating validation dataset...")
    val_dataset = LungNoduleDataset(
        data_dir=data_dir,
        mode='val',
        input_size=unetr_input_size,
        use_augmentation=False,  # No augmentation for validation
        volume_depth=volume_depth
    )
    
    print("Creating test dataset...")
    test_dataset = LungNoduleDataset(
        data_dir=data_dir,
        mode='test',
        input_size=unetr_input_size,
        use_augmentation=False,  # No augmentation for testing
        volume_depth=volume_depth
    )
    
    def dict_collate(batch):
        """Collate function that handles dictionary outputs from the dataset.
        
        Args:
            batch: A batch of samples from the dataset
            
        Returns:
            Dictionary of batched tensors or the original batch if processing fails
        """
        print(f"\n=== dict_collate called with batch type: {type(batch).__name__} ===")
        
        # Handle empty batch
        if not batch:
            print("Warning: Empty batch received in dict_collate")
            return {}
            
        # Debug print the first few elements
        print(f"Batch length: {len(batch)}")
        if len(batch) > 0:
            print(f"First element type: {type(batch[0]).__name__}")
            if isinstance(batch[0], (list, tuple)) and len(batch[0]) > 0:
                print(f"  First element of first item type: {type(batch[0][0]).__name__}")
            elif hasattr(batch[0], 'shape'):
                print(f"  First element shape: {batch[0].shape}")
        
        # If batch is already a dictionary, ensure it's properly formatted
        if isinstance(batch, dict):
            print("Batch is already a dictionary")
            # Ensure all values are tensors
            result = {}
            for key, value in batch.items():
                try:
                    if isinstance(value, (list, tuple)) and len(value) > 0 and torch.is_tensor(value[0]):
                        result[key] = torch.stack(value)
                    elif torch.is_tensor(value):
                        result[key] = value
                    else:
                        result[key] = torch.tensor(value)
                except Exception as e:
                    print(f"Error processing key '{key}' with value type {type(value)}: {e}")
                    result[key] = value  # Keep the original value if conversion fails
            return result
            
        # If batch is a list of dictionaries, convert to dictionary of tensors
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            if isinstance(batch[0], dict):
                print("Batch is a list of dictionaries")
                keys = batch[0].keys()
            else:
                # Handle case where batch is a list of tensors or other objects
                print("Batch is a list of non-dictionary items")
                try:
                    # Try to stack tensors if possible
                    if all(torch.is_tensor(item) for item in batch):
                        return {'data': torch.stack(batch)}
                    # Otherwise return as is
                    return {'data': batch}
                except Exception as e:
                    print(f"Error processing batch of non-dict items: {e}")
                    return {'data': batch}
            result = {}
            for key in keys:
                values = []
                for sample in batch:
                    if key in sample:
                        values.append(sample[key])
                    else:
                        print(f"Warning: Key '{key}' missing in sample")
                        values.append(torch.zeros_like(batch[0][key]))
                
                # Stack tensors if possible
                if all(torch.is_tensor(v) for v in values):
                    result[key] = torch.stack(values)
                else:
                    result[key] = values
            return result
        
        # If batch is a list of tensors, stack them
        if isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]):
            print("Batch is a list of tensors, stacking...")
            return torch.stack(batch)
            
        # If batch is a list of lists/tuples, convert to tensor
        if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], (list, tuple)):
            print("Batch is a list of lists, converting to tensor...")
            return torch.tensor(batch)
            
        # For any other case, try to convert to tensor
        print(f"Warning: Unhandled batch type: {type(batch).__name__}")
        try:
            return torch.tensor(batch)
        except Exception as e:
            print(f"Error converting batch to tensor: {e}")
            return batch
    
    # Create data loaders with custom collate function
    print("\nCreating train loader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dict_collate
    )
    
    print("Creating validation loader...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dict_collate
    )
    
    print("Creating test loader...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Typically use batch size 1 for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dict_collate
    )
    
    print("\nData loaders created successfully")
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
