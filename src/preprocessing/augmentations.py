import albumentations as A
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class LungAugmentor:
    """
    Class for applying augmentations to lung CT images and their corresponding masks.
    Uses Albumentations library for efficient image augmentations.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (512, 512), mode: str = 'train'):
        """
        Initialize the augmentor with specified configuration.
        
        Args:
            input_size (tuple): Target size for resizing (height, width)
            mode (str): One of 'train', 'val', or 'test' to determine which augmentations to apply
        """
        self.input_size = input_size
        self.mode = mode
        
        # Define base transforms that are always applied
        self.base_transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1], always_apply=True)
        ], additional_targets={'mask': 'mask'})
        
        # Training augmentations
        if mode == 'train':
            self.transform = A.Compose([
                # Spatial transforms
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.RandomRotate90(p=0.3),
                    A.Transpose(p=0.1),
                ], p=0.8),
                
                # Elastic transforms
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
                    A.GridDistortion(p=0.2),
                    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.3),
                ], p=0.5),
                
                # Intensity transforms
                A.OneOf([
                    A.CLAHE(clip_limit=2.0, p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                ], p=0.5),
                
                # Noise and blur
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.MotionBlur(blur_limit=3, p=0.2),
                ], p=0.3),
                
                # Dropout
                A.OneOf([
                    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                                  min_holes=1, min_height=8, min_width=8, 
                                  fill_value=0, p=0.3),
                    A.GridDropout(ratio=0.1, p=0.2),
                ], p=0.3),
                
                # Normalization
                A.Normalize(
                    mean=[0.485],
                    std=[0.229],
                    max_pixel_value=255.0,
                    p=1.0
                )
            ], additional_targets={'mask': 'mask'})
        
        # Validation/Test transforms (only resize and normalize)
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=[0.485],
                    std=[0.229],
                    max_pixel_value=255.0,
                    p=1.0
                )
            ], additional_targets={'mask': 'mask'})
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Union[Dict, Tuple]:
        """
        Apply augmentations to the image and mask.
        
        Args:
            image (np.ndarray): Input image (H, W) or (H, W, C)
            mask (np.ndarray, optional): Corresponding mask (H, W) or (H, W, C)
            
        Returns:
            If mask is provided: tuple of (augmented_image, augmented_mask)
            If no mask: augmented_image
        """
        # Ensure image is in the right format (H, W, 1) for grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            
        # Apply base transforms (resize)
        if mask is not None:
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)
                
            augmented = self.base_transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            
            # Apply additional transforms
            augmented = self.transform(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        else:
            augmented = self.base_transform(image=image)
            augmented = self.transform(image=augmented['image'])
            return augmented['image']


def get_validation_augmentations(input_size: Tuple[int, int] = (512, 512)):
    """
    Get validation augmentations (only resize and normalize).
    
    Args:
        input_size (tuple): Target size for resizing (height, width)
        
    Returns:
        albumentations.Compose: Composition of augmentations
    """
    return A.Compose([
        A.Resize(height=input_size[0], width=input_size[1], always_apply=True),
        A.Normalize(
            mean=[0.485],
            std=[0.229],
            max_pixel_value=255.0,
        )
    ], additional_targets={'mask': 'mask'})


def get_training_augmentations(input_size: Tuple[int, int] = (512, 512)):
    """
    Get training augmentations with moderate geometric and intensity transformations.
    
    Args:
        input_size (tuple): Target size for resizing (height, width)
        
    Returns:
        albumentations.Compose: Composition of augmentations
    """
    return A.Compose([
        A.Resize(height=input_size[0], width=input_size[1], always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Normalize(
            mean=[0.485],
            std=[0.229],
            max_pixel_value=255.0,
        )
    ], additional_targets={'mask': 'mask'})
