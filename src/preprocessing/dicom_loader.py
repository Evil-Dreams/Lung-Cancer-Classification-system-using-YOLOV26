import os
import pydicom
import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2
from pathlib import Path

class DICOMLoader:
    """
    Class for loading and processing DICOM files from CT scans.
    Handles loading of single slices or complete CT scan volumes.
    """
    
    def __init__(self, window_center: int = 40, window_width: int = 400):
        """
        Initialize DICOM loader with default window settings for lung CT.
        
        Args:
            window_center (int): Center of the window for windowing (HU units)
            window_width (int): Width of the window for windowing (HU units)
        """
        self.window_center = window_center
        self.window_width = window_width
    
    def load_dicom_series(self, directory: str) -> np.ndarray:
        """
        Load a complete DICOM series from a directory.
        
        Args:
            directory (str): Path to directory containing DICOM files
            
        Returns:
            np.ndarray: 3D numpy array of the CT scan with shape (depth, height, width)
        """
        files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.lower().endswith('.dcm') or f.lower().endswith('.dicom')]
        
        if not files:
            raise ValueError(f"No DICOM files found in {directory}")
        
        # Load all slices
        slices = [pydicom.dcmread(f, force=True) for f in files]
        
        # Sort slices by their z-position
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2])
                   if hasattr(x, 'ImagePositionPatient') else 0, reverse=True)
        
        # Convert to Hounsfield Units (HU)
        volume = np.stack([self._dicom_to_hu(slice) for slice in slices])
        
        return volume
    
    def _dicom_to_hu(self, dicom_slice) -> np.ndarray:
        """Convert DICOM pixel array to Hounsfield Units (HU)."""
        # Apply rescale slope and intercept
        pixel_array = dicom_slice.pixel_array.astype(np.float32)
        
        if hasattr(dicom_slice, 'RescaleSlope'):
            pixel_array = pixel_array * float(dicom_slice.RescaleSlope)
        if hasattr(dicom_slice, 'RescaleIntercept'):
            pixel_array = pixel_array + float(dicom_slice.RescaleIntercept)
            
        return pixel_array
    
    def apply_window(self, image: np.ndarray, window_center: Optional[int] = None, 
                    window_width: Optional[int] = None) -> np.ndarray:
        """
        Apply windowing to a CT slice.
        
        Args:
            image (np.ndarray): CT slice in HU units
            window_center (int, optional): Center of the window. Defaults to None.
            window_width (int, optional): Width of the window. Defaults to None.
            
        Returns:
            np.ndarray: Windowed image (0-255 uint8)
        """
        window_center = window_center if window_center is not None else self.window_center
        window_width = window_width if window_width is not None else self.window_width
        
        min_value = window_center - window_width // 2
        max_value = window_center + window_width // 2
        
        windowed = np.clip(image, min_value, max_value)
        windowed = ((windowed - min_value) / (max_value - min_value) * 255).astype(np.uint8)
        
        return windowed
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize volume to 0-1 range.
        
        Args:
            volume (np.ndarray): 3D volume to normalize
            
        Returns:
            np.ndarray: Normalized volume
        """
        # Clip to reasonable HU range for lung CT
        volume = np.clip(volume, -1000, 1000)
        
        # Normalize to 0-1
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        return volume.astype(np.float32)
    
    def save_volume_as_npy(self, volume: np.ndarray, output_path: str):
        """Save volume as .npy file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, volume)
        
    def load_volume_from_npy(self, npy_path: str) -> np.ndarray:
        """Load volume from .npy file."""
        return np.load(npy_path)


def create_lung_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for the lungs in a CT slice.
    
    Args:
        image (np.ndarray): CT slice (2D array)
        
    Returns:
        np.ndarray: Binary mask where 1 represents lung tissue
    """
    # Convert to 8-bit
    image_8bit = ((image - image.min()) * (255.0 / (image.max() - image.min()))).astype(np.uint8)
    
    # Threshold to get lung region
    _, binary = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask
    mask = np.zeros_like(image_8bit)
    
    # Fill contours (assuming lungs are among the largest contours)
    if contours:
        # Sort contours by area, descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Take the two largest contours (assuming they are the lungs)
        for i, contour in enumerate(contours[:2]):
            cv2.fillPoly(mask, [contour], 1)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask.astype(bool)
