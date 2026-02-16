import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch
from ultralytics import YOLO
import yaml

class YOLONoduleDetector:
    """
    YOLOv8-based lung nodule detector.
    Handles loading the YOLO model, preprocessing, inference, and post-processing.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path (str, optional): Path to the YOLO model weights (.pt file).
                                      If None, will use the default pretrained model.
            conf_threshold (float): Confidence threshold for detection.
            iou_threshold (float): IoU threshold for NMS.
            device (str): Device to run inference on ('cuda' or 'cpu').
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load the model
        self.model = self._load_model(model_path)
        
        # Get class names
        self.class_names = self._get_class_names()
    
    def _load_model(self, model_path: Optional[str] = None) -> YOLO:
        """Load the YOLO model from file or download if not found."""
        if model_path is None:
            # Use a default YOLOv8 model (you may want to replace with your trained model)
            model = YOLO('yolov8n.pt')
            print("Using default YOLOv8n model. It's recommended to train a custom model for lung nodule detection.")
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = YOLO(model_path)
        
        # Set model parameters
        model.overrides['conf'] = self.conf_threshold
        model.overrides['iou'] = self.iou_threshold
        model.overrides['device'] = self.device
        
        return model
    
    def _get_class_names(self) -> List[str]:
        """Get class names from model or use default."""
        if hasattr(self.model, 'names') and self.model.names:
            return list(self.model.names.values())
        else:
            # Default class names if not available in the model
            return ['nodule']
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for YOLO model.
        
        Args:
            image (np.ndarray): Input image (H, W, C) in BGR format (OpenCV default)
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to 3 channels if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        return image
    
    def detect(self, 
               image: Union[str, np.ndarray], 
               return_image: bool = False) -> Union[Dict, Tuple[Dict, np.ndarray]]:
        """
        Detect lung nodules in the input image.
        
        Args:
            image (str or np.ndarray): Input image path or numpy array
            return_image (bool): Whether to return the annotated image
            
        Returns:
            If return_image is False: Dict with detection results
            If return_image is True: Tuple of (results_dict, annotated_image)
        """
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = cv2.imread(image)
        
        # Store original image for visualization
        original_image = image.copy()
        
        # Preprocess the image
        processed_image = self.preprocess(image)
        
        # Run inference
        results = self.model(processed_image)
        
        # Process results
        detections = self._process_detections(results[0])
        
        if return_image:
            # Draw detections on the image
            annotated_image = self._draw_detections(original_image, detections)
            return detections, annotated_image
        else:
            return detections
    
    def _process_detections(self, results) -> Dict:
        """Process YOLO detection results into a structured format."""
        # Initialize output dictionary
        output = {
            'boxes': [],
            'scores': [],
            'class_ids': [],
            'class_names': [],
            'count': 0
        }
        
        # Extract detections
        for result in results.boxes.data.tolist():
            # Extract bounding box and score
            x1, y1, x2, y2, score, class_id = result
            
            # Filter by confidence threshold
            if score < self.conf_threshold:
                continue
                
            # Convert to integers for bounding box coordinates
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get class name
            class_id = int(class_id)
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            
            # Add to output
            output['boxes'].append([x1, y1, x2, y2])
            output['scores'].append(score)
            output['class_ids'].append(class_id)
            output['class_names'].append(class_name)
            output['count'] += 1
        
        return output
    
    def _draw_detections(self, 
                        image: np.ndarray, 
                        detections: Dict,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
        """
        Draw detection bounding boxes on the image.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            detections (Dict): Detection results from detect()
            color (tuple): BGR color for bounding boxes
            thickness (int): Line thickness
            
        Returns:
            np.ndarray: Image with drawn detections
        """
        result = image.copy()
        
        for i in range(detections['count']):
            x1, y1, x2, y2 = detections['boxes'][i]
            score = detections['scores'][i]
            class_name = detections['class_names'][i]
            
            # Draw rectangle
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Create label
            label = f"{class_name}: {score:.2f}"
            
            # Get text size
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw filled rectangle for label background
            cv2.rectangle(
                result, 
                (x1, y1 - label_height - 10), 
                (x1 + label_width, y1), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                result, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0),  # Black text
                1, 
                cv2.LINE_AA
            )
        
        return result
    
    def process_volume(self, 
                      volume: np.ndarray,
                      batch_size: int = 8,
                      show_progress: bool = True) -> Dict[int, Dict]:
        """
        Process a 3D volume to detect nodules in each slice.
        
        Args:
            volume (np.ndarray): 3D volume (D, H, W) or (D, H, W, C)
            batch_size (int): Number of slices to process in parallel
            show_progress (bool): Whether to show progress bar
            
        Returns:
            Dict[int, Dict]: Dictionary mapping slice index to detection results
        """
        # Ensure volume is 3D (D, H, W)
        if len(volume.shape) == 4:
            volume = volume[..., 0]  # Take first channel if multi-channel
        
        num_slices = volume.shape[0]
        all_detections = {}
        
        # Process in batches
        for i in range(0, num_slices, batch_size):
            batch = volume[i:i+batch_size]
            batch_images = []
            
            # Preprocess each slice in the batch
            for j in range(batch.shape[0]):
                # Convert to 3-channel grayscale for YOLO
                img = batch[j]
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                batch_images.append(img)
            
            # Run batch inference
            with torch.no_grad():
                results = self.model(batch_images)
            
            # Process results
            for j, result in enumerate(results):
                slice_idx = i + j
                if slice_idx >= num_slices:
                    continue
                    
                detections = self._process_detections(result)
                if detections['count'] > 0:
                    all_detections[slice_idx] = detections
            
            # Print progress
            if show_progress:
                print(f"Processed {min(i + batch_size, num_slices)}/{num_slices} slices", end='\r')
        
        if show_progress:
            print(f"\nDone. Found nodules in {len(all_detections)} slices.")
            
        return all_detections
    
    def save_detections_to_file(self, 
                              detections: Dict[int, Dict], 
                              output_path: str,
                              class_names: Optional[List[str]] = None):
        """
        Save detection results to a file in YOLO format.
        
        Args:
            detections (Dict[int, Dict]): Detection results from process_volume()
            output_path (str): Output file path
            class_names (List[str], optional): List of class names
        """
        with open(output_path, 'w') as f:
            for slice_idx, dets in detections.items():
                for i in range(dets['count']):
                    x1, y1, x2, y2 = dets['boxes'][i]
                    score = dets['scores'][i]
                    class_id = dets['class_ids'][i]
                    class_name = dets['class_names'][i]
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    img_width = 512  # Update with actual image dimensions if different
                    img_height = 512
                    
                    x_center = (x1 + x2) / (2 * img_width)
                    y_center = (y1 + y2) / (2 * img_height)
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Write to file: class_id x_center y_center width height confidence
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n")
