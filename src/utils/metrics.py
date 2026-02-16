"""
Metrics for evaluating segmentation performance.
"""

import torch
import numpy as np

def calculate_metrics(preds, targets, smooth=1e-5):
    """
    Calculate various segmentation metrics.
    
    Args:
        preds: Predicted masks (B, C, H, W) where C=2 for binary segmentation
        targets: Ground truth masks (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dictionary containing various metrics
    """
    # Convert predictions to binary (0 or 1) and take channel 1 (foreground)
    if preds.shape[1] == 2:  # If model outputs 2 channels (background + foreground)
        preds = torch.softmax(preds, dim=1)[:, 1:2]  # Take only foreground channel
    preds = (preds > 0.5).float()
    
    # Ensure targets are binary (0 or 1)
    targets = (targets > 0.5).float()
    
    # Flatten predictions and targets
    preds_flat = preds.reshape(preds.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)
    
    # Calculate true positives, false positives, false negatives, true negatives
    tp = (preds_flat * targets_flat).sum(dim=1)
    fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
    fn = ((1 - preds_flat) * targets_flat).sum(dim=1)
    tn = ((1 - preds_flat) * (1 - targets_flat)).sum(dim=1)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    specificity = tn / (tn + fp + smooth)
    dice = (2 * tp) / (2 * tp + fp + fn + smooth)
    iou = tp / (tp + fp + fn + smooth)
    
    return {
        'dice': dice.mean().item(),
        'iou': iou.mean().item(),
        'accuracy': accuracy.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'specificity': specificity.mean().item()
    }

def print_metrics(metrics):
    """Print formatted metrics."""
    print("\nModel Performance Metrics:")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"{name.upper():<12}: {value:.4f}")
    print("-" * 30)
