"""Test script to verify data loading functionality."""

import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.dataset import LungNoduleDataset

def test_dataloading():
    print("=== Testing Data Loading ===")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = LungNoduleDataset(
        data_dir='data',
        mode='train',
        input_size=(256, 256),
        use_augmentation=False,
        volume_depth=32
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test getting a single item
    print("\nTesting single item access:")
    try:
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value).__name__}")
    except Exception as e:
        print(f"Error getting sample: {e}")
        import traceback
        traceback.print_exc()
    
    # Create data loader with batch size 2
    print("\nCreating data loader...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Set to 0 for easier debugging
        pin_memory=True,
        drop_last=True
    )
    
    # Get first batch
    print("\nGetting first batch...")
    try:
        batch = next(iter(dataloader))
        print(f"Batch type: {type(batch)}")
        
        if isinstance(batch, dict):
            print("Batch keys:", batch.keys())
            for key, value in batch.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value).__name__} of length {len(value)}")
                    if len(value) > 0 and hasattr(value[0], 'shape'):
                        print(f"    First item shape: {value[0].shape}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        else:
            print("Batch is not a dictionary. Content:", batch)
            
    except Exception as e:
        print(f"Error getting batch: {e}")
        import traceback
        traceback.print_exc()
    
    # Test iterating through the dataloader
    print("\nTesting iteration through dataloader:")
    try:
        for i, batch in enumerate(dataloader):
            print(f"\nBatch {i+1}:")
            if isinstance(batch, dict):
                print("  Batch keys:", batch.keys())
                for key, value in batch.items():
                    if hasattr(value, 'shape'):
                        print(f"    {key}: {value.shape} ({value.dtype})")
                    else:
                        print(f"    {key}: {type(value).__name__}")
            else:
                print("  Batch is not a dictionary:", batch)
            
            # Only process first few batches for testing
            if i >= 2:
                break
                
    except Exception as e:
        print(f"Error during iteration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataloading()
