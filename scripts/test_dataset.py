"""Test script to verify dataset and data loader functionality."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from src.preprocessing.dataset import LungNoduleDataset, create_data_loaders

def test_dataset():
    print("=== Testing Dataset ===")
    
    # Create a dataset instance
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
    for i in range(min(3, len(dataset))):  # Test first 3 items
        try:
            sample = dataset[i]
            print(f"Sample {i}:")
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: {type(value).__name__}")
        except Exception as e:
            print(f"Error getting sample {i}: {e}")
            import traceback
            traceback.print_exc()

def test_dataloader():
    print("\n=== Testing DataLoader ===")
    
    # Create data loaders
    loaders = create_data_loaders(
        data_dir='data',
        batch_size=2,
        num_workers=0,
        input_size=(256, 256),
        use_augmentation=False,
        volume_depth=32
    )
    
    # Test training data loader
    train_loader = loaders['train']
    print(f"Number of training batches: {len(train_loader)}")
    
    # Get first batch
    print("\nGetting first training batch:")
    try:
        batch = next(iter(train_loader))
        print(f"Batch type: {type(batch)}")
        
        if isinstance(batch, dict):
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
            print("Batch is not a dictionary. Contents:")
            print(batch)
            
    except Exception as e:
        print(f"Error getting batch: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
    test_dataloader()
