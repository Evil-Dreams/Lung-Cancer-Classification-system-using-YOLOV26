import sys
import os
import torch
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.dataset import LungNoduleDataset, create_data_loaders

def test_dataset():
    print("Testing dataset loading...")
    
    # Create a dataset instance
    data_dir = os.path.join(project_root, 'data')
    dataset = LungNoduleDataset(
        data_dir=data_dir,
        mode='train',
        input_size=(256, 256),
        use_augmentation=True
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Test loading a single sample
    print("\nTesting single sample loading...")
    for i in range(min(2, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {value} (type: {type(value).__name__})")
    
    # Test data loader
    print("\nTesting data loader...")
    loaders = create_data_loaders(
        data_dir=data_dir,
        batch_size=2,
        num_workers=0,  # Set to 0 for easier debugging
        input_size=(256, 256)
    )
    
    # Get a batch from the training loader
    train_loader = loaders['train']
    print(f"Number of batches in train_loader: {len(train_loader)}")
    
    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i}:")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {value} (type: {type(value).__name__})")
        
        # Only process the first batch for testing
        if i >= 0:
            break

if __name__ == "__main__":
    test_dataset()
