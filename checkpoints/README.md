# Model Checkpoint Placeholder

This directory should contain the trained UNETR model file `model_best.pth`.

## How to obtain the model:

1. **Train your own model:**
   ```bash
   python train_model.py --data-dir data/ --num-epochs 100 --batch-size 8
   ```

2. **Download pre-trained model:**
   - Contact the repository maintainer for access to pre-trained weights
   - Or use the training pipeline to train on your own dataset

## Model Specifications:
- **Architecture**: UNETR (UNet Transformer)
- **Input Size**: 256x256
- **Feature Size**: 16
- **Hidden Size**: 256
- **Number of Classes**: 2 (background, nodule)

## Expected File:
- `model_best.pth` - Best model checkpoint from training (approximately 86MB)

## Note:
The model file is too large for GitHub storage. Please obtain or train the model separately and place it in this directory.
