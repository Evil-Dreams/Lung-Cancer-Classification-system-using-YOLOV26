# Lung Cancer Detection and Segmentation Web Application

## Project Overview
This project implements a comprehensive web-based solution for lung cancer diagnosis using deep learning. It features a user-friendly interface for uploading CT scans, with a backend powered by UNETR (UNet Transformer) for precise segmentation of pulmonary nodules, enabling accurate cancer risk assessment and analysis. The system includes an intelligent AI chat assistant specialized in lung cancer discussions and patient education.

## Key Features

### Web Application
- **Interactive UI**: Modern, responsive design with drag-and-drop file upload
- **Real-time Analysis**: Instant processing of CT scans with detailed results
- **Visualization**: Side-by-side comparison of original and segmented images
- **Risk Assessment**: Clear diagnosis with color-coded risk levels (Low/Medium/High)
- **Detailed Report**: Comprehensive analysis including nodule count and affected area percentage
- **AI Chat Assistant**: Interactive medical consultation with context-aware responses

### Backend Processing
- **Segmentation**: UNETR-based model for precise nodule detection
- **Analysis**: Automated nodule counting and affected area calculation
- **Risk Scoring**: Probability-based risk assessment with medical recommendations
- **RESTful API**: Clean endpoints for easy integration
- **Medical Knowledge Base**: Comprehensive lung cancer information system

### AI Chat Assistant
- **Medical Specialization**: Focused exclusively on lung cancer discussions
- **Context-Aware Responses**: Analyzes actual scan results for personalized feedback
- **Risk Assessment Logic**: Low/Moderate/High risk categorization with recommendations
- **Medical Information**: Symptoms, risk factors, prevention strategies, screening guidelines
- **Professional Tone**: Medical-appropriate language with proper disclaimers

## Technical Implementation

### Model Architecture
- **Detection**: YOLOv8 with custom head for nodule detection
- **Segmentation**: UNETR (UNet Transformer) with the following specifications:
  - Input size: 256x256
  - Feature size: 16
  - Hidden size: 256
  - MLP dimension: 512
  - Number of attention heads: 8
  - Spatial dimensions: 2D
  - Attention mechanisms: Self-attention for global context
  - Multi-scale features: Encoder-decoder with skip connections

### Training Process
1. **Data Preparation**
   - Images are resized to 256x256
   - Normalized using mean=[0.485], std=[0.229]
   - Augmentations include:
     - Random rotations
     - Horizontal/Vertical flips
     - Elastic transformations
     - Brightness/Contrast adjustments
     - Gaussian noise

2. **Training Command**
   ```bash
   python train_model.py \
     --data_dir data \
     --num_epochs 100 \
     --batch_size 8 \
     --learning_rate 1e-4 \
     --checkpoint_dir checkpoints
   ```

3. **GPU Acceleration**
   The training script automatically detects and uses available GPUs. To ensure GPU usage:
   - Install CUDA-compatible PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
   - Verify GPU availability: `torch.cuda.is_available()`

### Model Saving
- Checkpoints are saved in the `checkpoints/` directory
- Best model (based on validation loss) is saved as `model_best.pth`
- Training can be resumed from the last checkpoint
- Early stopping prevents overfitting with patience of 15 epochs

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA support (recommended)
- Web browser with JavaScript enabled
- 8GB+ RAM (16GB+ recommended for training)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/lung-cancer-prediction.git
   cd lung-cancer-prediction
   ```

2. **Set up a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements-web.txt
   ```

4. **Create necessary directories**:
   ```bash
   mkdir -p static/uploads static/results
   ```

5. **Download pre-trained model** (if not included):
   ```bash
   # Place model_best.pth in checkpoints/ directory
   ```

## Running the Application

### Option 1: Using Batch File (Windows)
```bash
run.bat
```

### Option 2: Manual Start
1. **Start the Flask development server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Upload a CT scan image** using the drag-and-drop interface or file selector

## Usage Guide

### Web Interface
1. **Upload a CT scan** (PNG, JPG, or JPEG format, max 10MB)
2. **Wait for analysis** to complete (typically a few seconds)
3. **View the results** including:
   - Original and segmented images side by side
   - Cancer risk assessment with probability
   - Nodule count and affected area percentage
   - Detailed diagnosis and medical recommendations
   - Interactive AI chat assistant for questions

4. **Interact with AI Assistant**:
   - Ask about your scan results
   - Get information about symptoms and risk factors
   - Learn about screening and prevention
   - Receive personalized medical guidance

### API Endpoints

#### GET `/`
Main web interface for the application.

#### POST `/predict`
Upload a CT scan for analysis.

**Request:**
```bash
Content-Type: multipart/form-data
file: [CT scan image]
```

**Response:**
```json
{
  "success": true,
  "original": "/static/uploads/filename.jpg",
  "mask": "/static/results/mask_filename.png",
  "analysis": {
    "probability": 12.5,
    "diagnosis": "Small nodules detected",
    "severity": "Moderate Risk",
    "severity_level": "medium",
    "recommendation": "Consult a pulmonologist for further evaluation...",
    "nodule_count": 3,
    "affected_area": "8.3%",
    "timestamp": "2025-11-11T13:45:30.123456",
    "details": {
      "total_pixels_analyzed": 65536,
      "affected_pixels": 5432
    }
  },
  "metadata": {
    "model": "UNETR-2D",
    "version": "1.0.0",
    "processing_time_ms": 0
  }
}
```

#### POST `/chat`
Interact with the AI medical assistant.

**Request:**
```json
Content-Type: application/json
{
  "message": "What are the symptoms of lung cancer?"
}
```

**Response:**
```json
{
  "success": true,
  "response": "**Common Lung Cancer Symptoms:**\n• Persistent cough\n• Shortness of breath\n• Chest pain\n• Unexplained weight loss\n• Fatigue\n\n**Less Common Symptoms:**\n• Hoarseness\n• Difficulty swallowing\n• Finger clubbing\n• Repeated infections\n\n**Important:** Many of these symptoms can be caused by conditions other than lung cancer...",
  "timestamp": "2025-11-11T13:45:30.123456"
}
```

#### GET `/static/*`
Serve static files (uploaded images, processed results, CSS, JS).

## Project Structure
```
Lung Cancer Prediction/
├── app.py                    # Main Flask application with AI chat integration
├── requirements.txt            # Core Python dependencies
├── requirements-web.txt        # Web application dependencies
├── run.bat                   # Windows launcher script
├── train_model.py            # Complete training pipeline
├── prepare_data.py           # Data preparation and validation
├── configs/
│   └── config.yaml          # Configuration settings
├── src/                     # Source code
│   ├── preprocessing/        # DICOM loading and data augmentation
│   │   ├── dicom_loader.py
│   │   ├── dataset.py
│   │   └── augmentations.py
│   ├── segmentation/         # UNETR model implementation
│   │   └── unetr.py
│   ├── detection/           # YOLOv8 detector
│   │   └── yolo_detector.py
│   └── utils/              # Utility functions
│       ├── metrics.py
│       ├── config.py
│       └── visualization.py
├── templates/               # HTML templates
│   └── index.html          # Main web interface with chat
├── static/                 # Static files
│   ├── uploads/            # User uploaded CT scans
│   └── results/            # Processed segmentation masks
├── checkpoints/            # Model checkpoints
│   └── model_best.pth     # Pre-trained UNETR model
└── scripts/               # Training and evaluation scripts
    ├── train_segmentation.py
    └── evaluate.py
```

## Data Preparation

### Data Organization
The system expects the following data structure for training:
```
data/
├── train/
│   ├── images/           # Training CT scan images
│   └── masks/            # Corresponding segmentation masks
├── val/
│   ├── images/           # Validation images
│   └── masks/            # Validation masks
└── test/
    ├── images/           # Test images
    └── masks/            # Test masks
```

### Supported Formats
- **Images**: PNG, JPG, JPEG (for web interface)
- **Medical**: DICOM files (for training pipeline)
- **Masks**: PNG format with binary values (0=background, 1=nodule)

### Data Preparation Script
Use the provided script to organize your data:
```bash
# Check existing data structure
python prepare_data.py --action check --source-dir data

# Organize scattered data
python prepare_data.py --action organize --source-dir /path/to/data --target-dir data

# Create sample data for testing
python prepare_data.py --action create-sample --target-dir data --num-samples 100
```

## Training

### Training Pipeline
1. **Prepare Data**: Use `prepare_data.py` to organize your dataset
2. **Start Training**: Run `train_model.py` with appropriate parameters
3. **Monitor Progress**: Check TensorBoard logs and console output
4. **Model Selection**: Best model automatically saved as `model_best.pth`

### Training Features
- **Early Stopping**: Prevents overfitting with patience parameter
- **Learning Rate Scheduling**: Reduces LR when validation plateaus
- **Checkpoint Management**: Saves best and regular checkpoints
- **TensorBoard Integration**: Real-time training metrics
- **GPU Support**: Automatic CUDA detection and usage
- **Batch Processing**: Memory-efficient data loading

### Training Commands
```bash
# Basic training
python train_model.py --data-dir data/ --num-epochs 50 --batch-size 4

# Advanced training with custom parameters
python train_model.py \
  --data-dir data/ \
  --checkpoint-dir checkpoints/ \
  --num-epochs 100 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --patience 15
```

## Performance

### Model Evaluation Metrics (Test Set)
- **Dice Coefficient**: 0.8328
- **IoU (Jaccard Index)**: 0.7790
- **Accuracy**: 0.8344
- **Precision**: 0.8170
- **Recall**: 0.8673
- **Specificity**: 0.8154

### Training Performance
- **Training Time**: ~2s per batch on NVIDIA T4 GPU
- **Memory Usage**: ~8GB VRAM with batch size 8
- **Validation Dice Score**: ~0.84 (on validation set)
- **Inference Speed**: ~50ms per image on CPU, ~5ms on GPU

### System Requirements
- **Minimum**: 8GB RAM, 4GB VRAM (GPU optional)
- **Recommended**: 16GB+ RAM, 8GB+ VRAM (GPU)
- **Storage**: 2GB+ for models and data
- **OS**: Windows 10/11, Linux, macOS

## Troubleshooting

### Common Issues

1. **Upload Fails**
   - Ensure file is in PNG, JPG, or JPEG format
   - Check file size (max 10MB)
   - Verify write permissions in `static/uploads/` and `static/results/`
   - Check browser console for JavaScript errors

2. **Model Not Loading**
   - Ensure `checkpoints/model_best.pth` exists
   - Check CUDA availability if using GPU
   - Verify PyTorch and MONAI versions
   - Check model file integrity

3. **Dependency Issues**
   - Recreate virtual environment
   - Install exact versions from requirements file
   - Update pip: `pip install --upgrade pip`
   - Use Python 3.8+ (recommended 3.9+)

4. **Chat Not Working**
   - Verify latest_analysis variable is being set
   - Check network connectivity for API calls
   - Ensure chat endpoint is accessible
   - Check browser console for JavaScript errors

5. **Performance Issues**
   - Reduce batch size for training
   - Use GPU acceleration if available
   - Close other applications using GPU memory
   - Optimize image input size

### Error Messages and Solutions

**Error**: "Model not loaded. Please check the model file."
- **Solution**: Ensure `checkpoints/model_best.pth` exists and is not corrupted

**Error**: "File type not allowed"
- **Solution**: Upload only PNG, JPG, or JPEG files

**Error**: "File size exceeds limit"
- **Solution**: Compress image or use smaller file (<10MB)

**Error**: "CUDA out of memory"
- **Solution**: Reduce batch size or use CPU for training

## API Integration

### Python Client Example
```python
import requests
import json

# Upload image for analysis
with open('ct_scan.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    result = response.json()

# Chat with AI assistant
chat_data = {'message': 'What are the risk factors for lung cancer?'}
response = requests.post('http://localhost:5000/chat', json=chat_data)
ai_response = response.json()
print(ai_response['response'])
```

### JavaScript Client Example
```javascript
// Upload image
const formData = new FormData();
formData.append('file', imageFile);

fetch('/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// Chat with AI
fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: 'Hello'})
})
.then(response => response.json())
.then(data => console.log(data.response));
```

## Security and Privacy

### Data Protection
- **Local Processing**: All analysis happens on your local server
- **No Cloud Upload**: Medical data never leaves your system
- **Input Validation**: Comprehensive file validation and sanitization
- **XSS Protection**: Chat messages are properly escaped
- **File Size Limits**: Prevents denial of service attacks

### Medical Disclaimer
- This tool is for research and educational purposes only
- Not a substitute for professional medical diagnosis
- Always consult with qualified healthcare professionals
- Results should be discussed with medical practitioners

## Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment and install dependencies
3. Make your changes following the existing code style
4. Add tests for new features
5. Submit a pull request with detailed description

### Code Style
- Follow PEP 8 Python style guidelines
- Use descriptive variable and function names
- Add comprehensive docstrings
- Include error handling and logging
- Write unit tests for new functionality

### Contribution Areas
- Model architecture improvements
- Additional medical knowledge for chat bot
- UI/UX enhancements
- Performance optimizations
- New features and endpoints

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **UNETR**: UNETR: Transformers for 3D Medical Image Segmentation (https://arxiv.org/abs/2103.10504)
- **YOLOv8**: Ultralytics YOLOv8 (https://github.com/ultralytics/ultralytics)
- **MONAI**: Medical Open Network for AI (https://monai.io/)
- **Flask**: Web framework (https://flask.palletsprojects.com/)
- **Tailwind CSS**: Utility-first CSS framework (https://tailwindcss.com/)
- **PyTorch**: Deep learning framework (https://pytorch.org/)

## Version History

### Version 1.0.0
- Initial release with UNETR segmentation
- Web interface with drag-and-drop upload
- AI chat assistant integration
- Real-time analysis and visualization
- Comprehensive medical knowledge base
- Risk assessment and recommendations

### Future Roadmap
- Classification model integration (ResNet50)
- 3D volume processing support
- Multi-language support
- Mobile application
- DICOM file upload support
- Advanced visualization tools
- Patient history tracking
- Export functionality for results
