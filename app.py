from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from datetime import datetime
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
from src.segmentation.unetr import get_unetr_model
import io
import cv2
import json
from pathlib import Path
import re
import random

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MODEL_PATH = 'checkpoints/model_best.pth'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Ensure folders exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_unetr_model()
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model = model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
# Global variable to store latest analysis results
latest_analysis = {}

class LungCancerChatBot:
    """AI Chat Bot specialized in lung cancer discussions and patient assessment."""
    
    def __init__(self):
        self.medical_responses = {
            'greeting': [
                "Hello! I'm your lung cancer specialist AI assistant. I can help you understand your CT scan results and provide information about lung health.",
                "Welcome! I'm here to discuss your lung cancer screening results and answer any questions you may have about lung health.",
                "Hi! I specialize in lung cancer analysis. Based on your scan results, I can provide insights and answer your health questions."
            ],
            'no_results': [
                "I don't see any recent scan results. Please upload your CT scan first so I can provide personalized analysis.",
                "To give you the best assessment, I need to see your CT scan results. Please upload an image first.",
                "I'd be happy to help, but I need your scan results to provide accurate information about your lung health."
            ],
            'low_risk': [
                "Based on your scan results, I don't see significant concerns. Your lungs appear healthy with no major nodules detected.",
                "Good news! Your CT scan shows minimal or no suspicious nodules. Your lung health appears to be in good condition.",
                "Your results look reassuring. I don't see significant abnormalities that would require immediate concern."
            ],
            'moderate_risk': [
                "I've detected some small nodules in your lungs. While not immediately alarming, I recommend consulting with a pulmonologist for further evaluation.",
                "Your scan shows some small nodules that warrant attention. These are common and often benign, but professional follow-up is advised.",
                "I found some nodules that should be monitored. Many small nodules are benign, but a specialist consultation is recommended."
            ],
            'high_risk': [
                "Your scan shows significant nodules that require immediate medical attention. Please consult with an oncologist as soon as possible.",
                "I'm seeing concerning findings that need urgent medical evaluation. Please schedule an appointment with a lung specialist immediately.",
                "Your results indicate significant abnormalities that require prompt medical attention. Don't delay in seeking specialist care."
            ],
            'general_info': [
                "I can provide information about lung cancer screening, symptoms, risk factors, and prevention strategies.",
                "Feel free to ask me about lung health, cancer screening guidelines, or what to expect during the diagnostic process.",
                "I'm here to help with questions about lung cancer detection, treatment options, and preventive care measures."
            ]
        }
        
        self.medical_knowledge = {
            'symptoms': {
                'common': ['persistent cough', 'shortness of breath', 'chest pain', 'unexplained weight loss', 'fatigue'],
                'less_common': ['hoarseness', 'difficulty swallowing', 'finger clubbing', 'repeated infections']
            },
            'risk_factors': [
                'smoking', 'secondhand smoke', 'radon exposure', 'asbestos exposure', 
                'family history', 'age over 50', 'air pollution'
            ],
            'prevention': [
                'quit smoking', 'avoid secondhand smoke', 'test home for radon', 
                'exercise regularly', 'eat healthy diet', 'regular screenings'
            ]
        }
    
    def get_response(self, user_message, analysis_data=None):
        """Generate AI response based on user message and analysis data."""
        user_message = user_message.lower().strip()
        
        # Check for greetings
        if any(word in user_message for word in ['hello', 'hi', 'hey', 'greetings']):
            return random.choice(self.medical_responses['greeting'])
        
        # Check if user is asking about their results
        if any(word in user_message for word in ['results', 'scan', 'analysis', 'diagnosis', 'what do you see', 'findings']):
            if analysis_data:
                return self._analyze_results(analysis_data)
            else:
                return random.choice(self.medical_responses['no_results'])
        
        # Provide general medical information
        if any(word in user_message for word in ['symptoms', 'signs', 'warning signs']):
            return self._provide_symptoms_info()
        
        if any(word in user_message for word in ['risk', 'causes', 'prevention']):
            return self._provide_risk_info()
        
        if any(word in user_message for word in ['screening', 'test', 'detection']):
            return self._provide_screening_info()
        
        # Default response
        return "I'm here to help with lung cancer-related questions. You can ask me about your scan results, symptoms, risk factors, or screening guidelines. For specific medical advice, always consult with a healthcare professional."
    
    def _analyze_results(self, analysis_data):
        """Analyze scan results and provide assessment."""
        try:
            probability = analysis_data.get('probability', 0)
            severity_level = analysis_data.get('severity_level', 'low')
            nodule_count = analysis_data.get('nodule_count', 0)
            affected_area = analysis_data.get('affected_area', '0%')
            
            if severity_level == 'low' or probability < 5:
                response = random.choice(self.medical_responses['low_risk'])
                response += f"\n\n**Detailed Analysis:**\n"
                response += f"• Cancer Probability: {probability}%\n"
                response += f"• Nodules Detected: {nodule_count}\n"
                response += f"• Affected Lung Area: {affected_area}\n"
                response += f"\n**Recommendation:** Continue with regular health check-ups and maintain a healthy lifestyle."
                
            elif severity_level == 'medium' or 5 <= probability < 15:
                response = random.choice(self.medical_responses['moderate_risk'])
                response += f"\n\n**Detailed Analysis:**\n"
                response += f"• Cancer Probability: {probability}%\n"
                response += f"• Nodules Detected: {nodule_count}\n"
                response += f"• Affected Lung Area: {affected_area}\n"
                response += f"\n**Recommendation:** Schedule a consultation with a pulmonologist within 2-4 weeks for further evaluation."
                
            else:  # high risk
                response = random.choice(self.medical_responses['high_risk'])
                response += f"\n\n**Detailed Analysis:**\n"
                response += f"• Cancer Probability: {probability}%\n"
                response += f"• Nodules Detected: {nodule_count}\n"
                response += f"• Affected Lung Area: {affected_area}\n"
                response += f"\n**Recommendation:** Seek immediate medical attention from an oncologist or lung specialist."
            
            return response
            
        except Exception as e:
            return "I'm having trouble analyzing your results. Please ensure your scan has been processed properly."
    
    def _provide_symptoms_info(self):
        """Provide information about lung cancer symptoms."""
        symptoms = self.medical_knowledge['symptoms']
        response = "**Common Lung Cancer Symptoms:**\n"
        response += "• " + "\n• ".join(symptoms['common']) + "\n\n"
        response += "**Less Common Symptoms:**\n"
        response += "• " + "\n• ".join(symptoms['less_common']) + "\n\n"
        response += "**Important:** Many of these symptoms can be caused by conditions other than lung cancer. If you experience any persistent symptoms, please consult a healthcare provider."
        return response
    
    def _provide_risk_info(self):
        """Provide information about risk factors and prevention."""
        response = "**Major Risk Factors:**\n"
        response += "• " + "\n• ".join(self.medical_knowledge['risk_factors'][:4]) + "\n\n"
        response += "**Prevention Strategies:**\n"
        response += "• " + "\n• ".join(self.medical_knowledge['prevention'][:4]) + "\n\n"
        response += "The single most important step you can take is to avoid or quit smoking."
        return response
    
    def _provide_screening_info(self):
        """Provide information about lung cancer screening."""
        response = "**Lung Cancer Screening Guidelines:**\n"
        response += "• Recommended for adults aged 50-80 with a 20+ pack-year smoking history\n"
        response += "• Current smokers or those who quit within the past 15 years\n"
        response += "• Annual low-dose CT scans are the standard screening method\n"
        response += "• Early detection through screening can significantly improve outcomes\n\n"
        response += "**Screening Benefits:**\n"
        response += "• Detects cancer at earlier, more treatable stages\n"
        response += "• Reduces lung cancer mortality by 20-25%\n"
        response += "• Non-invasive and quick procedure (about 10 minutes)"
        return response

# Initialize chat bot
chat_bot = LungCancerChatBot()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file):
    """Validate image file before processing"""
    if not file:
        return False, "No file provided"
    
    if not allowed_file(file.filename):
        return False, "File type not allowed. Please upload a PNG, JPG, or JPEG image."
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File size exceeds {MAX_FILE_SIZE//(1024*1024)}MB limit."
    
    try:
        # Verify it's a valid image
        img = Image.open(file)
        img.verify()
        file.seek(0)
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def preprocess_image(image):
    """Preprocess the image for model inference"""
    try:
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Store original size for later use
        original_size = image.size
        
        # Resize to 256x256 while maintaining aspect ratio
        image = ImageOps.pad(image, (256, 256), method=Image.Resampling.LANCZOS, color=0)
        
        # Convert to numpy array and normalize
        image_np = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch and channel dimensions
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)
        
        return image_tensor, image, original_size
    except Exception as e:
        raise Exception(f"Error in image preprocessing: {str(e)}")

def postprocess_output(output, original_size):
    """Process model output to create segmentation mask"""
    try:
        # Apply softmax and get the predicted class (0: background, 1: nodule)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
        
        # Convert to numpy and scale to 0-255
        pred_np = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
        
        # Convert to PIL Image and resize back to original dimensions
        mask = Image.fromarray(pred_np)
        mask = mask.resize(original_size, Image.Resampling.NEAREST)
        
        # Apply some smoothing to the mask
        mask = mask.filter(ImageFilter.MedianFilter(3))
        
        return mask
    except Exception as e:
        raise Exception(f"Error in output postprocessing: {str(e)}")

def analyze_nodules(mask_array):
    """Analyze the segmentation mask to extract nodule information"""
    try:
        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask_array.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter small contours (noise reduction)
        min_contour_area = 10  # Minimum area to be considered a nodule
        nodules = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                nodules.append({
                    'area': int(area),
                    'bounding_box': [int(x), int(y), int(w), int(h)]
                })
        
        # Calculate total affected area as percentage
        total_area = mask_array.size
        affected_area = np.sum(mask_array > 0)
        affected_percentage = (affected_area / total_area) * 100
        
        return {
            'nodule_count': len(nodules),
            'nodules': nodules,
            'affected_area': round(float(affected_percentage), 2),
            'total_area': total_area
        }
    except Exception as e:
        print(f"Error in nodule analysis: {str(e)}")
        return {
            'nodule_count': 0,
            'nodules': [],
            'affected_area': 0.0,
            'total_area': 0
        }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request', 'success': False}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        # Validate file
        is_valid, error_msg = validate_image(file)
        if not is_valid:
            return jsonify({'error': error_msg, 'success': False}), 400
        
        # Generate unique filenames with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        original_filename = f'original_{timestamp}.png'
        mask_filename = f'mask_{timestamp}.png'
        
        # Ensure upload directory exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(RESULT_FOLDER, exist_ok=True)
        
        # Save original image
        original_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.seek(0)
        file.save(original_path)
        
        # Process image
        file.seek(0)
        try:
            image = Image.open(io.BytesIO(file.read()))
            if image is None:
                raise ValueError("Failed to read the image file")
                
            image_tensor, processed_image, original_size = preprocess_image(image)
            
            # Run model
            if model is None:
                raise Exception("Model not loaded. Please check the model file.")
                
            with torch.no_grad():
                output = model(image_tensor)
            
            # Process output
            mask = postprocess_output(output, image.size)  # Use original image size
            
            # Save mask
            mask_path = os.path.join(RESULT_FOLDER, mask_filename)
            mask.save(mask_path)
            
            # Convert mask to numpy array for analysis
            mask_array = np.array(mask) > 0
            
            # Analyze nodules
            nodule_analysis = analyze_nodules(mask_array.astype(np.uint8) * 255)
            
            # Calculate cancer probability (simplified example)
            cancer_probability = min(99.9, nodule_analysis['affected_area'] * 1.5)  # Scale for demo
            
            # Determine diagnosis based on probability
            if cancer_probability < 5:
                diagnosis = "No significant nodules detected"
                severity = "Low Risk"
                recommendation = "Routine check-up recommended. No immediate action required."
                severity_level = "low"
            elif cancer_probability < 15:
                diagnosis = "Small nodules detected"
                severity = "Moderate Risk"
                recommendation = "Consult a pulmonologist for further evaluation. Regular monitoring advised."
                severity_level = "medium"
            else:
                diagnosis = "Significant nodules detected"
                severity = "High Risk"
                recommendation = "Immediate consultation with an oncologist is strongly recommended. Further diagnostic tests needed."
                severity_level = "high"
            
            # Prepare response
            response = {
                'success': True,
                'original': f'/static/uploads/{original_filename}',
                'mask': f'/static/results/{mask_filename}',
                'analysis': {
                    'probability': round(float(cancer_probability), 1),
                    'diagnosis': diagnosis,
                    'severity': severity,
                    'severity_level': severity_level,
                    'recommendation': recommendation,
                    'nodule_count': nodule_analysis['nodule_count'],
                    'affected_area': f"{round(nodule_analysis['affected_area'], 1)}%",
                    'timestamp': datetime.now().isoformat(),
                    'details': {
                        'total_pixels_analyzed': int(nodule_analysis['total_area']),
                        'affected_pixels': int(nodule_analysis['total_area'] * nodule_analysis['affected_area'] / 100)
                    }
                },
                'metadata': {
                    'model': 'UNETR-2D',
                    'version': '1.0.0',
                    'processing_time_ms': 0  # Would be calculated in a real implementation
                }
            }
            
            # Store latest analysis for chat bot
            global latest_analysis
            latest_analysis = response['analysis']
            
            return jsonify(response)
            
        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}',
                'details': str(e)
            }), 500
            
    except Exception as e:
        app.logger.error(f"Unexpected error in predict endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with AI assistant."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided', 'success': False}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Empty message', 'success': False}), 400
        
        # Get AI response
        ai_response = chat_bot.get_response(user_message, latest_analysis)
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your message',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
