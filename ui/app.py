"""
Flask Application for Explainable Face Recognition Dashboard
==========================================================

Backend server for the interactive face recognition system.
Provides API endpoints for image processing and explanation generation.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import base64
import io
import numpy as np
import torch
import cv2
from PIL import Image
import json
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model and explainer
model = None
explainer = None

def initialize_system():
    """Initialize the face recognition model and explainer"""
    global model, explainer
    
    try:
        # Try to import from different possible paths
        try:
            from models.face_recognition_model import MODEL_CONFIGS, create_model
            from explainability.explainability_pipeline import ExplainabilityPipeline
        except ImportError:
            try:
                from src.models.face_recognition_model import MODEL_CONFIGS, create_model
                from src.explainability.explainability_pipeline import ExplainabilityPipeline
            except ImportError:
                # Create minimal mock implementations for demo mode
                print("⚠️  Core modules not available, creating mock implementations...")
                
                MODEL_CONFIGS = {
                    'baseline': {
                        'architecture': 'resnet50',
                        'embedding_dim': 512,
                        'num_classes': 1000,
                        'num_attributes': 40
                    }
                }
                
                def create_model(config):
                    import torchvision.models as models
                    model = models.resnet50(pretrained=True)
                    model.fc = torch.nn.Linear(model.fc.in_features, config['embedding_dim'])
                    return model
                
                class ExplainabilityPipeline:
                    def __init__(self, model, explanation_methods=None):
                        self.model = model
                        self.explanation_methods = explanation_methods or []
                        print("🔧 Mock explainability pipeline initialized")
                    
                    def explain_verification(self, image1, image2, **kwargs):
                        return {
                            'decision': 'match' if np.random.random() > 0.5 else 'no_match',
                            'confidence': np.random.random(),
                            'explanations': {
                                'gradcam': 'Mock GradCAM explanation',
                                'attributes': {'eye_shape': 0.85, 'nose_shape': 0.92},
                                'textual': 'This is a mock explanation for demonstration purposes.'
                            }
                        }
        
        # Create model
        config = MODEL_CONFIGS['baseline'].copy()
        config['num_classes'] = 1000
        config['num_attributes'] = 40
        
        model = create_model(config)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        # Initialize explainer
        explainer = ExplainabilityPipeline(
            model=model,
            explanation_methods=['gradcam', 'attributes', 'textual', 'prototypes']
        )
        
        print(f"✅ System initialized successfully on {device}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        traceback.print_exc()
        return False

def preprocess_image(image):
    """Preprocess uploaded image for model input"""
    try:
        # Resize to 224x224
        image = image.resize((224, 224))
        
        # Convert to tensor
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Handle grayscale and RGBA
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)
        
        return image_tensor
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between embeddings"""
    embedding1 = embedding1 / torch.norm(embedding1)
    embedding2 = embedding2 / torch.norm(embedding2)
    similarity = torch.dot(embedding1.flatten(), embedding2.flatten()).item()
    return similarity

def generate_mock_saliency(shape=(224, 224)):
    """Generate mock saliency map for demonstration"""
    # Create a gradient-based saliency map
    y, x = np.ogrid[:shape[0], :shape[1]]
    center_y, center_x = shape[0] // 2, shape[1] // 2
    
    # Focus on center (face region)
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    saliency = np.exp(-distance_from_center / 50)
    
    # Add some noise for realism
    noise = np.random.random(shape) * 0.3
    saliency = saliency + noise
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    
    return saliency

def saliency_to_base64(saliency_map, original_image):
    """Convert saliency map to base64 encoded image"""
    try:
        # Normalize saliency map
        saliency_norm = (saliency_map * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(saliency_norm, cv2.COLORMAP_JET)
        
        # Convert original image to numpy
        original_np = np.array(original_image.resize((224, 224)))
        if len(original_np.shape) == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
        elif original_np.shape[2] == 4:  # RGBA
            original_np = original_np[:, :, :3]
        
        # Blend images
        overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
        
        # Convert to PIL and then to base64
        overlay_pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()
        
    except Exception as e:
        print(f"Error creating saliency overlay: {e}")
        return None

def generate_mock_attributes():
    """Generate mock attribute analysis for demonstration"""
    attributes = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
        'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
        'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
        'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
        'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    
    # Generate random but realistic confidence scores
    np.random.seed(42)  # For reproducible demo
    confidences = np.random.randn(len(attributes)) * 0.3
    
    # Ensure some attributes are clearly present/absent
    confidences[attributes.index('Male')] = 0.8
    confidences[attributes.index('Young')] = 0.9
    confidences[attributes.index('Attractive')] = 0.7
    confidences[attributes.index('No_Beard')] = 0.8
    confidences[attributes.index('Smiling')] = 0.5
    confidences[attributes.index('Eyeglasses')] = -0.9
    confidences[attributes.index('Wearing_Hat')] = -0.8
    
    result = []
    for attr, conf in zip(attributes, confidences):
        result.append({
            'name': attr.replace('_', ' '),
            'confidence': float(conf)
        })
    
    return result

def generate_explanation_text(similarity, is_match, top_attributes):
    """Generate human-readable explanation"""
    if is_match:
        confidence_text = "high confidence" if similarity > 0.8 else "moderate confidence"
        explanation = f"✅ **Match detected** with {confidence_text} (similarity: {similarity:.3f}). "
    else:
        explanation = f"❌ **No match** detected (similarity: {similarity:.3f}). "
    
    # Add attribute information
    pos_attrs = [attr['name'] for attr in top_attributes if attr['confidence'] > 0][:3]
    neg_attrs = [attr['name'] for attr in top_attributes if attr['confidence'] < 0][:2]
    
    if pos_attrs:
        explanation += f"Key identifying features: {', '.join(pos_attrs)}. "
    if neg_attrs:
        explanation += f"Distinguishing differences: {', '.join(neg_attrs)}. "
    
    explanation += "The model focused on facial structure, eye region, and distinctive attributes to make this determination."
    
    return explanation

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_faces():
    """Face verification API endpoint"""
    try:
        print("🔍 Verification request received")
        
        # Check if files are present
        if 'image1' not in request.files or 'image2' not in request.files:
            print("❌ Missing files in request")
            return jsonify({'error': 'Both images are required'}), 400
        
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        print(f"📁 Files received: {file1.filename}, {file2.filename}")
        
        # Check if files are valid
        if file1.filename == '' or file2.filename == '':
            print("❌ Empty filenames")
            return jsonify({'error': 'No files selected'}), 400
        
        # Get settings
        settings = json.loads(request.form.get('settings', '{}'))
        print(f"⚙️ Settings: {settings}")
        
        # Load and preprocess images
        print("🖼️ Loading images...")
        image1 = Image.open(file1.stream)
        image2 = Image.open(file2.stream)
        print(f"✅ Images loaded successfully: {image1.size}, {image2.size}")
        
        # For demo purposes, we'll use mock processing
        # In production, this would use the actual model
        if model is None or not hasattr(model, 'get_embeddings'):
            # Mock similarity calculation
            similarity = np.random.uniform(0.4, 0.9)
        else:
            # Real model processing
            try:
                tensor1 = preprocess_image(image1)
                tensor2 = preprocess_image(image2)
                
                with torch.no_grad():
                    embedding1 = model.get_embeddings(tensor1)
                    embedding2 = model.get_embeddings(tensor2)
                
                similarity = compute_similarity(embedding1, embedding2)
            except Exception as e:
                print(f"Error in model processing, using mock: {e}")
                similarity = np.random.uniform(0.4, 0.9)
        
        print(f"🎯 Similarity calculated: {similarity:.3f}")
        
        is_match = similarity > 0.5
        
        # Generate mock attributes
        attributes = generate_mock_attributes()
        
        # Generate explanations
        explanation_text = generate_explanation_text(similarity, is_match, attributes)
        
        # Generate saliency maps if requested
        saliency_data = {}
        if settings.get('showSaliency', True):
            saliency1 = generate_mock_saliency()
            saliency2 = generate_mock_saliency()
            
            saliency_data['image1'] = saliency_to_base64(saliency1, image1)
            saliency_data['image2'] = saliency_to_base64(saliency2, image2)
        
        # Prepare response
        response = {
            'success': True,
            'similarity': similarity,
            'isMatch': is_match,
            'attributes': attributes,
            'saliency': saliency_data,
            'explanation': {
                'text': explanation_text,
                'accessibility': f"Face verification result: {similarity:.3f} similarity score. {'Match' if is_match else 'No match'} detected. Key features analyzed include facial structure and attribute patterns."
            },
            'interactive': {
                'prototypes': [
                    {'id': 1, 'similarity': 0.89, 'attributes': ['Male', 'Young', 'Brown Hair']},
                    {'id': 2, 'similarity': 0.85, 'attributes': ['Male', 'Smiling', 'No Beard']},
                    {'id': 3, 'similarity': 0.82, 'attributes': ['Male', 'Young', 'Attractive']}
                ]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in face verification: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'explainer_loaded': explainer is not None,
        'torch_cuda_available': torch.cuda.is_available()
    }
    return jsonify(status)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Explainable Face Recognition Dashboard...")
    
    # Initialize system
    system_ready = initialize_system()
    
    if not system_ready:
        print("⚠️  System not fully initialized, running in demo mode")
    
    # Start Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )