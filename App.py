import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import albumentations as A
import plotly.graph_objects as go
import plotly.express as px
import json
import io
import time
from datetime import datetime

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
class Config:
    IMG_SIZE = 224
    CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
    NUM_CLASSES = len(CLASS_NAMES)

# Advanced CSS with modern animations and glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Animated Header */
    .main-header {
        font-size: 4rem;
        font-weight: 700;
        color: transparent;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        animation: gradientShift 4s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        border-radius: 20px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .glass-card:hover::before {
        opacity: 0.3;
    }
    
    .glass-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
    }
    
    /* Prediction Result with Advanced Animations */
    .prediction-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        backdrop-filter: blur(30px);
        padding: 2rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        animation: float 6s ease-in-out infinite;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .prediction-box h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Status Boxes with Gradient Borders */
    .status-box {
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1rem 0;
        position: relative;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 2px solid rgba(34, 197, 94, 0.3);
        color: #10b981;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        border: 2px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
    }
    
    .confidence-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(29, 78, 216, 0.1) 100%);
        border: 2px solid rgba(59, 130, 246, 0.3);
        color: #3b82f6;
    }
    
    /* Animated Progress Bars */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 8px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    .progress-bar {
        height: 25px;
        border-radius: 12px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 2s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Floating Action Buttons */
    .fab {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        animation: bounce 2s infinite;
    }
    
    .fab:hover {
        transform: scale(1.1);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.6);
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    /* Loading Spinner */
    .loading-spinner {
        border: 4px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Sidebar Enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(20px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
    }
    
    /* Button Enhancements */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* File Uploader Styling */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Notification Animation */
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        animation: slideInRight 0.5s ease-out;
        z-index: 1000;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Image Container with Hover Effects */
    .image-container {
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
    }
    
    .image-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .image-container:hover .image-overlay {
        opacity: 1;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
</style>
""", unsafe_allow_html=True)

# Refined Focal Loss class (needed for model loading)
@keras.utils.register_keras_serializable()
class RefinedFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, name='refined_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
    
    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
        p_t = tf.reduce_sum(y_true * y_pred, axis=1)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        focal_loss = focal_weight * ce
        return tf.reduce_mean(focal_loss)

def create_animated_loader():
    """Create an animated loading indicator"""
    return st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100px;">
        <div class="loading-spinner"></div>
    </div>
    """, unsafe_allow_html=True)

def show_notification(message, type="info"):
    """Show animated notification"""
    color_map = {
        "success": "#10b981",
        "error": "#ef4444",
        "warning": "#f59e0b",
        "info": "#3b82f6"
    }
    
    return st.markdown(f"""
    <div class="notification" style="background: {color_map.get(type, '#3b82f6')};">
        {message}
    </div>
    """, unsafe_allow_html=True)

def get_augmentation():
    """Get validation augmentation pipeline"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image, img_size=224):
    """Preprocess image for model prediction with improved handling"""
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in correct format
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 1:  # Single channel
                image = np.squeeze(image, axis=2)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize image
        image = cv2.resize(image, (img_size, img_size))
        
        # Apply augmentation (normalization)
        aug = get_augmentation()
        augmented = aug(image=image)
        image = augmented['image']
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def find_target_layer(model):
    """Find the best layer for Grad-CAM visualization with improved detection"""
    suitable_layers = []
    
    # Look for convolutional layers first
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            suitable_layers.append(layer.name)
        elif hasattr(layer, 'layers'):  # For layers that contain other layers
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    suitable_layers.append(sublayer.name)
    
    # If no Conv2D found, look for layers with 4D output
    if not suitable_layers:
        for layer in reversed(model.layers):
            try:
                if hasattr(layer, 'output_shape'):
                    output_shape = layer.output_shape
                    if isinstance(output_shape, list) and len(output_shape) > 0:
                        output_shape = output_shape[0] if isinstance(output_shape[0], tuple) else output_shape
                    if isinstance(output_shape, tuple) and len(output_shape) == 4:
                        suitable_layers.append(layer.name)
            except:
                continue
    
    return suitable_layers[0] if suitable_layers else None

def generate_gradcam_improved(model, image, class_idx, layer_name=None):
    """Generate Grad-CAM heatmap with robust error handling"""
    try:
        # Ensure image is in correct format
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Find suitable layer if not specified
        if layer_name is None:
            layer_name = find_target_layer(model)
            if layer_name is None:
                st.error("No suitable layer found for Grad-CAM visualization")
                return generate_gradcam_simple_alternative(model, image, class_idx)
        
        st.info(f"Using layer: {layer_name} for Grad-CAM")
        
        # Verify layer exists
        try:
            target_layer = model.get_layer(layer_name)
        except ValueError:
            st.error(f"Layer '{layer_name}' not found in model")
            return generate_gradcam_simple_alternative(model, image, class_idx)
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(image)
            model_outputs = grad_model(image)
            
            # Handle different output formats
            if isinstance(model_outputs, list) and len(model_outputs) == 2:
                conv_outputs, predictions = model_outputs
            else:
                st.error("Unexpected model output format")
                return generate_gradcam_simple_alternative(model, image, class_idx)
            
            # Handle case where conv_outputs might be a list
            if isinstance(conv_outputs, list):
                if len(conv_outputs) > 0:
                    conv_outputs = conv_outputs[0]  # Take first element
                else:
                    st.error("Empty conv_outputs list")
                    return generate_gradcam_simple_alternative(model, image, class_idx)
            
            # Ensure conv_outputs is a tensor
            if not tf.is_tensor(conv_outputs):
                conv_outputs = tf.convert_to_tensor(conv_outputs)
            
            # Get the score for the predicted class
            if isinstance(predictions, list):
                predictions = predictions[0] if len(predictions) > 0 else predictions
            
            if not tf.is_tensor(predictions):
                predictions = tf.convert_to_tensor(predictions)
            
            if class_idx < predictions.shape[-1]:
                loss = predictions[:, class_idx]
            else:
                st.error(f"Class index {class_idx} out of range")
                return generate_gradcam_simple_alternative(model, image, class_idx)
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            st.error("Could not compute gradients - trying alternative approach")
            return generate_gradcam_simple_alternative(model, image, class_idx)
        
        # Handle case where grads might be a list
        if isinstance(grads, list):
            if len(grads) > 0:
                grads = grads[0]
            else:
                st.error("Empty gradients list")
                return generate_gradcam_simple_alternative(model, image, class_idx)
        
        # Ensure grads is a tensor
        if not tf.is_tensor(grads):
            grads = tf.convert_to_tensor(grads)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps by pooled gradients
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        
        # Ensure dimensions match
        if len(pooled_grads.shape) > 0 and len(conv_outputs.shape) > 2:
            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        else:
            st.error("Dimension mismatch in Grad-CAM computation")
            return generate_gradcam_simple_alternative(model, image, class_idx)
        
        # Apply ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        return heatmap.numpy()
    
    except Exception as e:
        st.error(f"Grad-CAM generation failed: {str(e)}")
        return generate_gradcam_simple_alternative(model, image, class_idx)

def generate_gradcam_simple_alternative(model, image, class_idx):
    """Simple alternative visualization using prediction differences"""
    try:
        st.info("Using simple attention visualization based on input perturbation...")
        
        # Ensure image is tensor
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Get original prediction
        original_pred = model(image)[0][class_idx]
        
        # Create attention map by occlusion
        image_np = image.numpy()[0]
        h, w = image_np.shape[:2]
        patch_size = max(8, min(h, w) // 16)  # Adaptive patch size
        
        attention_map = np.zeros((h, w))
        
        # Grid of patches to occlude
        step_size = patch_size // 2
        for i in range(0, h - patch_size + 1, step_size):
            for j in range(0, w - patch_size + 1, step_size):
                # Create occluded image
                occluded_image = image_np.copy()
                occluded_image[i:i+patch_size, j:j+patch_size] = 0
                
                # Get prediction for occluded image
                occluded_tensor = tf.expand_dims(tf.convert_to_tensor(occluded_image), 0)
                occluded_pred = model(occluded_tensor)[0][class_idx]
                
                # Calculate importance (drop in prediction)
                importance = max(0, original_pred - occluded_pred)
                attention_map[i:i+patch_size, j:j+patch_size] += importance
        
        # Normalize attention map
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        
        # Smooth the attention map
        from scipy import ndimage
        attention_map = ndimage.gaussian_filter(attention_map, sigma=1.0)
        
        return attention_map
    
    except Exception as e:
        st.error(f"Simple alternative visualization also failed: {str(e)}")
        # Return a dummy heatmap as last resort
        if isinstance(image, tf.Tensor):
            shape = image.shape[1:3]  # Get H, W from (batch, H, W, C)
        else:
            shape = image.shape[:2] if len(image.shape) >= 2 else (224, 224)
        
        return np.random.rand(*shape) * 0.1  # Very faint random pattern

def generate_gradcam_alternative(model, image, class_idx):
    """Alternative Grad-CAM approach using integrated gradients concept"""
    try:
        st.info("Trying integrated gradients approach...")
        
        # Ensure image is tensor
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Get model prediction function
        def predict_fn(x):
            preds = model(x)
            if isinstance(preds, list):
                preds = preds[0] if len(preds) > 0 else preds
            return preds[:, class_idx]
        
        # Create baseline (black image)
        baseline = tf.zeros_like(image)
        
        # Create interpolated images
        num_steps = 50
        alphas = tf.linspace(0.0, 1.0, num_steps)
        
        interpolated_images = []
        for alpha in alphas:
            interpolated = baseline + alpha * (image - baseline)
            interpolated_images.append(interpolated)
        
        interpolated_images = tf.concat(interpolated_images, axis=0)
        
        # Get gradients for interpolated images
        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            target_class_predictions = predict_fn(interpolated_images)
        
        grads = tape.gradient(target_class_predictions, interpolated_images)
        
        if grads is not None:
            # Handle case where grads might be a list
            if isinstance(grads, list):
                grads = grads[0] if len(grads) > 0 else None
            
            if grads is not None:
                # Average gradients and create heatmap
                avg_grads = tf.reduce_mean(grads, axis=0)
                
                # Create attention map by taking absolute values and averaging across channels
                if len(avg_grads.shape) == 3:  # (H, W, C)
                    attention_map = tf.reduce_mean(tf.abs(avg_grads), axis=-1)
                else:
                    attention_map = tf.abs(avg_grads)
                
                # Normalize
                max_val = tf.reduce_max(attention_map)
                if max_val > 0:
                    attention_map = attention_map / max_val
                
                return attention_map.numpy()
        
        # If all else fails, use the simple alternative
        return generate_gradcam_simple_alternative(model, image, class_idx)
            
    except Exception as e:
        st.error(f"Integrated gradients approach failed: {str(e)}")
        return generate_gradcam_simple_alternative(model, image, class_idx)

def overlay_heatmap_robust(original_image, heatmap, alpha=0.4):
    """Robust heatmap overlay with better error handling"""
    try:
        # Ensure original image is in correct format
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Handle different image formats
        if len(original_image.shape) == 2:  # Grayscale
            original_image = np.stack((original_image,)*3, axis=-1)
        elif len(original_image.shape) == 3:
            if original_image.shape[2] == 4:  # RGBA
                original_image = original_image[..., :3]
            elif original_image.shape[2] == 1:  # Single channel
                original_image = np.concatenate([original_image]*3, axis=-1)
        # Ensure uint8 format
        if original_image.dtype != np.uint8:
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            else:
                original_image = original_image.astype(np.uint8)
        
        # Resize heatmap to match original image
        if heatmap.shape != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Normalize heatmap to 0-255
        heatmap_norm = np.uint8(255 * heatmap)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images
        overlayed = cv2.addWeighted(original_image, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlayed
    
    except Exception as e:
        st.error(f"Heatmap overlay failed: {str(e)}")
        return original_image

def create_confidence_chart(predictions, class_names):
    """Create interactive confidence chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=predictions,
            marker_color=['#ef4444' if pred == max(predictions) else '#6b7280' for pred in predictions],
            text=[f'{pred:.1%}' for pred in predictions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence by Class",
        xaxis_title="Diabetic Retinopathy Stage",
        yaxis_title="Confidence",
        yaxis=dict(tickformat='.0%'),
        height=400,
        showlegend=False
    )
    
    return fig

def get_severity_info(class_name):
    """Get information about each severity level"""
    info = {
        'No_DR': {
            'description': 'No signs of diabetic retinopathy detected.',
            'recommendations': 'Continue regular eye exams and maintain good diabetes control.',
            'color': '#10b981',
            'urgency': 'Routine'
        },
        'Mild': {
            'description': 'Mild non-proliferative diabetic retinopathy. Small areas of bleeding in the retina.',
            'recommendations': 'Monitor closely. Follow up with eye care professional within 6-12 months.',
            'color': '#f59e0b',
            'urgency': 'Monitor'
        },
        'Moderate': {
            'description': 'Moderate non-proliferative diabetic retinopathy. More blood vessels are blocked.',
            'recommendations': 'Requires medical attention. Follow up within 3-6 months.',
            'color': '#f97316',
            'urgency': 'Medical Attention'
        },
        'Severe': {
            'description': 'Severe non-proliferative diabetic retinopathy. Many blood vessels are blocked.',
            'recommendations': 'Urgent medical attention required. Risk of progression to proliferative stage.',
            'color': '#dc2626',
            'urgency': 'Urgent'
        },
        'Proliferate_DR': {
            'description': 'Proliferative diabetic retinopathy. New blood vessels are growing in the retina.',
            'recommendations': 'Immediate medical treatment required. Risk of severe vision loss.',
            'color': '#7c2d12',
            'urgency': 'Emergency'
        }
    }
    return info.get(class_name, {})

@st.cache_resource
def load_model_safe(model_path):
    """Safely load the trained model with comprehensive error handling"""
    try:
        # Register custom objects
        custom_objects = {
            'RefinedFocalLoss': RefinedFocalLoss,
            'refined_focal_loss': RefinedFocalLoss
        }
        
        # Try loading with custom objects
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        # Recompile the model to ensure it works
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Trying to load without custom objects...")
        
        try:
            # Try loading without custom objects
            model = keras.models.load_model(model_path, compile=False)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e2:
            st.error(f"Failed to load model even without custom objects: {str(e2)}")
            return None

def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. Upload a retinal fundus image
    2. Wait for the AI analysis
    3. Review the prediction and confidence scores
    4. Examine the attention visualization
    5. Consult with a medical professional for diagnosis
    """)
    
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    This AI model analyzes retinal images to detect diabetic retinopathy severity levels:
    - **No DR**: No diabetic retinopathy
    - **Mild**: Mild non-proliferative DR
    - **Moderate**: Moderate non-proliferative DR  
    - **Severe**: Severe non-proliferative DR
    - **Proliferative**: Proliferative DR
    """)
    
    st.sidebar.warning("⚠️ This tool is for educational purposes only. Always consult a medical professional for diagnosis.")
    
    # Model upload section
    st.header("🔧 Model Setup")
    model_file = st.file_uploader(
        "Upload your trained model (.h5 file)", 
        type=['h5'],
        help="Upload the trained diabetic retinopathy detection model"
    )
    
    if model_file is not None:
        # Save uploaded model temporarily
        try:
            with open("temp_model.h5", "wb") as f:
                f.write(model_file.getbuffer())
            
            # Load model
            with st.spinner("Loading model..."):
                model = load_model_safe("temp_model.h5")
            
            if model is not None:
                st.success("✅ Model loaded successfully!")
                
                # Display model info
                with st.expander("📊 Model Information"):
                    st.write(f"**Input Shape:** {model.input_shape}")
                    st.write(f"**Output Shape:** {model.output_shape}")
                    st.write(f"**Total Parameters:** {model.count_params():,}")
                    st.write(f"**Number of Layers:** {len(model.layers)}")
                
                # Image upload section
                st.header("📸 Upload Retinal Image")
                uploaded_file = st.file_uploader(
                    "Choose a retinal fundus image...", 
                    type=['jpg', 'jpeg', 'png', 'tiff'],
                    help="Upload a clear retinal fundus photograph"
                )
                
                if uploaded_file is not None:
                    # Display uploaded image
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📷 Original Image")
                        image = Image.open(uploaded_file)
                        original_image = np.array(image.convert('RGB'))
                        st.image(image, caption="Uploaded Retinal Image", use_column_width=True)
                    
                    # Preprocess and predict
                    with st.spinner("🔍 Analyzing image..."):
                        processed_image = preprocess_image(image, Config.IMG_SIZE)
                        
                        if processed_image is not None:
                            predictions = model.predict(processed_image, verbose=0)
                            predicted_class = np.argmax(predictions[0])
                            confidence = np.max(predictions[0])
                            class_name = Config.CLASS_NAMES[predicted_class]
                        else:
                            st.error("Failed to preprocess image")
                            return
                    
                    with col2:
                        st.subheader("🎯 Prediction Results")
                        
                        # Main prediction
                        severity_info = get_severity_info(class_name)
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>Predicted Class: {class_name.replace('_', ' ')}</h2>
                            <h3>Confidence: {confidence:.2%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Severity information
                        if severity_info:
                            box_class = "warning-box" if severity_info['urgency'] in ['Urgent', 'Emergency'] else "confidence-box"
                            st.markdown(f"""
                            <div class="{box_class}">
                                <h4>Description:</h4>
                                <p>{severity_info['description']}</p>
                                <h4>Recommendations:</h4>
                                <p>{severity_info['recommendations']}</p>
                                <h4>Urgency Level: {severity_info['urgency']}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Confidence chart
                    st.subheader("📊 Detailed Confidence Scores")
                    confidence_chart = create_confidence_chart(predictions[0], Config.CLASS_NAMES)
                    st.plotly_chart(confidence_chart, use_container_width=True)
                    
                    # Model attention visualization
                    st.subheader("🔥 Model Attention Visualization")
                    st.info("""
                    This visualization shows regions that influenced the model's decision:
                    - 🔴 Red areas: High model attention (critical regions)
                    - 🔵 Blue areas: Low model attention
                    """)
                    
                    # Generate visualization
                    viz_option = st.selectbox(
                        "Choose visualization method:",
                        ["Grad-CAM", "Alternative (Integrated Gradients-like)"],
                        help="Select the type of attention visualization"
                    )
                    
                    if st.button("🔍 Generate Visualization"):
                        with st.spinner("Generating attention visualization..."):
                            if viz_option == "Grad-CAM":
                                heatmap = generate_gradcam_improved(model, processed_image, predicted_class)
                            else:
                                heatmap = generate_gradcam_alternative(model, processed_image, predicted_class)
                        
                        if heatmap is not None:
                            col3, col4 = st.columns([1, 1])
                            
                            with col3:
                                st.subheader("🌡️ Attention Heatmap")
                                fig, ax = plt.subplots(figsize=(8, 8))
                                im = ax.imshow(heatmap, cmap='jet', alpha=0.8)
                                ax.axis('off')
                                ax.set_title('Model Attention Heatmap')
                                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                                st.pyplot(fig)
                                plt.close()
                            
                            with col4:
                                st.subheader("🔍 Overlay on Original Image")
                                overlay = overlay_heatmap_robust(original_image, heatmap)
                                fig, ax = plt.subplots(figsize=(8, 8))
                                ax.imshow(overlay)
                                ax.axis('off')
                                ax.set_title('Diagnostic Focus Areas')
                                st.pyplot(fig)
                                plt.close()
                        else:
                            st.warning("Could not generate attention visualization. This might be due to model architecture incompatibility.")
                    
                    # Additional analysis
                    st.subheader("📈 Risk Analysis")
                    
                    # Calculate risk score
                    risk_weights = [0, 0.25, 0.5, 0.75, 1.0]  # Risk weights for each class
                    risk_score = sum(pred * weight for pred, weight in zip(predictions[0], risk_weights))
                    
                    col5, col6, col7 = st.columns(3)
                    
                    with col5:
                        st.metric("Risk Score", f"{risk_score:.3f}", help="Overall risk assessment (0-1 scale)")
                    
                    with col6:
                        st.metric("Highest Confidence", f"{confidence:.2%}", help="Confidence in predicted class")
                    
                    with col7:
                        second_highest = np.partition(predictions[0], -2)[-2]
                        st.metric("Second Highest", f"{second_highest:.2%}", help="Second most likely class confidence")
                    
                    # Warning for high-risk cases
                    if predicted_class >= 3:  # Severe or Proliferative
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ High Risk Detection</h4>
                            <p>The model has detected signs of advanced diabetic retinopathy. 
                            Immediate consultation with an ophthalmologist is strongly recommended.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif predicted_class == 0:
                        st.markdown("""
                        <div class="success-box">
                            <h4>✅ No Diabetic Retinopathy Detected</h4>
                            <p>Continue regular eye examinations and maintain good diabetes control.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Export results
                    st.subheader("💾 Export Results")
                    results_dict = {
                        'predicted_class': class_name,
                        'confidence': float(confidence),
                        'risk_score': float(risk_score),
                        'urgency_level': severity_info.get('urgency', 'Unknown'),
                        'all_predictions': {Config.CLASS_NAMES[i]: float(pred) for i, pred in enumerate(predictions[0])},
                        'model_info': {
                            'input_shape': str(model.input_shape),
                            'total_parameters': int(model.count_params())
                        }
                    }
                    
                    st.json(results_dict)
                    
                    # Download results
                    results_json = json.dumps(results_dict, indent=2)
                    st.download_button(
                        label="📄 Download Results (JSON)",
                        data=results_json,
                        file_name=f"dr_analysis_{class_name}.json",
                        mime="application/json"
                    )
            
            else:
                st.error("❌ Failed to load model. Please check the model file format and try again.")
        
        except Exception as e:
            st.error(f"❌ Error handling model file: {str(e)}")
    
    else:
        st.info("👆 Please upload your trained model to begin analysis.")
        
        # Show example of how to use
        st.subheader("🔧 How to get your model file")
        st.markdown("""
        1. Train your model using the provided training code
        2. The model will be saved as 'dr_enhanced_best.h5'
        3. Upload that file using the file uploader above
        4. Start analyzing retinal images!
        
        **Model Requirements:**
        - Input shape: (224, 224, 3)
        - Output: 5 classes (No_DR, Mild, Moderate, Severe, Proliferate_DR)
        - Format: Keras .h5 file
        """)

if __name__ == "__main__":
    main()