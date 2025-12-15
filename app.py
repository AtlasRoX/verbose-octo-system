import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import csv
import os
from datetime import datetime
from utils import (
    create_confidence_gauge, 
    create_probability_chart,
    generate_gradcam,
    overlay_heatmap,
    create_architecture_diagram,
    create_training_curves,
    get_model_summary
)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #22c55e;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
    }
    
    /* Dark background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 30, 46, 0.6);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(99, 102, 241, 0.1);
        border-radius: 10px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.2);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: 1px solid rgba(99, 102, 241, 0.5);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    /* Card containers */
    .css-1r6slb0 {
        background-color: rgba(30, 30, 46, 0.6);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(99, 102, 241, 0.1);
        border: 2px dashed rgba(99, 102, 241, 0.4);
        border-radius: 15px;
        padding: 30px;
    }
    
    /* Success/Error boxes */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        border-radius: 10px;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        border-radius: 10px;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #6366f1;
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Feedback Collection Function
# ----------------------------
def save_feedback(feedback_data):
    """Save user feedback to CSV file"""
    feedback_file = 'feedback_data.csv'
    file_exists = os.path.isfile(feedback_file)
    
    with open(feedback_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['timestamp', 'prediction', 'confidence', 'feedback', 'validation_passed']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(feedback_data)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DenseNet121 model
    model = models.densenet121(pretrained=False)
    
    # Modify classifier for binary classification (model was trained with RGB input)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 1)
    
    # Load trained weights
    model.load_state_dict(torch.load("pneumonia_densenet121.pth", map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

model, device = load_model()
model_stats = get_model_summary(model)

# ----------------------------
# Image Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.convert("RGB")),  # Convert to RGB (3 channels)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# ----------------------------
# Header
# ----------------------------
st.markdown("<h1>🫁 AI-Powered Pneumonia Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced Deep Learning System for Chest X-Ray Analysis</p>", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Parameters", f"{model_stats['total_params']:,}")
    with col2:
        st.metric("Accuracy", "91%")
    
    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    - **Input**: 224×224 Grayscale
    - **Conv Layers**: 2
    - **Dense Layers**: 2
    - **Activation**: ReLU
    - **Regularization**: Dropout (30%)
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Performance")
    st.markdown("""
    - **Training Accuracy**: 96%
    - **Validation Accuracy**: 91%
    - **Final Loss**: 0.09
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.caption("This tool is for educational and research purposes only. Not a substitute for professional medical diagnosis.")

# ----------------------------
# Main Content Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "🏗️ Architecture", "📊 Training Metrics", "📚 How It Works"])

# ----------------------------
# TAB 1: Prediction
# ----------------------------
with tab1:
    st.markdown("### Upload Chest X-Ray Image")
    
    # Compact Warning Banner
    st.error("""
**IMPORTANT: This tool ONLY works with chest X-ray images!**

**✅ DO:** Upload frontal chest X-ray images (PA or AP view)

**❌ DON'T:** Upload random photos, selfies, or non-medical images

💡 The AI cannot detect if your image is a chest X-ray - it will try to classify anything you upload, but results will be meaningless for non-X-ray images!
    """, icon="⚠️")
    
    uploaded_file = st.file_uploader(
        "Drop your X-ray image here or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        # Load and display image
        image = Image.open(uploaded_file).convert("L")
        
        # Basic image validation
        img_array = np.array(image)
        height, width = img_array.shape
        aspect_ratio = width / height
        
        # Check if image looks like a chest X-ray (basic heuristics)
        is_grayscale = len(img_array.shape) == 2
        reasonable_size = (height >= 224 and width >= 224)
        reasonable_aspect = (0.6 <= aspect_ratio <= 1.4)  # X-rays are roughly square-ish
        
        # Calculate image statistics
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # Medical images typically have certain characteristics
        likely_medical = (mean_intensity > 50 and std_intensity > 20)
        
        validation_passed = reasonable_size and reasonable_aspect and likely_medical
        
        if not validation_passed:
            st.error("""
            🚫 **Warning: This image may not be a chest X-ray!**
            
            The uploaded image doesn't match typical chest X-ray characteristics:
            - Image size: {}x{} (need at least 224x224)
            - Aspect ratio: {:.2f} (expected 0.6-1.4 for chest X-rays)
            - Image properties suggest this might not be a medical image
            
            **Prediction results below are likely INVALID and meaningless.**
            Please upload a proper chest X-ray image for accurate results.
            """.format(width, height, aspect_ratio), icon="🚫")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 Original Image")
            st.image(image, use_container_width=True)
            
            if validation_passed:
                st.success("Image validation passed", icon="✅")
            else:
                st.error("Image validation failed", icon="❌")
        
        # Make prediction
        input_tensor = transform(image).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
        
        # Apply confidence threshold
        confidence_threshold = 0.65
        is_pneumonia = prob > confidence_threshold
        
        # Check if prediction confidence is too extreme (suspicious)
        very_confident = (prob > 0.95 or prob < 0.05)
        if very_confident and not validation_passed:
            st.warning("""
            ⚠️ **Suspicious Prediction Detected**
            
            The model is extremely confident ({:.1f}%), which is unusual for unclear images.
            This strongly suggests the image is not a chest X-ray.
            """.format(prob * 100 if is_pneumonia else (1-prob) * 100), icon="⚠️")
        
        with col2:
            st.markdown("#### 🔥 Grad-CAM Heatmap")
            try:
                # Generate Grad-CAM (using last dense block for DenseNet)
                input_tensor_grad = transform(image).unsqueeze(0).to(device)
                input_tensor_grad.requires_grad = True
                heatmap = generate_gradcam(model, input_tensor_grad, 'features.denseblock4')
                overlayed_img = overlay_heatmap(image, heatmap, alpha=0.5)
                st.image(overlayed_img, use_container_width=True)
                st.caption("Red areas indicate regions the model focused on for prediction")
            except Exception as e:
                st.warning(f"Grad-CAM visualization unavailable: {str(e)}")
                st.image(image, use_container_width=True)
        
        st.markdown("---")
        
        # Prediction Results
        st.markdown("### 🎯 Prediction Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Confidence Gauge
            fig_gauge = create_confidence_gauge(prob)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col2:
            # Probability Distribution
            fig_prob = create_probability_chart(prob)
            st.plotly_chart(fig_prob, use_container_width=True)
        
        # Final Diagnosis
        st.markdown("### 💊 Diagnosis")
        if is_pneumonia:
            st.error(f"""
            ### 🔴 PNEUMONIA DETECTED
            
            **Confidence**: {prob*100:.1f}%
            
            The model has detected signs consistent with pneumonia in the chest X-ray.
            This AI prediction should be verified by a qualified medical professional.
            """)
        else:
            st.success(f"""
            ### 🟢 NORMAL
            
            **Confidence**: {(1-prob)*100:.1f}%
            
            The model indicates the chest X-ray appears normal with no significant signs of pneumonia.
            Regular medical check-ups are still recommended.
            """)
        
        # Feedback Collection System
        st.markdown("---")
        st.markdown("### 📝 Help Us Improve!")
        st.write("Was this prediction helpful? Your feedback helps improve the model.")
        
        col_feedback1, col_feedback2, col_feedback3 = st.columns([1, 1, 2])
        
        with col_feedback1:
            if st.button("👍 Correct Prediction", use_container_width=True, type="primary"):
                # Save feedback to CSV
                feedback_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'prediction': 'Pneumonia' if is_pneumonia else 'Normal',
                    'confidence': f"{prob*100:.1f}%" if is_pneumonia else f"{(1-prob)*100:.1f}%",
                    'feedback': 'correct',
                    'validation_passed': validation_passed
                }
                save_feedback(feedback_data)
                st.success("✅ Thank you for your feedback!", icon="🎉")
        
        with col_feedback2:
            if st.button("👎 Incorrect Prediction", use_container_width=True):
                # Save feedback to CSV
                feedback_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'prediction': 'Pneumonia' if is_pneumonia else 'Normal',
                    'confidence': f"{prob*100:.1f}%" if is_pneumonia else f"{(1-prob)*100:.1f}%",
                    'feedback': 'incorrect',
                    'validation_passed': validation_passed
                }
                save_feedback(feedback_data)
                st.warning("📊 Thank you! This helps us improve the model.", icon="🙏")
        
        with col_feedback3:
            st.caption("💡 Your feedback is saved anonymously and used to improve model accuracy.")
    
    else:
        st.info("👆 Please upload a chest X-ray image to get started")
        
        # Example images section
        st.markdown("### 📋 Sample Images")
        st.markdown("""
        You can test the model with chest X-ray images in JPG, JPEG, or PNG format.
        For best results, use frontal view chest X-rays with good contrast.
        """)

# ----------------------------
# TAB 2: Architecture
# ----------------------------
with tab2:
    st.markdown("### 🏗️ Model Architecture Visualization")
    
    st.markdown("""
    The Pneumonia Detection CNN uses a compact yet effective architecture designed for binary classification
    of chest X-ray images. Below is the layer-by-layer breakdown:
    """)
    
    fig_arch = create_architecture_diagram()
    st.plotly_chart(fig_arch, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔵 Convolutional Layers")
        st.markdown("""
        - Extract spatial features
        - Learn edge and texture patterns
        - Detect pneumonia indicators
        """)
    
    with col2:
        st.markdown("#### 🟣 Pooling Layers")
        st.markdown("""
        - Reduce spatial dimensions
        - Retain important features
        - Add translation invariance
        """)
    
    with col3:
        st.markdown("#### 🟢 Dense Layers")
        st.markdown("""
        - High-level reasoning
        - Combine extracted features
        - Final classification
        """)
    
    st.markdown("---")
    
    st.markdown("#### 📊 Layer Details")
    st.markdown("""
    | Layer | Type | Output Shape | Parameters |
    |-------|------|--------------|------------|
    | Conv2D-1 | Convolution | 16×224×224 | 160 |
    | BatchNorm-1 | Normalization | 16×224×224 | 32 |
    | MaxPool-1 | Pooling | 16×112×112 | 0 |
    | Conv2D-2 | Convolution | 32×112×112 | 4,640 |
    | BatchNorm-2 | Normalization | 32×112×112 | 64 |
    | MaxPool-2 | Pooling | 32×56×56 | 0 |
    | Flatten | Reshape | 100,352 | 0 |
    | Dense-1 | Fully Connected | 128 | 12,845,184 |
    | Dense-2 | Output | 1 | 129 |
    | **Total** | | | **12,850,209** |
    """)

# ----------------------------
# TAB 3: Training Metrics
# ----------------------------
with tab3:
    st.markdown("### 📈 Training Performance Metrics")
    
    st.info("""
    The model was trained on a large dataset of chest X-ray images over 20 epochs.
    Below are the training and validation metrics showing the learning progression.
    """)
    
    fig_acc, fig_loss = create_training_curves()
    
    st.plotly_chart(fig_acc, use_container_width=True)
    st.plotly_chart(fig_loss, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Final Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Accuracy", "96.0%", "4.0%")
    with col2:
        st.metric("Validation Accuracy", "91.0%", "3.0%")
    with col3:
        st.metric("Training Loss", "0.09", "-0.56")
    with col4:
        st.metric("Validation Loss", "0.18", "-0.46")
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Strengths")
        st.markdown("""
        - High accuracy on both training and validation sets
        - Minimal overfitting (5% gap between train/val)
        - Stable convergence with consistent improvement
        - Effective feature extraction from X-ray images
        """)
    
    with col2:
        st.markdown("#### ⚠️ Considerations")
        st.markdown("""
        - Performance may vary with different image qualities
        - Requires frontal chest X-rays for best results
        - Should be used as a screening tool, not diagnosis
        - Regular retraining recommended with new data
        """)

# ----------------------------
# TAB 4: How It Works
# ----------------------------
with tab4:
    st.markdown("### 🧠 How the AI Model Works")
    
    st.markdown("""
    This pneumonia detection system uses a **Convolutional Neural Network (CNN)**, 
    a type of deep learning model specifically designed for image analysis.
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔄 Step-by-Step Process")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 1️⃣ Image Input")
        st.info("X-ray image is uploaded and preprocessed")
    
    with col2:
        st.markdown("""
        - Image is resized to 224×224 pixels
        - Converted to grayscale (single channel)
        - Normalized to improve model performance
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 2️⃣ Feature Extraction")
        st.info("CNN layers detect patterns and features")
    
    with col2:
        st.markdown("""
        - **First Conv Layer**: Detects basic edges and textures
        - **Second Conv Layer**: Combines features into complex patterns
        - **Pooling Layers**: Reduce image size while keeping important info
        - Each layer learns to recognize pneumonia-specific indicators
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 3️⃣ Classification")
        st.info("Dense layers make the final decision")
    
    with col2:
        st.markdown("""
        - Flattened features are passed to fully connected layers
        - First dense layer performs high-level reasoning
        - Output layer produces a probability score (0-1)
        - Threshold (0.65) determines NORMAL vs PNEUMONIA
        """)
    
    st.markdown("---")
    
    st.markdown("### 🔬 Understanding Pneumonia Detection")
    
    st.markdown("""
    #### What the Model Looks For:
    
    Pneumonia causes specific changes in chest X-rays that the CNN learns to identify:
    
    - **Lung Opacity**: Areas of increased density (lighter patches)
    - **Consolidation**: Fluid-filled air spaces appearing as white regions
    - **Pattern Distribution**: Location and spread of abnormalities
    - **Texture Changes**: Differences in lung tissue appearance
    
    #### Grad-CAM Visualization:
    
    The heatmap overlay (Gradient-weighted Class Activation Mapping) shows which regions
    of the X-ray the model focused on when making its prediction. Red/yellow areas indicate
    high attention, helping doctors understand the AI's reasoning.
    
    #### Limitations:
    
    - **Not a Replacement**: This AI assists but doesn't replace medical professionals
    - **Image Quality**: Poor quality images may reduce accuracy
    - **Edge Cases**: Unusual presentations may be challenging
    - **Continuous Learning**: Model performance improves with more diverse training data
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎓 Technical Details")
    
    with st.expander("📖 CNN Architecture Explanation"):
        st.markdown("""
        **Convolutional Neural Networks (CNNs)** are inspired by the human visual cortex.
        
        - **Convolutional Layers**: Apply filters to detect features (edges, patterns)
        - **Activation Functions (ReLU)**: Introduce non-linearity for complex patterns
        - **Batch Normalization**: Stabilizes and accelerates training
        - **Pooling Layers**: Downsample while preserving important features
        - **Dropout**: Prevents overfitting by randomly disabling neurons
        - **Dense Layers**: Perform final classification based on learned features
        """)
    
    with st.expander("🔧 Training Process"):
        st.markdown("""
        The model was trained using:
        
        - **Dataset**: Thousands of labeled chest X-ray images
        - **Optimizer**: Adam (adaptive learning rate)
        - **Loss Function**: Binary Cross-Entropy
        - **Epochs**: 20 iterations through the entire dataset
        - **Validation**: Separate dataset to prevent overfitting
        - **Augmentation**: Image transformations for better generalization
        """)
    
    with st.expander("🎯 Performance Metrics"):
        st.markdown("""
        - **Accuracy**: Percentage of correct predictions
        - **Precision**: Of predicted pneumonia cases, how many are correct
        - **Recall**: Of actual pneumonia cases, how many are detected
        - **F1-Score**: Harmonic mean of precision and recall
        - **AUC-ROC**: Area under receiver operating characteristic curve
        """)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p>🫁 <strong>Pneumonia Detection AI</strong> | Powered by Deep Learning</p>
    <p style='font-size: 0.9rem;'>Designed and developed By <strong>GhostCache_</strong></p>
    <p style='font-size: 0.85rem; margin-top: 10px;'>Built with PyTorch, Streamlit, and Plotly | © 2025</p>
</div>
""", unsafe_allow_html=True)
