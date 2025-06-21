import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import io

gdrive_url = "https://drive.google.com/file/d/1EPL1P53QVTI8qqC0oZjnJqXXghStKVBQ/view?usp=drive_link"

import requests
import tempfile
from tensorflow.keras.models import load_model

def gdrive_to_direct_url(gdrive_url):
    file_id = gdrive_url.split("/d/")[1].split("/")[0]
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def load_model_from_gdrive(gdrive_url):
    direct_url = gdrive_to_direct_url(gdrive_url)
    response = requests.get(direct_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
        tmp_file.write(response.content)
        return load_model(tmp_file.name)



# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #333;
    }
    .positive-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        color: #d32f2f;
    }
    .negative-result {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
        color: #333;
        border: 1px solid #e0e0e0;
    }
    .info-box h3 {
        color: #1f77b4;
        margin-top: 0;
    }
    .info-box p, .info-box ul, .info-box li {
        color: #555;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
        color: #333;
    }
    .metric-card h2 {
        color: #1f77b4;
        margin: 0.5rem 0;
    }
    .metric-card h3 {
        color: #666;
        margin: 0;
    }
    .metric-card p {
        color: #666;
        margin: 0;
    }
    .result-box h3, .result-box h4 {
        margin-top: 0;
    }
    .result-box p {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model function (you'll need to modify this based on your model)
@st.cache_resource
@st.cache_resource
def load_brain_tumor_model():
    try:
        gdrive_url = "https://drive.google.com/file/d/1EPL1P53QVTI8qqC0oZjnJqXXghStKVBQ/view?usp=drive_link"
        model = load_model_from_gdrive(gdrive_url)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Image preprocessing function
def preprocess_image(img, target_size=(128, 128)):
    """
    Preprocess the image for model prediction
    Modify this function based on your model's requirements
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Prediction function
def predict_tumor(model, img_array):
    """
    Make prediction using the loaded model
    Modify this based on your model's output format
    """
    try:
        prediction = model.predict(img_array)
        
        # Assuming binary classification (tumor/no tumor)
        # Modify this based on your model's output
        if len(prediction[0]) == 1:  # Binary classification
            confidence = float(prediction[0][0])
            if confidence > 0.5:
                result = "Tumor Detected"
                probability = confidence * 100
            else:
                result = "No Tumor Detected"
                probability = (1 - confidence) * 100
        else:  # Multi-class classification
            class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']  # Modify as needed
            predicted_class = np.argmax(prediction[0])
            result = class_names[predicted_class]
            probability = float(np.max(prediction[0]) * 100)
        
        return result, probability, prediction[0]
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Detection", "About", "Statistics"])
    
    if page == "Home":
        show_home_page()
    elif page == "Detection":
        show_detection_page()
    elif page == "About":
        show_about_page()
    elif page == "Statistics":
        show_statistics_page()

def show_home_page():
    st.markdown('<h2 class="sub-header">Welcome to Brain Tumor Detection System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Our Mission</h3>
        <p>To provide accurate and efficient brain tumor detection using advanced machine learning technology, 
        helping medical professionals make informed decisions quickly.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🔬 Technology</h3>
        <p>Our system uses deep learning algorithms trained on thousands of brain MRI images to detect 
        various types of brain tumors with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>⚡ Features</h3>
        <ul>
        <li>Fast and accurate detection</li>
        <li>Support for multiple image formats</li>
        <li>Detailed confidence scores</li>
        <li>User-friendly interface</li>
        <li>Secure and private</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>📊 Performance</h3>
        <p>Our model achieves high accuracy on validation datasets and continues to improve with 
        ongoing research and development.</p>
        </div>
        """, unsafe_allow_html=True)

def show_detection_page():
    st.markdown('<h2 class="sub-header">🔍 Brain Tumor Detection</h2>', unsafe_allow_html=True)
    
    # Load model
    model = load_brain_tumor_model()
    
    if model is None:
        st.error("Model could not be loaded. Please check if the model file exists.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a brain MRI image",
        type=['jpg', 'jpeg', 'png', 'dicom', 'dcm'],
        help="Supported formats: JPG, JPEG, PNG, DICOM"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image_data = Image.open(uploaded_file)
            st.image(image_data, caption="Brain MRI Scan", use_column_width=True)
            
            # Image info
            st.markdown(f"""
            **Image Details:**
            - Format: {image_data.format}
            - Size: {image_data.size}
            - Mode: {image_data.mode}
            """)
        
        with col2:
            st.subheader("Analysis Results")
            
            # Analyze button
            if st.button("🔬 Analyze Image", type="primary"):
                with st.spinner("Analyzing image... Please wait."):
                    # Preprocess image
                    processed_image = preprocess_image(image_data)
                    
                    # Make prediction
                    result, confidence, raw_prediction = predict_tumor(model, processed_image)
                    
                    if result is not None:
                        # Display results
                        if "Tumor Detected" in result or result in ["Glioma", "Meningioma", "Pituitary"]:
                            st.markdown(f"""
                            <div class="result-box positive-result">
                            <h3>⚠️ Detection Result</h3>
                            <h4>{result}</h4>
                            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-box negative-result">
                            <h3>✅ Detection Result</h3>
                            <h4>{result}</h4>
                            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Confidence visualization
                        if len(raw_prediction) > 1:  # Multi-class
                            class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
                            probabilities = raw_prediction * 100
                            
                            fig = px.bar(
                                x=class_names,
                                y=probabilities,
                                title="Prediction Probabilities",
                                labels={'x': 'Class', 'y': 'Probability (%)'},
                                color=probabilities,
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:  # Binary classification
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = confidence,
                                title = {'text': "Confidence Level"},
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "red" if confidence > 70 else "orange" if confidence > 50 else "green"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "gray"},
                                        {'range': [80, 100], 'color': "darkgray"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Disclaimer
                        st.warning("""
                        **⚠️ Medical Disclaimer:** This tool is for educational and research purposes only. 
                        It should not be used as a substitute for professional medical diagnosis. 
                        Always consult with qualified healthcare professionals for medical decisions.
                        """)
                        
                        # Save results option
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        results_data = {
                            'timestamp': timestamp,
                            'result': result,
                            'confidence': confidence,
                            'filename': uploaded_file.name
                        }
                        
                        st.download_button(
                            label="📄 Download Results",
                            data=str(results_data),
                            file_name=f"brain_tumor_analysis_{timestamp}.txt",
                            mime="text/plain"
                        )

def show_about_page():
    st.markdown('<h2 class="sub-header">ℹ️ About This System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧠 What is Brain Tumor Detection?
        
        Brain tumor detection is a critical medical imaging task that involves identifying abnormal 
        tissue growth in the brain using MRI scans. Early detection is crucial for effective treatment 
        and improved patient outcomes.
        
        ### 🔬 Our Approach
        
        Our system uses deep learning techniques, specifically Convolutional Neural Networks (CNNs), 
        to analyze brain MRI images and detect the presence of tumors. The model has been trained on 
        a large dataset of brain MRI scans with expert annotations.
        
        ### 📊 Model Performance
        
        - **Accuracy**: High accuracy on validation datasets
        - **Speed**: Fast inference time for real-time analysis
        - **Reliability**: Consistent performance across different image qualities
        """)
    
    with col2:
        st.markdown("""
        ### 🏥 Types of Brain Tumors Detected
        
        Our system can identify several types of brain tumors:
        
        - **Glioma**: Tumors that arise from glial cells
        - **Meningioma**: Tumors that develop in the meninges
        - **Pituitary**: Tumors in the pituitary gland
        - **No Tumor**: Normal brain tissue
        
        ### ⚖️ Ethical Considerations
        
        - Patient privacy and data security
        - Transparent AI decision-making
        - Complementing, not replacing, medical expertise
        - Continuous model improvement and validation
        
        ### 🔧 Technical Stack
        
        - **Framework**: TensorFlow/Keras
        - **Interface**: Streamlit
        - **Visualization**: Plotly
        - **Image Processing**: OpenCV, PIL
        """)

def show_statistics_page():
    st.markdown('<h2 class="sub-header">📊 System Statistics</h2>', unsafe_allow_html=True)
    
    # Mock statistics - replace with actual data from your system
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🔍</h3>
        <h2>1,234</h2>
        <p>Images Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>✅</h3>
        <h2>95.6%</h2>
        <p>Accuracy Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>⚡</h3>
        <h2>2.3s</h2>
        <p>Avg Processing Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3>👥</h3>
        <h2>567</h2>
        <p>Users Served</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample charts
    st.subheader("Detection Statistics")
    
    # Sample data for visualization
    sample_data = {
        'Tumor Type': ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary'],
        'Count': [450, 320, 280, 184]
    }
    
    fig = px.pie(
        values=sample_data['Count'],
        names=sample_data['Tumor Type'],
        title="Distribution of Detected Cases"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly analysis trend
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    analyses = [45, 52, 61, 58, 67, 73]
    
    fig2 = px.line(
        x=months,
        y=analyses,
        title="Monthly Analysis Trend",
        labels={'x': 'Month', 'y': 'Number of Analyses'}
    )
    st.plotly_chart(fig2, use_container_width=True)
    

if __name__ == "__main__":
    main()