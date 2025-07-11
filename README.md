﻿

#  Brain Tumor Detection System

A deep learning-powered web application for automated brain tumor detection from MRI scans. This system uses advanced Convolutional Neural Networks (CNNs) to classify brain MRI images and detect various types of tumors with high accuracy.

## Features

- ** Accurate Detection**: High-precision brain tumor classification using deep learning
- ** User-Friendly Interface**: Clean, intuitive web interface built with Streamlit
- ** Real-time Analysis**: Fast processing and instant results
- ** Detailed Visualization**: Interactive charts and confidence scores
- ** Multiple Tumor Types**: Detects Glioma, Meningioma, Pituitary tumors, and normal cases
- ** Multiple Formats**: Supports JPG, PNG, and DICOM image formats
- ** Results Export**: Download analysis results for record-keeping
- ** Privacy Focused**: No data storage, secure processing

##  Live Demo

**[Try the Live Application](https://braintumordetection-t2vtx3icsayv2spp3or56h.streamlit.app/)**

##  Screenshots

### Home Page
![Home Page]("C:\Users\Asus\Downloads\Home_BTD.png")

### Detection Interface
![Detection Interface]("C:\Users\Asus\Downloads\detection_BTD.png")


## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, Flask (API)
- **Machine Learning**: TensorFlow/Keras
- **Image Processing**: OpenCV, PIL
- **Visualization**: Plotly, Matplotlib
- **Deployment**: Streamlit Cloud

##  Prerequisites

- Python 3.8 or higher
- Git

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/hardik2712-ai/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Create Virtual Environment
```bash
# Using conda
conda create -n brain-tumor python=3.8
conda activate brain-tumor

# Using venv
python -m venv brain-tumor-env
source brain-tumor-env/bin/activate  # On Windows: brain-tumor-env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Files
Place your trained model file (`brain_tumor_model.h5`) in the project root directory.

### 5. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

##  Project Structure

```
brain-tumor-detection/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── LICENSE                  # License file
│
├── screenshots/             # Application screenshots
│   ├── home_page.png
│   ├── detection_page.png
│   └── results_page.png
│
├── models/                  # Model training files
│   ├── train_model.py
│   └── model_utils.py
│
├── data/                    # Sample data (if any)
│   └── sample_images/
│
└── docs/                    # Additional documentation
    └── model_architecture.md
```

##  Model Information

### Architecture
- **Base Model**: Convolutional Neural Network (CNN)
- **Input Size**: 128x128x3
- **Classes**: 4 (No Tumor, Glioma, Meningioma, Pituitary)
- **Framework**: TensorFlow/Keras

### Performance Metrics
- **Accuracy**: 95.6%
- **Precision**: 94.2%
- **Recall**: 93.8%
- **F1-Score**: 94.0%

### Dataset
- **Training Images**: 3,000+
- **Validation Images**: 800+
- **Test Images**: 500+
- **Data Sources**: Public medical datasets

## Usage

### Web Interface

1. **Home Page**: Overview of the system and its capabilities
2. **Detection Page**: 
   - Upload brain MRI image
   - Click "Analyze Image" 
   - View results and confidence scores
3. **About Page**: Technical details and system information
4. **Statistics Page**: Usage statistics and performance metrics

### API Usage (if Flask backend is running)

```python
import requests

# Upload and analyze image
url = "http://localhost:5000/predict"
files = {"file": open("brain_scan.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

## Deployment

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the app

### Heroku Deployment

1. Create `Profile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Docker Deployment

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

##  Requirements

```txt
streamlit>=1.28.0
tensorflow>=2.10.0
pillow>=9.0.0
opencv-python>=4.5.0
plotly>=5.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex functions
- Include unit tests for new features
- Update documentation as needed

## Testing

Run tests using pytest:

```bash
pip install pytest
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Medical Disclaimer

**IMPORTANT**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## Acknowledgments

- Medical imaging datasets providers
- Open-source community
- TensorFlow and Streamlit teams
- Healthcare professionals who provided domain expertise

## 📞 Contact & Support

- **Author**: Hardik Tyagi
- **Email**: hrktyagi2022@gmail.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/hardik-tyagi2712)
- **GitHub**: [Your GitHub Profile](https://github.com/hardik2712-ai)

### Issues & Bug Reports

If you encounter any issues, please:
1. Check existing [Issues](https://github.com/hardik2712-ai/brain-tumor-detection/issues)
2. Create a new issue with detailed description
3. Include screenshots and error messages

### Feature Requests

We're always looking to improve! Submit feature requests through GitHub Issues with the "enhancement" label.

## Changelog

### v1.0.0 (Current)
- Initial release
- Basic tumor detection functionality
- Web interface with Streamlit
- Support for multiple image formats

### Future Updates
- [ ] Mobile app version
- [ ] Integration with DICOM viewers
- [ ] Multi-language support
- [ ] Advanced visualization features
- [ ] Batch processing capabilities

## Additional Resources

- [Model Training Notebook](brain_tumour_detection_using_deep_learning.ipynb)

---
