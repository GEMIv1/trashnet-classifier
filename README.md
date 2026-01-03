# [TrashNet Classifier](https://trashnet-classifier-desjhwxrzwf9eueucgqwtp.streamlit.app/) 🗑️♻️

A hybrid deep learning and machine learning waste classification system achieving **92.49% accuracy**. Combines EfficientNetB0 feature extraction with SVM classification to identify and categorize 6 types of waste materials. Includes an interactive Streamlit web app with camera support for real-time classification.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Technologies Used](#technologies-used)

## Overview

TrashNet Classifier is a production-ready hybrid AI system that combines deep learning feature extraction with classical machine learning for automated waste categorization. The system uses **EfficientNetB0** (pre-trained on ImageNet) as a feature extractor, followed by an optimized **SVM classifier** that achieves **92.49% accuracy** on waste classification.

This innovative approach offers:
- **High Accuracy**: 92.49% classification accuracy with balanced performance
- **Efficient Computation**: Fast inference (~100-200ms per image)
- **Robust Features**: Transfer learning from ImageNet provides excellent visual representations
- **Production Ready**: Includes confidence thresholding and multiple deployment options
- **Interactive Interface**: Streamlit app with image upload, live camera, and camera capture

## Features

- **Hybrid Architecture**: Combines EfficientNetB0 (deep learning) with SVM (classical ML)
- **High Accuracy**: Achieves 92.49% classification accuracy on test set
- **Advanced Data Augmentation**: Custom pipeline with 2.4x dataset expansion
- **Balanced Training**: Addresses class imbalance (600 images per class)
- **Multiple Classifier Comparison**: Comprehensive SVM vs KNN evaluation
- **Interactive Web App**: Streamlit interface with three input modes
  - 📤 Image upload (drag-and-drop)
  - 📹 Live camera feed
  - 📸 Camera screenshot capture
- **Real-time Classification**: Instant predictions with confidence scores
- **Confidence Thresholding**: 50% threshold with "Unknown Object" fallback
- **Comprehensive Documentation**: Full ML report with performance analysis
- **Production Ready**: Optimized for deployment with <200ms inference time

## Dataset

The project uses the **TrashNet dataset**, which contains images of waste sorted into the following categories:

- **Cardboard** - Cardboard boxes and packaging
- **Glass** - Glass bottles and containers
- **Metal** - Metal cans and objects
- **Paper** - Paper waste and documents
- **Plastic** - Plastic bottles and packaging
- **Trash** - Non-recyclable waste

The dataset consists of 1,865 valid images (after removing 100 corrupted files) resized to 224×224 pixels for optimal EfficientNetB0 processing.

**Dataset Source**: [TrashNet on Kaggle](https://www.kaggle.com/datasets/feyzazkefe/trashnet)

## Methodology

### System Workflow

![ML project pipeline (1)](https://github.com/user-attachments/assets/a8458061-e6d4-4d13-84ce-2dd45937e049)

The complete pipeline follows these stages:

1. **Images** → Raw input images of waste items
2. **Preprocessing and Augmentation** → Image resizing, normalization, and augmentation (4x multiplier)
3. **CNN (EfficientNetB3)** → Feature extraction using pre-trained model
4. **Classification** → Two classifiers tested:
   - **SVM** → Selected (92.5% accuracy) ✅
   - **KNN** → Comparison model (~88-90% accuracy)
5. **Output** → Predicted waste category with confidence score

### Architecture Overview

The system uses a two-stage pipeline:

1. **Feature Extraction**: EfficientNetB3 (frozen, pre-trained on ImageNet)
2. **Classification**: Support Vector Machine (SVM) with RBF kernel

```
Input Image (300x300x3) 
    ↓
EfficientNetB3 Feature Extractor (frozen)
    ↓
Global Average Pooling
    ↓
Feature Vector
    ↓
SVM Classifier
    ↓
Predicted Class
```

### Feature Extraction

```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D

# Load pre-trained EfficientNetB0
base_model = EfficientNetB0(
    weights='imagenet', 
    include_top=False, 
    input_shape=(300, 300, 3)
)
base_model.trainable = False  # Freeze the base model

# Create feature extractor
model = Sequential([
    base_model,
    GlobalAveragePooling2D()
])
```

**Why EfficientNetB0?**
- State-of-the-art accuracy with minimal computational cost
- Pretrained on ImageNet provides robust visual features (1,300-dimensional feature vectors)
- Compact model suitable for deployment
- Faster inference compared to larger models

### Data Augmentation Strategy

Custom augmentation pipeline designed specifically for waste classification to address class imbalance:

**Target:** 600 images per class for balanced training

```python
ImageDataGenerator(
            rotation_range=20,           # Moderate rotation (trash can be at any angle)
            width_shift_range=0.15,      # Slight horizontal shifts
            height_shift_range=0.15,     # Slight vertical shifts
            zoom_range=0.2,              # Zoom to simulate different distances
            shear_range=0.15,            # Slight shearing
            horizontal_flip=True,        # Trash items can appear mirrored
            vertical_flip=False,         # Don't flip vertically (unnatural)
            brightness_range=[0.8, 1.3], # Lighting variations
            fill_mode='nearest',         # Fill missing pixels
            channel_shift_range=0.1      # Color variations
        )
```

**Augmentation Results:**
- Original training set: 1,492 images (80% of dataset)
- After augmentation: 3,600 images (perfectly balanced, 600 per class)
- Augmentation multiplier: 2.4x dataset size
- Eliminated class imbalance (4.2:1 ratio → 1:1 ratio)

### Classification Models

Two classifiers were trained and comprehensively compared:

1. **Support Vector Machine (SVM)** ✅ **Selected for Production**
   - Kernel: RBF (Radial Basis Function)
   - C parameter: 10 (regularization strength)
   - Gamma: scale (automatic)
   - Class weight: balanced (handles class imbalance)
   - Probability estimates: enabled
   - **Accuracy: 92.49%**
   - **Advantages:** Superior accuracy, robust to overfitting, excellent generalization, balanced performance across all categories

2. **K-Nearest Neighbors (KNN)** (Comparison Model)
   - Number of neighbors: 7
   - Distance metric: Minkowski (p=2, Euclidean distance)
   - Weighting: distance-based (closer neighbors have more influence)
   - **Accuracy: 87.13%**
   - **Advantages:** Fast training, simple algorithm, perfect recall on cardboard
   - **Limitations:** Lower overall accuracy, struggles with glass and trash categories

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/AhmedMohamedKame1/trashnet-classifier.git
cd trashnet-classifier
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages

```
tensorflow==2.17.0
tf-keras==2.17.0
streamlit
streamlit-webrtc
opencv-python-headless
pillow
scikit-learn
joblib
```

## Usage

### Running the Streamlit App

Launch the web application:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the App
1. Upload an image of trash using the file uploader
2. The app will automatically extract features using EfficientNetB3
3. SVM classifier predicts the waste category
4. View the predicted category and confidence score


## Training Pipeline

The complete training pipeline is available in the Kaggle notebook: [ML Pipeline](https://www.kaggle.com/code/ahmedkamel111/ml-pipeline)

### Training Workflow

#### 1. Data Loading & Augmentation
```python
from preprocessing import augment_and_load_images

# Load training data with augmentation
X_train, y_train = augment_and_load_images(
    train_paths, 
    train_labels, 
    target_size=(300, 300), 
    augment=True
)

# Load validation/test data without augmentation
X_val, y_val = augment_and_load_images(
    val_paths, 
    val_labels, 
    target_size=(300, 300), 
    augment=False
)
```

#### 2. Feature Extraction
```python
# Extract features using EfficientNetB3
print("Extracting features from training data...")
train_features = feature_extractor.predict(X_train, batch_size=32, verbose=1)

print("Extracting features from validation data...")
val_features = feature_extractor.predict(X_val, batch_size=32, verbose=1)
```

#### 3. Classifier Training
```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Train SVM
svm_classifier = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)
svm.fit(train_features, y_train)

# Train KNN for comparison
knn_classifier = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='auto', metric='minkowski', p=2, n_jobs=-1)
knn.fit(train_features, y_train)
```

#### 4. Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix

# Evaluate SVM
svm_predictions = svm.predict(val_features)
print(classification_report(y_val, svm_predictions, target_names=classes))

# Evaluate KNN
knn_predictions = knn.predict(val_features)
print(classification_report(y_val, knn_predictions, target_names=classes))
```

## Results

### Overall Model Performance

| Classifier | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Inference Time |
|------------|----------|-------------------|----------------|------------------|----------------|
| **SVM (Production)** | **92.49%** | **93%** | **92%** | **92%** | ~100-200ms |
| KNN (Comparison) | 87.13% | 85% | 87% | 86% | ~50-100ms |

**Performance Gain:** SVM outperforms KNN by **+5.36%** in accuracy

### Per-Category Performance Comparison

#### Precision
| Category | SVM | KNN | Winner |
|----------|-----|-----|--------|
| **Cardboard** | 96% | 94% | SVM (+2%) |
| **Glass** | 90% | 79% | **SVM (+11%)** |
| **Metal** | 85% | 87% | KNN (+2%) |
| **Paper** | 95% | 94% | SVM (+1%) |
| **Plastic** | **97%** | 91% | **SVM (+6%)** |
| **Trash** | 95% | 67% | **SVM (+28%)** |

#### Recall
| Category | SVM | KNN | Winner |
|----------|-----|-----|--------|
| **Cardboard** | 98% | **100%** | KNN (+2%) |
| **Glass** | 92% | 90% | SVM (+2%) |
| **Metal** | 92% | 75% | **SVM (+17%)** |
| **Paper** | 96% | 90% | **SVM (+6%)** |
| **Plastic** | 88% | 84% | SVM (+4%) |
| **Trash** | 86% | 86% | Tie |

#### F1-Score (Balanced Metric)
| Category | SVM F1 | KNN F1 | Difference |
|----------|--------|--------|------------|
| **Cardboard** | 97% | 97% | Tie |
| **Glass** | 91% | 84% | **+7%** |
| **Metal** | 89% | 80% | **+9%** |
| **Paper** | 95% | 92% | +3% |
| **Plastic** | 92% | 87% | +5% |
| **Trash** | 90% | 75% | **+15%** |

**Result:** SVM wins in **5 out of 6 categories** for F1-score

### Why SVM Was Selected for Production

✅ **Higher Overall Accuracy**: 92.49% vs 87.13%  
✅ **Superior Performance on Challenging Classes**: Significantly better on Glass (+7% F1), Trash (+15% F1), and Metal (+9% F1)  
✅ **Balanced Performance**: Consistent 85-97% precision across all categories  
✅ **Better Generalization**: More robust to test data variations  
✅ **Smaller Deployment Size**: Only classifier weights needed (vs. entire KNN dataset)  
✅ **Confidence Scores**: Probability estimates enable confidence thresholding  

### Category-Specific Insights

**Best Performing Classes (SVM):**
- **Plastic**: 97% precision, 92% F1-score - Highest precision
- **Cardboard**: 96% precision, 98% recall - Excellent detection
- **Paper**: 95% precision, 95% F1-score - Largest test set, very stable

**Challenging Classes:**
- **Trash**: Smallest training set (106 images), most diverse category
- **Glass**: Transparency creates similarity with plastic
- **Metal**: Reflective surfaces can be confused with plastic

**Common Misclassifications:**
1. Glass ↔ Plastic (transparency similarity)
2. Metal ↔ Plastic (reflective surfaces)
3. Trash misclassified as other categories (high variability)

### Detailed Classification Report (SVM)

```
                precision    recall  f1-score   support

    cardboard       0.96      0.98      0.97        49
        glass       0.90      0.92      0.91        77
        metal       0.85      0.92      0.89        63
        paper       0.95      0.96      0.95        90
      plastic       0.97      0.88      0.92        73
        trash       0.95      0.86      0.90        21

     accuracy                           0.92       373
    macro avg       0.93      0.92      0.92       373
 weighted avg       0.93      0.92      0.93       373
```

### Training Efficiency

**Data Split:**
- Training: 1,492 images (80%) → Augmented to 3,600 images
- Testing: 373 images (20%)

**Feature Extraction Performance:**
- Training set processing: ~2-3 minutes (3,600 images)
- Test set processing: ~15 seconds (373 images)
- Per-image feature extraction: ~50ms

**Model Training:**
- SVM training time: ~30 seconds
- KNN training time: Instant (lazy learner)

### Production Performance Metrics

**Expected Real-World Performance:**
- Accuracy: ~92% on similar data
- Inference time: 100-200ms per image
  - Feature extraction: 80-150ms
  - SVM prediction: 1-5ms
- Confidence threshold: 50% (adjustable)

**Success Criteria Met:**
✓ Achieved >90% overall accuracy (92.49%)  
✓ Balanced performance across categories (85-97% precision)  
✓ Fast inference (<200ms per image)  
✓ Production-ready with confidence thresholding  

## Streamlit App

The Streamlit application provides an intuitive, multi-modal interface for waste classification using the trained SVM model.

### App Features

#### Multiple Input Methods
- **Upload Images**: Drag-and-drop or browse for image files (JPG, PNG, JPEG)
- **Live Camera**: Real-time classification using your device's camera
- **Camera Screenshots**: Capture and classify images directly from camera

#### AI-Powered Classification
- **Automatic Processing**: Features extracted via EfficientNetB0 (1,300-dimensional vectors)
- **SVM Classification**: Real-time predictions with 92.49% accuracy
- **Confidence Scores**: Visual representation of prediction probabilities for all 6 categories
- **Confidence Threshold**: 50% threshold with "Unknown Object" fallback for ambiguous predictions

#### Interactive Results
- **Image Preview**: View uploaded/captured images before classification
- **Prediction Breakdown**: See confidence scores for all waste categories
- **Category Information**: Descriptions and recycling guidelines for each class
- **Processing Time**: Real-time inference performance metrics

#### Comprehensive Documentation
- **Model Report**: Access the full ML performance analysis report
- **Technical Details**: Architecture, training methodology, and evaluation metrics
- **Usage Guide**: Step-by-step instructions for using the app

### Screenshots

#### Main Interface
<img width="1913" height="907" alt="TN-1" src="https://github.com/user-attachments/assets/1f48d588-96a0-4038-a7a7-10925c592577"/>

#### Classification Results
<img width="1917" height="900" alt="TN-2" src="https://github.com/user-attachments/assets/1ce713a7-98e7-45e6-a678-70e51c19c88d"/>

### Using the App

#### Option 1: Upload Image
1. Click "Browse files" or drag-and-drop an image
2. Image automatically processed through EfficientNetB0
3. SVM classifier predicts the waste category
4. View prediction with confidence scores

#### Option 2: Live Camera
1. Click "Use Live Camera"
2. Grant camera permissions
3. Point camera at waste item
4. Real-time classification updates
5. Capture screenshot to save prediction

#### Option 3: Camera Screenshot
1. Click "Take Photo"
2. Capture image using device camera
3. Review captured image
4. Classify and view results

## Technologies Used

### Deep Learning
- **TensorFlow/Keras 2.x** - Feature extraction framework
- **EfficientNetB0** - Pre-trained CNN for feature extraction (ImageNet weights, 1,300-dim features)

### Machine Learning
- **scikit-learn** - SVM and KNN classifiers with hyperparameter optimization
- **Support Vector Machine (SVM)** - Primary classifier with RBF kernel (C=7.7, gamma='scale')
- **K-Nearest Neighbors (KNN)** - Comparison model (n=7, distance-weighted)

### Data Processing
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data manipulation and analysis
- **PIL/Pillow** - Image loading and preprocessing
- **OpenCV** - Image processing and camera handling
- **ImageDataGenerator** - Data augmentation pipeline

### Web Application
- **Streamlit** - Interactive web interface with camera support

### Analysis
- **scikit-learn metrics** - Performance evaluation (precision, recall, F1-score, confusion matrix)

### Development & Deployment
- **Kaggle Notebooks** - GPU-accelerated training environment
- **Git/GitHub** - Version control and collaboration
- **joblib** - Model serialization

Kaggle Notebook: [ML Pipeline](https://www.kaggle.com/code/ahmedkamel111/ml-pipeline)
---
