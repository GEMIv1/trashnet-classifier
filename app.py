import streamlit as st
import numpy as np
import joblib
import keras
import tensorflow as tf
from PIL import Image
import os
import cv2
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page configuration
st.set_page_config(
    page_title="TrashNet Classifier",
    page_icon="bin",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #2E7D32;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
    }
    .prediction-known {
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
        color: #2E7D32;
    }
    .prediction-unknown {
        background-color: #FFF3E0;
        border: 2px solid #FF9800;
        color: #E65100;
    }
    </style>
""", unsafe_allow_html=True)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.50  

# Fixed FPS and prediction interval
TARGET_FPS = 30
PREDICTION_INTERVAL = 15 

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


@st.cache_resource
def load_models():
    """Load all models from the correct path"""
    try:
        models_dir = r'D:\AI\ML_University_project\application\saved_models\trashnet_models'
        
        if not os.path.exists(models_dir):
            return None
        
        config_path = os.path.join(models_dir, 'config.pkl')
        config = joblib.load(config_path)
        
        label_encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
        label_encoder = joblib.load(label_encoder_path)
        
        savedmodel_path = os.path.join(models_dir, 'feature_extractor_savedmodel')
        
        if os.path.exists(savedmodel_path):
            try:
                feature_extractor = keras.layers.TFSMLayer(
                    savedmodel_path, 
                    call_endpoint='serving_default'
                )
            except Exception as e:
                feature_extractor = tf.saved_model.load(savedmodel_path)
        else:
            return None
        
        svm_path = os.path.join(models_dir, 'svm_classifier.pkl')
        svm_classifier = joblib.load(svm_path)
        
        return {
            'config': config,
            'label_encoder': label_encoder,
            'feature_extractor': feature_extractor,
            'svm_classifier': svm_classifier
        }
        
    except FileNotFoundError as e:
        return None
    except Exception as e:
        return None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.efficientnet.preprocess_input(img_array)
    
    return img_array


def extract_features(feature_extractor, processed_img):
    """Extract features using the feature extractor model"""
    try:
        input_tensor = tf.constant(processed_img, dtype=tf.float32)
        
        if hasattr(feature_extractor, 'predict'):
            features = feature_extractor.predict(processed_img, verbose=0, batch_size=1)
        else:
            features = feature_extractor(input_tensor)
        
        if isinstance(features, dict):
            if len(features) == 1:
                features = list(features.values())[0]
            elif 'output_0' in features:
                features = features['output_0']
            elif 'predictions' in features:
                features = features['predictions']
            elif 'global_average_pooling2d' in features:
                features = features['global_average_pooling2d']
            else:
                features = list(features.values())[0]
        
        if hasattr(features, 'numpy'):
            features = features.numpy()
        
        features = np.array(features)
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        elif len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        return features
        
    except Exception as e:
        raise e


def predict_image(image, models):
    """Make prediction on image using SVM"""
    processed_img = preprocess_image(image, tuple(models['config']['target_size']))
    features = extract_features(models['feature_extractor'], processed_img)
    
    classifier = models['svm_classifier']
    
    prediction = classifier.predict(features)
    probabilities = classifier.predict_proba(features)[0]
    
    predicted_class = models['label_encoder'].inverse_transform(prediction)[0]
    confidence = max(probabilities)
    
    class_probs = {
        models['label_encoder'].inverse_transform([i])[0]: prob 
        for i, prob in enumerate(probabilities)
    }
    
    # Check if confidence is below threshold
    is_unknown = confidence < CONFIDENCE_THRESHOLD
    
    return predicted_class, class_probs, confidence, is_unknown


def get_class_info(class_name, is_unknown=False):
    """Get information for each class"""
    
    if is_unknown:
        return {
            'tips': 'The AI could not confidently classify this item. Please try taking a clearer photo or manually identify the material.',
            'bin': 'Unable to determine - please check manually',
            'color': '#9E9E9E'
        }
    
    info = {
        'cardboard': {
            'tips': 'Flatten boxes before recycling. Remove tape and labels if possible.',
            'bin': 'Blue Recycling Bin',
            'color': '#8D6E63'
        },
        'glass': {
            'tips': 'Rinse bottles and jars. Remove caps and lids.',
            'bin': 'Blue Recycling Bin (or separate glass bin)',
            'color': '#42A5F5'
        },
        'metal': {
            'tips': 'Rinse cans and containers. Aluminum and steel are highly recyclable.',
            'bin': 'Blue Recycling Bin',
            'color': '#78909C'
        },
        'paper': {
            'tips': 'Keep paper dry and clean. Shredded paper should be in a paper bag.',
            'bin': 'Blue Recycling Bin',
            'color': '#FFEB3B'
        },
        'plastic': {
            'tips': 'Check the number on the bottom. Rinse containers. #1 and #2 are most recyclable.',
            'bin': 'Blue Recycling Bin (check local guidelines)',
            'color': '#FF9800'
        },
        'trash': {
            'tips': 'This item cannot be recycled. Dispose in regular trash.',
            'bin': 'General Waste Bin',
            'color': '#757575'
        }
    }
    
    return info.get(class_name.lower(), {
        'tips': 'Unknown classification',
        'bin': 'Check local guidelines',
        'color': '#9E9E9E'
    })


def flip_image_horizontal(image):
    """Flip image horizontally"""
    try:
        try:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        except AttributeError:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
    except Exception as e:
        return image


def display_prediction_results(predicted_class, class_probs, confidence, is_unknown):
    """Display prediction results with styling"""
    sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    
    if is_unknown:
        st.markdown("### Predicted Class: **UNKNOWN**")
        st.warning(f"Confidence ({confidence:.1%}) is below the {CONFIDENCE_THRESHOLD:.0%} threshold")
        st.markdown(
            f"**Best guess:** <span style='color:gray; font-size:1em;'>{predicted_class.title()} ({confidence:.1%})</span>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"### Predicted Class: **{predicted_class.upper()}**")
        
        if confidence > 0.8:
            conf_color = "green"
        elif confidence > 0.6:
            conf_color = "orange"
        else:
            conf_color = "red"
        
        st.markdown(
            f"**Confidence:** <span style='color:{conf_color}; font-size:1.2em; font-weight:bold;'>{confidence:.1%}</span>", 
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown("### All Class Probabilities")
    
    for class_name, prob in sorted_probs:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(float(prob))
        with col2:
            st.write(f"**{class_name.title()}**: {prob:.1%}")
    
    st.markdown("---")
    st.caption(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")


# Global variable to store the latest prediction for real-time display
class PredictionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.predicted_class = None
        self.confidence = 0.0
        self.is_unknown = True
        self.class_probs = {}
        self.frame_count = 0


# Create a global prediction state
if 'prediction_state' not in st.session_state:
    st.session_state.prediction_state = PredictionState()


class VideoProcessor:
    """Video processor for real-time classification"""
    
    def __init__(self, models):
        self.models = models
        self.frame_count = 0
        self.last_prediction = "Waiting..."
        self.last_confidence = 0.0
        self.is_unknown = True
        self.class_probs = {}
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Flip the image horizontally (mirror effect)
        img = cv2.flip(img, 1)
        
        self.frame_count += 1
        
        # Only predict every N frames to reduce lag
        if self.frame_count % PREDICTION_INTERVAL == 0:
            try:
                # Convert BGR to RGB for prediction
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Make prediction
                predicted_class, class_probs, confidence, is_unknown = predict_image(
                    rgb_frame, self.models
                )
                
                self.last_prediction = predicted_class
                self.last_confidence = confidence
                self.is_unknown = is_unknown
                self.class_probs = class_probs
                
                # Update global state for display
                with st.session_state.prediction_state.lock:
                    st.session_state.prediction_state.predicted_class = predicted_class
                    st.session_state.prediction_state.confidence = confidence
                    st.session_state.prediction_state.is_unknown = is_unknown
                    st.session_state.prediction_state.class_probs = class_probs
                    st.session_state.prediction_state.frame_count = self.frame_count
                    
            except Exception as e:
                pass  # Silently handle errors to keep stream running
        
        # Draw prediction on frame
        self._draw_prediction_overlay(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _draw_prediction_overlay(self, img):
        """Draw prediction overlay on the frame"""
        height, width = img.shape[:2]
        
        # Create semi-transparent overlay at the top
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Determine display text and color
        if self.is_unknown:
            display_class = "UNKNOWN"
            color = (0, 165, 255)  # Orange in BGR
            sub_text = f"Best guess: {self.last_prediction.title()}"
        else:
            display_class = self.last_prediction.upper()
            if self.last_confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif self.last_confidence > 0.6:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            sub_text = ""
        
        # Draw main prediction text
        cv2.putText(
            img, 
            f"Class: {display_class}", 
            (20, 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            color, 
            2
        )
        
        # Draw confidence
        conf_text = f"Confidence: {self.last_confidence:.1%}"
        cv2.putText(
            img, 
            conf_text, 
            (20, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Draw sub text if unknown
        if sub_text:
            cv2.putText(
                img, 
                sub_text, 
                (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (200, 200, 200), 
                1
            )


def main():
    # Header
    st.markdown('<p class="main-header">TrashNet Waste Classifier</p>', unsafe_allow_html=True)    
    # Sidebar
    with st.sidebar:
        st.markdown("## Settings")
        
        st.markdown("### Classifier")
        st.info("**SVM Classifier** - Support Vector Machine")
        
        st.markdown("---")
        
        # Input mode selection
        st.markdown("## Input Mode")
        input_mode = st.radio(
            "Choose input method",
            options=['upload', 'snapshot', 'realtime'],
            format_func=lambda x: {
                'upload': 'Upload Image',
                'snapshot': 'Camera Snapshot', 
                'realtime': 'Real-time Camera'
            }[x],
            help="Real-time mode provides live classification"
        )
        
        st.markdown("---")
        
        # Display current threshold
        st.markdown("## Confidence Threshold")
        st.info(f"**Current threshold:** {CONFIDENCE_THRESHOLD:.0%}\n\nPredictions below this confidence will be marked as 'Unknown'")
        
        st.markdown("---")
        st.markdown("## Model Information")
        st.info("""
        
        **Feature Extraction:** EfficientNetB0 (pre-trained on ImageNet)
        
        **Classification:** SVM (Support Vector Machine)
        
        **Training Dataset:** TrashNet
        
        **Classes:** 6 waste categories
        """)
        
        st.markdown("---")
        st.markdown("## Waste Categories")
        categories = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
        for cat in categories:
            st.markdown(f"- **{cat}**")
        
    # Load models silently
    models = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check that the model files exist in the correct directory.")
        st.stop()
    
    # Display categories
    st.markdown("### Classification Categories")
    cols = st.columns(6)
    for idx, class_name in enumerate(models['config']['classes']):
        with cols[idx]:
            st.markdown(
                f"<div style='text-align:center'><b>{class_name.title()}</b></div>", 
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    
    # ==================== REAL-TIME CAMERA MODE ====================
    if input_mode == 'realtime':
        st.markdown("### Real-time Camera Classification")
        st.info("Point your camera at a waste item for real-time classification. The prediction will update automatically.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # WebRTC streamer with fixed FPS
            webrtc_ctx = webrtc_streamer(
                key="waste-classifier",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: VideoProcessor(models),
                media_stream_constraints={
                    "video": {
                        "frameRate": {"ideal": TARGET_FPS, "max": TARGET_FPS},
                        "width": {"ideal": 640},
                        "height": {"ideal": 480}
                    }, 
                    "audio": False
                },
                async_processing=True,
            )
        
        with col2:
            st.markdown("### Live Results")
            
            # Display current prediction status
            if webrtc_ctx.state.playing:
                st.success("Camera is active")
                st.markdown("""
                **Tips for best results:**
                - Hold item steady in frame
                - Ensure good lighting
                - Center the item
                - Wait for prediction to stabilize
                """)
            else:
                st.warning("Click 'START' to begin")
                st.markdown("""
                **Instructions:**
                1. Click the START button
                2. Allow camera access
                3. Point camera at waste item
                4. View real-time predictions
                """)
            
            # Show quick reference
            st.markdown("---")
            st.markdown("### Quick Reference")
            
            for cat in ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']:
                st.markdown(f"- **{cat.title()}**")
    
    # ==================== UPLOAD MODE ====================
    elif input_mode == 'upload':
        st.markdown("### Upload Waste Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image of a waste item",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a single waste item for best results"
        )
        
        if uploaded_file is not None:
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                st.caption(f"Image size: {image.size[0]} x {image.size[1]} pixels")
            
            with col2:
                st.markdown("### Classification Results")
                
                with st.spinner("Analyzing image"):
                    try:
                        predicted_class, class_probs, confidence, is_unknown = predict_image(
                            image, models
                        )
                        display_prediction_results(predicted_class, class_probs, confidence, is_unknown)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        return
            
            # Disposal information
            st.markdown("---")
            st.markdown("### Disposal Information")
            
            info = get_class_info(predicted_class, is_unknown)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Disposal Bin:**\n\n{info['bin']}")
            
            with col2:
                if is_unknown:
                    st.markdown(f"**Category:**\n\n**Unknown**\n\n(Best guess: {predicted_class.title()})")
                else:
                    st.markdown(f"**Category:**\n\n**{predicted_class.title()}**")
            
            st.markdown("### Disposal Tips")
            if is_unknown:
                st.warning(info['tips'])
            else:
                st.info(info['tips'])
        else:
            st.info("Upload an image to start classification")
    
    # ==================== SNAPSHOT MODE ====================
    elif input_mode == 'snapshot':
        st.markdown("### Camera Snapshot")
        
        camera_image = st.camera_input("Take a picture of the waste item")
        
        if camera_image:
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Captured Image")
                image = Image.open(camera_image)
                # Flip camera image to match preview
                image = flip_image_horizontal(image)
                st.image(image, use_container_width=True)
                st.caption(f"Camera image - {image.size[0]} x {image.size[1]} pixels")
            
            with col2:
                st.markdown("### Classification Results")
                
                with st.spinner("Analyzing image..."):
                    try:
                        predicted_class, class_probs, confidence, is_unknown = predict_image(
                            image, models
                        )
                        display_prediction_results(predicted_class, class_probs, confidence, is_unknown)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        return
            
            # Disposal information
            st.markdown("---")
            st.markdown("### Disposal Information")
            
            info = get_class_info(predicted_class, is_unknown)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Disposal Bin:**\n\n{info['bin']}")
            
            with col2:
                if is_unknown:
                    st.markdown(f"**Category:**\n\n**Unknown**\n\n(Best guess: {predicted_class.title()})")
                else:
                    st.markdown(f"**Category:**\n\n**{predicted_class.title()}**")
            
            st.markdown("### Disposal Tips")
            if is_unknown:
                st.warning(info['tips'])
            else:
                st.info(info['tips'])
        else:
            st.info("Click the camera button to take a snapshot")
    
    # Footer tips (when no image)
    if input_mode != 'realtime' and ((input_mode == 'upload' and 'uploaded_file' not in dir()) or 
                                       (input_mode == 'snapshot' and 'camera_image' not in dir())):
        st.markdown("---")
        st.markdown("### Tips for Best Results")
        
        tip_col1, tip_col2, tip_col3 = st.columns(3)
        
        with tip_col1:
            st.markdown("#### Good Lighting")
            st.markdown("Ensure the item is well-lit with minimal shadows.")
        
        with tip_col2:
            st.markdown("#### Center the Item")
            st.markdown("Place the waste item in the center of the frame.")
        
        with tip_col3:
            st.markdown("#### Single Item")
            st.markdown("Photograph one item at a time for better accuracy.")


if __name__ == "__main__":
    main()