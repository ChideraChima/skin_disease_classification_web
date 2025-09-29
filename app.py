import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("best_model.keras")
        return model
    except:
        try:
            model = tf.keras.models.load_model("skin_classifier_model_1.keras")
            return model
        except:
            return None

# Class names
CLASS_NAMES = ["acne", "eczema", "healthy", "psoriasis", "ringworm"]

def preprocess_image(image):
    """Preprocess image for prediction"""
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_skin_disease(image):
    """Predict skin disease from image"""
    model = load_model()
    if model is None:
        return "Model not loaded", 0.0, {}
    
    # Preprocess image
    img_array = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = float(probabilities[predicted_class_idx])
    
    # Create probability dictionary
    prob_dict = {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, probabilities)}
    
    return predicted_class, confidence, prob_dict

# Streamlit UI
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Skin Disease Classifier")
st.markdown("Upload an image to get a skin disease diagnosis with confidence score")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image of the skin area you want to analyze"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, probabilities = predict_skin_disease(image)
            
            if predicted_class == "Model not loaded":
                st.error("❌ Model could not be loaded. Please check the server logs.")
            else:
                # Display results
                st.success(f"✅ **Predicted Class:** {predicted_class.title()}")
                st.info(f"🎯 **Confidence:** {confidence:.1%}")
                
                # Display all probabilities
                st.subheader("📊 All Probabilities:")
                for class_name, prob in probabilities.items():
                    st.write(f"**{class_name.title()}:** {prob:.1%}")

# Footer
st.markdown("---")
st.markdown("**Model Performance:** 86% accuracy on test set")
st.markdown("**Classes:** Acne, Eczema, Healthy, Psoriasis, Ringworm")
