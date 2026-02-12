import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- CONFIGURATIONS ---
st.set_page_config(page_title="SolarGuard", page_icon="☀️", layout="wide")

# --- MODEL PATH HANDLING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_model", "solar_panel_model_final.h5")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Define class names (must match training order)
class_names = [
    'Bird-drop',
    'Clean',
    'Dusty',
    'Electrical-damage',
    'Physical-Damage',
    'Snow-Covered'
]

# --- HELPER FUNCTION ---
def preprocess_image(image):
    """
    Resize to 224x224, normalize, and add batch dimension.
    """
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.asarray(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# --- STREAMLIT UI ---
st.title("☀️ SolarGuard: Solar Panel Defect Detection")
st.write(
    "Upload an image of a solar panel and the AI model will classify "
    "its condition into one of six categories."
)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Choose a solar panel image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("Prediction Results")

    if uploaded_file:
        try:
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image, verbose=0)

            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = float(np.max(prediction) * 100)

            st.success(f"**Condition:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            st.divider()
            st.write("### Prediction Probabilities")

            for i, name in enumerate(class_names):
                prob = float(prediction[0][i] * 100)
                st.write(f"{name}: {prob:.2f}%")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    else:
        st.info("Please upload an image to see the prediction.")
