import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- CONFIGURATIONS ---
st.set_page_config(page_title="SolarGuard", page_icon="☀️", layout="wide")

# --- MODEL LOADING ---
# Load your trained model.
# Make sure the path is correct, relative to where you run the streamlit command.
# If you run 'streamlit run app.py' from the 'scripts' folder, the path is correct.
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('saved_model/solar_panel_model_final.h5')
    return model

model = load_model()

# Define the class names in the order your model was trained on.
class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']


# --- HELPER FUNCTION ---
def preprocess_image(image):
    """
    Takes an image, resizes it to 224x224, normalizes it,
    and prepares it for the model.
    """
    # Convert the image to RGB
    image = image.convert('RGB')
    # Resize the image
    image = image.resize((224, 224))
    # Convert image to numpy array
    image_array = np.asarray(image)
    # Normalize the image
    image_array = image_array / 255.0
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# --- STREAMLIT APP LAYOUT ---

st.title("☀️ SolarGuard: Solar Panel Defect Detection")
st.write("Upload an image of a solar panel, and the AI will classify its condition into one of six categories.")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    # Image uploader
    uploaded_file = st.file_uploader("Choose a solar panel image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)

with col2:
    st.subheader("Prediction Results")
    if uploaded_file is not None:
        # Preprocess the image and make a prediction
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(prediction) * 100

        # Display the prediction
        st.success(f"**Condition:** {predicted_class_name}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Optional: Display prediction probabilities
        st.write("---")
        st.write("Prediction Probabilities:")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")
    else:
        st.info("Please upload an image to see the prediction.")