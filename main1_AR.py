import streamlit as st
import tensorflow as tf
import numpy as np
import os
from gtts import gTTS
from io import BytesIO
import base64

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # conversion into batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

def read_file_content(disease_name):
    # Search for the disease file in a specific directory
    directory = 'disease_descriptions/'
    file_name = f"{disease_name}.txt"
    file_path = os.path.join(directory, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        return "No description available for this disease."

def text_to_speech(text):
    tts = gTTS(text)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def download_button(mp3_fp):
    b64 = base64.b64encode(mp3_fp.read()).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="disease_description.mp3">Download the audio file</a>'
    return href

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION CENTER")
    # Same Home content...

elif app_mode == "About":
    st.header("About the Project")
    # Same About content...

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    st.markdown("Upload an image of a plant leaf to get a disease prediction.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = model_prediction(uploaded_file)
        
        # Define class names
        class_names = [
           "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry(sweet or sour) - Powdery Mildew",
    "Cherry(sweet or sour) - Healthy",
    "Corn(maize) - Gray Leaf Spot",
    "Corn(maize) - Common Rust",
    "Corn(maize) - Northern Leaf Blight",
    "Corn(maize) - Healthy",
    "Grape - Black Rot",
    "Grape - Esca (Black Measles)",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape - Healthy",
    "Orange - Haunglongbing (Citrus Greening)",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Bell Pepper - Bacterial Spot",
    "Bell Pepper - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Two Spotted Spider Mite",
    "Tomato - Target Spot",
    "Tomato - Tomato Yellow Leaf Curl Virus",
    "Tomato - Tomato Mosaic Virus",
    "Tomato - Healthy"
        ]
        
        disease_name = class_names[label]
        st.success(f"Prediction: {disease_name}")
        
        # Find and display disease file content
        disease_description = read_file_content(disease_name)
        st.write(f"Description for {disease_name}:")
        st.text(disease_description)
        
        # Text-to-Speech
        if st.button("Play Audio Description"):
            mp3_fp = text_to_speech(disease_description)
            st.audio(mp3_fp)

            # Close the BytesIO stream after use
            mp3_fp.close()
            
        # Generate a new audio stream for download since the previous one is closed
        mp3_fp_download = text_to_speech(disease_description)
        st.markdown(download_button(mp3_fp_download), unsafe_allow_html=True)
