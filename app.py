import tensorflow as tf
import pandas as pd
import os
import streamlit as st
from PIL import Image
import numpy as np

loaded_model = tf.keras.models.load_model("snake_classification.h5")

st.header("SerpAI - A Snake Species Detection Model🐍")
st.text('Try to upload a clear image of the snake for accurate prediction')

# Upload an image
img_file_buffer = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# Preprocessing
def preprocess_image(img_file, target_size=(128, 128)):
    img = Image.open(img_file).convert("RGB")  
    img = img.resize(target_size)  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    st.subheader('Input Image')
    st.image(img_array[0], caption="Input Image",  use_container_width=True)  # Display the image 
    return img_array

# Prediction function
def prediction(img_file):
    input_image = preprocess_image(img_file)

    # Prediction
    prediction = loaded_model.predict(input_image)

    predicted_class = tf.argmax(prediction, axis=-1).numpy()[0]

    # Getting Class Name using predicted_class
    df = pd.read_csv("Snake_Species.csv")

    snake_labels = list(df['Index '].values)
    snake_name = list(df['Snake'].values)

    snake_predicted = None
    for i, label in enumerate(snake_labels):
        if label == predicted_class:
            snake_predicted = snake_name[i]
            break
    
    if (predicted_class >=0 and predicted_class < 7):
        snake_predicted = snake_predicted + " and it is Venemous...."
    else:
        snake_predicted = snake_predicted + " and it is Non-Venemous...."
        
    return snake_predicted

if st.button("Predict"):
    if img_file_buffer is not None:
        snake_prediction = prediction(img_file_buffer)  
        st.subheader(f"Predicted Snake Species : {snake_prediction}")  
    else:
        st.error("Please upload an image")