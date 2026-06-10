import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("food_spoiled_classifier_fine_tuned.h5")

classes = [
    'freshapples', 'freshbanana', 'freshbittergroud',
    'freshcapsicum', 'freshcucumber', 'freshokra',
    'freshoranges', 'freshpotato', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottenbittergroud',
    'rottencapsicum', 'rottencucumber', 'rottenokra',
    'rottenoranges', 'rottenpotato', 'rottentomato'
]

st.title("Food Spoilage Classification")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    st.success(f"Prediction: {classes[class_index]}")
