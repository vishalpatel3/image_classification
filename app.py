import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("model/cnn_model.h5")
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

st.title("Image Classification using CNN")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((32,32))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    st.write(f"Prediction: **{class_names[class_index]}**")
