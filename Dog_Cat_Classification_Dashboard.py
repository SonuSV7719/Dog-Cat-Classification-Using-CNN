import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

model = keras.models.load_model('./Trained_Model/dog_cat_classifire.h5')

reshaped_img = ''
st.set_page_config(page_title='Dog Cat Classification', page_icon='🐶')
st.title("Dog Cat Classification")
uploaded_image = st.file_uploader(label='Upload Dog or Cat image only...', type=['png', 'jpg', 'jpeg'])
if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    resized_img = cv2.resize(img, (256, 256))
    reshaped_img = resized_img.reshape((1, 256, 256, 3))
    st.image(resized_img, caption='Uploaded Image') 
else:
    st.write("No image file uploaded")

if not hasattr(st.session_state, 'output'):
    st.session_state.output = ""

def predict():
    if uploaded_image is not None:
        prediction = model.predict(reshaped_img)
        if prediction > 0:
            # st.write("Given Image is of Dog")
            st.session_state.output = "<h3>Classified as: </h3><p style='color:red;'>Given Image is of Dog</p>"
        else:
            # st.write("Given Image is of Cat")
            st.session_state.output = "<h3>Classified as: </h3><p style='color:red;'>Given Image is of Cat</p>"
    else:
        st.write("Please Upload image")

st.markdown(st.session_state.output, unsafe_allow_html=True)
button = st.button(label="Classify", on_click=predict)


