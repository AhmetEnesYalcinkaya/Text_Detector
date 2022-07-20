from charset_normalizer import detect
import streamlit as st
from PIL import Image
from craft_text_detector import Craft
import numpy as np

st.header('Text Detector')
st.write("Choose any image")

uploaded_file = st.file_uploader("Choose an image...")


def detection(img1):
    
    image =  np.array(img1)
    print(image.shape)
    print(type(image))
    # set image path and export folder directory
    # can be filepath, PIL image or numpy array
    output_dir = 'outputs/'

    # create a craft instance
    craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

    # apply craft text detection and export detected regions to output directory
    prediction_result = craft.detect_text(image)

    # unload models from ram/gpu
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    
    #print(type(image))
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    new_image = image.resize((600, 400))
    st.image(new_image, caption='Input Image',use_column_width=True)

    
    im = detection(new_image)
    st.header('Predicted Image')
    st.image(im,caption='Output Image',use_column_width=True)