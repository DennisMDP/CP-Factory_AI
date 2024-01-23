import streamlit as st
from backend.ImageClassifier import ImageClassifier
from PIL import Image, ImageTk
import io
from time import sleep


# create image classifier object
image_classifier = ImageClassifier()

# Set Streamlit title
st.title("Image Classifier")

# Set default image
default_img = Image.open("/Users/denniskollmann/Desktop/CPF_AI/Classification_App/CP-Factory_AI/default_img.png")

# Display default image
st.image(default_img, caption="Default Image", use_column_width=True)

# Text label for predicted class
class_label = st.empty()

def update_gui():
    '''
    Check web service if there is a new picture.
    If there is a new picture: Predict the class and update GUI with new picture and associated class.
    '''
    
    # call current image from web service
    img = image_classifier.get_image()
    
    # check if the image has been changed
    if not image_classifier.are_images_equal():
        # convert image for GUI label
        # img_pil = Image.open(io.BytesIO(img)).convert("RGB")
        # Display the updated image
        st.image(img, use_column_width=True)
        # predict image class
        image_class = image_classifier.predict_class()
        # update class label
        class_label.text(f"Klasse: {image_class}")
        # warning bell if the class is "handyschale_falsch"
        if image_class == "handyschale_falsch":
            st.balloons()  # Streamlit balloons effect as a substitute for the root.bell()
    
    # repeat update-function every 2 seconds
    sleep(2)
    update_gui()

# Start update-loop
# update_gui()