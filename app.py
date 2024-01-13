import tkinter as tk
from backend.ImageClassifier import ImageClassifier
from PIL import Image, ImageTk
import io


# create image classifier object
image_classifier = ImageClassifier()

# create gui root window
root = tk.Tk()
# window title
root.title("Image Classifier")

# GUI start configuration
# set image label with default image
default_img = ImageTk.PhotoImage(Image.open("/Users/denniskollmann/Desktop/CPF_AI/Classification_App/CP-Factory_AI/default_img.png"))
image_label = tk.Label(root, image=default_img)
image_label.pack(side="top")
# text label for predicted class
class_label = tk.Label(root, text="Klasse: ", font=("Helvetica", 30))
class_label.pack(side="top")


def update_gui():
    '''
    Check web service if there is a new picture.
    If there is a new picture: Predict the class and  update GUI with new picture and associated class.
    '''
    
    # call current image from web service
    img = image_classifier.get_image()
    
    # check if the image has been changed
    if not image_classifier.are_images_equal():
        # convert image for GUI label
        img_tk = ImageTk.PhotoImage(Image.open(io.BytesIO(img)).convert("RGB"))
        # update image label
        image_label.configure(image=img_tk)
        image_label.image = img_tk
        # predict image class
        image_class = image_classifier.predict_class()
        # update class label
        class_label.configure(text=f"Klasse: {image_class}")
        # warning bell if the class is "handyschale_falsch"
        if image_class == "handyschale_falsch":
            root.bell()
    # repeat update-function every 2 seconds
    root.after(2000, update_gui)

# start update-loop
update_gui()

# start GUI main loop
root.mainloop()
