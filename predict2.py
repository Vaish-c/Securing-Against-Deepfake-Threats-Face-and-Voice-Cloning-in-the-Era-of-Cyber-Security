import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
from PIL import Image, ImageTk

# Load the saved model using TFSMLayer
model_path = r"C:\Users\vaishnavi\onedrive_backup\Desktop\BE\projectFinal\deepfakemodules\captured_frames\given\deepfake_verification_model\deepfake_verification"
model = tf.keras.Sequential([
    TFSMLayer(model_path, call_endpoint='serving_default')
])

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to classify the image
def classify_image(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Predict using the loaded model
    prediction = model(preprocessed_image)
    # Extract the actual prediction value from the dictionary
    prediction_value = prediction['dense_1'][0][0]  # Use 'dense_1' key to access prediction value
    # Determine the class label
    if prediction_value < 0.5:
        class_label = "Real"
    else:
        class_label = "Fake"
    return class_label, preprocessed_image

# Function to select an image
def select_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Classify the selected image
        prediction_result, preprocessed_image = classify_image(file_path)
        # Cast the preprocessed image data to the correct data type
        preprocessed_image = (preprocessed_image * 255).astype(np.uint8)
        
        # Display the image in the GUI
        img = Image.fromarray(cv2.cvtColor(preprocessed_image[0], cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img)
        panel.img_tk = img_tk  # Keep a reference to avoid garbage collection
        panel.config(image=img_tk)
        
        # Show the prediction result
        messagebox.showinfo("Prediction Result", f"This image is {prediction_result}")

# Create the main window
root = tk.Tk()
root.title("Deepfake Verification")

# Create a button to select an image
btn = tk.Button(root, text="Select Image", command=select_image)
btn.pack()

# Create a panel to display the image
panel = tk.Label(root)
panel.pack()

# Start the GUI event loop
root.mainloop()
