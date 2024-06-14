


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define the root directory where the dataset is located
root_dir = r'"C:\Users\vaishnavi\onedrive_backup\Desktop\BE\projectFinal\deepfakemodules\given\TrimDataset\TrimDataset"'

# Define paths to different parts of the dataset
dataset_dir = os.path.join(root_dir, "Dataset")
test_dir = os.path.join(dataset_dir, "Test")
train_dir = os.path.join(dataset_dir, "Train")
validation_dir = os.path.join(dataset_dir, "Validation")
deepfakegenerated_dir = os.path.join(root_dir, "deepfakegenerated")

# Define paths to subdirectories within each part of the dataset
test_fake_dir = os.path.join(test_dir, "Fake")
test_real_dir = os.path.join(test_dir, "Real")
train_fake_dir = os.path.join(train_dir, "Fake")
train_real_dir = os.path.join(train_dir, "Real")
validation_fake_dir = os.path.join(validation_dir, "Fake")
validation_real_dir = os.path.join(validation_dir, "Real")

# Function to count the number of files in a directory
def count_files(directory):
    return len(os.listdir(directory))

# Print the number of files in each directory
print("Number of files in Test/Fake directory:", count_files(test_fake_dir))
print("Number of files in Test/Real directory:", count_files(test_real_dir))
print("Number of files in Train/Fake directory:", count_files(train_fake_dir))
print("Number of files in Train/Real directory:", count_files(train_real_dir))
print("Number of files in Validation/Fake directory:", count_files(validation_fake_dir))
print("Number of files in Validation/Real directory:", count_files(validation_real_dir))

# Optional: Print the contents of the deepfakegenerated directory
print("Contents of deepfakegenerated directory:", os.listdir(deepfakegenerated_dir))

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define image dimensions and batch size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Create data generators with data augmentation for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load pre-trained MobileNetV2 model (excluding top layers)
base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers for deepfake detection
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Combine base model with custom top layers
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model with fewer epochs for faster training
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=5,  # Adjust the number of epochs as per your liking
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator, steps=len(validation_generator))
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)




# Evaluate the model
loss, accuracy = model.evaluate(validation_generator, steps=len(validation_generator))
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)




# Define the path where you want to save the model
model_save_path = r"C:\Users\vaishnavi\onedrive_backup\Desktop\BE\projectFinal\deepfakemodules\given\model"

# Save the model
model.save(model_save_path)

print("Model saved successfully at:", model_save_path)




# Save the model with the name "deepfake_verification"
model.save(r"C:\Users\vaishnavi\onedrive_backup\Desktop\BE\projectFinal\deepfakemodules\given\deepfake_verification_model\deepfake_verification")





import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the saved model
model_path = r'"C:\Users\vaishnavi\onedrive_backup\Desktop\BE\projectFinal\deepfakemodules\given\deepfake_verification_model\deepfake_verification"'
model = tf.keras.models.load_model(model_path)

# Define function to preprocess image
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
    prediction = model.predict(preprocessed_image)
    # Determine the class label
    if prediction < 0.5:
        class_label = "Real"
    else:
        class_label = "Fake"
    return class_label, preprocessed_image

# Set the image path
image_path = r'"C:\Users\vaishnavi\onedrive_backup\Desktop\newProject\Images\gjfhut.jpg"'

# Verify whether the input image is fake or real and get the preprocessed image
prediction_result, preprocessed_image = classify_image(image_path)

# Cast the preprocessed image data to the correct data type
preprocessed_image = (preprocessed_image * 255).astype(np.uint8)

# Display the result along with the image using Matplotlib
plt.imshow(cv2.cvtColor(preprocessed_image[0], cv2.COLOR_BGR2RGB))
plt.title(f"This image is {prediction_result}")
plt.axis('off')
plt.show()






