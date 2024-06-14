import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

audio_dataset_path = r"C:\Users\vaishnavi\onedrive_backup\Desktop\BE\projectFinal\AI FAKE VOICE DETECTION\archive (8)\KAGGLE\AUDIO"
num_labels = 2

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename()
    return file_path

def soundPredict():
    filename = select_file()
    print("Selected file:", filename)

    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    print(mfccs_scaled_features)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    print(mfccs_scaled_features)
    print(mfccs_scaled_features.shape)
    mfccs_scaled_features = sc.transform(mfccs_scaled_features)
    predicted_label = classifier.predict(mfccs_scaled_features)
    print(predicted_label)

    predicted_class = disease[predicted_label[0]]

    return predicted_class

# Load the data
extracted_features = []

for folder in os.listdir(audio_dataset_path):
    print(folder)
    for file_name in os.listdir(audio_dataset_path + '/' + folder + '/'):
        data = features_extractor(audio_dataset_path + '/' + folder + '/' + file_name)
        extracted_features.append([data, folder])

extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])

X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

# Feature scaling
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_scaled, y_encoded)

# Define the disease classes
disease = [
    "Fake",
    "Real",
]

print(soundPredict())
