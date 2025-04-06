import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier



# Configuration
languages = ['Hindi', 'Tamil', 'Marathi'] 
dataset_path = '/kaggle/input/audio-dataset-with-10-indian-languages/Language Detection Dataset/'  
num_samples_to_visualize = 2
num_samples_for_stats = 50 
n_mfcc = 13

def compute_mfcc(file_path):
    """Load audio and compute MFCCs."""
    y, sr = librosa.load(file_path, sr=22050)  # Standardize sampling rate
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

plt.figure(figsize=(15, 10))
for i, lang in enumerate(languages):
    lang_dir = os.path.join(dataset_path, lang)
    audio_files = [f for f in os.listdir(lang_dir) if f.endswith('.mp3')]
    selected_files = audio_files[:num_samples_to_visualize]
    for j, file in enumerate(selected_files):
        file_path = os.path.join(lang_dir, file)
        mfccs = compute_mfcc(file_path)
        plt.subplot(len(languages), num_samples_to_visualize, i * num_samples_to_visualize + j + 1)
        librosa.display.specshow(mfccs, x_axis='time', sr=22050)
        plt.colorbar()
        plt.title(f'{lang} Sample {j+1}')
plt.tight_layout()
plt.show()


languages = ['Hindi', 'Tamil', 'Bengali', 'Gujarati', 'Kannada', 
             'Malayalam', 'Marathi', 'Urdu', 'Punjabi', 'Telugu']



stats = {}
for lang in languages:
    lang_dir = os.path.join(dataset_path, lang)
    audio_files = [f for f in os.listdir(lang_dir) if f.endswith('.mp3')][:num_samples_for_stats]
    all_mfccs = []
    for file in audio_files:
        file_path = os.path.join(lang_dir, file)
        mfccs = compute_mfcc(file_path)
        all_mfccs.append(mfccs)
    if all_mfccs:
        all_mfccs_concat = np.concatenate(all_mfccs, axis=1)
        stats[lang] = {
            'mean': np.mean(all_mfccs_concat, axis=1),
            'variance': np.var(all_mfccs_concat, axis=1)
        }

# Plot Mean MFCCs
plt.figure(figsize=(10, 6))
for lang in languages:
    if lang in stats:
        plt.plot(stats[lang]['mean'], label=lang, marker='o')
plt.xlabel('MFCC Coefficient Index')
plt.ylabel('Mean Value')
plt.title('Mean MFCC Coefficients Across Languages')
plt.legend()
plt.grid()
plt.show()

# Plot Variance
plt.figure(figsize=(10, 6))
for lang in languages:
    if lang in stats:
        plt.plot(stats[lang]['variance'], label=lang, marker='o')
plt.xlabel('MFCC Coefficient Index')
plt.ylabel('Variance')
plt.title('Variance of MFCC Coefficients Across Languages')
plt.legend()
plt.grid()
plt.show()



