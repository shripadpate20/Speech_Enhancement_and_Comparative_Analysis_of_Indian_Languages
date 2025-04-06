# Question 2: Indian Language Audio Analysis and Classification

## Overview
This repository contains code for analyzing and classifying Indian languages from audio files. The project uses machine learning techniques to identify spoken languages based on audio features extracted using librosa.

## Dataset
The code works with the "Audio Dataset with 10 Indian Languages" from Kaggle, which includes audio samples of:
- Hindi
- Tamil
- Bengali
- Gujarati
- Kannada
- Malayalam
- Marathi
- Urdu
- Punjabi
- Telugu

## Features
This implementation offers:
1. Audio feature extraction using MFCCs (Mel-Frequency Cepstral Coefficients)
2. Visualization of audio features across different languages
3. Statistical analysis of MFCC distributions by language
4. Multiple classification models:
   - Neural network classifier using PyTorch
   - Random Forest classifier
   - SVM classifier
   - MLP classifier

## Code Structure
The repository contains:
1. **Audio Visualization:** Code to visualize MFCCs for different language samples
2. **Feature Analysis:** Statistical plots showing means and variances of MFCCs across languages
3. **Model Implementation:** Complete neural network pipeline including:
   - Data preprocessing and normalization
   - Model architecture definition
   - Training and evaluation functions
   - Performance visualization

## Neural Network Architecture
The PyTorch model uses a 3-layer neural network with:
- Input layer matching feature dimensions
- Two hidden layers (128 and 64 neurons)
- Output layer for 10 classes (languages)
- Dropout regularization (0.3)
- ReLU activation functions

## Results
The neural network achieves competitive accuracy in distinguishing between Indian languages based solely on audio features. Performance metrics include:
- Classification accuracy
- Per-language precision and recall
- Confusion matrix visualization
- Training and validation curves

## Usage
1. Set up the dataset path
2. Run the visualization code to understand audio features
3. Execute the model training pipeline
4. Evaluate model performance with test data
5. Use the trained model to predict languages for new audio samples

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Librosa
- PyTorch
- Matplotlib
- Seaborn
- Scikit-learn
- tqdm

## Future Improvements
Potential enhancements include:
- Incorporating additional audio features beyond MFCCs
- Experimenting with CNN and RNN architectures
- Implementing data augmentation techniques
- Exploring pre-trained audio models and transfer learning approaches
