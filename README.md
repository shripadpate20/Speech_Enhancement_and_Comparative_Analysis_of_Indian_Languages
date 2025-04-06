# Question 1
# Speaker Analysis Project

This repository contains code for two main audio processing tasks:

1. **Speaker Identification**: Using the UniSpeechSat model with LoRA fine-tuning to identify speakers from voice samples
2. **Speaker Separation**: Using SepFormer to separate mixed audio into individual speaker tracks

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Speaker Identification](#speaker-identification)
  - [Features](#features)
  - [Dataset](#dataset)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Speaker Separation](#speaker-separation)
  - [Features](#features-1)
  - [Methodology](#methodology)
  - [Evaluation Metrics](#evaluation-metrics)
- [Usage Examples](#usage-examples)
- [Results](#results)

## Requirements

```
torch
torchaudio
transformers
peft
scikit-learn
numpy
tqdm
soundfile
speechbrain
pesq
mir_eval
```

## Project Structure

The project consists of two main scripts:

1. `speaker_identification.py` - Trains and evaluates a speaker identification model using UniSpeechSat with LoRA
2. `speaker_separation.py` - Demonstrates speaker separation using SepFormer and evaluates the separation quality

## Speaker Identification

### Features

- Uses Microsoft's UniSpeechSat model for X-vector extraction
- Implements Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Uses ArcFace loss for better speaker discrimination
- Evaluates using Equal Error Rate (EER) and True Accept Rate (TAR)

### Dataset

The code uses VoxCeleb datasets:
- **VoxCeleb2**: Used for training the speaker identification model
- **VoxCeleb1**: Used for evaluation

### Training

The training process includes:

1. Loading the UniSpeechSat base model
2. Applying LoRA to specific attention modules
3. Adding ArcFace loss for improved speaker embedding discrimination
4. Fine-tuning for 10 epochs with evaluation every 2 epochs

### Evaluation

The model is evaluated using:
- **Equal Error Rate (EER)**: Lower is better
- **True Accept Rate at 1% False Accept Rate (TAR@1%FAR)**: Higher is better
- **Speaker Identification Accuracy**: Higher is better

## Speaker Separation

### Features

- Uses SpeechBrain's SepFormer model for separating mixed audio
- Creates a dataset of mixed speaker recordings from VoxCeleb
- Preserves original source files for evaluation

### Methodology

1. Speaker audio files are loaded from VoxCeleb
2. Two speakers' audio files are combined to create mixed audio
3. SepFormer model separates the mixed audio into individual speakers
4. Evaluation metrics compare the separated audio with original sources

### Evaluation Metrics

The separation performance is measured using:
- **Signal-to-Distortion Ratio (SDR)**: Higher is better
- **Signal-to-Interference Ratio (SIR)**: Higher is better
- **Signal-to-Artifact Ratio (SAR)**: Higher is better
- **Perceptual Evaluation of Speech Quality (PESQ)**: Higher is better

## Usage Examples

### Speaker Identification

```python
# The main() function runs the entire training and evaluation pipeline
# Set your paths first
vox2_root = "/path/to/voxceleb2"
vox1_root = "/path/to/voxceleb1"
trials_file = "/path/to/veri_test.txt"

# Run training and evaluation
main()
```

### Speaker Separation

```python
# Set paths to your VoxCeleb dataset
voxceleb_root = "/path/to/voxceleb"
train_output_dir = "mixed_train"
test_output_dir = "mixed_test"

# Get speaker files and create mixed audio
train_mixed, train_sources = mix_speakers(train_files, train_output_dir)
test_mixed, test_sources = mix_speakers(test_files, test_output_dir)

# Load SepFormer model and evaluate separation
sepformer = separator.from_hparams(source="speechbrain/sepformer-whamr", 
                                  savedir='pretrained_models/sepformer-whamr')
metrics = separate_and_evaluate(test_mixed, test_sources, sepformer)
```

## Results

The speaker identification model compares base performance versus fine-tuned performance:
- Base Model: EER, TAR@1%FAR, and Accuracy metrics
- Fine-tuned Model: Improved EER, TAR@1%FAR, and Accuracy metrics
- Detailed improvement metrics are reported during evaluation

The speaker separation evaluation provides:
- Individual metrics for each separated file
- Average SDR, SIR, SAR, and PESQ across all test files

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
