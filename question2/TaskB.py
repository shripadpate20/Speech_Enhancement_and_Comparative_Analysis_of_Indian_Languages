import os
import numpy as np
import librosa
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths and languages
dataset_path = '/kaggle/input/audio-dataset-with-10-indian-languages/Language Detection Dataset'
languages = ['Hindi', 'Tamil', 'Bengali', 'Gujarati', 'Kannada', 
             'Malayalam', 'Marathi', 'Urdu', 'Punjabi', 'Telugu']
MAX_SAMPLES_PER_LANGUAGE = 5000

def extract_mfcc(file_path, n_mfcc=13):
    """Extract MFCC features from an audio file"""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_features(file_path, n_mfcc=13):
    """Extract and process MFCC features from an audio file"""
    mfcc = extract_mfcc(file_path, n_mfcc)
    if mfcc is None:
        return None
    # Calculate statistical features from MFCC
    mean = np.mean(mfcc, axis=1)  # Mean across time
    std = np.std(mfcc, axis=1)    # Standard deviation across time
    return np.concatenate([mean, std])  # Combine features

# Define neural network model
class LanguageClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout_rate=0.3):
        super(LanguageClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size2, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
        
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = valid_correct / valid_total
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
    
    return train_losses, train_accs, valid_losses, valid_accs

# Collect features and labels
X, y = [], []
file_counts = {}

# Process each language
for lang in languages:
    lang_path = os.path.join(dataset_path, lang)
    if not os.path.exists(lang_path):
        print(f"Directory not found: {lang_path}")
        continue
        
    # Get all audio files for this language
    audio_files = [f for f in os.listdir(lang_path) if f.endswith('.mp3')]
    total_files = len(audio_files)
    
    # Randomly select MAX_SAMPLES_PER_LANGUAGE files or use all if there are fewer
    if total_files > MAX_SAMPLES_PER_LANGUAGE:
        selected_files = random.sample(audio_files, MAX_SAMPLES_PER_LANGUAGE)
        print(f"{lang}: Randomly selected {MAX_SAMPLES_PER_LANGUAGE} from {total_files} files")
    else:
        selected_files = audio_files
        print(f"{lang}: Using all {total_files} files (less than {MAX_SAMPLES_PER_LANGUAGE})")
    
    file_counts[lang] = len(selected_files)
    
    # Process selected files
    for file in selected_files:
        file_path = os.path.join(lang_path, file)
        features = get_features(file_path)
        if features is not None:
            X.append(features)
            y.append(lang)

if len(X) == 0:
    raise ValueError("No audio files loaded. Check dataset path and structure!")

# Convert to numpy arrays and encode labels
X = np.array(X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Display dataset information
print(f"Total samples: {len(X)}")
print(f"Feature vector length: {X.shape[1]}")
print("Files per language:")
for lang, count in file_counts.items():
    print(f"  - {lang}: {count}")

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Initialize the model
input_dim = X_train.shape[1]
num_classes = 10
model = LanguageClassifier(input_dim, 128, 64, num_classes, dropout_rate=0.3)
model = model.to(device)
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
train_losses, train_accs, valid_losses, valid_accs = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
)

# Evaluate the model
model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_correct / test_total
print(f'Test Accuracy: {test_acc:.4f}')

# Generate detailed classification report
class_report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
print("\nClassification Report:")
print(class_report)

# Create confusion matrix visualization
plt.figure(figsize=(12, 10))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(train_accs)
plt.plot(valid_accs)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(train_losses)
plt.plot(valid_losses)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()

# Save the model and preprocessing components
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'input_dim': input_dim,
    'num_classes': num_classes
}, '/kaggle/working/language_model.pth')

import pickle
with open('/kaggle/working/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('/kaggle/working/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and preprocessing components saved to /kaggle/working/")

# Function to make predictions on new data
def predict_language(audio_path, model, scaler, label_encoder):
    # Extract features
    features = get_features(audio_path)
    if features is None:
        return "Error extracting features"
    
    # Normalize features
    features = scaler.transform(features.reshape(1, -1))
    
    # Convert to tensor and move to device
    features_tensor = torch.FloatTensor(features).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs, 1)
        
    # Convert to language name
    language = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
    
    return language