import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, UniSpeechSatForXVector
from peft import LoraConfig, get_peft_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F
from tqdm import tqdm
from sklearn import metrics

# Define dataset class for VoxCeleb2 (training)
class VoxCeleb2Dataset(Dataset):
    def __init__(self, root_dir, identities, feature_extractor, max_frames=32000):
        self.root_dir = root_dir
        self.identities = identities
        self.feature_extractor = feature_extractor
        self.max_frames = max_frames
        self.data = self.load_data()
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform([d[0] for d in self.data])
    
    def load_data(self):
        data = []
        for identity in self.identities:
            id_path = os.path.join(self.root_dir, identity)
            if os.path.isdir(id_path):
                for folder in os.listdir(id_path):
                    folder_path = os.path.join(id_path, folder)
                    if os.path.isdir(folder_path):
                        for file in os.listdir(folder_path):
                            if file.endswith(".m4a"):
                                data.append((identity, os.path.join(folder_path, file)))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def process_audio(self, file_path):
        wav, sample_rate = torchaudio.load(file_path)
        num_frames = wav.shape[1]
        if num_frames >= self.max_frames:
            wav = wav[:, :self.max_frames]
        else:
            pad_size = self.max_frames - num_frames
            wav = F.pad(wav, (0, pad_size), value=0)
        return wav, sample_rate
    
    def __getitem__(self, idx):
        label, file_path = self.data[idx]
        wav, _ = self.process_audio(file_path)
        wav = wav.squeeze(0)  # Convert (1, T) -> (T)
        features = self.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_values.squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

# Define dataset class for VoxCeleb1 (testing/evaluation)
class VoxCeleb1Dataset(Dataset):
    def __init__(self, root_dir, trials_file, max_frames=32000):
        self.root_dir = root_dir
        self.trials_file = trials_file
        self.max_frames = max_frames
        self.data = self.read_file()
    
    def read_file(self):
        data = []
        with open(self.trials_file, 'r') as fil:
            for line in fil:
                label, first, second = line.strip().split()
                first_path = os.path.join(self.root_dir, first)
                second_path = os.path.join(self.root_dir, second)
                if os.path.exists(first_path) and os.path.exists(second_path):
                    data.append((int(label), first_path, second_path))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def process_sample(self, file_path):
        wav, sample_rate = torchaudio.load(file_path)
        # Resample if necessary
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            wav = resampler(wav)
        
        # Handle mono/stereo
        if wav.shape[0] > 1:  # If stereo, convert to mono
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        # Handle length
        num_frames = wav.shape[1]
        if num_frames >= self.max_frames:
            wav = wav[:, :self.max_frames]
        else:
            pad_size = self.max_frames - num_frames
            wav = F.pad(wav, (0, pad_size), value=0)
            
        return wav.squeeze(0)  # Return as (T) not (1, T)
    
    def __getitem__(self, idx):
        label, first_path, second_path = self.data[idx]
        first_wav = self.process_sample(first_path)
        second_wav = self.process_sample(second_path)
        label = torch.tensor(label, dtype=torch.float32)
        return first_wav, second_wav, label

# Evaluation metrics functions
def compute_eer(labels, preds):
    """Compute Equal Error Rate"""
    fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label=1)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    return eer * 100

def compute_tar_far(labels, preds, far_target=0.01):
    """Compute True Accept Rate at specified False Accept Rate"""
    fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label=1)
    far_index = np.where(fpr <= far_target)[0]
    tar = tpr[far_index[-1]] if len(far_index) > 0 else 0.0
    return tar * 100

def compute_speaker_identification_accuracy(labels, preds, threshold=0.5):
    """Compute Speaker Identification Accuracy"""
    pred_labels = (np.array(preds) >= threshold).astype(int)
    accuracy = metrics.accuracy_score(labels, pred_labels)
    return accuracy * 100

# Evaluation function
def evaluate(model, test_loader, feature_extractor, device):
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    model.eval()
    all_labels, all_scores = [], []
    
    with torch.no_grad():
        for batch_idx, (wav1, wav2, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            batch_size = wav1.size(0)
            wav1_embeds = []
            wav2_embeds = []
            
            # Process each audio sample individually to avoid dimension issues
            for i in range(batch_size):
                # Process first waveform
                audio1 = feature_extractor(
                    wav1[i].numpy(), 
                    return_tensors="pt", 
                    sampling_rate=16000
                ).input_values.to(device)
                
                # Process second waveform
                audio2 = feature_extractor(
                    wav2[i].numpy(), 
                    return_tensors="pt", 
                    sampling_rate=16000
                ).input_values.to(device)
                
                # Extract embeddings
                emb1 = F.normalize(model(input_values=audio1).embeddings, dim=1)
                emb2 = F.normalize(model(input_values=audio2).embeddings, dim=1)
                
                wav1_embeds.append(emb1)
                wav2_embeds.append(emb2)
            
            # Stack all embeddings
            wav1_embeds = torch.cat(wav1_embeds, dim=0)
            wav2_embeds = torch.cat(wav2_embeds, dim=0)
            
            # Compute similarities
            similarity = torch.sigmoid(cos_sim(wav1_embeds, wav2_embeds)).cpu().numpy()
            
            all_scores.extend(similarity)
            all_labels.extend(labels.numpy())
            
            if batch_idx % 20 == 0 and batch_idx > 0:
                print(f"Processed {batch_idx}/{len(test_loader)} batches")
    
    eer = compute_eer(all_labels, all_scores)
    tar = compute_tar_far(all_labels, all_scores)
    accuracy = compute_speaker_identification_accuracy(all_labels, all_scores)
    
    print(f"Evaluation Results:")
    print(f"EER: {eer:.2f}%")
    print(f"TAR@1%FAR: {tar:.2f}%")
    print(f"Speaker Identification Accuracy: {accuracy:.2f}%")
    
    return eer, tar, accuracy

# Main function to run both training and evaluation
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up paths and configuration
    vox2_root = "/kaggle/input/voxceleb1-and-2/vox2_test_aac/aac"  # Change this to your VoxCeleb2 path
    vox1_root = "/kaggle/input/voxceleb1-and-2/vox1_test_wav/wav"  # Change this to your VoxCeleb1 path
    trials_file = "/kaggle/input/text-data/veri_test2.txt"  # Change this to your trials file
    
    # Initialize feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/unispeech-sat-large-sv')
    
    # Load the training dataset
    identities = sorted(os.listdir(vox2_root))
    train_identities = identities[:100]
    test_identities = identities[100:118]
    
    print("Loading training dataset...")
    train_dataset = VoxCeleb2Dataset(vox2_root, train_identities, feature_extractor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Load model and apply LoRA
    print("Loading model and applying LoRA...")
    model = UniSpeechSatForXVector.from_pretrained("microsoft/unispeech-sat-large-sv")
    
    target_modules = ["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"]
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["classifier"]  # Keep original classifier trainable
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Verify LoRA application
    model.to(device)
    
    # ArcFace Loss
    class ArcFaceLoss(nn.Module):
        def __init__(self, embedding_size, num_classes, s=30.0, m=0.5):
            super(ArcFaceLoss, self).__init__()
            self.s = s
            self.m = m
            self.W = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
            nn.init.xavier_uniform_(self.W)
        
        def forward(self, embeddings, labels):
            cosine = F.linear(F.normalize(embeddings), F.normalize(self.W))
            theta = torch.acos(torch.clamp(cosine, -1.0, 1.0))
            target_logit = torch.cos(theta + self.m)
            one_hot = F.one_hot(labels, num_classes=self.W.shape[0]).float()
            logits = one_hot * target_logit + (1 - one_hot) * cosine
            logits *= self.s
            return logits
    
    # Define loss and optimizer
    num_classes = len(set(train_dataset.labels))
    arcface_loss = ArcFaceLoss(embedding_size=512, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    epochs = 10
    best_eer = float('inf')
    best_model_path = "best_model.pth"
    
    # Prepare evaluation dataset early to be able to evaluate during training
    print("Loading evaluation dataset...")
    eval_dataset = VoxCeleb1Dataset(vox1_root, trials_file)
    
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Evaluate base model before fine-tuning
    print("Evaluating base model before fine-tuning...")
    pre_eer, pre_tar, pre_acc = evaluate(model, eval_loader, feature_extractor, device)
    
    print("Starting fine-tuning...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(input_values=inputs).embeddings
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            logits = arcface_loss(normalized_embeddings, labels)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluate after each epoch
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:  # Evaluate every 2 epochs or at the end
            print(f"Evaluating after epoch {epoch+1}...")
            current_eer, current_tar, current_acc = evaluate(model, eval_loader, feature_extractor, device)
            
            # Save best model based on EER
            if current_eer < best_eer:
                best_eer = current_eer
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss/len(train_loader),
                    'eer': current_eer,
                    'tar': current_tar,
                    'accuracy': current_acc
                }, best_model_path)
                print(f"New best model saved with EER: {best_eer:.2f}%")
    
    print("Training complete!")
    
    # Load best model for final evaluation
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Performing final evaluation on VoxCeleb1...")
    final_eer, final_tar, final_acc = evaluate(model, eval_loader, feature_extractor, device)
    
    print("Base Model vs Fine-tuned Model Comparison:")
    print(f"Base Model - EER: {pre_eer:.2f}%, TAR@1%FAR: {pre_tar:.2f}%, Accuracy: {pre_acc:.2f}%")
    print(f"Fine-tuned Model - EER: {final_eer:.2f}%, TAR@1%FAR: {final_tar:.2f}%, Accuracy: {final_acc:.2f}%")
    print(f"Improvement - EER: {pre_eer - final_eer:.2f}%, TAR@1%FAR: {final_tar - pre_tar:.2f}%, Accuracy: {final_acc - pre_acc:.2f}%")

if __name__ == "__main__":
    main()