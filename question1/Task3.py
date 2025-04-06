import os
import random
import torchaudio
import numpy as np
import soundfile as sf
from speechbrain.inference.separation import SepformerSeparation as separator
from pesq import pesq
from mir_eval.separation import bss_eval_sources

def get_speaker_files(voxceleb_root, speaker_ids):
    """Recursively get all .m4a audio files for the given speaker IDs."""
    speaker_files = {}
    
    for speaker in speaker_ids:
        speaker_dir = os.path.join(voxceleb_root, speaker)
        if os.path.isdir(speaker_dir):
            # Fixed typo: speaker*dir -> speaker_dir
            files = [os.path.join(root, f) 
                     for root, _, filenames in os.walk(speaker_dir) 
                     for f in filenames if f.endswith('.m4a')]
            
            if files:
                speaker_files[speaker] = files
            else:
                print(f"Warning: No .m4a files found for {speaker}")
    
    return speaker_files

def mix_speakers(speaker_files, output_dir, sample_rate=16000):
    """Mix two speakers' utterances and save the mixed audio with original sources."""
    os.makedirs(output_dir, exist_ok=True)
    mixed_files = []
    sources_dict = {}
    
    speakers = list(speaker_files.keys())
    for i in range(0, len(speakers), 2):
        if i+1 >= len(speakers):
            break
            
        spk1, spk2 = speakers[i], speakers[i+1]
        if not speaker_files[spk1] or not speaker_files[spk2]:
            print(f"Skipping {spk1} or {spk2} due to no available audio files.")
            continue
            
        file1 = random.choice(speaker_files[spk1])
        file2 = random.choice(speaker_files[spk2])
        
        wav1, sr1 = torchaudio.load(file1)
        wav2, sr2 = torchaudio.load(file2)
        
        if sr1 != sample_rate:
            wav1 = torchaudio.transforms.Resample(sr1, sample_rate)(wav1)
        if sr2 != sample_rate:
            wav2 = torchaudio.transforms.Resample(sr2, sample_rate)(wav2)
            
        min_len = min(wav1.shape[1], wav2.shape[1])
        wav1, wav2 = wav1[:, :min_len], wav2[:, :min_len]
        
        # Mix audio
        mixed_wav = (wav1 + wav2) / 2
        
        # Save mixed audio
        mixed_file = os.path.join(output_dir, f'mixed_{spk1}_{spk2}.wav')
        sf.write(mixed_file, mixed_wav.squeeze().numpy(), sample_rate, format='WAV')
        
        # Save original sources for PESQ calculation
        src1_file = os.path.join(output_dir, f'src1_{spk1}_{spk2}.wav')
        src2_file = os.path.join(output_dir, f'src2_{spk1}_{spk2}.wav')
        
        sf.write(src1_file, wav1.squeeze().numpy(), sample_rate, format='WAV')
        sf.write(src2_file, wav2.squeeze().numpy(), sample_rate, format='WAV')
        
        # Store sources info
        sources_dict[mixed_file] = {
            'sources': np.vstack([wav1.squeeze().numpy(), wav2.squeeze().numpy()]),
            'source_files': [src1_file, src2_file],
            'sample_rate': sample_rate
        }
        
        mixed_files.append(mixed_file)
        
    return mixed_files, sources_dict

def separate_and_evaluate(mixed_files, sources_dict, model):
    """Perform speaker separation and evaluate using SIR, SAR, SDR, and PESQ."""
    metrics = {}
    
    for mixed_file in mixed_files:
        try:
            # Get original mixed audio to compare length
            mixed_audio, sr = torchaudio.load(mixed_file)
            mixed_length = mixed_audio.shape[1]
            
            # Get original sources
            ref_sources = sources_dict[mixed_file]['sources']  # Shape: [num_spk, num_samples]
            sample_rate = sources_dict[mixed_file]['sample_rate']
            
            # Debug info
            print(f"Mixed file: {mixed_file}")
            print(f"Reference sources shape: {ref_sources.shape}")
            
            # Get separated sources from model
            est_sources = model.separate_file(path=mixed_file)
            est_sources = est_sources.cpu().numpy()  # Shape: [1, num_spk, num_samples]
            est_sources = est_sources.squeeze(0)  # Remove batch dimension: [num_spk, num_samples]
            
            print(f"Estimated sources shape: {est_sources.shape}")
            
            # Ensure both have the same number of sources
            if ref_sources.shape[0] != est_sources.shape[0]:
                print(f"Warning: Number of sources don't match. "
                      f"Reference: {ref_sources.shape[0]}, Estimated: {est_sources.shape[0]}")
                # If needed, adjust number of sources to match
                min_sources = min(ref_sources.shape[0], est_sources.shape[0])
                ref_sources = ref_sources[:min_sources]
                est_sources = est_sources[:min_sources]
            
            # Ensure same length (truncate to shorter one)
            min_len = min(ref_sources.shape[1], est_sources.shape[1])
            ref_sources = ref_sources[:, :min_len]
            est_sources = est_sources[:, :min_len]
            
            print(f"After adjustment - Reference: {ref_sources.shape}, Estimated: {est_sources.shape}")
            
            # Transpose if needed - mir_eval expects shape [n_sources, n_samples]
            if ref_sources.shape[0] > ref_sources.shape[1]:
                ref_sources = ref_sources.T
            if est_sources.shape[0] > est_sources.shape[1]:
                est_sources = est_sources.T
                
            # Calculate BSS metrics - use permutation invariant version
            sdr, sir, sar, _ = bss_eval_sources(ref_sources, est_sources)
            
            # Calculate PESQ (use first separated source against first reference)
            try:
                pesq_score = pesq(sample_rate, ref_sources[0], est_sources[0], 'wb')
            except Exception as e:
                print(f"PESQ calculation failed: {e}")
                pesq_score = float('nan')
            
            metrics[os.path.basename(mixed_file)] = {
                'SDR': float(np.mean(sdr)),
                'SIR': float(np.mean(sir)),
                'SAR': float(np.mean(sar)),
                'PESQ': float(pesq_score)
            }
            
        except Exception as e:
            print(f"Error processing {mixed_file}: {e}")
            metrics[os.path.basename(mixed_file)] = {
                'SDR': float('nan'),
                'SIR': float('nan'),
                'SAR': float('nan'),
                'PESQ': float('nan')
            }
    
    return metrics

# Paths
voxceleb_root = "/kaggle/input/voxceleb1-and-2/vox2_test_aac/aac"
train_output_dir = "mixed_train"
test_output_dir = "mixed_test"

# Get speaker lists
all_speakers = sorted(os.listdir(voxceleb_root))
train_speakers = all_speakers[:50]
test_speakers = all_speakers[50:100]

# Get speaker audio files
train_files = get_speaker_files(voxceleb_root, train_speakers)
test_files = get_speaker_files(voxceleb_root, test_speakers)

# Mix speakers - here's where we save the original reference sources
train_mixed, train_sources = mix_speakers(train_files, train_output_dir)
test_mixed, test_sources = mix_speakers(test_files, test_output_dir)

# Print debug info about the first mixed file
if test_mixed:
    first_file = test_mixed[0]
    print(f"First mixed file: {first_file}")
    if first_file in test_sources:
        print(f"Sources shape: {test_sources[first_file]['sources'].shape}")
        print(f"Sample rate: {test_sources[first_file]['sample_rate']}")

# Load SepFormer model
sepformer = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')

# Evaluate separation
metrics = separate_and_evaluate(test_mixed, test_sources, sepformer)

# Print metrics
print("\nSeparation Performance Metrics:")
for file, file_metrics in metrics.items():
    print(f"\n{file}:")
    for metric, value in file_metrics.items():
        print(f"  {metric}: {value:.4f}")

# Calculate average metrics (excluding NaN values)
avg_metrics = {
    metric: np.nanmean([m[metric] for m in metrics.values()]) 
    for metric in ['SDR', 'SIR', 'SAR', 'PESQ']
}

print("\nAverage Metrics:")
for metric, value in avg_metrics.items():
    print(f"  {metric}: {value:.4f}")