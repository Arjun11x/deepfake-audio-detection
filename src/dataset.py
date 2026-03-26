import warnings
warnings.filterwarnings("ignore", message=".*TorchCodec.*")

import os
import torch
import torchaudio
import torchaudio.functional as F_audio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class AudioDeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, target_sample_rate=16000, max_seconds=4, is_training=True):
        self.file_paths        = file_paths
        self.labels            = labels
        self.target_sample_rate = target_sample_rate
        self.max_length        = target_sample_rate * max_seconds
        self.is_training       = is_training

        # Mel spectrogram pipeline
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.target_sample_rate,
            n_mels      = 64,
            n_fft       = 1024,
            hop_length  = 512
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # SpecAugment (training only)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=35)

        # Compute fallback shape from a dummy waveform
        dummy_wave         = torch.zeros(1, self.max_length)
        self.fallback_mel  = self.db_transform(self.mel_transform(dummy_wave))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        try:
            waveform, sr = torchaudio.load(filepath, backend="soundfile")
        except Exception as e:
            print(f"[WARN] Failed to load {filepath}: {e}")
            return (
                torch.zeros(self.max_length),
                self.fallback_mel.clone(),
                torch.tensor(self.labels[idx], dtype=torch.long)
            )

        # Resample if needed
        if sr != self.target_sample_rate:
            waveform = F_audio.resample(waveform, orig_freq=sr, new_freq=self.target_sample_rate)

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or trim to max_length
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            pad_amount = self.max_length - waveform.shape[1]
            waveform   = F.pad(waveform, (0, pad_amount))

        # Time-domain augmentation (training only)
        if self.is_training and torch.rand(1).item() > 0.5:
            noise_amplitude = 0.01 * torch.rand(1).item()
            waveform        = waveform + (torch.randn_like(waveform) * noise_amplitude)

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        # Raw audio for Teacher (Wav2Vec)
        raw_audio = waveform.squeeze(0)

        # Mel spectrogram for Student (CNN)
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec)

        assert mel_spec.shape == self.fallback_mel.shape, \
            f"[ERROR] Mel shape mismatch at {filepath}: got {mel_spec.shape}, expected {self.fallback_mel.shape}"

        # Frequency-domain augmentation (training only)
        if self.is_training:
            mel_spec = self.freq_masking(mel_spec)
            mel_spec = self.time_masking(mel_spec)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return raw_audio, mel_spec, label


# ==========================================
# Quick self-test
# ==========================================
if __name__ == "__main__":
    import soundfile as sf
    import numpy as np

    print("=" * 50)
    print("   Deepfake Audio Dataset — Pipeline Tests")
    print("=" * 50)

    def create_dummy_wav(path, duration_sec, sample_rate):
        samples = np.random.randn(int(duration_sec * sample_rate)).astype(np.float32)
        sf.write(path, samples, sample_rate)

    dummy_files = []
    create_dummy_wav("test_normal.wav",  4.0, 16000); dummy_files.append("test_normal.wav")
    create_dummy_wav("test_short.wav",   1.0, 16000); dummy_files.append("test_short.wav")
    create_dummy_wav("test_long.wav",    6.0, 16000); dummy_files.append("test_long.wav")
    create_dummy_wav("test_48k.wav",     4.0, 48000); dummy_files.append("test_48k.wav")
    dummy_files.append("test_missing_file.wav")

    test_labels = [0, 1, 0, 1, 0]

    print("\n[ TEST A ] — Training Mode")
    dataset_train = AudioDeepfakeDataset(dummy_files, test_labels, is_training=True)
    descs = ["Normal 4s @ 16kHz", "Short 1s (padding)", "Long 6s (truncation)",
             "48kHz (resample)", "Missing (fallback)"]
    for i, desc in enumerate(descs):
        raw, mel, lbl = dataset_train[i]
        print(f"\n  Sample {i+1} — {desc}")
        print(f"    raw_audio : {raw.shape}   mel: {mel.shape}   label: {lbl.item()}")
        assert raw.shape == torch.Size([64000])
        assert mel.shape[1] == 64
        assert lbl.dtype == torch.long
        print(f"    ✅ OK")

    print("\n[ TEST B ] — DataLoader Batch")
    loader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0)
    for raw_b, mel_b, lbl_b in loader:
        print(f"  raw_batch: {raw_b.shape}  mel_batch: {mel_b.shape}")
        assert mel_b.shape[1] == 1
        print(f"  ✅ OK")
        break

    for f in dummy_files:
        if os.path.exists(f):
            os.remove(f)

    print("\n✅ ALL TESTS PASSED")
