import os
import random
import torch
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


# ==========================================
# Dataset Loader
# ==========================================
def load_asvspoof2019(dataset_root, audio_dirs, protocol_files, subset="train", max_samples=None, balanced=True):
    """
    Load ASVspoof 2019 LA file paths and labels from protocol file.

    Args:
        dataset_root   : root path of the LA folder
        audio_dirs     : dict mapping subset → audio folder path  (from config)
        protocol_files : dict mapping subset → protocol .txt path (from config)
        subset         : "train" | "dev" | "eval"
        max_samples    : cap total samples (balanced: half real, half fake)
        balanced       : if True, enforce equal real/fake counts (use for train/sweep)
                         if False, use natural distribution  (use for dev/eval EER)

    Returns:
        file_paths : list of .flac file paths
        labels     : list of ints (0=real, 1=fake)
    """
    audio_dir     = audio_dirs[subset]
    protocol_path = protocol_files[subset]

    assert os.path.exists(audio_dir),     f"[ERROR] Audio folder not found: {audio_dir}"
    assert os.path.exists(protocol_path), f"[ERROR] Protocol file not found: {protocol_path}"

    real_paths = []
    fake_paths = []

    with open(protocol_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts     = line.strip().split()
        file_id   = parts[1]
        label_str = parts[4]
        filepath  = os.path.join(audio_dir, file_id + ".flac")

        if not os.path.exists(filepath):
            print(f"[WARN] File not found, skipping: {filepath}")
            continue

        if label_str == "bonafide":
            real_paths.append(filepath)
        else:
            fake_paths.append(filepath)

    random.shuffle(real_paths)
    random.shuffle(fake_paths)

    if balanced and max_samples is not None:
        samples_per_class = max_samples // 2
        real_paths = real_paths[:samples_per_class]
        fake_paths = fake_paths[:samples_per_class]
    elif not balanced and max_samples is not None:
        # Keep natural ratio but cap total
        total       = len(real_paths) + len(fake_paths)
        real_ratio  = len(real_paths) / total
        real_cap    = int(max_samples * real_ratio)
        fake_cap    = max_samples - real_cap
        real_paths  = real_paths[:real_cap]
        fake_paths  = fake_paths[:fake_cap]

    file_paths = real_paths + fake_paths
    labels     = [0] * len(real_paths) + [1] * len(fake_paths)

    combined = list(zip(file_paths, labels))
    random.shuffle(combined)
    file_paths, labels = zip(*combined)
    file_paths = list(file_paths)
    labels     = list(labels)

    print(f"[INFO] {subset} → {len(file_paths)} samples | Real: {labels.count(0)} | Fake: {labels.count(1)}")
    return file_paths, labels


# ==========================================
# EER Computation
# ==========================================
def compute_eer(labels, scores):
    """
    Compute Equal Error Rate (EER).

    Args:
        labels : list of ints  — 0=real/bonafide, 1=spoof/fake
        scores : list of floats — higher = more likely FAKE

    Returns:
        eer      : EER as percentage (e.g. 5.71 means 5.71%)
        fpr, tpr : ROC curve arrays (for plotting)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100.0, fpr, tpr, thresholds


# ==========================================
# Knowledge Distillation Loss
# ==========================================
def kd_loss(student_logits, teacher_logits, true_labels, temperature, alpha,
            hard_loss_fn, soft_loss_fn):
    """
    Combined KD loss = alpha * soft_loss + (1-alpha) * hard_loss.

    Args:
        student_logits : [B, 2] raw student outputs
        teacher_logits : [B, 2] raw teacher outputs (no_grad)
        true_labels    : [B]    ground truth
        temperature    : float  — softens probability distributions
        alpha          : float  — weight for soft KD loss
        hard_loss_fn   : CrossEntropyLoss instance
        soft_loss_fn   : KLDivLoss(reduction="batchmean") instance

    Returns:
        total_loss : scalar tensor
        hard_loss  : scalar tensor (for logging)
        soft_loss  : scalar tensor (for logging)
    """
    import torch.nn.functional as F

    hard_loss         = hard_loss_fn(student_logits, true_labels)
    soft_targets      = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss         = soft_loss_fn(student_log_probs, soft_targets) * (temperature ** 2)
    total_loss        = (alpha * soft_loss) + ((1.0 - alpha) * hard_loss)

    return total_loss, hard_loss, soft_loss


# ==========================================
# Mel Spectrogram Preprocessing (for inference)
# ==========================================
def preprocess_chunk(waveform_chunk, config, device):
    """
    Convert a raw waveform chunk to a mel spectrogram tensor.
    Matches training preprocessing exactly.

    Args:
        waveform_chunk : [1, N] tensor
        config         : config module
        device         : torch device

    Returns:
        mel_spec : [1, 1, 64, T] tensor ready for student model
    """
    import torchaudio
    import torch.nn.functional as F

    target_length = config.MAX_LENGTH

    # Pad or trim
    if waveform_chunk.shape[1] < target_length:
        pad_length     = target_length - waveform_chunk.shape[1]
        waveform_chunk = F.pad(waveform_chunk, (0, pad_length))
    else:
        waveform_chunk = waveform_chunk[:, :target_length]

    # Normalize
    waveform_chunk = waveform_chunk / (waveform_chunk.abs().max() + 1e-8)

    # Mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate = config.SAMPLE_RATE,
        n_fft       = config.N_FFT,
        hop_length  = config.HOP_LENGTH,
        n_mels      = config.N_MELS,
        f_min       = config.F_MIN,
        f_max       = config.F_MAX,
    ).to(device)

    waveform_chunk = waveform_chunk.to(device)
    mel_spec       = mel_transform(waveform_chunk)
    mel_spec       = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel_spec       = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    mel_spec       = mel_spec.unsqueeze(0)   # add batch dim → [1, 1, 64, T]

    return mel_spec
