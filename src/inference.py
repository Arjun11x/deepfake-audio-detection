"""
inference.py — Deepfake audio detection on any audio file.

Uses multi-chunk analysis (50% overlap) for reliable predictions.

Usage:
    python src/inference.py --audio path/to/file.wav
    python src/inference.py --audio path/to/file.wav --env colab
    python src/inference.py --audio path/to/file.wav --model /path/to/model.pth

For Colab:
    Set ENV = "colab" in config.py, or pass --env colab
    Upload audio via files.upload() then pass the saved path to this script,
    or call run_inference() directly from a notebook cell.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models import MobileStudentCNN
from utils import preprocess_chunk


def parse_args():
    parser = argparse.ArgumentParser(description="Run deepfake detection on an audio file")
    parser.add_argument("--audio", type=str, required=True, help="Path to .wav or .flac file")
    parser.add_argument("--env",   type=str, choices=["local", "colab"], default=None)
    parser.add_argument("--model", type=str, default=None, help="Path to .pth file (overrides config)")
    return parser.parse_args()


def load_model(model_path, device):
    """Load student model from checkpoint."""
    assert os.path.exists(model_path), f"[ERROR] Model not found: {model_path}"
    student = MobileStudentCNN().to(device)
    student.load_state_dict(torch.load(model_path, map_location=device))
    student.eval()
    num_params = sum(p.numel() for p in student.parameters())
    print(f"✅ Model loaded — {num_params:,} parameters | {os.path.getsize(model_path)/1024/1024:.1f} MB")
    return student


def run_inference(filepath, student=None, device=None, model_path=None, save_plots=True):
    """
    Run multi-chunk deepfake detection on a single audio file.

    Can be called from a notebook directly:
        from inference import run_inference
        result = run_inference("/path/to/audio.wav")

    Args:
        filepath   : path to .wav or .flac file
        student    : pre-loaded MobileStudentCNN (optional — avoids reloading)
        device     : torch.device (optional — auto-detected if None)
        model_path : override model path (optional)
        save_plots : save visualization to SAVE_DIR

    Returns:
        dict with keys: prediction, confidence, avg_real, avg_fake,
                        chunk_real_probs, chunk_fake_probs, inference_ms
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if student is None:
        mp = model_path or config.BEST_MODEL_PATH
        student = load_model(mp, device)

    assert os.path.exists(filepath), f"[ERROR] Audio file not found: {filepath}"

    print(f"\n{'='*55}")
    print(f"  Analyzing: {os.path.basename(filepath)}")
    print(f"{'='*55}")

    # ==========================================
    # Load + Preprocess Audio
    # ==========================================
    waveform, sr = torchaudio.load(filepath)
    size_kb      = os.path.getsize(filepath) / 1024
    duration     = waveform.shape[1] / sr

    print(f"  File size   : {size_kb:.1f} KB")
    print(f"  Duration    : {duration:.2f}s")
    print(f"  Sample rate : {sr} Hz")
    print(f"  Channels    : {waveform.shape[0]}")

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        print(f"  Converted   : stereo → mono")

    # Resample
    if sr != config.SAMPLE_RATE:
        waveform = F_audio.resample(waveform, sr, config.SAMPLE_RATE)
        print(f"  Resampled   : {sr} Hz → {config.SAMPLE_RATE} Hz")
        sr = config.SAMPLE_RATE

    # ==========================================
    # Split into overlapping chunks
    # ==========================================
    total_samples    = waveform.shape[1]
    chunks           = []
    start            = 0

    while start < total_samples:
        end   = start + config.CHUNK_LENGTH
        chunk = waveform[:, start:min(end, total_samples)]
        if chunk.shape[1] >= config.MIN_CHUNK_SAMPLES:
            chunks.append(chunk)
        start += config.CHUNK_STEP
        if len(chunks) >= config.MAX_CHUNKS:
            break

    print(f"  Chunks      : {len(chunks)} × 4s windows (2s overlap)")

    # ==========================================
    # Inference per chunk
    # ==========================================
    chunk_real_probs = []
    chunk_fake_probs = []
    start_time       = time.time()

    with torch.no_grad():
        for chunk in chunks:
            mel_spec  = preprocess_chunk(chunk, config, device)
            logits    = student(mel_spec)
            probs     = F.softmax(logits, dim=1)
            chunk_real_probs.append(probs[0][0].item() * 100)
            chunk_fake_probs.append(probs[0][1].item() * 100)

    inference_ms = (time.time() - start_time) * 1000

    # ==========================================
    # Aggregate
    # ==========================================
    avg_real   = float(np.mean(chunk_real_probs))
    avg_fake   = float(np.mean(chunk_fake_probs))
    max_fake   = float(np.max(chunk_fake_probs))
    fake_votes = sum(1 for p in chunk_fake_probs if p > 50)
    total_v    = len(chunks)
    vote_pct   = 100.0 * fake_votes / total_v

    prediction = "FAKE" if avg_fake > avg_real else "REAL"
    confidence = max(avg_real, avg_fake)

    # ==========================================
    # Print Results
    # ==========================================
    print(f"\n  {'='*45}")
    icon = "🚨" if prediction == "FAKE" else "✅"
    print(f"  {icon} PREDICTION  : {prediction}")
    print(f"  {'='*45}")
    print(f"  Avg Real     : {avg_real:.2f}%")
    print(f"  Avg Fake     : {avg_fake:.2f}%")
    print(f"  Confidence   : {confidence:.2f}%")
    print(f"  Max fake     : {max_fake:.2f}%")
    print(f"  Fake votes   : {fake_votes}/{total_v} chunks ({vote_pct:.1f}%)")
    print(f"  Inference    : {inference_ms:.1f} ms")
    print(f"  {'='*45}")

    if confidence > 90:
        print(f"  📊 Very high confidence")
    elif confidence > 75:
        print(f"  📊 High confidence")
    elif confidence > 60:
        print(f"  📊 Moderate confidence")
    else:
        print(f"  📊 Low confidence — borderline case")

    # Per-chunk table
    print(f"\n  Per-chunk breakdown:")
    print(f"  {'Chunk':<8} {'Real%':<10} {'Fake%':<10} {'Decision'}")
    print(f"  {'-'*38}")
    for i, (r, f) in enumerate(zip(chunk_real_probs, chunk_fake_probs)):
        decision = "FAKE 🚨" if f > r else "REAL ✅"
        print(f"  {i+1:<8} {r:<10.1f} {f:<10.1f} {decision}")

    # ==========================================
    # Visualization
    # ==========================================
    if save_plots:
        config.make_dirs()
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Panel 1 — Mel spectrogram
        mel_display = preprocess_chunk(
            waveform[:, :min(config.CHUNK_LENGTH, total_samples)], config, device
        ).squeeze().cpu().numpy()
        im = axes[0].imshow(mel_display, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title(f"Mel Spectrogram (first 4s)\n{os.path.basename(filepath)}")
        axes[0].set_xlabel("Time Frames"); axes[0].set_ylabel("Mel Bins")
        plt.colorbar(im, ax=axes[0])

        # Panel 2 — Chunk fake probabilities
        chunk_indices = list(range(1, len(chunks) + 1))
        colors = ["tomato" if f > 50 else "royalblue" for f in chunk_fake_probs]
        axes[1].bar(chunk_indices, chunk_fake_probs, color=colors, edgecolor="black", linewidth=0.8)
        axes[1].axhline(y=50,       color="gray", linestyle="--", linewidth=1.5, label="Threshold (50%)")
        axes[1].axhline(y=avg_fake, color="red",  linestyle="-",  linewidth=2,   label=f"Avg fake ({avg_fake:.1f}%)")
        axes[1].set_xlabel("Chunk"); axes[1].set_ylabel("Fake Probability (%)")
        axes[1].set_title("Fake Probability Per Chunk"); axes[1].set_ylim(0, 110)
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

        # Panel 3 — Final averaged result
        bar_colors = [
            "royalblue"  if prediction == "REAL" else "lightblue",
            "tomato"     if prediction == "FAKE" else "lightsalmon"
        ]
        bars = axes[2].bar(["Real Audio", "Fake Audio"], [avg_real, avg_fake],
                           color=bar_colors, width=0.4, edgecolor="black", linewidth=1.2)
        axes[2].set_ylim(0, 110)
        axes[2].set_ylabel("Average Probability (%)")
        axes[2].set_title(f"Final: {prediction}\n({confidence:.1f}% conf, {fake_votes}/{total_v} chunks FAKE)")
        axes[2].axhline(y=50, color="gray", linestyle="--", linewidth=1)
        for bar, prob in zip(bars, [avg_real, avg_fake]):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                         f"{prob:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

        plt.tight_layout()
        plot_path = os.path.join(config.SAVE_DIR, "inference_result.png")
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"\n✅ Plot saved → {plot_path}")

    return {
        "prediction"       : prediction,
        "confidence"       : confidence,
        "avg_real"         : avg_real,
        "avg_fake"         : avg_fake,
        "max_fake"         : max_fake,
        "fake_votes"       : fake_votes,
        "total_chunks"     : total_v,
        "chunk_real_probs" : chunk_real_probs,
        "chunk_fake_probs" : chunk_fake_probs,
        "inference_ms"     : inference_ms,
    }


if __name__ == "__main__":
    args   = parse_args()

    if args.env:
        config.ENV = args.env
        import importlib; importlib.reload(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp     = args.model or config.BEST_MODEL_PATH
    model  = load_model(mp, device)

    result = run_inference(args.audio, student=model, device=device, save_plots=True)
    print(f"\n✅ Done — {result['prediction']} ({result['confidence']:.1f}% confidence)")
