"""
evaluate.py — Evaluate student_best.pth on ASVspoof 2019 eval set.

Usage:
    python src/evaluate.py                     # uses ENV from config.py
    python src/evaluate.py --env colab         # override ENV at runtime
    python src/evaluate.py --model /path/to/model.pth  # custom model path
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models import MobileStudentCNN
from dataset import AudioDeepfakeDataset
from utils import load_asvspoof2019, compute_eer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model on eval set")
    parser.add_argument("--env",   type=str, choices=["local", "colab"], default=None)
    parser.add_argument("--model", type=str, default=None, help="Path to .pth file (overrides config)")
    return parser.parse_args()


def evaluate(env_override=None, model_path_override=None):

    if env_override:
        config.ENV = env_override
        import importlib; importlib.reload(config)

    config.print_config()
    config.make_dirs()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_path_override or config.BEST_MODEL_PATH

    # ==========================================
    # Load Model
    # ==========================================
    print(f"\n{'='*55}")
    print(f"  FINAL EVALUATION — ASVspoof 2019 Eval Set")
    print(f"{'='*55}")

    assert os.path.exists(model_path), f"[ERROR] Model not found: {model_path}"

    student = MobileStudentCNN().to(device)
    student.load_state_dict(torch.load(model_path, map_location=device))
    student.eval()

    size_mb    = os.path.getsize(model_path) / (1024 * 1024)
    num_params = sum(p.numel() for p in student.parameters())
    print(f"\n✅ Loaded: {os.path.basename(model_path)} ({size_mb:.1f} MB)")
    print(f"   Parameters : {num_params:,}")
    print(f"   Device     : {device}")

    # ==========================================
    # Load Eval Dataset
    # ==========================================
    print(f"\nLoading eval dataset...")
    eval_files, eval_labels = load_asvspoof2019(
        config.DATASET_ROOT, config.AUDIO_DIRS, config.PROTOCOL_FILES,
        subset="eval", max_samples=None, balanced=False
    )

    print(f"[INFO] Total: {len(eval_files)} | Real: {eval_labels.count(0)} | Fake: {eval_labels.count(1)}")
    print(f"[INFO] Contains UNSEEN attack types A07–A19")

    eval_dataset = AudioDeepfakeDataset(eval_files, eval_labels, is_training=False)
    eval_loader  = DataLoader(
        eval_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    print(f"[INFO] Eval batches: {len(eval_loader)}")

    # ==========================================
    # Run Evaluation
    # ==========================================
    print(f"\nRunning evaluation on {len(eval_files):,} samples...")
    print("(~20-30 minutes on Colab GPU)\n")

    all_labels, all_scores, all_preds = [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (_, mel_specs, true_labels) in enumerate(eval_loader):
            mel_specs   = mel_specs.to(device)
            true_labels = true_labels.to(device)

            logits      = student(mel_specs)
            fake_scores = F.softmax(logits, dim=1)[:, 1]
            predicted   = torch.argmax(logits, dim=1)

            correct += (predicted == true_labels).sum().item()
            total   += true_labels.size(0)

            all_labels.extend(true_labels.cpu().numpy().tolist())
            all_scores.extend(fake_scores.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())

            if (batch_idx + 1) % 500 == 0:
                pct = 100.0 * (batch_idx + 1) / len(eval_loader)
                acc = 100.0 * correct / total
                print(f"  Progress: {pct:.1f}% | Acc so far: {acc:.1f}%")

    # ==========================================
    # Metrics
    # ==========================================
    eval_acc = 100.0 * correct / total

    try:
        eval_eer, fpr, tpr, _ = compute_eer(all_labels, all_scores)
    except Exception as e:
        print(f"❌ EER failed: {e}")
        eval_eer, fpr, tpr = 99.0, None, None

    cm           = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    far          = 100.0 * fp / (fp + tn)
    frr          = 100.0 * fn / (fn + tp)

    print(f"\n{'='*55}")
    print(f"  ✅ FINAL EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"  Total samples : {total:,}")
    print(f"  Accuracy      : {eval_acc:.2f}%")
    print(f"  EER           : {eval_eer:.2f}%  ← TRUE BENCHMARK")
    print(f"  FAR           : {far:.2f}%")
    print(f"  FRR           : {frr:.2f}%")
    print(f"{'='*55}")
    print(f"  Confusion Matrix:")
    print(f"  Real accepted  (TN) : {tn:,}")
    print(f"  Fake accepted  (FP) : {fp:,}")
    print(f"  Real rejected  (FN) : {fn:,}")
    print(f"  Fake caught    (TP) : {tp:,}")
    print(f"{'='*55}")
    print(f"\n  vs published work:")
    print(f"  ASVspoof baseline : ~8-11% EER")
    print(f"  Your model        : {eval_eer:.2f}% EER")

    # Save results text
    results_path = os.path.join(config.SAVE_DIR, "eval_results.txt")
    with open(results_path, "w") as f:
        f.write(f"EER     : {eval_eer:.2f}%\n")
        f.write(f"Accuracy: {eval_acc:.2f}%\n")
        f.write(f"FAR     : {far:.2f}%\n")
        f.write(f"FRR     : {frr:.2f}%\n")
        f.write(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")
    print(f"\n✅ Results saved → {results_path}")

    # ==========================================
    # Plots — ROC + Score Distribution
    # ==========================================
    if fpr is not None:
        all_labels_np = np.array(all_labels)
        all_scores_np = np.array(all_scores)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(fpr, tpr, color="royalblue", linewidth=2, label=f"ROC (EER={eval_eer:.2f}%)")
        axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        axes[0].axvline(x=eval_eer/100, color="red", linestyle="--", linewidth=1.5, label=f"EER Point")
        axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve — Eval Set"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        real_scores = all_scores_np[all_labels_np == 0]
        fake_scores = all_scores_np[all_labels_np == 1]
        axes[1].hist(real_scores, bins=50, alpha=0.6, color="royalblue", label="Real", density=True)
        axes[1].hist(fake_scores, bins=50, alpha=0.6, color="tomato",    label="Fake", density=True)
        axes[1].set_xlabel("Fake Probability Score"); axes[1].set_ylabel("Density")
        axes[1].set_title("Score Distribution"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(config.SAVE_DIR, "eval_results.png")
        plt.savefig(plot_path, dpi=150)
        print(f"✅ Plots saved → {plot_path}")

    return eval_eer, eval_acc, far, frr


if __name__ == "__main__":
    args = parse_args()
    evaluate(env_override=args.env, model_path_override=args.model)
