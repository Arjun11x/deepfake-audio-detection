"""
train.py — Full training loop for deepfake audio detection.

Usage:
    python src/train.py                        # uses ENV from config.py
    python src/train.py --env colab            # override ENV at runtime

For Colab:
    Set ENV = "colab" in config.py, or pass --env colab
    Paths are picked up automatically from config.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Allow running from project root or src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models import MobileStudentCNN, Wav2VecTeacher
from dataset import AudioDeepfakeDataset
from utils import load_asvspoof2019, compute_eer, kd_loss


# ==========================================
# Argument Parser — override ENV without editing config
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train deepfake audio detection student model")
    parser.add_argument("--env", type=str, choices=["local", "colab"], default=None,
                        help="Override ENV from config.py (local | colab)")
    return parser.parse_args()


# ==========================================
# Main Training Function
# ==========================================
def train(env_override=None):

    # Apply ENV override if passed
    if env_override:
        config.ENV = env_override
        # Re-derive paths after ENV change
        import importlib
        importlib.reload(config)

    config.print_config()
    config.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ==========================================
    # Load Dataset (full, unbalanced for dev)
    # ==========================================
    print("\nLoading full training dataset...")
    train_files, train_labels = load_asvspoof2019(
        config.DATASET_ROOT, config.AUDIO_DIRS, config.PROTOCOL_FILES,
        subset="train", max_samples=None, balanced=True
    )
    val_files, val_labels = load_asvspoof2019(
        config.DATASET_ROOT, config.AUDIO_DIRS, config.PROTOCOL_FILES,
        subset="dev", max_samples=None, balanced=False   # natural distribution for true EER
    )

    train_dataset = AudioDeepfakeDataset(train_files, train_labels, is_training=True)
    val_dataset   = AudioDeepfakeDataset(val_files,   val_labels,   is_training=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True,  num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ==========================================
    # Models
    # ==========================================
    print("\nLoading Teacher (frozen)...")
    teacher = Wav2VecTeacher().to(device)
    teacher.eval()

    student = MobileStudentCNN().to(device)

    # ==========================================
    # Optimizer + Scheduler + Loss
    # ==========================================
    optimizer    = optim.Adam(student.parameters(), lr=config.LR)
    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = config.SCHEDULER_FACTOR,
        patience = config.SCHEDULER_PATIENCE,
    )
    hard_loss_fn = nn.CrossEntropyLoss()
    soft_loss_fn = nn.KLDivLoss(reduction="batchmean")

    # ==========================================
    # Resume from checkpoint if available
    # ==========================================
    start_epoch                = 1
    best_eer                   = float("inf")
    epochs_without_improvement = 0
    ideal_epoch                = 0
    train_loss_curve           = []
    val_loss_curve             = []
    val_acc_curve              = []
    val_eer_curve              = []
    lr_curve                   = []

    if os.path.exists(config.CHECKPOINT_PATH):
        print(f"\n🔄 Checkpoint found — resuming...")
        ckpt = torch.load(config.CHECKPOINT_PATH, map_location=device)
        student.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch                = ckpt["epoch"] + 1
        best_eer                   = ckpt["best_eer"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        ideal_epoch                = ckpt["ideal_epoch"]
        train_loss_curve           = ckpt["train_loss_curve"]
        val_loss_curve             = ckpt["val_loss_curve"]
        val_acc_curve              = ckpt["val_acc_curve"]
        val_eer_curve              = ckpt["val_eer_curve"]
        lr_curve                   = ckpt.get("lr_curve", [])
        print(f"  Resumed from epoch {start_epoch - 1} | Best EER: {best_eer:.2f}%")
    else:
        print(f"\n🆕 No checkpoint found — starting fresh")

    # ==========================================
    # Training Loop
    # ==========================================
    print(f"\nStarting training from epoch {start_epoch}...\n")
    print(f"  {'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<10} {'EER':<10} {'LR'}")
    print(f"  {'-'*65}")

    for epoch in range(start_epoch, config.FULL_MAX_EPOCHS + 1):

        current_lr = optimizer.param_groups[0]["lr"]

        # --- Train ---
        student.train()
        train_loss = 0.0

        for raw_audio, mel_specs, true_labels in train_loader:
            raw_audio   = raw_audio.to(device)
            mel_specs   = mel_specs.to(device)
            true_labels = true_labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(raw_audio)

            student_logits = student(mel_specs)
            loss, _, _     = kd_loss(
                student_logits, teacher_logits, true_labels,
                config.TEMPERATURE, config.ALPHA,
                hard_loss_fn, soft_loss_fn
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- Validate ---
        student.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_labels, all_scores = [], []

        with torch.no_grad():
            for raw_audio, mel_specs, true_labels in val_loader:
                raw_audio   = raw_audio.to(device)
                mel_specs   = mel_specs.to(device)
                true_labels = true_labels.to(device)

                teacher_logits = teacher(raw_audio)
                student_logits = student(mel_specs)
                loss, _, _     = kd_loss(
                    student_logits, teacher_logits, true_labels,
                    config.TEMPERATURE, config.ALPHA,
                    hard_loss_fn, soft_loss_fn
                )

                val_loss    += loss.item()
                predicted    = torch.argmax(student_logits, dim=1)
                val_correct += (predicted == true_labels).sum().item()
                val_total   += true_labels.size(0)

                fake_scores = F.softmax(student_logits, dim=1)[:, 1]
                all_labels.extend(true_labels.cpu().numpy().tolist())
                all_scores.extend(fake_scores.cpu().numpy().tolist())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        avg_val_acc    = 100.0 * val_correct / val_total

        try:
            epoch_eer, _, _, _ = compute_eer(all_labels, all_scores)
        except Exception as e:
            epoch_eer = 99.0
            print(f"  [WARN] EER failed: {e}")

        train_loss_curve.append(avg_train_loss)
        val_loss_curve.append(avg_val_loss)
        val_acc_curve.append(avg_val_acc)
        val_eer_curve.append(epoch_eer)
        lr_curve.append(current_lr)

        print(
            f"  [{epoch:02d}/{config.FULL_MAX_EPOCHS}]   "
            f"{avg_train_loss:<12.4f} {avg_val_loss:<12.4f} "
            f"{avg_val_acc:<10.1f}% {epoch_eer:<10.2f}% {current_lr:.6f}"
        )

        scheduler.step(epoch_eer)

        # Best model
        if epoch_eer < best_eer:
            best_eer                   = epoch_eer
            ideal_epoch                = epoch
            epochs_without_improvement = 0
            torch.save(student.state_dict(), config.BEST_MODEL_PATH)
            print(f"  💾 Best model saved — EER: {epoch_eer:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"  ⏳ No improvement ({epochs_without_improvement}/{config.FULL_PATIENCE})")

        # Checkpoint every epoch
        torch.save({
            "epoch"                      : epoch,
            "model_state"                : student.state_dict(),
            "optimizer_state"            : optimizer.state_dict(),
            "scheduler_state"            : scheduler.state_dict(),
            "best_eer"                   : best_eer,
            "epochs_without_improvement" : epochs_without_improvement,
            "ideal_epoch"                : ideal_epoch,
            "train_loss_curve"           : train_loss_curve,
            "val_loss_curve"             : val_loss_curve,
            "val_acc_curve"              : val_acc_curve,
            "val_eer_curve"              : val_eer_curve,
            "lr_curve"                   : lr_curve,
        }, config.CHECKPOINT_PATH)

        # Early stopping
        if epochs_without_improvement >= config.FULL_PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch} | Best EER: {best_eer:.2f}% at epoch {ideal_epoch}")
            break

    # Final save
    torch.save(student.state_dict(), config.FINAL_MODEL_PATH)
    print(f"\n✅ Final model → {config.FINAL_MODEL_PATH}")
    print(f"✅ Best model  → {config.BEST_MODEL_PATH}")

    # ==========================================
    # Plot training curves — 4 panels
    # ==========================================
    fig, axes = plt.subplots(1, 4, figsize=(22, 4))

    axes[0].plot(train_loss_curve, label="Train", color="royalblue",     linewidth=2)
    axes[0].plot(val_loss_curve,   label="Val",   color="tomato",        linewidth=2)
    axes[0].axvline(x=ideal_epoch-1, color="green", linestyle="--", linewidth=1.5, label=f"Best ({ideal_epoch})")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_acc_curve, color="mediumseagreen", linewidth=2, label="Val Acc")
    axes[1].axvline(x=ideal_epoch-1, color="green", linestyle="--", linewidth=1.5, label=f"Best ({ideal_epoch})")
    axes[1].set_title("Validation Accuracy (%)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(val_eer_curve, color="darkorchid", linewidth=2, label="EER")
    axes[2].axvline(x=ideal_epoch-1, color="green", linestyle="--", linewidth=1.5, label=f"Best ({ideal_epoch})")
    axes[2].set_title("EER % (lower = better)"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    axes[3].plot(lr_curve, color="darkorange", linewidth=2, label="LR")
    axes[3].axvline(x=ideal_epoch-1, color="green", linestyle="--", linewidth=1.5, label=f"Best ({ideal_epoch})")
    axes[3].set_title("Learning Rate"); axes[3].legend(); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(config.SAVE_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    print(f"✅ Curves saved → {curve_path}")

    # Cleanup checkpoint
    if os.path.exists(config.CHECKPOINT_PATH):
        os.remove(config.CHECKPOINT_PATH)
        print("✅ Checkpoint cleaned up")

    # Summary
    print(f"\n{'='*50}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Epoch : {ideal_epoch}")
    print(f"  Best EER   : {best_eer:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    args = parse_args()
    train(env_override=args.env)
