# =============================================================
# config.py — Central configuration for all paths + hyperparams
# Change ENV to switch between local and Colab environments
# =============================================================

import os

# ------------------------------------------------------------------
# 🔧 CHANGE THIS LINE ONLY — everything else adapts automatically
ENV = "local"   # Options: "local" | "colab"
# ------------------------------------------------------------------

# =============================================================
# PATH CONFIGURATION
# =============================================================

if ENV == "colab":
    # --- Google Colab paths ---
    PROJECT_ROOT = "/content/project"
    DATASET_ROOT = "/content/asvspoof2019/LA"          # extracted from LA.zip
    DRIVE_ROOT   = "/content/drive/MyDrive/deepfake_detector"
    SAVE_DIR     = f"{DRIVE_ROOT}/models"              # plots + checkpoints saved here
    SRC_DIR      = PROJECT_ROOT                        # src/ files live here on Colab

else:
    # --- Local machine paths ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_ROOT = os.environ.get("ASVSPOOF_ROOT") or os.path.join(PROJECT_ROOT, "data", "LA")
    SAVE_DIR     = os.path.join(PROJECT_ROOT, "outputs")
    SRC_DIR      = os.path.join(PROJECT_ROOT, "src")

# Dataset subdirectory structure (same for both envs)
AUDIO_DIRS = {
    "train" : os.path.join(DATASET_ROOT, "ASVspoof2019_LA_train", "flac"),
    "dev"   : os.path.join(DATASET_ROOT, "ASVspoof2019_LA_dev",   "flac"),
    "eval"  : os.path.join(DATASET_ROOT, "ASVspoof2019_LA_eval",  "flac"),
}

PROTOCOL_FILES = {
    "train" : os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev"   : os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval"  : os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt"),
}

# Model checkpoint paths
BEST_MODEL_PATH       = os.path.join(SAVE_DIR, "student_best.pth")
FINAL_MODEL_PATH      = os.path.join(SAVE_DIR, "student_final.pth")
MEDIUM_MODEL_PATH     = os.path.join(SAVE_DIR, "student_medium_best.pth")
CHECKPOINT_PATH       = os.path.join(SAVE_DIR, "training_checkpoint.pth")

# =============================================================
# AUDIO CONFIGURATION
# =============================================================
SAMPLE_RATE    = 16000
MAX_SECONDS    = 4
MAX_LENGTH     = SAMPLE_RATE * MAX_SECONDS   # 64000 samples

# Mel spectrogram parameters (must match between train + inference)
N_MELS         = 64
N_FFT          = 1024
HOP_LENGTH     = 512
F_MIN          = 0.0
F_MAX          = 8000.0

# =============================================================
# BEST HYPERPARAMETERS (confirmed from sweep)
# =============================================================
LR             = 0.0005
TEMPERATURE    = 6.0
ALPHA          = 0.7     # 70% soft KD loss + 30% hard CE loss

# =============================================================
# TRAINING CONFIGURATION
# =============================================================
BATCH_SIZE     = 16
NUM_WORKERS    = 2

# Full training
FULL_MAX_EPOCHS     = 25
FULL_PATIENCE       = 6
SCHEDULER_FACTOR    = 0.5
SCHEDULER_PATIENCE  = 3

# Medium training (confirmation run)
MEDIUM_MAX_EPOCHS   = 40
MEDIUM_PATIENCE     = 5
MEDIUM_TRAIN_SAMPLES = 5000
MEDIUM_VAL_SAMPLES   = 1000

# Sweep training
SWEEP_EPOCHS        = 5
SWEEP_TRAIN_SAMPLES = 1000
SWEEP_VAL_SAMPLES   = 200

SWEEP_GRID = [
    # (learning_rate, temperature, alpha)
    (0.001,  4.0, 0.7),
    (0.001,  6.0, 0.7),
    (0.001,  4.0, 0.5),
    (0.0005, 4.0, 0.7),
    (0.0005, 6.0, 0.7),
    (0.0005, 4.0, 0.5),
    (0.0001, 4.0, 0.7),
    (0.0001, 6.0, 0.7),
    (0.0001, 4.0, 0.5),
]

# =============================================================
# INFERENCE CONFIGURATION
# =============================================================
CHUNK_LENGTH        = MAX_LENGTH      # 4 seconds
CHUNK_STEP          = SAMPLE_RATE * 2 # 2 second step (50% overlap)
MAX_CHUNKS          = 20
MIN_CHUNK_SAMPLES   = SAMPLE_RATE     # at least 1 second to be valid

# =============================================================
# UTILITY
# =============================================================
def make_dirs():
    """Create all output directories if they don't exist."""
    os.makedirs(SAVE_DIR, exist_ok=True)

def print_config():
    """Print active configuration — useful at top of each notebook."""
    print(f"{'='*50}")
    print(f"  Active ENV       : {ENV}")
    print(f"  Project root     : {PROJECT_ROOT}")
    print(f"  Dataset root     : {DATASET_ROOT}")
    print(f"  Save directory   : {SAVE_DIR}")
    print(f"{'='*50}")
    print(f"  LR               : {LR}")
    print(f"  Temperature      : {TEMPERATURE}")
    print(f"  Alpha            : {ALPHA}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"{'='*50}")
