# Deepfake Audio Detection
### Lightweight Cross-Modal Knowledge Distillation

A lightweight deepfake audio detection system that uses **Knowledge Distillation** to compress a large Wav2Vec2 teacher model (94.4M parameters) into a tiny MobileStudentCNN (0.23M parameters) — achieving **4.75% EER** on the ASVspoof 2019 LA eval set while running in <10ms on CPU.

---

## Results

| Metric | Value |
|--------|-------|
| Dev EER (validation, natural distribution) | **1.22%** |
| Eval EER (unseen attacks A07–A19) | **4.75%** ← TRUE BENCHMARK |
| Accuracy | 90.59% |
| FAR (Real wrongly predicted as Fake) | 1.18% |
| FRR (Fake wrongly predicted as Real) | 10.36% |
| Model size | 0.9 MB |
| Parameters | 230,058 (~0.23M) |
| Inference speed | <10ms on CPU |

### vs Published Work

| System | EER |
|--------|-----|
| ASVspoof 2019 baseline | ~8–11% |
| Lightweight CNN (typical) | ~3–5% |
| Published SOTA | ~0.5–1% |
| **This model** | **4.75%** ✅ beats baseline |

---

## Architecture

### Two-Phase Training Pipeline

```
Phase 1 — Teacher Pre-training
  facebook/wav2vec2-base backbone (frozen, 94.4M params)
           ↓  raw waveform [B, 64000]
  Classifier head (trainable, 198,914 params)
           ↓
  Trained on true labels via CrossEntropy
  Result: 93.8% validation accuracy

Phase 2 — Knowledge Distillation
  Trained Teacher (fully frozen)
           ↓  soft probability distributions
           ↓  KL Divergence (temperature=4.0)
  Student: MobileStudentCNN (0.23M params, trainable)
           ↑  mel spectrogram [B, 1, 64, 126]
```

A dedicated teacher pre-training phase was added because the Wav2Vec2 classifier head starts randomly initialized — without first training it on the real/fake task, its soft targets carry no useful signal for distillation.

**Loss function:**
```
Total Loss = α × Soft KD Loss + (1-α) × Hard CE Loss
           = 0.7 × KLDiv(student/T, teacher/T) × T² + 0.3 × CrossEntropy
```

### Student Model — MobileStudentCNN

```
Input: [B, 1, 64, 126] mel spectrogram
  → Conv2d(1→16) + BN + AvgPool
  → Conv2d(16→32) + BN + AvgPool
  → Conv2d(32→64) + BN
  → ResBlock(64) + SEBlock (channel attention)
  → AdaptiveAvgPool → [B, 64, 4, 4]
  → FC(1024→128) + Dropout(0.3)
  → FC(128→2)
Output: [B, 2] logits
```

Key design choices:
- **SEBlock** — learns which mel frequency channels matter most
- **ResBlock** — prevents vanishing gradients in deeper layers
- **AvgPool** over MaxPool — preserves more spectral information
- **Single channel input** — grayscale mel spectrogram, not RGB

---

## Project Structure

```
deepfake-audio-detection/
├── configs/
│   └── best_config.yaml   ← confirmed best hyperparameters
├── notebooks/
│   ├── 1_setup_dataset.ipynb        ← GPU verify, install, mount Drive, extract dataset
│   ├── 2_EDA.ipynb                  ← waveforms, spectrograms, class distribution
│   └── 3_hyperparameter_sweep.ipynb ← 9 combos × 5 epochs sweep
├── outputs/
│   ├── eval_results.png   ← ROC curve + score distribution (eval set)
│   ├── eval_results.txt   ← final EER / accuracy / FAR / FRR
│   └── training_curves.png ← loss / accuracy / EER / LR over training
├── src/
│   ├── __init__.py
│   ├── config.py          ← all paths + hyperparams, ENV switching
│   ├── dataset.py         ← AudioDeepfakeDataset with SpecAugment
│   ├── evaluate.py        ← EER / FAR / FRR / confusion matrix / plots
│   ├── inference.py       ← multi-chunk inference on any audio file
│   ├── models.py          ← MobileStudentCNN + Wav2VecTeacher
│   ├── train.py           ← train_teacher() + full KD training loop
│   └── utils.py           ← load_asvspoof2019, compute_eer, kd_loss
├── .gitignore
├── README.md
└── requirements.txt
```

Trained model weights (`teacher_best.pth`, `student_best.pth`, `student_final.pth`) are not committed to GitHub due to file size — they are excluded via `.gitignore` and stored locally / on Drive.

Full training and evaluation are run directly via `src/train.py` and `src/evaluate.py` rather than dedicated notebooks.

---

## Dataset

**ASVspoof 2019 Logical Access (LA)**

| Subset | Samples | Real | Fake | Attack Types |
|--------|---------|------|------|--------------|
| Train | 25,380 | 2,580 | 22,800 | A01–A06 (seen) |
| Dev | 24,844 | 2,548 | 22,296 | A01–A06 (seen) |
| Eval | 71,237 | 7,355 | 63,882 | A07–A19 (unseen) |

Expected folder structure after extraction:
```
LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
```

---

## Quick Start

### On Google Colab
Run the setup and exploration notebooks in order:
1. `1_setup_dataset.ipynb` — run once per session
2. `2_EDA.ipynb` — explore the dataset
3. `3_hyperparameter_sweep.ipynb` — find best hyperparameters

Then switch `ENV = "colab"` in `config.py` and run `src/train.py` / `src/evaluate.py` directly.

### On Local Machine

**1. Clone the repo**
```bash
git clone https://github.com/Arjun11x/deepfake-audio-detection.git
cd deepfake-audio-detection
```

**2. Install PyTorch with CUDA first** (adjust CUDA version to match your GPU/driver)
```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. Install remaining dependencies**
```bash
pip install -r requirements.txt
```

**4. Set dataset path**

Either place the dataset at `data/LA/` inside the project folder, or set an environment variable:
```bash
# Linux/Mac
export ASVSPOOF_ROOT=/path/to/LA

# Windows
set ASVSPOOF_ROOT=C:\path\to\LA
```

**5. Train the teacher, then the student**
```bash
# Phase 1 + Phase 2 are both triggered from train.py
python src/train.py

# Skip student training if already trained
python src/train.py --skip_if_trained
```

**6. Evaluate**
```bash
python src/evaluate.py
```

**7. Run inference on any audio file**
```bash
python src/inference.py --audio path/to/audio.wav
```

---

## Hyperparameters

Best parameters confirmed from sweep, re-validated after adding teacher pre-training:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 0.0005 | Adam optimizer |
| Temperature | 4.0 | Softens KD distributions |
| Alpha | 0.7 | 70% soft + 30% hard loss |
| Batch Size | 16 | |
| Scheduler | ReduceLROnPlateau | factor=0.5, patience=3 |
| Early Stopping | patience=6 | based on EER |
| Max Epochs | 30 | |
| Best Epoch | 22 | |
| Stopped At | Epoch 28 | |

---

## Training Details

### Why Pre-train the Teacher?
`Wav2Vec2Model` is loaded from Hugging Face with its backbone frozen, but the classifier head on top is randomly initialized. Without training that head first, the teacher's output logits are close to random noise — and distilling from random noise teaches the student nothing useful. `train.py` includes a `train_teacher()` function that trains only the classifier head (the backbone stays frozen throughout) for 10 epochs before the student's KD training begins.

### Data Augmentation
- **Time domain** — random Gaussian noise (p=0.5, amplitude=0.01)
- **Frequency domain** — SpecAugment (FreqMask=15, TimeMask=35)
- **Balanced sampling** — equal real/fake during training

### Why EER over Accuracy?
With ~89% fake samples in the eval set, a naive model predicting everything as FAKE achieves ~89% accuracy. EER forces balanced performance on both real and fake — a model with no real discriminative ability would score ~50% EER. Our model achieves **4.75% EER**, confirming it genuinely learned to distinguish real from fake rather than exploiting class imbalance.

---

## Inference

The inference script uses **multi-chunk analysis** with 50% overlap for reliable predictions on audio of any length:

```python
from src.inference import run_inference

result = run_inference("path/to/audio.wav")
print(result['prediction'])    # "FAKE" or "REAL"
print(result['confidence'])    # e.g. 94.7
print(result['fake_votes'])    # e.g. 2 out of 2 chunks
```

Confidence scales with how much of the 4-second input window is real signal versus zero-padding — short clips that get heavily padded naturally produce lower-confidence predictions.

---

## Requirements

PyTorch must be installed separately with the correct CUDA build for your GPU (see Quick Start above). Remaining dependencies:

```
transformers>=5.0.0
numpy<2.5.0
scipy>=1.17.0
scikit-learn>=1.9.0
matplotlib>=3.11.0
soundfile>=0.14.0
PyYAML>=6.0.0
```

---

## Limitations

- FRR of 10.36% means a meaningful fraction of real audio gets flagged as fake — higher than FAR (1.18%), indicating some unseen attack types are harder to distinguish from genuine speech than others
- Eval EER (4.75%) is higher than Dev EER (1.22%) — expected generalization drop when testing on attack types (A07–A19) never seen during training
- Training data is heavily class-imbalanced (~9:1 fake:real); training uses balanced sampling, but dev/eval are evaluated on the natural distribution
- Confidence on very short audio clips (<2s) is lower due to zero-padding diluting the input signal
