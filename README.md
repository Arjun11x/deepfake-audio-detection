# Deepfake Audio Detection
### Lightweight Cross-Modal Knowledge Distillation

A lightweight deepfake audio detection system that uses **Knowledge Distillation** to compress a large Wav2Vec2 teacher model (94.4M parameters) into a tiny MobileStudentCNN (0.23M parameters) — achieving **5.71% EER** on the ASVspoof 2019 LA eval set while running in **<10ms on CPU**.

---

## Results

| Metric | Value |
|--------|-------|
| Dev EER (seen attacks A01–A06) | **0.30%** |
| Eval EER (unseen attacks A07–A19) | **5.71%** ← TRUE BENCHMARK |
| Accuracy | 90.84% |
| FAR (Fake wrongly accepted) | 1.09% |
| FRR (Real wrongly rejected) | 10.09% |
| Model size | 0.9 MB |
| Parameters | 230,058 (~0.23M) |
| Inference speed | <10ms on CPU |

### vs Published Work

| System | EER |
|--------|-----|
| ASVspoof 2019 baseline | ~8–11% |
| Lightweight CNN (typical) | ~3–5% |
| Published SOTA | ~0.5–1% |
| **This model** | **5.71%** ✅ beats baseline |

---

## Architecture

### Knowledge Distillation Setup

```
Teacher: facebook/wav2vec2-base  (94.4M params, frozen)
           ↓  raw waveform [B, 64000]
           ↓  soft probability distributions
           ↓  KL Divergence (temperature=6.0)
Student: MobileStudentCNN        (0.23M params, trainable)
           ↑  mel spectrogram [B, 1, 64, 126]
```

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
│   ├── 1_setup_dataset.ipynb      ← GPU verify, install, mount Drive, extract dataset
│   ├── 2_EDA.ipynb                ← waveforms, spectrograms, class distribution
│   ├── 3_hyperparameter_sweep.ipynb ← 9 combos × 5 epochs + medium training
│   ├── 4_training.ipynb           ← full training via train.py
│   ├── 5_evaluation.ipynb         ← eval on 71,237 samples + EER explanation
│   └── 6_inference.ipynb          ← inference on any uploaded audio file
├── src/
│   └── __init__.py
│   ├── config.py          ← all paths + hyperparams, ENV switching
│   ├── dataset.py         ← AudioDeepfakeDataset with SpecAugment
│   ├── evaluate.py        ← EER / FAR / FRR / confusion matrix / plots
│   ├── inference.py       ← multi-chunk inference on any audio file
│   ├── models.py          ← MobileStudentCNN + Wav2VecTeacher
│   ├── train.py           ← full training loop + --skip_if_trained flag
│   ├── utils.py           ← load_asvspoof2019, compute_eer, kd_loss
├── .gitignore
├── README.md
└── requirements.txt
```

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

### On Google Colab (recommended)
Open and run the notebooks in order:
1. `1_setup_dataset.ipynb` — run once per session
2. `2_EDA.ipynb` — explore the dataset
3. `3_hyperparameter_sweep.ipynb` — find best hyperparameters
4. `4_training.ipynb` — train the model
5. `5_evaluation.ipynb` — evaluate on eval set
6. `6_inference.ipynb` — test on any audio file

### On Local Machine

**1. Clone the repo**
```bash
git clone https://github.com/Arjun11x/deepfake-audio-detection.git
cd deepfake-audio-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set dataset path**

Either place the dataset at `data/LA/` inside the project folder, or set an environment variable:
```bash
# Linux/Mac
export ASVSPOOF_ROOT=/path/to/LA

# Windows
set ASVSPOOF_ROOT=C:\path\to\LA
```

**4. Train**
```bash
# Fresh training
python src/train.py

# Skip if already trained
python src/train.py --skip_if_trained
```

**5. Evaluate**
```bash
python src/evaluate.py
```

**6. Run inference on any audio file**
```bash
python src/inference.py --audio path/to/audio.wav
```

---

## Hyperparameters

Best parameters confirmed from sweep + medium training:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 0.0005 | Adam optimizer |
| Temperature | 6.0 | Softens KD distributions |
| Alpha | 0.7 | 70% soft + 30% hard loss |
| Batch Size | 16 | |
| Scheduler | ReduceLROnPlateau | factor=0.5, patience=3 |
| Early Stopping | patience=6 | based on EER |
| Best Epoch | 13 | |
| Stopped At | Epoch 19 | |

---

## Training Details

### Data Augmentation
- **Time domain** — random Gaussian noise (p=0.5, amplitude=0.01)
- **Frequency domain** — SpecAugment (FreqMask=15, TimeMask=35)
- **Balanced sampling** — equal real/fake during training

### Why EER over Accuracy?
With ~89% fake samples in the eval set, a naive model predicting everything as FAKE achieves ~89% accuracy. EER forces balanced performance on both real and fake — a model predicting all FAKE would score ~50% EER (random). Our model achieves **5.71% EER** confirming it genuinely learned to distinguish real from fake.

### Scheduler Choice
Medium training used `CosineAnnealingLR` which caused visible oscillation in curves. Full training switches to `ReduceLROnPlateau` — automatically halves LR when EER stops improving, producing smoother convergence.

---

## Inference

The inference script uses **multi-chunk analysis** with 50% overlap for reliable predictions on audio of any length:

```python
from src.inference import run_inference

result = run_inference("path/to/audio.wav")
print(result['prediction'])    # "FAKE" or "REAL"
print(result['confidence'])    # e.g. 87.3
print(result['fake_votes'])    # e.g. 5 out of 6 chunks
```

---

## Requirements

```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
soundfile>=0.12.0
pyyaml>=6.0
```

---

## Limitations

- Model is trained on ASVspoof 2019 LA attack types — confidence scores are moderate (~58%) on out-of-distribution TTS systems
- FRR of 10.09% means some real audio gets flagged as fake
- Eval EER (5.71%) is higher than Dev EER (0.30%) — expected generalization drop on unseen attack types A07–A19
