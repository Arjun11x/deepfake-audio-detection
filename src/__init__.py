"""
src/ — Deepfake Audio Detection package.

Imports available at package level:
    from src import MobileStudentCNN, Wav2VecTeacher
    from src import AudioDeepfakeDataset
    from src import load_asvspoof2019, compute_eer, kd_loss, preprocess_chunk
    from src import config
"""

from models  import MobileStudentCNN, Wav2VecTeacher
from dataset import AudioDeepfakeDataset
from utils   import load_asvspoof2019, compute_eer, kd_loss, preprocess_chunk
import config

__all__ = [
    "MobileStudentCNN",
    "Wav2VecTeacher",
    "AudioDeepfakeDataset",
    "load_asvspoof2019",
    "compute_eer",
    "kd_loss",
    "preprocess_chunk",
    "config",
]
