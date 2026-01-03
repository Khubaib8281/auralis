from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = "/home/khubaib/projects/vocal_fatigue_scoring/models/ecapa_supcon_model.pth"
REF_EMB = "/home/khubaib/projects/vocal_fatigue_scoring/data/reference_embeddings_192-d.npy"
REF_C_H = "/home/khubaib/projects/vocal_fatigue_scoring/data/centroid_healthy.npy"
FATIGUE_AXIS = "/home/khubaib/projects/vocal_fatigue_scoring/data/fatigue_axis.npy"
LOW_PERCENTILE = "/home/khubaib/projects/vocal_fatigue_scoring/data/low_percentile.npz"
HIGH_PERCENTILE = "/home/khubaib/projects/vocal_fatigue_scoring/data/high_percentile.npz"

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SEC = 5
N_MELS = 80
TARGET_LEN = SAMPLE_RATE * TARGET_SEC

print(f"Model directory is set to: {MODEL_DIR}")
print(f"base dir: {BASE_DIR}")
print(f"base dir: {REF_EMB}")