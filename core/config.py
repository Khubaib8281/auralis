from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = "/home/khubaib/projects/vocal_fatigue_scoring/models/ecapa_supcon_model.pth"
REF_EMB = "/home/khubaib/projects/vocal_fatigue_scoring/data/reference_embeddings_192-d.npy"

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SEC = 5
TARGET_LEN = SAMPLE_RATE * TARGET_SEC

print(f"Model directory is set to: {MODEL_DIR}")
print(f"base dir: {BASE_DIR}")
print(f"base dir: {REF_EMB}")