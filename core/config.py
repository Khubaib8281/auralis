from pathlib import Path
import torch
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = f"{BASE_DIR}/models/ecapa_supcon_model.pth"
REF_EMB = f"{BASE_DIR}/data/reference_embeddings_192-d.npy"
REF_C_H = f"{BASE_DIR}/data/centroid_healthy.npy"
FATIGUE_AXIS = f"{BASE_DIR}/data/fatigue_axis.npy"
LOW_PERCENTILE = f"{BASE_DIR}/data/low_percentile.npz"
HIGH_PERCENTILE = f"{BASE_DIR}/data/high_percentile.npz"
CONFIG_PATH = f"{BASE_DIR}/model/config.yaml"

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SEC = 5
N_MELS = 80
TARGET_LEN = SAMPLE_RATE * TARGET_SEC
MAX_DURATION_SEC = 10.0
MIN_DURATION_SEC = 5.0
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

print(f"Model directory is set to: {MODEL_DIR}")
print(f"base dir: {BASE_DIR}")
print(f"ref emb path: {REF_EMB}")