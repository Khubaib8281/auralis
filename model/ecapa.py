import torch
import torchaudio
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from app.core.config import MODEL_DIR, DEVICE
import numpy as np

class ECAPAENCODER:
    def __init__(self):
        self.ecapa = torch.load(MODEL_SIR, map_location=DEVICE)
        self.ecapa.eval()

    @torch.no_grad()
    def encode(self, features: torch.Tensor) -> torch.Tesnor:
        emb = self.ecapa(features)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb