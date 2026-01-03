import torch
import torchaudio
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from app.core.config import MODEL_DIR, DEVICE, N_MELS
import numpy as np

class ECAPAENCODER:
    def __init__(self):
        self.ecapa = ECAPA_TDNN(
            input_size = N_MELS,
            lin_neurons = 192,
            channels = [512, 512, 512],
            kernel_sizes = [5, 3, 3],
            dilations = [1, 2 , 3]
        ).to(DEVICE)

        self.ecapa.load_state_dict(torch.load(MODEL_DIR, map_location=DEVICE))
        self.ecapa.eval()

    @torch.no_grad()
    def encode(self, mel_features: torch.Tensor) -> torch.Tensor:
        emb = self.ecapa(mel_features)
        return emb.squeeze(0).cpu().numpy()