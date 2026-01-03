import torch
import torchaudio
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from core.config import MODEL_DIR, DEVICE, N_MELS
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

        checkpoint = torch.load(MODEL_DIR, map_location = DEVICE)
        self.ecapa.load_state_dict(checkpoint['ecapa_state_dict'])
        self.ecapa.eval()

    @torch.no_grad()
    def encode(self, waveform):
        """
        waveform: Tensor [T]
        returns: np.ndarray [192]
        """

        # ---- safety checks ----
        if waveform.dim() != 1:
            raise ValueError(f"Expected waveform [T], got {waveform.shape}")

        waveform = waveform.float().to(DEVICE)
        waveform = waveform.unsqueeze(0)          # [1, T]

        mel = mel_transform(waveform)              # [1, n_mels, frames]
        mel = amp_to_db(mel)

        if mel.dim() == 4:
            mel = mel.squeeze(1)

        mel = mel.transpose(1, 2).contiguous()     # [1, T, n_mels]

        # ---- critical debug line (keep this while testing) ----
        print("ECAPA INPUT SHAPE:", mel.shape)

        emb = self.ecapa(mel)                     # [1, 192]
        return emb.squeeze(0).cpu().numpy()
