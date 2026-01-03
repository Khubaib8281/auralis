import torch
import torchaudio
import torch.nn.functional as F
from core.config import SAMPLE_RATE, DEVICE, N_MELS, TARGET_LEN
from Pydub import AudioSegment
import numpy as np

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft = 400,
    hop_length = 256,
    n_mels = N_MELS    
).to(DEVICE)

amp_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

# def load_audio(path: str) -> torch.Tensor:
#     wav, sr = torchaudio.load(path)
#     if sr != SAMPLE_RATE:
#         wav = torchaudio.transforms.Resample(wav, sr, SAMPLE_RATE)
#     if wav.shape[0] > 1:
#         wav = wav.mean(dim = 0)
#     return wav.to(DEVICE)  

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from pydub import AudioSegment

class AudioLoadError(Exception):
    pass

def load_audio(path: str) -> torch.Tensor:
    waveform = None
    sr = None

    # --- primary loader ---
    try:
        waveform, sr = torchaudio.load(path)
    except Exception as e1:
        # --- fallback loader ---
        try:
            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if samples.size == 0:
                raise AudioLoadError("Empty audio file")

            waveform = torch.from_numpy(samples)
            sr = SAMPLE_RATE

        except Exception as e2:
            raise AudioLoadError(
                f"Failed to decode audio file: {str(e2)}"
            ) from e2

    # ---- sanity checks ----
    if waveform is None or waveform.numel() == 0:
        raise AudioLoadError("Loaded audio is empty")

    # mono
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)

    # resample
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    # duration control
    if waveform.numel() < TARGET_LEN:
        raise AudioLoadError("Audio too short for analysis")

    if waveform.numel() > TARGET_LEN:
        waveform = waveform[:TARGET_LEN]
    else:
        waveform = F.pad(waveform, (0, TARGET_LEN - waveform.numel()))

    return waveform.float()



def waveform_to_mel(waveform: torch.Tensor):
    """
    waveform: [T]
    returns: [1, T, N_MELS]
    """
    mel = mel_transform(waveform.unsqueeze(0))   # [1, n_mels, frames]
    mel = amp_to_db(mel)
    mel = mel.transpose(1, 2)                     # [1, frames, n_mels]
    return mel

def pad_time_dim(mel):
    T = mel.shape[1]
    pad_len = (8 - (T % 8)) % 8
    if pad_len > 0:
        mel = F.pad(mel, (0, 0, 0, pad_len))
    return mel


def extract_features(wav: torch.Tensor) -> torch.Tensor:
    mel = mel_transform(wav.unsqueeze(0))
    mel = amp_to_db(mel)
    if mel.dim == 4:
        mel = mel.squeeze(1)

    mel.transpose(1, 2)  # [B, T, N_MELS]
    mel = pad_time_dim(mel)
    return mel