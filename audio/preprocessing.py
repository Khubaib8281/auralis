import torch
import torchaudio
from app.core.config import SAMPLE_RATE, DEVICE, N_MELS

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft = 400,
    hop_length = 256,
    n_mels = N_MELS    
).to(DEVICE)

amp_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

def load_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transform.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim = 0)
    return wav.to(DEVICE)  

def extract_features(wav: torch.Tensor) -> torch.Tensor:
    mel = mel_transform(wav.unsqueeze(0))
    mel = amp_to_db(mel)
    if mel.dim == 4:
        mel = mel.squeeze(0)
    return mel.transpose(1, 2)  # [B, T, N_MELS]