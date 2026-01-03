import torch
import torchaudio
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_MELS = 80
checkpoint = torch.load("/home/khubaib/projects/vocal_fatigue_scoring/models/ecapa_supcon_model.pth", map_location=DEVICE)   

ecapa = ECAPA_TDNN(input_size=N_MELS, lin_neurons=192, channels = [512, 512, 512],kernel_sizes=[5, 3, 3], dilations=[1, 2 , 3])

ecapa.load_state_dict(checkpoint['ecapa_state_dict'])

print("loaded")   