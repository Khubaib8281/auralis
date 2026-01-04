# import numpy as np
# import parselmouth
# from core.config import SAMPLE_RATE

# def get_prosody_stats(waveforms, sr=SAMPLE_RATE):
#     feats = {"pitch_mean" : [], "pitch_std" : [], "jitter" : [], "shimmer" : [], "hnr" : []}

#     for wav in waveforms:
#         snd = parselmouth.Sound(wav.numpy, sampling_frequency=sr)
#         pitch = snd.to_pitch()

#         feats["pitch_mean"].append(pitch.mean())
#         feats["pitch_std"].append(pitch.stdev())
#         feats["jitter"].append(snd.get_jitter_local())
#         feats["shimmer"].append(snd.get_shimmer_local())
#         feats["hnr"].append(snd.to_harmonicity().mean())

#         thresholds = {}

#         for k, v in feats.items():
#             thresholds[k] = (np.percentile(v, 5), np.percentile(v, 95))
#         return thresholds