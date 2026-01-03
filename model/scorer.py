from core.config import LOW_PERCENTILE, HIGH_PERCENTILE, FATIGUE_AXIS, REF_C_H
import numpy as np


C_h = np.load(REF_C_H)
fatigue_axis = np.load(FATIGUE_AXIS)
low = float(np.load(LOW_PERCENTILE)["arr_0"])
high = float(np.load(HIGH_PERCENTILE)["arr_0"])

# def fatigue_score_0_to_100(emb: np.ndarray) -> float:
#     raw = np.dot(emb - C_h, fatigue_axis)
#     raw = np.clip(raw, low , high)
#     return 100 * (raw - low) / (high - low)

def fatigue_score_0_to_100(embedding, C_h, fatigue_axis, raw_low, raw_high, method='linear'):
    """
    embedding: 192-d numpy array
    C_h: healthy centroid
    fatigue_axis: unit vector (healthy -> fatigued)
    raw_low, raw_high: percentiles from training embeddings for scaling
    method: 'linear' or 'sigmoid'
    """
    raw = np.dot(embedding - C_h, fatigue_axis)

    if method == 'linear':
        # clip to training range (optional: allow some overshoot)
        raw = np.clip(raw, raw_low, raw_high)
        score = 100 * (raw - raw_low) / (raw_high - raw_low)
    elif method == 'sigmoid':
        # smooth bounded score
        midpoint = (raw_low + raw_high) / 2
        scale = (raw_high - raw_low) / 4  # controls steepness
        score = 1 / (1 + np.exp(-(raw - midpoint) / scale))
        score = score * 100
    else:
        raise ValueError("method must be 'linear' or 'sigmoid'")
    
    return score