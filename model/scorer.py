from core.config import LOW_PERCENTILE, HIGH_PERCENTILE, FATIGUE_AXIS, REF_C_H
import numpy as np
from core.config import CONFIG


C_h = np.load(REF_C_H)
fatigue_axis = np.load(FATIGUE_AXIS)
low = float(np.load(LOW_PERCENTILE)["arr_0"])
high = float(np.load(HIGH_PERCENTILE)["arr_0"])

# def fatigue_score_0_to_100(emb: np.ndarray) -> float:
#     raw = np.dot(emb - C_h, fatigue_axis)
#     raw = np.clip(raw, low , high)
#     return 100 * (raw - low) / (high - low)

def fatigue_score_0_to_100(embedding, C_h, fatigue_axis, raw_low, raw_high, method='sigmoid'):
    """
    Compute a continuous fatigue score (0-100) from an embedding.

    embedding: 192-d numpy array
    C_h: healthy centroid (192-d)
    fatigue_axis: unit vector from healthy -> fatigued (192-d)
    raw_low, raw_high: training percentile values along the fatigue axis
    method: 'linear', 'sigmoid', or 'smooth_linear'

    Returns: float [0, 100]
    """
    # Project embedding along fatigue axis
    raw = np.dot(embedding - C_h, fatigue_axis)

    # Normalize raw value to [0, 1] within training range
    normalized = (raw - raw_low) / (raw_high - raw_low)

    # Clamp slightly beyond training range to avoid extreme scores
    normalized = np.clip(normalized, -0.05, 1.05)

    if method == 'linear':
        score = normalized * 100  # simple linear scaling
    elif method == 'sigmoid':
        # Smooth sigmoid, less steep
        midpoint = 0.5
        scale = 0.25  # tune this for slope; bigger = smoother
        score = 1 / (1 + np.exp(-(normalized - midpoint) / scale)) * 100
    elif method == 'smooth_linear':
        # Combine linear scaling with mild sigmoid smoothing at ends
        # This gives a natural 0-100 spread but saturates near extremes
        scale = 10  # controls smoothness near 0 and 100
        score = normalized * 100
        score = 100 / (1 + np.exp(- (score - 50) / scale))
    else:
        raise ValueError("method must be 'linear', 'sigmoid', or 'smooth_linear'")

    # Ensure the output is float and bounded
    return float(np.clip(score, 0, 100))


# def prosody_score(prosody_feats):
#     report = []
#     thresholds = CONFIG['prosody_thresholds']

#     for feat, val in prosody_feats.items():
#         low, high = thresholds[feat]
#         if val < low:
#             report.append(f"{feat} is low → potential fatigue")
#         elif val > high:
#             report.append(f"{feat} is high → potential fatigue")
    
#     score = len(report)  # simple count, or map to 0-100 if needed
#     return score, report