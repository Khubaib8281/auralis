from app.core.config import LOW_PERCENTILE, HIGH_PERCENTILE, FATIGUE_AXIS, REF_C_H
import numpy as np


C_h = np.load(REF_C_H)
fatigue_axis = np.load(FATIGUE_AXIS)
low = np.float(np.load(LOW_PERCENTILE))
high = np.float(np.load(HIGH_PERCENTILE))

def fatigue_score_0_to_100(emb: np.ndarray) -> float:
    raw = np.dot(emb - C_h, fatigue_axis)
    raw = np.clip(raw, low , high)
    return 100 * (raw - low) / (high - low)