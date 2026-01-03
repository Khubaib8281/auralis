from fastapi import File, UploadFile, APIRouter
from audio.preprocessing import load_audio, extract_features
from model.ecapa import ECAPAENCODER
from model.scorer import fatigue_score_0_to_100
from fastapi.responses import JSONResponse
import numpy as np
from utils.file_utils import save_temp_audio
from core.config import LOW_PERCENTILE, HIGH_PERCENTILE, FATIGUE_AXIS, REF_C_H

C_h = np.load(REF_C_H)
fatigue_axis = np.load(FATIGUE_AXIS)
low = float(np.load(LOW_PERCENTILE)["arr_0"])
high = float(np.load(HIGH_PERCENTILE)["arr_0"])



router = APIRouter()

encoder = ECAPAENCODER()

@router.post("/score")
async def score_voice(file: UploadFile = File(...)):
    path = save_temp_audio(file)
    wav = load_audio(path)
    features = extract_features(wav)
    wav = wav.squeeze()
    emb = encoder.encode(wav)
    score = float(fatigue_score_0_to_100(emb, C_h, fatigue_axis, low, high))
    return {"fatigue_score": score}   