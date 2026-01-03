from fastapi import File, UploadFile, APIRouter
from audio.preprocessing import load_audio, extract_features
from model.ecapa import ECAPAENCODER
from model.scorer import fatigue_score_0_to_100
from fastapi.responses import JSONResponse
import numpy as np
from utils.file_utils import save_temp_audio

router = APIRouter()

encoder = ECAPAENCODER()

@router.post("/score")
async def score_voice(file: UploadFile = File(...)):
    path = save_temp_audio(file)
    wav = load_audio(path)
    features = extract_features(wav)
    wav = wav.squeeze()
    emb = encoder.encode(wav)
    score = float(fatigue_score_0_to_100(emb))
    return {"fatigue_score": score}   