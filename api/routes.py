from fastapi import File, UploadFile, APIRouter
from audio.preprocessing import load_audio, extract_features
from model.ecapa import ECAPAENCODER
# from model.scorer import fatigue_score_0_to_100, prosody_score  ## for prosody scoring
from model.scorer import fatigue_score_0_to_100
from fastapi.responses import JSONResponse
import numpy as np
from utils.logger import logger
from utils.file_utils import save_temp_audio
from core.config import LOW_PERCENTILE, HIGH_PERCENTILE, FATIGUE_AXIS, REF_C_H
# from audio.feature_extractor import get_prosody_stats
from fastapi import HTTPException, status


C_h = np.load(REF_C_H)
fatigue_axis = np.load(FATIGUE_AXIS)
low = float(np.load(LOW_PERCENTILE)["arr_0"])
high = float(np.load(HIGH_PERCENTILE)["arr_0"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}

def validate_audio_file(filename: str):
    ext  = filename.lower().rsplit(".", 1)[-1]
    ext = "." + ext
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Unsupported file format received: {filename}")
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "Unsupported file type. Allowed formats are: " + ", ".join(ALLOWED_EXTENSIONS)
        )
router = APIRouter()

encoder = ECAPAENCODER()

@router.post("/score")
async def score_voice(file: UploadFile = File(...)):
    validate_audio_file(file.filename)
    path = save_temp_audio(file)
    wav = load_audio(path)
    # prosody_features = get_prosody_stats(wav)
    # p_score, report = prosody_score(prosody_features)
    features = extract_features(wav)
    wav = wav.squeeze()
    emb = encoder.encode(wav)
    score = float(fatigue_score_0_to_100(emb, C_h, fatigue_axis, low, high))
    # return {"fatigue_score": score, "prosody_score": p_score, "prosody_report": report}
    return {"fatigue_score" : score}