from fastapi import File, UploadFile, APIRouter
from audio.preprocessing import load_audio, extract_features
from model.ecapa import ECAPAENCODER
# from model.scorer import fatigue_score_0_to_100, prosody_score  ## for prosody scoring
from model.scorer import fatigue_score_0_to_100
from fastapi.responses import JSONResponse
import numpy as np
from utils.logger import logger
from utils.file_utils import save_temp_audio
from core.config import LOW_PERCENTILE, HIGH_PERCENTILE, FATIGUE_AXIS, REF_C_H, MAX_DURATION_SEC
# from audio.feature_extractor import get_prosody_stats
from fastapi import HTTPException, status
from audio.validators import validate_audio_duration, validate_audio_file, AudioValidationError


C_h = np.load(REF_C_H)
fatigue_axis = np.load(FATIGUE_AXIS)
low = float(np.load(LOW_PERCENTILE)["arr_0"])
high = float(np.load(HIGH_PERCENTILE)["arr_0"])

router = APIRouter()

encoder = ECAPAENCODER()

@router.post("/score")
async def score_voice(file: UploadFile = File(...)):
    try:

        path = save_temp_audio(file)
        validate_audio_file(
            file_path=path,
            original_filename=file.filename
        )
        wav = load_audio(path)
        # prosody_features = get_prosody_stats(wav)
        # p_score, report = prosody_score(prosody_features)
        features = extract_features(wav)
        wav = wav.squeeze()
        emb = encoder.encode(wav)
        score = float(fatigue_score_0_to_100(emb, C_h, fatigue_axis, low, high))
        # return {"fatigue_score": score, "prosody_score": p_score, "prosody_report": report}
        return {"fatigue_score" : score}

    except AudioValidationError as e:
        logger.warning(str(e))
        raise HTTPException(
            status_code= status.HTTP_400_BAD_REQUEST,
            detail = str(e)
        )
    except Exception as e:
        logger.exception("Unexpected server error")
        raise HTTPException(
            status_code= status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = "Unexpected server error."
        )