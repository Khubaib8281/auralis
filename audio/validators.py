import soundfile as sf
from core.config import MAX_DURATION_SEC, MIN_DURATION_SEC, ALLOWED_EXTENSIONS
from utils.logger import logger

class AudioValidationError(ValueError):
    pass


def validate_audio_duration(filepath: str, max_duration: float = MAX_DURATION_SEC):
    try:
        info = sf.info(filepath)
    except RuntimeError:
        raise AudioValidationError("Invalid or corrupted audio file.")

    duration = info.frames / float(info.samplerate)

    if duration > max_duration or duration < MIN_DURATION_SEC:
        logger.warning(f"Unsupported file length received: {original_filename}")
        raise AudioValidationError(
            f"Audio duration {duration:.2f}s invalid. "
            f"Allowed range: {MIN_DURATION_SEC:.2f}s – {max_duration:.2f}s."
        )

    return duration

def validate_audio_file(file_path: str, original_filename: str):
    ext = original_filename.lower().rsplit(".", 1)[-1]
    ext = "." + ext

    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Unsupported file format received: {original_filename}")
        raise AudioValidationError(
            f"Unsupported file type {ext}. Allowed formats are: " + ", ".join(ALLOWED_EXTENSIONS)
            )

    validate_audio_duration(file_path)