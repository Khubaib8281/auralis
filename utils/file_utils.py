import tempfile
from fastapi import UploadFile

def save_temp_audio(file: UploadFile) -> str:
    suffix = file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
        tmp_file.write(file.file.read())
        return tmp_file.name