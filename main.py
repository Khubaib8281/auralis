from fastapi import FastAPI, APIRouter, Request
from api.routes import router
import time
import logging

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title = "Vocal fatigue scoring API")

@app.middleware("HTTP")   
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    logging.info(
        f"{request.method} {request.url.path} "
        f"status_code = {response.status_code} "
        f"time = {duration:.3f}"
    )
    return response

api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(router, prefix="/voice")

app.include_router(api_v1)

@app.get("/health")
def health():
    return {"status" : "ok"}  
