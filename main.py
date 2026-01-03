from fastapi import FastAPI
from api.routes import router

app = FastAPI(title = "Vocal fatigue scoring API")

app.include_router(router, prefix="/api")

@app.get("/health")
def health():
    return {"status" : "ok"}  