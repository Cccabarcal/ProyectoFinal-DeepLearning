"""API FastAPI del clasificador de noticias.

Expone los cuatro modelos y sirve el frontend estático. Si los pesos aún no se
han entrenado, los modelos responden en MODO DEMO (campo `demo: true`), de modo
que la interfaz es totalmente usable antes de tener los checkpoints.

Ejecutar desde `app/backend/`:
    uvicorn main:app --reload --port 8000
Luego abrir http://localhost:8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from registry import CLASS_NAMES, REGISTRY

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Clasificador de noticias", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model: str | None = Field(None, description="clave de modelo; si se omite, usa todos")


@app.get("/api/health")
def health():
    return {"status": "ok", "classes": CLASS_NAMES}


@app.get("/api/models")
def models():
    return {"models": REGISTRY.list_info(), "classes": CLASS_NAMES}


@app.post("/api/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="El texto está vacío.")
    if req.model:
        m = REGISTRY.get(req.model)
        if m is None:
            raise HTTPException(status_code=404, detail=f"Modelo desconocido: {req.model}")
        r = m.predict(text)
        r.update({"key": m.key, "name": m.name})
        return {"text": text, "classes": CLASS_NAMES, "results": [r]}
    return {"text": text, "classes": CLASS_NAMES, "results": REGISTRY.predict_all(text)}


# --- Servir el frontend estático ---
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
