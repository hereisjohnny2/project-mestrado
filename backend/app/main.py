from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.projects import router as projects_router
from .core.config import get_settings
from .db.session import init_db

app = FastAPI(title="Rock Segmentation API", version="0.1.0")

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
