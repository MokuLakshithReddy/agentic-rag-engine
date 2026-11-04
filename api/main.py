"""
Agentic RAG Engine - FastAPI Application Entrypoint.
Configures safe CORS origins, exception handling, and mounts the research UI.
"""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.routes import router

app = FastAPI(
    title="Agentic RAG Engine API",
    description="Production-oriented Autonomous Knowledge Research & Grounded Synthesis System",
    version="1.0.0"
)

# Configurable CORS origins
raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include core API routes
app.include_router(router)

# Mount frontend web interface
frontend_dir = Path(__file__).resolve().parent.parent / "app"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
