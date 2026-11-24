"""
Network Intrusion Detection System (NIDS) - FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import database
import predict
from backend.app.schemas import HealthResponse, ModelInfoResponse
from backend.app.routes import prediction, analytics, history


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application startup: verify database and model loading
    database.init_db()
    predictor = predict.get_predictor()
    yield
    # Cleanup on shutdown if needed


app = FastAPI(
    title="Network Intrusion Detection System (NIDS) API",
    description="Multiclass SVM Cyberattack Detection API for CIC-IDS2017 Traffic",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration for Streamlit frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(prediction.router)
app.include_router(analytics.router)
app.include_router(history.router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Checks service operational status."""
    return {
        "status": "healthy",
        "model_loaded": predict.get_predictor().pipeline is not None,
        "version": "1.0.0"
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def get_model_information():
    """Returns technical metadata for the trained SVM model."""
    predictor = predict.get_predictor()
    meta = predictor.metadata
    return {
        "model_type": meta.get("model_type", "Support Vector Machine"),
        "kernel": meta.get("kernel", "rbf"),
        "best_params": meta.get("best_params", {}),
        "feature_count": meta.get("feature_count", len(predictor.expected_features)),
        "training_samples_count": meta.get("training_samples_count", 0),
        "classes": predictor.classes,
        "cv_macro_f1": meta.get("cv_macro_f1"),
        "training_duration_seconds": meta.get("training_duration_seconds")
    }


if __name__ == "__main__":
    import uvicorn
    import config
    uvicorn.run("backend.app.main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
