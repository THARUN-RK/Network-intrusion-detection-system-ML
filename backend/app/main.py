"""
Network Intrusion Detection System (NIDS) & Security Operations Center (SOC) API Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import database
import predict
from backend.app.schemas import HealthResponse, ModelInfoResponse
from backend.app.routes import prediction, analytics, history, auth_routes, soc_routes
from backend.app.websocket_manager import ws_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application startup: verify database and model loading
    database.init_db()
    predictor = predict.get_predictor()
    yield


app = FastAPI(
    title="Network Security Operations Center (SOC) & NIDS API",
    description="Multiclass SVM Threat Detection, IP Reputation, and Incident Response API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_routes.router)
app.include_router(soc_routes.router)
app.include_router(prediction.router)
app.include_router(analytics.router)
app.include_router(history.router)


# ==============================================================================
# WEBSOCKET REAL-TIME DISPATCH
# ==============================================================================

@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """WebSocket channel for live threat streaming and instant alert notifications."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep-alive ping loop
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Checks service operational status."""
    return {
        "status": "healthy",
        "model_loaded": predict.get_predictor().pipeline is not None,
        "version": "2.0.0"
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
