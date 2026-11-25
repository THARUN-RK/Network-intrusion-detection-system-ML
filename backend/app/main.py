"""
Network Intrusion Detection System (NIDS) & Security Operations Center (SOC) API Entry Point
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import database
import predict
from backend.app.schemas import HealthResponse, ModelInfoResponse
from backend.app.routes import prediction, analytics, history, auth_routes, soc_routes, api_live
from backend.app.websocket_manager import ws_manager
from backend.app.simulator import traffic_simulator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application startup: verify database and model loading
    database.init_db()
    predictor = predict.get_predictor()
    # Start real-time background traffic simulator
    traffic_simulator.start()
    yield
    # Graceful shutdown
    traffic_simulator.stop()


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
app.include_router(api_live.router)

INDEX_HTML_PATH = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse, tags=["Web Dashboard"], include_in_schema=False)
def serve_soc_dashboard():
    """Serves the fast, lightweight NIDS Security Operations Center web dashboard."""
    if INDEX_HTML_PATH.exists():
        return FileResponse(INDEX_HTML_PATH)
    return HTMLResponse("<h1>NIDS SOC API Active</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


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
