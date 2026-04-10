"""FastAPI application factory."""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app

from app.api.dependencies import PredictorSingleton, RecommendationEngineSingleton
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting up MoodSense AI...")

    # Initialize singletons
    predictor = PredictorSingleton.get_instance()
    engine = RecommendationEngineSingleton.get_instance()

    app.state.start_time = datetime.now()
    app.state.settings = get_settings()

    if predictor.is_ready():
        logger.info("MoodPredictor initialized successfully")
    else:
        logger.warning("MoodPredictor not ready - model may need training")

    yield

    # Shutdown
    logger.info("Shutting down MoodSense AI...")
    PredictorSingleton.reset_instance()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    # Setup logging
    setup_logging()

    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered mood detection and personalized content recommendation",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Include API routes
    app.include_router(router, prefix="/api/v1")
    app.include_router(router, prefix="")

    return app
