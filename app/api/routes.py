"""API routes for MoodSense AI."""

import time
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from prometheus_client import Counter, Histogram

from app.api.dependencies import (
    get_predictor,
    get_recommendation_engine,
    get_request_settings,
)
from app.api.schemas import (
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictBatchRequest,
    PredictRequest,
    PredictResponse,
)
from app.core.config import Settings
from app.core.constants import MOOD_DESCRIPTIONS
from app.core.logging import get_logger
from app.models.predictor import MoodPredictor
from app.services.recommendation import RecommendationEngine

logger = get_logger(__name__)
router = APIRouter()

# Prometheus metrics
prediction_counter = Counter(
    "moodsense_predictions_total",
    "Total predictions made",
    ["mood", "status"]
)
prediction_histogram = Histogram(
    "moodsense_prediction_duration_seconds",
    "Prediction duration in seconds",
)
error_counter = Counter(
    "moodsense_errors_total",
    "Total errors",
    ["error_type"]
)


@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to MoodSense AI v2",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse)
async def health_check(
    request: Request,
    predictor: MoodPredictor = Depends(get_predictor),
):
    """Health check endpoint."""
    start_time = getattr(request.app.state, "start_time", datetime.now())
    uptime = (datetime.now() - start_time).total_seconds()

    model_info = predictor.get_model_info()

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model_loaded=model_info.get("status") != "not_loaded",
        model_name=model_info.get("model_name"),
        timestamp=datetime.now(),
        uptime_seconds=uptime,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(
    predictor: MoodPredictor = Depends(get_predictor),
):
    """Get model information."""
    info = predictor.get_model_info()

    if info.get("status") == "not_loaded":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train a model first.",
        )

    return ModelInfoResponse(
        model_name=info["model_name"],
        model_type=info["model_type"],
        labels=info["labels"],
        num_classes=info["num_classes"],
        metrics=info.get("metrics"),
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    },
)
async def predict(
    request: PredictRequest,
    predictor: MoodPredictor = Depends(get_predictor),
    engine: RecommendationEngine = Depends(get_recommendation_engine),
):
    """Predict mood from text.

    Analyze the emotional tone of the provided text and return the detected
    mood with confidence scores and personalized recommendations.
    """
    if not predictor.is_ready():
        error_counter.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train a model first.",
        )

    with prediction_histogram.time():
        try:
            # Make prediction
            prediction = predictor.predict(
                text=request.text,
                return_all_probabilities=True,
            )

            prediction_counter.labels(
                mood=prediction.mood,
                status="success"
            ).inc()

            # Build response
            response_data = {
                "text": prediction.text,
                "mood": prediction.mood,
                "confidence": prediction.confidence,
                "all_probabilities": prediction.all_probabilities,
                "processing_time_ms": prediction.processing_time_ms,
            }

            # Add recommendations if requested
            if request.include_recommendations:
                rec_result = engine.get_hybrid_recommendations(
                    text=request.text,
                    mood=prediction.mood,
                    confidence=prediction.confidence,
                )
                response_data["recommendations"] = rec_result

                # Add explanation if requested
                if request.include_explanation:
                    explanation = engine.explain_recommendation(
                        text=request.text,
                        mood=prediction.mood,
                        confidence=prediction.confidence,
                        strategy=rec_result.get("strategy", "rule-based"),
                    )
                    response_data["explanation"] = explanation

            return PredictResponse(**response_data)

        except ValueError as e:
            error_counter.labels(error_type="validation_error").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            error_counter.labels(error_type="prediction_error").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error processing prediction",
            )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    },
)
async def predict_batch(
    request: PredictBatchRequest,
    predictor: MoodPredictor = Depends(get_predictor),
):
    """Predict mood for multiple texts in batch."""
    if not predictor.is_ready():
        error_counter.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train a model first.",
        )

    try:
        predictions = predictor.predict_batch(
            texts=request.texts,
            return_all_probabilities=True,
        )

        response_predictions = []
        errors = 0

        for pred in predictions:
            if pred is None:
                errors += 1
                continue

            response_predictions.append(
                PredictResponse(
                    text=pred.text,
                    mood=pred.mood,
                    confidence=pred.confidence,
                    all_probabilities=pred.all_probabilities,
                    processing_time_ms=pred.processing_time_ms,
                )
            )

            prediction_counter.labels(
                mood=pred.mood,
                status="success"
            ).inc()

        return BatchPredictResponse(
            predictions=response_predictions,
            total_processed=len(request.texts),
            errors=errors,
        )

    except ValueError as e:
        error_counter.labels(error_type="validation_error").inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        error_counter.labels(error_type="batch_prediction_error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing batch prediction",
        )


@router.get("/moods")
async def get_moods():
    """Get all available mood labels and descriptions."""
    return {
        "moods": [
            {
                "label": mood,
                "description": desc,
            }
            for mood, desc in MOOD_DESCRIPTIONS.items()
        ],
        "count": len(MOOD_DESCRIPTIONS),
    }


@router.get("/recommendations/{mood}")
async def get_mood_recommendations(
    mood: str,
    count: int = 5,
    engine: RecommendationEngine = Depends(get_recommendation_engine),
):
    """Get recommendations for a specific mood."""
    from app.core.constants import MoodLabel

    if mood not in [m.value for m in MoodLabel]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mood: {mood}. Valid moods: {[m.value for m in MoodLabel]}",
        )

    recs = engine.get_rule_based_recommendations(mood, count=count)
    activities = engine.get_activity_suggestions(mood, count=3)

    return {
        "mood": mood,
        "recommendations": recs,
        "activity_suggestions": activities,
    }
