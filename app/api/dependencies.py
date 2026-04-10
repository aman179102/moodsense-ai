"""FastAPI dependencies."""

from functools import lru_cache
from typing import Optional

from fastapi import Request

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.predictor import MoodPredictor
from app.services.recommendation import RecommendationEngine

logger = get_logger(__name__)


class PredictorSingleton:
    """Singleton for MoodPredictor instance."""

    _instance: Optional[MoodPredictor] = None

    @classmethod
    def get_instance(cls) -> MoodPredictor:
        """Get or create the predictor singleton."""
        if cls._instance is None:
            logger.info("Initializing MoodPredictor singleton")
            cls._instance = MoodPredictor()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance."""
        cls._instance = None


class RecommendationEngineSingleton:
    """Singleton for RecommendationEngine instance."""

    _instance: Optional[RecommendationEngine] = None

    @classmethod
    def get_instance(cls) -> RecommendationEngine:
        """Get or create the recommendation engine singleton."""
        if cls._instance is None:
            logger.info("Initializing RecommendationEngine singleton")
            predictor = PredictorSingleton.get_instance()
            from app.services.embeddings import EmbeddingService
            embedding_service = EmbeddingService()
            # Try to load embedding service from model path
            import os
            model_path = get_settings().model_path
            if os.path.exists(model_path):
                embedding_service.load(model_path)
            cls._instance = RecommendationEngine(embedding_service=embedding_service)
        return cls._instance


def get_predictor() -> MoodPredictor:
    """FastAPI dependency for getting predictor."""
    return PredictorSingleton.get_instance()


def get_recommendation_engine() -> RecommendationEngine:
    """FastAPI dependency for getting recommendation engine."""
    return RecommendationEngineSingleton.get_instance()


def get_request_settings(request: Request) -> Settings:
    """Get settings from request state."""
    return request.app.state.settings
