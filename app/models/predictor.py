"""Mood prediction model for inference."""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embeddings import EmbeddingService
from app.services.preprocessing import TextPreprocessor

logger = get_logger(__name__)


class MoodPrediction:
    """Result of a mood prediction."""

    def __init__(
        self,
        text: str,
        mood: str,
        confidence: float,
        all_probabilities: Dict[str, float],
        processing_time_ms: float,
    ):
        self.text = text
        self.mood = mood
        self.confidence = confidence
        self.all_probabilities = all_probabilities
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict:
        """Convert prediction to dictionary."""
        return {
            "text": self.text,
            "mood": self.mood,
            "confidence": round(self.confidence, 4),
            "all_probabilities": {
                k: round(v, 4) for k, v in self.all_probabilities.items()
            },
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class MoodPredictor:
    """Predictor for mood classification."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        """Initialize the mood predictor.

        Args:
            model_path: Path to saved model
            embedding_service: Embedding service instance
            preprocessor: Text preprocessor instance
        """
        self.settings = get_settings()
        self.model_path = model_path or self.settings.get_model_full_path()

        self.model = None
        self.model_name = None
        self.label_map: Dict[str, int] = {}
        self.reverse_label_map: Dict[int, str] = {}
        self.metrics = None

        self.embedding_service = embedding_service or EmbeddingService()
        self.preprocessor = preprocessor or TextPreprocessor(use_spacy=False)

        self._load_model()

    def _load_model(self) -> None:
        """Load the prediction model."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found at {self.model_path}. Need to train first.")
            return

        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.model_name = model_data.get("model_name", "Unknown")
            self.label_map = model_data.get("label_map", {})
            self.reverse_label_map = model_data.get("reverse_label_map", {})
            self.metrics = model_data.get("metrics", None)

            # Load embedding service state
            model_dir = os.path.dirname(self.model_path)
            self.embedding_service.load(model_dir)

            logger.info(f"Loaded model: {self.model_name}")
            logger.info(f"Available labels: {list(self.label_map.keys())}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(
        self,
        text: str,
        return_all_probabilities: bool = True,
    ) -> MoodPrediction:
        """Predict mood for a single text.

        Args:
            text: Input text to analyze
            return_all_probabilities: Whether to return all class probabilities

        Returns:
            MoodPrediction result
        """
        import time
        start_time = time.time()

        if self.model is None:
            raise RuntimeError("Model not loaded. Please train a model first.")

        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")

        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess(text)

            # Generate embeddings
            embeddings = self.embedding_service.get_combined_embeddings([processed_text])

            # Handle LightGBM feature names
            if hasattr(self.model, "feature_name"):
                embeddings_df = self._create_feature_dataframe(embeddings)
            else:
                embeddings_df = embeddings

            # Get predictions
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(embeddings_df)[0]
            else:
                # Fallback for models without probability
                pred = self.model.predict(embeddings_df)[0]
                n_classes = len(self.label_map)
                probabilities = np.zeros(n_classes)
                probabilities[pred] = 1.0

            # Get predicted class
            predicted_idx = int(np.argmax(probabilities))
            predicted_mood = self.reverse_label_map.get(predicted_idx, "unknown")
            confidence = float(probabilities[predicted_idx])

            # Build probability dict
            all_probabilities = {}
            if return_all_probabilities:
                for idx, prob in enumerate(probabilities):
                    mood = self.reverse_label_map.get(idx, f"class_{idx}")
                    all_probabilities[mood] = float(prob)

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            return MoodPrediction(
                text=text,
                mood=predicted_mood,
                confidence=confidence,
                all_probabilities=all_probabilities,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def predict_batch(
        self,
        texts: List[str],
        return_all_probabilities: bool = True,
    ) -> List[MoodPrediction]:
        """Predict mood for multiple texts.

        Args:
            texts: List of input texts
            return_all_probabilities: Whether to return all class probabilities

        Returns:
            List of MoodPrediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please train a model first.")

        results = []
        for text in texts:
            try:
                result = self.predict(text, return_all_probabilities)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for text '{text[:50]}...': {e}")
                results.append(None)

        return results

    def _create_feature_dataframe(self, embeddings: np.ndarray):
        """Create feature DataFrame for LightGBM compatibility.

        Args:
            embeddings: Embedding matrix

        Returns:
            DataFrame or array depending on model type
        """
        import pandas as pd

        if hasattr(self.model, "feature_name"):
            feature_names = self.model.feature_name()
            if feature_names:
                return pd.DataFrame(
                    embeddings,
                    columns=[f"feature_{i}" for i in range(embeddings.shape[1])]
                )

        return embeddings

    def get_model_info(self) -> Dict:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "labels": list(self.label_map.keys()),
            "num_classes": len(self.label_map),
            "model_path": self.model_path,
            "metrics": {
                "accuracy": getattr(self.metrics, "accuracy", None),
                "precision": getattr(self.metrics, "precision", None),
                "recall": getattr(self.metrics, "recall", None),
                "f1_score": getattr(self.metrics, "f1_score", None),
            } if self.metrics else None,
        }

    def is_ready(self) -> bool:
        """Check if predictor is ready for predictions.

        Returns:
            True if model is loaded
        """
        return self.model is not None
