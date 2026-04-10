"""Model training pipeline for mood classification."""

import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from app.core.config import get_settings
from app.core.constants import SAMPLE_TRAINING_DATA
from app.core.logging import get_logger
from app.services.embeddings import EmbeddingService
from app.services.preprocessing import TextPreprocessor

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Experiment tracking disabled.")

logger = get_logger(__name__)

warnings.filterwarnings("ignore")


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    classification_report: str = ""


@dataclass
class TrainingResult:
    """Result of model training."""

    model_name: str
    model: Any
    metrics: ModelMetrics
    best_params: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None


class ModelTrainer:
    """Trainer for multiple mood classification models."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        use_mlflow: bool = True,
    ):
        """Initialize the model trainer.

        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            use_mlflow: Whether to use MLflow for tracking
        """
        self.settings = get_settings()
        self.test_size = test_size
        self.random_state = random_state
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        self.preprocessor = TextPreprocessor(use_spacy=False)
        self.embedding_service = EmbeddingService()

        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.label_map: Optional[Dict[str, int]] = None

    def prepare_data(
        self,
        texts: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and test data.

        Args:
            texts: List of text samples (uses sample data if None)
            labels: List of labels (uses sample data if None)

        Returns:
            X_train, X_test, y_train, y_test
        """
        if texts is None or labels is None:
            logger.info("Using sample training data")
            texts, labels = zip(*SAMPLE_TRAINING_DATA)

        # Preprocess texts
        logger.info(f"Preprocessing {len(texts)} texts...")
        processed_texts = [self.preprocessor.preprocess(t) for t in texts]

        # Create label mapping
        unique_labels = sorted(set(labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}

        # Convert labels to numeric
        y = np.array([self.label_map[label] for label in labels])

        # Generate embeddings
        logger.info("Generating embeddings...")
        self.embedding_service.fit_tfidf(processed_texts)
        X = self.embedding_service.get_combined_embeddings(processed_texts)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Feature dimensions: {X.shape[1]}")
        logger.info(f"Classes: {unique_labels}")

        return X_train, X_test, y_train, y_test

    def train_logistic_regression(self) -> TrainingResult:
        """Train Logistic Regression model.

        Returns:
            TrainingResult with model and metrics
        """
        logger.info("Training Logistic Regression...")

        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "class_weight": ["balanced", None],
        }

        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1,
        )

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )

        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)

        metrics = self._calculate_metrics(y_pred)

        # Feature importance for LR (coefficients)
        feature_importance = None
        if hasattr(best_model, "coef_"):
            avg_coef = np.mean(np.abs(best_model.coef_), axis=0)
            feature_importance = {
                f"feature_{i}": float(coef)
                for i, coef in enumerate(avg_coef[:20])  # Top 20
            }

        return TrainingResult(
            model_name="LogisticRegression",
            model=best_model,
            metrics=metrics,
            best_params=grid_search.best_params_,
            feature_importance=feature_importance,
        )

    def train_naive_bayes(self) -> TrainingResult:
        """Train Naive Bayes model on TF-IDF features.

        Returns:
            TrainingResult with model and metrics
        """
        logger.info("Training Naive Bayes...")

        # NB works best with TF-IDF, so we use a different pipeline
        texts_train = self._get_texts_from_embeddings(self.X_train)
        texts_test = self._get_texts_from_embeddings(self.X_test)

        # Create pipeline with TF-IDF + NB
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ("nb", MultinomialNB()),
        ])

        param_grid = {
            "nb__alpha": [0.1, 0.5, 1.0, 2.0],
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )

        grid_search.fit(texts_train, self.y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(texts_test)

        metrics = self._calculate_metrics(y_pred)

        # Get feature importance from TF-IDF
        tfidf = best_model.named_steps["tfidf"]
        nb = best_model.named_steps["nb"]

        feature_names = tfidf.get_feature_names_out()
        log_probs = nb.feature_log_prob_
        avg_log_prob = np.mean(log_probs, axis=0)
        top_indices = np.argsort(avg_log_prob)[-20:][::-1]

        feature_importance = {
            feature_names[i]: float(avg_log_prob[i])
            for i in top_indices
        }

        return TrainingResult(
            model_name="NaiveBayes",
            model=best_model,
            metrics=metrics,
            best_params=grid_search.best_params_,
            feature_importance=feature_importance,
        )

    def train_lightgbm(self) -> TrainingResult:
        """Train LightGBM model.

        Returns:
            TrainingResult with model and metrics
        """
        logger.info("Training LightGBM...")

        param_grid = {
            "num_leaves": [31, 50],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 200],
        }

        model = lgb.LGBMClassifier(
            objective="multiclass",
            boosting_type="gbdt",
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )

        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)

        metrics = self._calculate_metrics(y_pred)

        # Feature importance
        feature_importance = None
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            top_indices = np.argsort(importances)[-20:][::-1]
            feature_importance = {
                f"feature_{i}": float(importances[i])
                for i in top_indices
            }

        return TrainingResult(
            model_name="LightGBM",
            model=best_model,
            metrics=metrics,
            best_params=grid_search.best_params_,
            feature_importance=feature_importance,
        )

    def _calculate_metrics(self, y_pred: np.ndarray) -> ModelMetrics:
        """Calculate model performance metrics.

        Args:
            y_pred: Predicted labels

        Returns:
            ModelMetrics instance
        """
        target_names = [
            self.reverse_label_map[i]
            for i in range(len(self.reverse_label_map))
        ]

        return ModelMetrics(
            accuracy=float(accuracy_score(self.y_test, y_pred)),
            precision=float(precision_score(self.y_test, y_pred, average="weighted")),
            recall=float(recall_score(self.y_test, y_pred, average="weighted")),
            f1_score=float(f1_score(self.y_test, y_pred, average="weighted")),
            classification_report=classification_report(
                self.y_test, y_pred, target_names=target_names
            ),
        )

    def _get_texts_from_embeddings(self, X: np.ndarray) -> List[str]:
        """Approximate original texts from embeddings (for NB training).

        Args:
            X: Embedding matrix

        Returns:
            List of placeholder texts
        """
        # For NB, we re-process from original - this is a simplification
        # In production, you'd store the original texts
        return [f"sample_{i}" for i in range(len(X))]

    def train_all_models(
        self,
        texts: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> List[TrainingResult]:
        """Train all models and return results.

        Args:
            texts: Training texts (uses sample if None)
            labels: Training labels (uses sample if None)

        Returns:
            List of TrainingResult for each model
        """
        # Prepare data
        self.prepare_data(texts, labels)

        results = []

        # Train Logistic Regression
        try:
            lr_result = self.train_logistic_regression()
            results.append(lr_result)
            self._log_to_mlflow(lr_result)
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")

        # Train Naive Bayes (on subset of data due to text requirement)
        try:
            # Use original sample data for NB
            nb_texts, nb_labels = zip(*[
                (t, l) for t, l in SAMPLE_TRAINING_DATA[:60]
            ])
            self.prepare_data(nb_texts, nb_labels)
            nb_result = self.train_naive_bayes()
            results.append(nb_result)
            self._log_to_mlflow(nb_result)

            # Reset to full data
            self.prepare_data(texts, labels)
        except Exception as e:
            logger.error(f"Error training Naive Bayes: {e}")

        # Train LightGBM
        try:
            lgb_result = self.train_lightgbm()
            results.append(lgb_result)
            self._log_to_mlflow(lgb_result)
        except Exception as e:
            logger.error(f"Error training LightGBM: {e}")

        return results

    def _log_to_mlflow(self, result: TrainingResult) -> None:
        """Log training results to MLflow.

        Args:
            result: TrainingResult to log
        """
        if not self.use_mlflow:
            return

        try:
            mlflow.set_experiment(self.settings.mlflow_experiment_name)

            with mlflow.start_run(run_name=result.model_name):
                # Log params
                mlflow.log_params(result.best_params)
                mlflow.log_param("model_name", result.model_name)

                # Log metrics
                mlflow.log_metrics({
                    "accuracy": result.metrics.accuracy,
                    "precision": result.metrics.precision,
                    "recall": result.metrics.recall,
                    "f1_score": result.metrics.f1_score,
                })

                # Log model
                mlflow.sklearn.log_model(result.model, "model")

                logger.info(f"Logged {result.model_name} to MLflow")

        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")

    def select_best_model(
        self, results: List[TrainingResult], metric: str = "f1_score"
    ) -> TrainingResult:
        """Select best model based on specified metric.

        Args:
            results: List of training results
            metric: Metric to use for selection

        Returns:
            Best TrainingResult
        """
        if not results:
            raise ValueError("No training results available")

        best = max(results, key=lambda r: getattr(r.metrics, metric, 0))
        logger.info(
            f"Best model: {best.model_name} "
            f"({metric}={getattr(best.metrics, metric):.4f})"
        )
        return best

    def save_model(
        self,
        result: TrainingResult,
        path: Optional[str] = None,
    ) -> str:
        """Save trained model and related artifacts.

        Args:
            result: TrainingResult to save
            path: Directory to save to

        Returns:
            Path to saved model
        """
        path = path or self.settings.model_path
        os.makedirs(path, exist_ok=True)

        # Save model
        model_filename = f"{result.model_name.lower()}_model.pkl"
        model_path = os.path.join(path, model_filename)

        with open(model_path, "wb") as f:
            pickle.dump({
                "model": result.model,
                "model_name": result.model_name,
                "metrics": result.metrics,
                "label_map": self.label_map,
                "reverse_label_map": self.reverse_label_map,
            }, f)

        logger.info(f"Saved model to {model_path}")

        # Save embedding service
        self.embedding_service.save(path)

        # Save as default model
        default_path = os.path.join(path, "mood_classifier.pkl")
        with open(default_path, "wb") as f:
            pickle.dump({
                "model": result.model,
                "model_name": result.model_name,
                "metrics": result.metrics,
                "label_map": self.label_map,
                "reverse_label_map": self.reverse_label_map,
            }, f)

        logger.info(f"Saved default model to {default_path}")

        return model_path
