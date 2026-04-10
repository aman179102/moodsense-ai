"""Application configuration settings."""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="MoodSense AI", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_workers: int = Field(default=1, alias="API_WORKERS")

    # Model
    model_path: str = Field(default="./models", alias="MODEL_PATH")
    default_model: str = Field(default="mood_classifier.pkl", alias="DEFAULT_MODEL")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    # NLP
    spacy_model: str = Field(default="en_core_web_sm", alias="SPACY_MODEL")
    use_gpu: bool = Field(default=False, alias="USE_GPU")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # MLflow
    mlflow_tracking_uri: Optional[str] = Field(
        default=None, alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="moodsense_experiments", alias="MLFLOW_EXPERIMENT_NAME"
    )

    # Hugging Face
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    hf_space_name: Optional[str] = Field(default=None, alias="HF_SPACE_NAME")

    # Recommendation
    max_recommendations: int = Field(default=5, alias="MAX_RECOMMENDATIONS")
    similarity_threshold: float = Field(default=0.7, alias="SIMILARITY_THRESHOLD")

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"

    def get_model_full_path(self) -> str:
        """Get the full path to the default model file."""
        return os.path.join(self.model_path, self.default_model)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
