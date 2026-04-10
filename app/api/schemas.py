"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request schema for mood prediction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for mood detection",
        examples=["I am feeling great today!"],
    )
    include_recommendations: bool = Field(
        default=True,
        description="Whether to include content recommendations",
    )
    include_explanation: bool = Field(
        default=True,
        description="Whether to include explanation of the recommendation",
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Ensure text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or just whitespace")
        return v.strip()


class PredictBatchRequest(BaseModel):
    """Request schema for batch mood prediction."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to analyze",
    )
    include_recommendations: bool = Field(default=False)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate that all texts are non-empty."""
        if not all(isinstance(t, str) and t.strip() for t in v):
            raise ValueError("All texts must be non-empty strings")
        return [t.strip() for t in v]


class MoodProbability(BaseModel):
    """Schema for mood probability."""

    mood: str
    probability: float = Field(ge=0.0, le=1.0)


class RecommendationItem(BaseModel):
    """Schema for a recommendation item."""

    type: str = Field(..., description="Type of recommendation (music, activity, movie, quote)")
    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None
    similarity_score: Optional[float] = None


class PredictResponse(BaseModel):
    """Response schema for mood prediction."""

    text: str
    mood: str
    confidence: float = Field(ge=0.0, le=1.0)
    all_probabilities: Dict[str, float]
    processing_time_ms: float
    recommendations: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None


class BatchPredictResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictResponse]
    total_processed: int
    errors: int


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    version: str
    model_loaded: bool
    model_name: Optional[str] = None
    timestamp: datetime
    uptime_seconds: Optional[float] = None


class ModelInfoResponse(BaseModel):
    """Model information response schema."""

    model_name: str
    model_type: str
    labels: List[str]
    num_classes: int
    metrics: Optional[Dict[str, Optional[float]]] = None


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
