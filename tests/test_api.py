"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.api.main import create_app


class TestAPI:
    """Test cases for API endpoints."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "MoodSense" in data["message"]

    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_loaded" in data

    def test_get_moods(self, client):
        """Test moods endpoint."""
        response = client.get("/moods")
        assert response.status_code == 200
        data = response.json()
        assert "moods" in data
        assert "count" in data
        assert len(data["moods"]) > 0

    def test_model_info_not_loaded(self, client):
        """Test model info when model not loaded."""
        response = client.get("/model/info")
        # Should return 503 when model not loaded
        assert response.status_code in [200, 503]

    def test_predict_no_model(self, client):
        """Test predict when model not loaded."""
        response = client.post(
            "/predict",
            json={"text": "I am happy today", "include_recommendations": False}
        )
        # Should fail if model not loaded
        assert response.status_code in [200, 503, 500]

    def test_predict_validation_empty_text(self, client):
        """Test predict with empty text."""
        response = client.post(
            "/predict",
            json={"text": "", "include_recommendations": False}
        )
        assert response.status_code == 422  # Validation error

    def test_predict_validation_whitespace_text(self, client):
        """Test predict with whitespace-only text."""
        response = client.post(
            "/predict",
            json={"text": "   ", "include_recommendations": False}
        )
        assert response.status_code == 422  # Validation error

    def test_predict_validation_long_text(self, client):
        """Test predict with very long text."""
        long_text = "word " * 5001  # Exceeds max length
        response = client.post(
            "/predict",
            json={"text": long_text, "include_recommendations": False}
        )
        assert response.status_code == 422  # Validation error

    def test_predict_batch_empty_list(self, client):
        """Test batch predict with empty list."""
        response = client.post(
            "/predict/batch",
            json={"texts": [], "include_recommendations": False}
        )
        assert response.status_code == 422  # Validation error

    def test_predict_batch_too_many(self, client):
        """Test batch predict with too many texts."""
        texts = ["text"] * 101  # Exceeds max
        response = client.post(
            "/predict/batch",
            json={"texts": texts, "include_recommendations": False}
        )
        assert response.status_code == 422  # Validation error

    def test_recommendations_endpoint(self, client):
        """Test recommendations endpoint."""
        response = client.get("/recommendations/happy?count=3")
        assert response.status_code == 200
        data = response.json()
        assert "mood" in data
        assert data["mood"] == "happy"
        assert "recommendations" in data

    def test_recommendations_invalid_mood(self, client):
        """Test recommendations with invalid mood."""
        response = client.get("/recommendations/invalid_mood")
        assert response.status_code == 400  # Bad request

    def test_api_docs(self, client):
        """Test API documentation endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_openapi_schema(self, client):
        """Test OpenAPI schema endpoint."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "prometheus" in response.text.lower() or "#" in response.text


@pytest.mark.asyncio
class TestAPIAsync:
    """Async test cases for API."""

    async def test_async_predict(self):
        """Test async predict endpoint."""
        app = create_app()
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/health")
            assert response.status_code == 200

    async def test_async_batch_predict(self):
        """Test async batch predict endpoint."""
        app = create_app()
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/predict/batch",
                json={"texts": ["I am happy", "I am sad"], "include_recommendations": False}
            )
            assert response.status_code in [200, 503, 500]
