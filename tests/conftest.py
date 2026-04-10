"""Pytest configuration and fixtures."""

import os
import tempfile

import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary test data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="session")
def sample_texts():
    """Sample texts for testing."""
    return [
        "I am so happy today! Everything is wonderful.",
        "I feel really sad and down right now.",
        "This makes me so angry and frustrated!",
        "I'm anxious about the upcoming meeting.",
        "Just a regular day, nothing special.",
    ]


@pytest.fixture(scope="session")
def sample_labels():
    """Sample labels for testing."""
    return ["happy", "sad", "angry", "anxious", "neutral"]


@pytest.fixture(scope="function")
def mock_env_vars(monkeypatch):
    """Set mock environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MODEL_PATH", "/tmp/test_models")
