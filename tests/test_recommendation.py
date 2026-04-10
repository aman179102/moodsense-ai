"""Tests for recommendation engine."""

import pytest

from app.core.constants import MOOD_RECOMMENDATIONS, MoodLabel
from app.services.recommendation import RecommendationEngine


class TestRecommendationEngine:
    """Test cases for RecommendationEngine."""

    @pytest.fixture(scope="class")
    def engine(self):
        """Create recommendation engine instance."""
        return RecommendationEngine(max_recommendations=5, similarity_threshold=0.5)

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.max_recommendations == 5
        assert engine.similarity_threshold == 0.5
        assert engine.content_db is not None

    def test_get_rule_based_recommendations(self, engine):
        """Test rule-based recommendations."""
        recs = engine.get_rule_based_recommendations(MoodLabel.HAPPY, count=3)
        assert isinstance(recs, list)
        assert len(recs) <= 3
        assert all("type" in rec for rec in recs)

    def test_get_rule_based_recommendations_unknown_mood(self, engine):
        """Test rule-based recommendations for unknown mood."""
        recs = engine.get_rule_based_recommendations("unknown_mood", count=3)
        # Should fall back to neutral
        assert isinstance(recs, list)

    def test_get_activity_suggestions(self, engine):
        """Test activity suggestions."""
        activities = engine.get_activity_suggestions(MoodLabel.SAD, count=3)
        assert isinstance(activities, list)
        assert len(activities) == 3
        assert all(isinstance(a, str) for a in activities)

    def test_get_activity_suggestions_unique(self, engine):
        """Test that activity suggestions are unique."""
        activities = engine.get_activity_suggestions(MoodLabel.HAPPY, count=5)
        assert len(activities) == len(set(activities))  # All unique

    def test_explain_recommendation_high_confidence(self, engine):
        """Test recommendation explanation for high confidence."""
        explanation = engine.explain_recommendation(
            text="I am so happy!",
            mood=MoodLabel.HAPPY,
            confidence=0.9,
            strategy="similarity-based",
        )
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "confident" in explanation.lower()

    def test_explain_recommendation_low_confidence(self, engine):
        """Test recommendation explanation for low confidence."""
        explanation = engine.explain_recommendation(
            text="I am so happy!",
            mood=MoodLabel.HAPPY,
            confidence=0.3,
            strategy="rule-based",
        )
        assert isinstance(explanation, str)
        assert "less certain" in explanation.lower() or "uncertainty" in explanation.lower()

    def test_content_database_complete(self, engine):
        """Test that content database has all moods."""
        for mood in MoodLabel:
            assert mood in engine.content_db
            assert len(engine.content_db[mood]) > 0

    def test_recommendation_types(self, engine):
        """Test that recommendations have valid types."""
        valid_types = {"music", "activity", "movie", "quote"}

        for mood in MoodLabel:
            recs = engine.get_rule_based_recommendations(mood, count=10)
            for rec in recs:
                assert rec["type"] in valid_types
