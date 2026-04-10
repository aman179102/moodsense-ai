"""Recommendation engine for mood-based content suggestions."""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.core.constants import MOOD_RECOMMENDATIONS, MoodLabel
from app.core.logging import get_logger
from app.services.embeddings import EmbeddingService

logger = get_logger(__name__)


class RecommendationEngine:
    """Hybrid recommendation engine combining rule-based and similarity-based methods."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        max_recommendations: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ):
        """Initialize the recommendation engine.

        Args:
            embedding_service: Service for computing text embeddings
            max_recommendations: Maximum number of recommendations to return
            similarity_threshold: Minimum similarity score for recommendations
        """
        self.settings = get_settings()
        self.max_recommendations = max_recommendations or self.settings.max_recommendations
        self.similarity_threshold = similarity_threshold or self.settings.similarity_threshold
        self.embedding_service = embedding_service

        # Mood-specific content database
        self.content_db = self._build_content_database()

    def _build_content_database(self) -> Dict[str, List[Dict]]:
        """Build the content database from constants."""
        return MOOD_RECOMMENDATIONS.copy()

    def get_rule_based_recommendations(
        self, mood: str, count: Optional[int] = None
    ) -> List[Dict]:
        """Get rule-based recommendations for a mood.

        Args:
            mood: Detected mood label
            count: Number of recommendations to return

        Returns:
            List of recommendation items
        """
        count = count or self.max_recommendations

        if mood not in self.content_db:
            logger.warning(f"Unknown mood: {mood}, returning neutral recommendations")
            mood = MoodLabel.NEUTRAL

        recommendations = self.content_db.get(mood, [])

        # Shuffle to provide variety
        shuffled = recommendations.copy()
        random.shuffle(shuffled)

        return shuffled[:count]

    def get_similarity_based_recommendations(
        self,
        text: str,
        mood: str,
        user_history: Optional[List[str]] = None,
        count: Optional[int] = None,
    ) -> List[Dict]:
        """Get similarity-based recommendations.

        Args:
            text: Input text for similarity computation
            mood: Detected mood label
            user_history: Optional list of previous user texts
            count: Number of recommendations to return

        Returns:
            List of recommendation items with similarity scores
        """
        if self.embedding_service is None:
            logger.warning("Embedding service not available, falling back to rule-based")
            return self.get_rule_based_recommendations(mood, count)

        count = count or self.max_recommendations

        # Get all content items for the mood
        mood_content = self.content_db.get(mood, [])

        if not mood_content:
            return []

        # Compute similarities
        similarities = []
        for item in mood_content:
            # Use title + description as the text to compare
            item_text = item.get("title", "") + " " + item.get("description", "")
            item_text += " " + item.get("content", "")

            try:
                similarity = self.embedding_service.compute_similarity(text, item_text)
                similarities.append((item, similarity))
            except Exception as e:
                logger.error(f"Error computing similarity: {e}")
                similarities.append((item, 0.0))

        # Sort by similarity and filter
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold and get top results
        filtered = [
            {**item, "similarity_score": round(sim, 3)}
            for item, sim in similarities
            if sim >= self.similarity_threshold
        ]

        return filtered[:count]

    def get_hybrid_recommendations(
        self,
        text: str,
        mood: str,
        confidence: float,
        user_history: Optional[List[str]] = None,
        count: Optional[int] = None,
    ) -> Dict:
        """Get hybrid recommendations combining both approaches.

        Args:
            text: Input text
            mood: Detected mood label
            confidence: Model confidence score
            user_history: Optional user history
            count: Number of recommendations to return

        Returns:
            Dictionary with recommendations and metadata
        """
        count = count or self.max_recommendations

        # Determine recommendation strategy based on confidence
        if confidence >= 0.8:
            # High confidence: Use similarity-based
            logger.info(f"High confidence ({confidence:.2f}), using similarity-based recommendations")
            recommendations = self.get_similarity_based_recommendations(
                text, mood, user_history, count
            )
            strategy = "similarity-based"
        elif confidence >= 0.5:
            # Medium confidence: Blend both approaches
            logger.info(f"Medium confidence ({confidence:.2f}), using hybrid approach")

            rule_based = self.get_rule_based_recommendations(mood, count // 2 + 1)
            similarity_based = self.get_similarity_based_recommendations(
                text, mood, user_history, count // 2 + 1
            )

            # Merge and deduplicate
            seen_titles = set()
            recommendations = []

            for rec in similarity_based + rule_based:
                title = rec.get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    recommendations.append(rec)

            recommendations = recommendations[:count]
            strategy = "hybrid"
        else:
            # Low confidence: Use rule-based
            logger.info(f"Low confidence ({confidence:.2f}), using rule-based recommendations")
            recommendations = self.get_rule_based_recommendations(mood, count)
            strategy = "rule-based"

        return {
            "recommendations": recommendations,
            "strategy": strategy,
            "count": len(recommendations),
            "mood": mood,
        }

    def get_activity_suggestions(self, mood: str, count: int = 3) -> List[str]:
        """Get activity suggestions based on mood.

        Args:
            mood: Detected mood
            count: Number of suggestions

        Returns:
            List of activity suggestions
        """
        activities_by_mood = {
            MoodLabel.HAPPY: [
                "Share your joy with a friend",
                "Write down 3 things you're grateful for",
                "Do something creative",
                "Go for a walk in nature",
                "Listen to upbeat music and dance",
            ],
            MoodLabel.SAD: [
                "Practice self-compassion meditation",
                "Reach out to someone you trust",
                "Write your feelings in a journal",
                "Take a warm bath or shower",
                "Watch a comforting movie",
            ],
            MoodLabel.ANGRY: [
                "Do intense physical exercise",
                "Practice deep breathing (4-7-8 technique)",
                "Write down your frustrations then tear the paper",
                "Take a cold shower to cool down",
                "Listen to calming music",
            ],
            MoodLabel.ANXIOUS: [
                "Try the 5-4-3-2-1 grounding technique",
                "Practice box breathing",
                "Go for a mindful walk",
                "Listen to binaural beats",
                "Do progressive muscle relaxation",
            ],
            MoodLabel.NEUTRAL: [
                "Try something new today",
                "Practice mindfulness meditation",
                "Read an interesting article",
                "Do light stretching",
                "Organize your space",
            ],
            MoodLabel.EXCITED: [
                "Channel energy into a productive task",
                "Share your excitement with others",
                "Set a goal for the day",
                "Do something adventurous",
                "Journal about your feelings",
            ],
            MoodLabel.BORED: [
                "Learn a new skill online",
                "Try a new recipe",
                "Start a creative project",
                "Explore a new hobby",
                "Call an old friend",
            ],
            MoodLabel.CONFUSED: [
                "Write down what you understand and don't understand",
                "Break the problem into smaller pieces",
                "Ask for help or clarification",
                "Take a break and come back fresh",
                "Research to fill knowledge gaps",
            ],
        }

        activities = activities_by_mood.get(mood, activities_by_mood[MoodLabel.NEUTRAL])
        return random.sample(activities, min(count, len(activities)))

    def explain_recommendation(
        self, text: str, mood: str, confidence: float, strategy: str
    ) -> str:
        """Generate a personalized explanation for the recommendation.

        Args:
            text: Input text
            mood: Detected mood
            confidence: Confidence score
            strategy: Recommendation strategy used

        Returns:
            Explanation string
        """
        mood_descriptions = {
            MoodLabel.HAPPY: "joyful and positive energy",
            MoodLabel.SAD: "need for comfort and self-care",
            MoodLabel.ANGRY: "intense energy that needs healthy outlet",
            MoodLabel.ANXIOUS: "state of worry that needs calming",
            MoodLabel.NEUTRAL: "balanced state open to gentle engagement",
            MoodLabel.EXCITED: "high energy ready for action",
            MoodLabel.BORED: "need for stimulation and novelty",
            MoodLabel.CONFUSED: "need for clarity and understanding",
        }

        explanation_parts = [
            f"Based on your message, I detected {mood_descriptions.get(mood, mood)}."
        ]

        if confidence >= 0.8:
            explanation_parts.append(
                f"I'm quite confident about this ({confidence:.0%})."
            )
        elif confidence >= 0.5:
            explanation_parts.append(
                f"I'm moderately confident ({confidence:.0%})."
            )
        else:
            explanation_parts.append(
                f"I'm less certain ({confidence:.0%}), so I've chosen widely helpful suggestions."
            )

        if strategy == "similarity-based":
            explanation_parts.append(
                "These recommendations are matched to the specific themes in your message."
            )
        elif strategy == "hybrid":
            explanation_parts.append(
                "I've combined mood-appropriate suggestions with personalized matches."
            )
        else:
            explanation_parts.append(
                "These are tried-and-true suggestions for this emotional state."
            )

        return " ".join(explanation_parts)
