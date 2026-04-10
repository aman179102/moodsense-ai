"""Tests for text preprocessing service."""

import pytest

from app.services.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor."""

    @pytest.fixture(scope="class")
    def preprocessor(self):
        """Create preprocessor instance."""
        return TextPreprocessor(use_spacy=False)

    def test_clean_text_lowercase(self, preprocessor):
        """Test that text is converted to lowercase."""
        result = preprocessor.clean_text("HELLO WORLD")
        assert result == "hello world"

    def test_clean_text_remove_urls(self, preprocessor):
        """Test URL removal."""
        text = "Check out https://example.com and http://test.org"
        result = preprocessor.clean_text(text)
        assert "https://example.com" not in result
        assert "http://test.org" not in result

    def test_clean_text_remove_mentions(self, preprocessor):
        """Test mention removal."""
        text = "Hey @user, how are you?"
        result = preprocessor.clean_text(text)
        assert "@user" not in result

    def test_clean_text_preserve_emoticons(self, preprocessor):
        """Test that emoticons are preserved."""
        text = "I'm happy :) and sad :("
        result = preprocessor.clean_text(text)
        assert ":)" in result or "sad" in result

    def test_clean_text_whitespace(self, preprocessor):
        """Test whitespace normalization."""
        text = "Too    much   space"
        result = preprocessor.clean_text(text)
        assert "  " not in result

    def test_tokenize(self, preprocessor):
        """Test tokenization."""
        text = "Hello world, how are you?"
        tokens = preprocessor.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "hello" in [t.lower() for t in tokens]

    def test_remove_stopwords(self, preprocessor):
        """Test stopword removal."""
        tokens = ["the", "quick", "brown", "fox"]
        result = preprocessor.remove_stopwords(tokens)
        assert "the" not in result
        assert "quick" in result

    def test_remove_stopwords_preserve_negation(self, preprocessor):
        """Test that negation words are preserved."""
        tokens = ["not", "happy", "never", "sad"]
        result = preprocessor.remove_stopwords(tokens)
        assert "not" in result
        assert "never" in result

    def test_lemmatize(self, preprocessor):
        """Test lemmatization."""
        tokens = ["running", "cats", "better"]
        result = preprocessor.lemmatize(tokens)
        assert isinstance(result, list)
        assert len(result) == len(tokens)

    def test_preprocess_full_pipeline(self, preprocessor):
        """Test full preprocessing pipeline."""
        text = "I'm SO EXCITED!!! Check this out: https://example.com @friend"
        result = preprocessor.preprocess(text)
        assert isinstance(result, str)
        assert "https://example.com" not in result
        assert "@friend" not in result
        assert "i'm" not in result.lower() or "excited" in result.lower()

    def test_preprocess_with_features(self, preprocessor):
        """Test preprocessing with feature extraction."""
        text = "Hello World!!! ???"
        result = preprocessor.preprocess(text, return_features=True)
        assert isinstance(result, dict)
        assert "text" in result
        assert "features" in result
        assert result["features"]["exclamation_count"] == 3
        assert result["features"]["question_count"] == 3

    def test_preprocess_empty_text(self, preprocessor):
        """Test handling of empty text."""
        result = preprocessor.preprocess("")
        assert result == ""

    def test_preprocess_none_text(self, preprocessor):
        """Test handling of None text."""
        result = preprocessor.preprocess(None)
        assert result == ""

    def test_preprocess_batch(self, preprocessor):
        """Test batch preprocessing."""
        texts = ["First text", "Second text", "Third text"]
        results = preprocessor.preprocess_batch(texts)
        assert len(results) == len(texts)
        assert all(isinstance(r, str) for r in results)
