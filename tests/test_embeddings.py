"""Tests for embedding service."""

import os
import tempfile

import numpy as np
import pytest

from app.services.embeddings import EmbeddingService


class TestEmbeddingService:
    """Test cases for EmbeddingService."""

    @pytest.fixture(scope="class")
    def embedding_service(self):
        """Create embedding service instance."""
        return EmbeddingService(
            embedding_model="all-MiniLM-L6-v2",
            use_tfidf=True,
            use_sentence_embeddings=True,
        )

    @pytest.fixture(scope="class")
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "I am happy today",
            "This is a test sentence",
            "Machine learning is fascinating",
            "Natural language processing",
            "Deep learning models",
        ]

    def test_initialization(self, embedding_service):
        """Test service initialization."""
        assert embedding_service.tfidf_vectorizer is not None
        assert embedding_service.sentence_transformer is not None

    def test_fit_tfidf(self, embedding_service, sample_texts):
        """Test TF-IDF fitting."""
        embedding_service.fit_tfidf(sample_texts)
        assert hasattr(embedding_service.tfidf_vectorizer, "vocabulary_")
        assert len(embedding_service.tfidf_vectorizer.vocabulary_) > 0

    def test_get_tfidf_embeddings(self, embedding_service, sample_texts):
        """Test TF-IDF embedding generation."""
        embedding_service.fit_tfidf(sample_texts)
        embeddings = embedding_service.get_tfidf_embeddings(sample_texts[:2])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0

    def test_get_sentence_embeddings(self, embedding_service, sample_texts):
        """Test sentence embedding generation."""
        embeddings = embedding_service.get_sentence_embeddings(sample_texts[:2])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384  # MiniLM dimension

    def test_get_combined_embeddings(self, embedding_service, sample_texts):
        """Test combined embedding generation."""
        embedding_service.fit_tfidf(sample_texts)
        embeddings = embedding_service.get_combined_embeddings(sample_texts[:2])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        # Combined dim = TF-IDF (5000) + sentence (384)
        assert embeddings.shape[1] > 384

    def test_get_embeddings_method_param(self, embedding_service, sample_texts):
        """Test different embedding methods."""
        embedding_service.fit_tfidf(sample_texts)

        # TF-IDF only
        tfidf_emb = embedding_service.get_embeddings(sample_texts[:2], method="tfidf")
        assert tfidf_emb.shape[0] == 2

        # Sentence only
        sent_emb = embedding_service.get_embeddings(sample_texts[:2], method="sentence")
        assert sent_emb.shape == (2, 384)

        # Combined
        comb_emb = embedding_service.get_embeddings(sample_texts[:2], method="combined")
        assert comb_emb.shape[0] == 2
        assert comb_emb.shape[1] > sent_emb.shape[1]

    def test_compute_similarity_cosine(self, embedding_service):
        """Test cosine similarity computation."""
        text1 = "I love machine learning"
        text2 = "Machine learning is great"
        similarity = embedding_service.compute_similarity(text1, text2, method="cosine")
        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Similar topics

    def test_compute_similarity_euclidean(self, embedding_service):
        """Test euclidean similarity computation."""
        text1 = "I love machine learning"
        text2 = "Machine learning is great"
        similarity = embedding_service.compute_similarity(text1, text2, method="euclidean")
        assert 0 <= similarity <= 1

    def test_embedding_dimension(self, embedding_service, sample_texts):
        """Test embedding dimension property."""
        embedding_service.fit_tfidf(sample_texts)
        dim = embedding_service.embedding_dimension
        assert dim > 0
        assert dim > 384  # Should include both TF-IDF and sentence

    def test_save_and_load(self, embedding_service, sample_texts):
        """Test saving and loading embedding service."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Fit and save
            embedding_service.fit_tfidf(sample_texts)
            embedding_service.save(tmpdir)

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "tfidf_vectorizer.pkl"))

            # Load into new service
            new_service = EmbeddingService(use_tfidf=True, use_sentence_embeddings=True)
            new_service.load(tmpdir)

            assert hasattr(new_service.tfidf_vectorizer, "vocabulary_")

    def test_unfitted_tfidf_error(self, embedding_service):
        """Test error when TF-IDF not fitted."""
        with pytest.raises(ValueError, match="not fitted"):
            embedding_service.get_tfidf_embeddings(["test text"])

    def test_invalid_method_error(self, embedding_service):
        """Test error for invalid embedding method."""
        with pytest.raises(ValueError, match="Unknown embedding method"):
            embedding_service.get_embeddings(["test"], method="invalid")
