"""Embedding service for feature extraction."""

import os
import pickle
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using multiple methods."""

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        use_tfidf: bool = True,
        use_sentence_embeddings: bool = True,
    ):
        """Initialize the embedding service.

        Args:
            embedding_model: Name of the sentence transformer model
            use_tfidf: Whether to use TF-IDF features
            use_sentence_embeddings: Whether to use sentence embeddings
        """
        self.settings = get_settings()
        self.embedding_model_name = embedding_model or self.settings.embedding_model
        self.use_tfidf = use_tfidf
        self.use_sentence_embeddings = use_sentence_embeddings

        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.sentence_transformer: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None

        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize embedding models."""
        try:
            if self.use_tfidf:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    stop_words="english",
                )
                logger.info("Initialized TF-IDF vectorizer")

            if self.use_sentence_embeddings:
                device = "cuda" if self.settings.use_gpu else "cpu"
                self.sentence_transformer = SentenceTransformer(
                    self.embedding_model_name,
                    device=device,
                )
                self._embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
                logger.info(
                    f"Loaded sentence transformer: {self.embedding_model_name} "
                    f"(dim={self._embedding_dim})"
                )

        except Exception as e:
            logger.error(f"Error initializing embedding models: {e}")
            raise

    def fit_tfidf(self, texts: List[str]) -> None:
        """Fit TF-IDF vectorizer on training texts.

        Args:
            texts: List of training texts
        """
        if self.tfidf_vectorizer is None:
            return

        try:
            self.tfidf_vectorizer.fit(texts)
            logger.info(
                f"Fitted TF-IDF vectorizer with {len(self.tfidf_vectorizer.vocabulary_)} features"
            )
        except Exception as e:
            logger.error(f"Error fitting TF-IDF: {e}")
            raise

    def get_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings.

        Args:
            texts: List of texts

        Returns:
            TF-IDF feature matrix
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not initialized")

        if not hasattr(self.tfidf_vectorizer, "vocabulary_"):
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")

        return self.tfidf_vectorizer.transform(texts).toarray()

    def get_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate sentence embeddings.

        Args:
            texts: List of texts

        Returns:
            Sentence embedding matrix
        """
        if self.sentence_transformer is None:
            raise ValueError("Sentence transformer not initialized")

        return self.sentence_transformer.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def get_combined_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate combined TF-IDF + sentence embeddings.

        Args:
            texts: List of texts

        Returns:
            Combined feature matrix
        """
        embeddings_list = []

        if self.use_tfidf and hasattr(self.tfidf_vectorizer, "vocabulary_"):
            tfidf_emb = self.get_tfidf_embeddings(texts)
            embeddings_list.append(tfidf_emb)

        if self.use_sentence_embeddings:
            sentence_emb = self.get_sentence_embeddings(texts)
            embeddings_list.append(sentence_emb)

        if not embeddings_list:
            raise ValueError("No embedding methods enabled")

        return np.hstack(embeddings_list)

    def get_embeddings(
        self,
        texts: List[str],
        method: str = "combined",
    ) -> np.ndarray:
        """Get embeddings using specified method.

        Args:
            texts: List of texts
            method: Embedding method ('tfidf', 'sentence', or 'combined')

        Returns:
            Feature matrix
        """
        if method == "tfidf":
            return self.get_tfidf_embeddings(texts)
        elif method == "sentence":
            return self.get_sentence_embeddings(texts)
        elif method == "combined":
            return self.get_combined_embeddings(texts)
        else:
            raise ValueError(f"Unknown embedding method: {method}")

    def compute_similarity(
        self, text1: str, text2: str, method: str = "cosine"
    ) -> float:
        """Compute similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            method: Similarity metric ('cosine' or 'euclidean')

        Returns:
            Similarity score
        """
        emb1 = self.get_sentence_embeddings([text1])[0]
        emb2 = self.get_sentence_embeddings([text2])[0]

        if method == "cosine":
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        elif method == "euclidean":
            return 1.0 / (1.0 + np.linalg.norm(emb1 - emb2))
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def save(self, path: str) -> None:
        """Save the embedding service state.

        Args:
            path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)

        # Save TF-IDF vectorizer
        if self.tfidf_vectorizer is not None:
            tfidf_path = os.path.join(path, "tfidf_vectorizer.pkl")
            with open(tfidf_path, "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
            logger.info(f"Saved TF-IDF vectorizer to {tfidf_path}")

        # Note: Sentence transformer model is not saved, just its name
        # It will be reloaded from HuggingFace on next init

    def load(self, path: str) -> None:
        """Load the embedding service state.

        Args:
            path: Directory path to load from
        """
        # Load TF-IDF vectorizer
        tfidf_path = os.path.join(path, "tfidf_vectorizer.pkl")
        if os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            logger.info(f"Loaded TF-IDF vectorizer from {tfidf_path}")

        # Sentence transformer is reinitialized in __init__

    @property
    def embedding_dimension(self) -> int:
        """Get the total embedding dimension."""
        dim = 0
        if self.use_tfidf and self.tfidf_vectorizer is not None:
            dim += self.tfidf_vectorizer.max_features
        if self.use_sentence_embeddings and self._embedding_dim is not None:
            dim += self._embedding_dim
        return dim
