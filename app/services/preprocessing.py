"""Text preprocessing service for NLP pipeline."""

import re
import string
from typing import List, Optional

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """Advanced text preprocessing pipeline for mood detection."""

    def __init__(self, use_spacy: bool = True, use_lemmatization: bool = True):
        """Initialize the text preprocessor.

        Args:
            use_spacy: Whether to use spaCy for advanced NLP
            use_lemmatization: Whether to apply lemmatization
        """
        self.settings = get_settings()
        self.use_spacy = use_spacy
        self.use_lemmatization = use_lemmatization
        self.nlp = None
        self.lemmatizer = None
        self.stop_words = set()

        self._initialize_nlp()

    def _initialize_nlp(self) -> None:
        """Initialize NLP components."""
        try:
            # Download required NLTK data
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)

            self.stop_words = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()

            if self.use_spacy:
                try:
                    self.nlp = spacy.load(self.settings.spacy_model)
                    logger.info(f"Loaded spaCy model: {self.settings.spacy_model}")
                except OSError:
                    logger.warning(
                        f"spaCy model {self.settings.spacy_model} not found. "
                        "Running without spaCy features."
                    )
                    self.use_spacy = False

        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
            self.use_spacy = False

    def clean_text(self, text: str) -> str:
        """Basic text cleaning.

        Args:
            text: Raw input text

        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove user mentions and hashtags (keep the text after #)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep emoticons
        text = self._preserve_emoticons(text)

        return text.strip()

    def _preserve_emoticons(self, text: str) -> str:
        """Preserve emoticons while cleaning other special characters.

        Args:
            text: Input text

        Returns:
            Text with emoticons preserved
        """
        # Common emoticon patterns
        emoticons = [
            r":\)", r":-\)", r":\(", r":-\(", r":D", r":-D", r":P", r":-P",
            r";\)", r";-\)", r":/", r":-\\", r":\|", r":-\|", r":\*", r":-\*",
            r":\)", r"<3", r"</3", r":\'\)", r":\'\(", r":O", r":-O", r":o", r":-o",
        ]

        # Replace emoticons with placeholders
        emoticon_map = {}
        for i, pattern in enumerate(emoticons):
            matches = re.findall(pattern, text)
            for j, match in enumerate(matches):
                placeholder = f" EMOTICON_{i}_{j} "
                emoticon_map[placeholder] = match
                text = text.replace(match, placeholder, 1)

        # Remove punctuation except emoticon placeholders
        text = text.translate(
            str.maketrans("", "", string.punctuation.replace("_", ""))
        )

        # Restore emoticons
        for placeholder, emoticon in emoticon_map.items():
            text = text.replace(placeholder.strip(), emoticon)

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text:
            return []

        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list.

        Args:
            tokens: List of tokens

        Returns:
            Filtered tokens
        """
        if not tokens:
            return []

        # Keep negation words as they affect sentiment
        negation_words = {
            "no", "not", "nor", "neither", "never", "nobody",
            "nothing", "nowhere", "hardly", "scarcely", "barely",
            "don", "doesn", "didn", "wasn", "weren", "won",
            "wouldn", "shouldn", "isn", "aren", "hasn", "haven",
        }

        filtered_stopwords = self.stop_words - negation_words

        return [
            token for token in tokens
            if token.lower() not in filtered_stopwords or len(token) <= 2
        ]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens.

        Args:
            tokens: List of tokens

        Returns:
            Lemmatized tokens
        """
        if not tokens or not self.use_lemmatization:
            return tokens

        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def extract_features(self, text: str) -> dict:
        """Extract linguistic features from text.

        Args:
            text: Input text

        Returns:
            Dictionary of features
        """
        features = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "emoji_count": len(re.findall(r"[\U0001F600-\U0001F64F]", text)),
        }

        if self.nlp and self.use_spacy:
            doc = self.nlp(text)
            features.update({
                "sentences": len(list(doc.sents)),
                "entities": len(doc.ents),
                "noun_chunks": len(list(doc.noun_chunks)),
            })

        return features

    def preprocess(self, text: str, return_features: bool = False) -> str | dict:
        """Full preprocessing pipeline.

        Args:
            text: Raw input text
            return_features: Whether to return features dict

        Returns:
            Preprocessed text or dict with text and features
        """
        if not text or not isinstance(text, str):
            if return_features:
                return {"text": "", "features": {}}
            return ""

        # Clean text
        cleaned = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(cleaned)

        # Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Lemmatize
        tokens = self.lemmatize(tokens)

        # Rejoin tokens
        processed_text = " ".join(tokens)

        if return_features:
            features = self.extract_features(text)
            return {"text": processed_text, "features": features}

        return processed_text

    def preprocess_batch(
        self, texts: List[str], return_features: bool = False
    ) -> List[str] | List[dict]:
        """Batch preprocessing.

        Args:
            texts: List of raw texts
            return_features: Whether to return features

        Returns:
            List of preprocessed texts or dicts
        """
        results = []
        for text in texts:
            result = self.preprocess(text, return_features=return_features)
            results.append(result)
        return results
