
<h1 align="center">
  MoodSense AI
</h1>

<p align="center">
  <strong>AI-powered emotion detection & personalized recommendations</strong>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://scikit-learn.org"><img src="https://img.shields.io/badge/Machine%20Learning-scikit--learn-F7931E?logo=scikit-learn&logoColor=white" alt="Machine Learning"></a>
  <a href="https://huggingface.co"><img src="https://img.shields.io/badge/Hugging%20Face-Ready-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="https://docker.com"><img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

<p align="center">
  <sub>Built by <strong>aman179102</strong></sub>
</p>

---

## Overview

MoodSense AI is a production-grade, full-stack application that detects emotional states from text and provides personalized content recommendations. Using state-of-the-art NLP techniques including TF-IDF, sentence embeddings, and ensemble machine learning models, it accurately classifies text into 8 distinct moods and suggests relevant music, activities, movies, and quotes.

### Key Capabilities

- **Text Analysis**: Understands emotional context from user messages
- **Confidence Scoring**: Provides probability distribution across all mood categories
- **Smart Recommendations**: Delivers personalized content based on detected mood
- **Production API**: RESTful API built with FastAPI for easy integration
- **Interactive UI**: Beautiful Gradio interface for immediate testing
- **Containerized**: Docker support for seamless deployment

---

## Features

| Feature | Description |
|---------|-------------|
| **Mood Detection (NLP)** | Classifies text into 8 moods: happy, sad, angry, anxious, neutral, excited, bored, confused |
| **Confidence Score** | Returns confidence percentage for the predicted mood |
| **Probability Distribution** | Shows likelihood scores for all mood categories |
| **Personalized Recommendations** | Suggests music, activities, movies, and quotes based on detected mood |
| **FastAPI Backend** | High-performance async API with automatic documentation |
| **Gradio UI** | Modern web interface with dark theme for interactive use |
| **Docker Support** | One-command deployment with Docker Compose |
| **Batch Processing** | Analyze multiple texts in a single API call |
| **Monitoring** | Prometheus metrics for production monitoring |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.11+ |
| **Web Framework** | FastAPI, Uvicorn |
| **Machine Learning** | scikit-learn, LightGBM |
| **NLP** | spaCy, NLTK, Sentence Transformers |
| **Deep Learning** | Hugging Face Transformers, PyTorch |
| **UI** | Gradio |
| **MLOps** | MLflow |
| **Testing** | pytest, httpx |
| **Deployment** | Docker, Docker Compose |
| **Monitoring** | Prometheus |

---

## Project Structure

```
moodsense-ai/
├── app/
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # Application factory
│   │   ├── routes.py          # API endpoints
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── dependencies.py    # FastAPI dependencies
│   ├── core/                   # Core utilities
│   │   ├── config.py          # Application settings
│   │   ├── logging.py         # Structured logging
│   │   └── constants.py       # Constants & sample data
│   ├── models/                 # Machine learning models
│   │   ├── trainer.py         # Model training pipeline
│   │   └── predictor.py       # Inference engine
│   ├── services/               # Business logic layer
│   │   ├── preprocessing.py   # Text preprocessing
│   │   ├── embeddings.py      # Feature extraction
│   │   └── recommendation.py  # Recommendation engine
│   ├── ui/                     # User interfaces
│   │   └── gradio_app.py      # Gradio web interface
│   └── cli/                    # Command-line tools
│       ├── train.py           # Training command
│       └── serve.py           # Serving command
├── data/                       # Data directory
├── models/                     # Saved model artifacts
├── notebooks/                  # Jupyter notebooks
│   └── 01_training_pipeline.ipynb
├── tests/                      # Test suite
├── configs/                    # Configuration files
├── Dockerfile                  # Docker build file
├── docker-compose.yml          # Docker Compose configuration
├── pyproject.toml             # Project metadata
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git

### Step-by-Step Setup

**1. Clone the repository**

```bash
git clone https://github.com/aman179102/moodsense-ai.git
cd moodsense-ai
```

**2. Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download NLP models**

```bash
python -m spacy download en_core_web_sm
```

---

## Run Locally

### 1. Train the Model

First, train the mood classification model:

```bash
python train.py
```

This trains three models (Logistic Regression, Naive Bayes, LightGBM) and automatically saves the best performing one to `./models/`.

### 2. Run the API Server

Start the FastAPI backend:

```bash
python serve.py
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics

### 3. Run the Gradio UI

In a new terminal window:

```bash
python gradio_app.py
```

Access the web interface at: http://localhost:7860

---

## API Usage

### Predict Mood

Send a text to analyze and get mood prediction with recommendations:

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I feel sad today",
    "include_recommendations": true,
    "include_explanation": true
  }'
```

**Response:**
```json
{
  "text": "I feel sad today",
  "mood": "sad",
  "confidence": 0.89,
  "all_probabilities": {
    "sad": 0.89,
    "anxious": 0.06,
    "neutral": 0.03,
    "happy": 0.01,
    "angry": 0.01,
    "bored": 0.00,
    "confused": 0.00,
    "excited": 0.00
  },
  "processing_time_ms": 42.15,
  "recommendations": {
    "recommendations": [
      {
        "type": "music",
        "title": "Fix You - Coldplay",
        "url": "https://open.spotify.com/track/..."
      },
      {
        "type": "activity",
        "title": "Practice self-compassion",
        "description": "Take a warm bath and be kind to yourself"
      }
    ],
    "strategy": "hybrid",
    "mood": "sad"
  },
  "explanation": "Based on your message, I detected a need for comfort and self-care. I'm quite confident about this (89%). These recommendations are matched to the specific themes in your message."
}
```

### Batch Prediction

Analyze multiple texts at once:

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I am so happy today!",
      "I am feeling really down",
      "This is so frustrating!"
    ]
  }'
```

### Health Check

Verify the service is running:

```bash
curl http://localhost:8000/health
```

### Get Available Moods

List all supported mood categories:

```bash
curl http://localhost:8000/moods
```

---

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py

# Run tests matching a pattern
pytest -k "test_embedding"
```

---

## Docker Setup

### Quick Start with Docker Compose

Build and run the entire stack:

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Manual Docker Build

```bash
# Build image
docker build -t moodsense-ai:latest .

# Run API container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  moodsense-ai:latest

# Run Gradio container
docker run -p 7860:7860 \
  -v $(pwd)/models:/app/models:ro \
  moodsense-ai:latest \
  python gradio_app.py --host 0.0.0.0
```

---

## Deployment Guide

### Hugging Face Spaces

Deploy the Gradio UI to Hugging Face Spaces for free hosting:

**1. Create a new Space**
- Go to [huggingface.co/spaces](https://huggingface.co/spaces)
- Click "Create new Space"
- Select "Gradio" as the SDK
- Choose a name (e.g., `moodsense-ai`)

**2. Upload files**
Create these files in your Space:

`app.py`:
```python
from app.ui.gradio_app import create_gradio_app
from app.models.predictor import MoodPredictor
from app.services.recommendation import RecommendationEngine
from app.services.embeddings import EmbeddingService

# Initialize services
predictor = MoodPredictor()
embedding_service = EmbeddingService()
embedding_service.load("./models")
engine = RecommendationEngine(embedding_service=embedding_service)

# Create and launch app
app = create_gradio_app(predictor=predictor, engine=engine)
app.launch()
```

`requirements.txt`: Copy from your project

**3. Upload model files**
- Create `models/` directory in your Space
- Upload the trained model file (`mood_classifier_v2.pkl`)
- Upload `tfidf_vectorizer.pkl`

**4. Deploy**
The Space will automatically build and deploy your app.

### Kaggle

Run the training notebook on Kaggle:

```python
# In a Kaggle notebook cell
!git clone https://github.com/aman179102/moodsense-ai.git
%cd moodsense-ai
!pip install -q -r requirements.txt
!python -m spacy download en_core_web_sm

# Run training
!python train.py --no-mlflow

# Test prediction
from app.models.predictor import MoodPredictor
predictor = MoodPredictor()
result = predictor.predict("I am so happy today!")
print(f"Mood: {result.mood}, Confidence: {result.confidence:.2%}")
```

### Local Production

Run FastAPI with production settings:

```bash
# Using uvicorn directly
uvicorn app.api.main:create_app --host 0.0.0.0 --port 8000 --workers 4

# Or using the CLI
python serve.py --host 0.0.0.0 --port 8000 --workers 4
```

---

## Screenshots

*Screenshots will be added here showing:*

- Gradio UI with mood prediction
 <img width="1680" height="1050" alt="2026-04-11-003406_1680x1050_scrot" src="https://github.com/user-attachments/assets/d6876032-7cfa-4532-802a-563e258eabd7" />
 
- API documentation page
 <img width="1680" height="1050" alt="2026-04-11-003126_1680x1050_scrot" src="https://github.com/user-attachments/assets/8f5d1c05-cd44-4417-8ab4-28d1b4c17183" />

- Recommendation results

  
- Probability distribution chart

  

---

## Future Improvements

- [ ] **Spotify API Integration**: Direct music playback links
- [ ] **YouTube API**: Video recommendations for each mood
- [ ] **User Authentication**: Save user history and preferences
- [ ] **Larger Dataset**: Train on more diverse, real-world data
- [ ] **Transformer Models**: Experiment with BERT/RoBERTa for better accuracy
- [ ] **Multi-language Support**: Detect moods in other languages
- [ ] **Voice Input**: Analyze tone and speech patterns
- [ ] **Mobile App**: React Native app for on-the-go use

---

## How to Replace Models

### Using Your Own Dataset

1. Prepare your data in CSV format:
```csv
text,mood
"I am so happy today!",happy
"I feel really down",sad
...
```

2. Update `app/core/constants.py`:
```python
SAMPLE_TRAINING_DATA = [
    # Replace with your data
    ("your text here", "mood_label"),
    ...
]
```

3. Retrain:
```bash
python train.py
```

### Using Pre-trained Models from Hugging Face

Replace the custom ML models with Hugging Face transformers:

```python
# In app/models/predictor.py
from transformers import pipeline

class MoodPredictor:
    def __init__(self):
        # Use a pre-trained sentiment model
        self.classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
```

### Using OpenAI API

Replace local models with OpenAI GPT:

```python
import openai

class MoodPredictor:
    def predict(self, text: str):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "Classify the mood of this text as: happy, sad, angry, anxious, neutral, excited, bored, or confused. Respond with just the mood."
            }, {
                "role": "user",
                "content": text
            }]
        )
        mood = response.choices[0].message.content.strip()
        return MoodPrediction(...)
```

---

## Author

**aman179102**

- GitHub: [@aman179102](https://github.com/aman179102)
- Project: [MoodSense AI](https://github.com/aman179102/moodsense-ai)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co) for transformer models and hosting
- [Sentence Transformers](https://www.sbert.net) for embeddings
- [FastAPI](https://fastapi.tiangolo.com) for the web framework
- [Gradio](https://gradio.app) for the UI components

---

<p align="center">
  Built with dedication by <strong>aman179102</strong>
</p>
