"""Modern Gradio UI for MoodSense AI."""

import os
from typing import Optional, Tuple

import gradio as gr
import pandas as pd

from app.core.config import get_settings
from app.core.constants import MOOD_COLORS, MOOD_DESCRIPTIONS, MOOD_RECOMMENDATIONS
from app.core.logging import get_logger
from app.models.predictor import MoodPredictor
from app.services.recommendation import RecommendationEngine

logger = get_logger(__name__)

# Custom CSS for dark modern theme
CUSTOM_CSS = """
/* MoodSense AI - Built by aman179102 */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #6366f1;
    --secondary-color: #8b5cf6;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-color: #334155;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
}

body {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-dark) !important;
}

.gradio-container {
    background: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    border-radius: 16px;
    margin-bottom: 1.5rem;
}

.main-header h1 {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

.main-header p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1rem !important;
    margin-top: 0.5rem !important;
}

.input-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}

.result-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin-top: 1rem !important;
}

.mood-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 1.1rem;
    text-transform: capitalize;
    color: white;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

.confidence-bar {
    height: 8px;
    background: var(--border-color);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 0.5rem;
}

.confidence-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

.recommendation-item {
    background: rgba(99, 102, 241, 0.1);
    border-left: 3px solid var(--primary-color);
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
}

.recommendation-item.music {
    border-left-color: #ec4899;
    background: rgba(236, 72, 153, 0.1);
}

.recommendation-item.activity {
    border-left-color: #10b981;
    background: rgba(16, 185, 129, 0.1);
}

.recommendation-item.movie {
    border-left-color: #f59e0b;
    background: rgba(245, 158, 11, 0.1);
}

.recommendation-item.quote {
    border-left-color: #6366f1;
    background: rgba(99, 102, 241, 0.1);
    font-style: italic;
}

button.primary {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3) !important;
}

.textarea-input textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
}

.textarea-input textarea:focus {
    border-color: var(--primary-color) !important;
    outline: none !important;
}

.probabilities-table {
    width: 100%;
    border-collapse: collapse;
}

.probabilities-table th,
.probabilities-table td {
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

.probabilities-table th {
    color: var(--text-secondary);
    font-weight: 500;
}

.footer {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
    font-size: 0.875rem;
}

.footer a {
    color: var(--primary-color);
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}
"""


def create_gradio_app(
    predictor: Optional[MoodPredictor] = None,
    engine: Optional[RecommendationEngine] = None,
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
) -> gr.Blocks:
    """Create the Gradio web interface.

    Args:
        predictor: MoodPredictor instance
        engine: RecommendationEngine instance
        share: Whether to create a public shareable link
        server_name: Server host
        server_port: Server port

    Returns:
        Gradio Blocks interface
    """
    # Initialize services if not provided
    if predictor is None:
        predictor = MoodPredictor()
    if engine is None:
        from app.services.embeddings import EmbeddingService
        embedding_service = EmbeddingService()
        settings = get_settings()
        if os.path.exists(settings.model_path):
            embedding_service.load(settings.model_path)
        engine = RecommendationEngine(embedding_service=embedding_service)

    def predict_mood(text: str) -> Tuple:
        """Process text and return all outputs."""
        if not text or not text.strip():
            return (
                "Please enter some text to analyze.",
                None,
                None,
                None,
                None,
            )

        if not predictor.is_ready():
            return (
                "⚠️ Model not loaded. Please train a model first using `python -m app.cli.train`",
                None,
                None,
                None,
                None,
            )

        try:
            # Get prediction
            prediction = predictor.predict(text, return_all_probabilities=True)

            # Get recommendations
            rec_result = engine.get_hybrid_recommendations(
                text=text,
                mood=prediction.mood,
                confidence=prediction.confidence,
            )

            # Get explanation
            explanation = engine.explain_recommendation(
                text=text,
                mood=prediction.mood,
                confidence=prediction.confidence,
                strategy=rec_result.get("strategy", "rule-based"),
            )

            # Build mood display
            mood_color = MOOD_COLORS.get(prediction.mood, "#6366f1")
            mood_description = MOOD_DESCRIPTIONS.get(prediction.mood, prediction.mood)
            mood_html = f"""
            <div style="text-align: center; padding: 1rem;">
                <div class="mood-badge" style="background: {mood_color};">
                    {prediction.mood.upper()}
                </div>
                <p style="color: var(--text-secondary); margin-top: 0.5rem;">
                    {mood_description}
                </p>
            </div>
            """

            # Build confidence display
            confidence_percent = int(prediction.confidence * 100)
            confidence_color = "#10b981" if confidence_percent >= 80 else "#f59e0b" if confidence_percent >= 50 else "#ef4444"
            confidence_html = f"""
            <div style="padding: 1rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: var(--text-secondary);">Confidence</span>
                    <span style="color: {confidence_color}; font-weight: 600;">{confidence_percent}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_percent}%; background: {confidence_color};"></div>
                </div>
            </div>
            """

            # Build probabilities table
            probs_df = pd.DataFrame([
                {"Mood": mood.replace("_", " ").title(), "Probability": f"{prob:.1%}"}
                for mood, prob in sorted(
                    prediction.all_probabilities.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ])

            # Build recommendations
            recs = rec_result.get("recommendations", [])
            recs_html = "<div style='margin-top: 1rem;'>"
            for rec in recs:
                rec_type = rec.get("type", "general")
                title = rec.get("title", "")
                desc = rec.get("description", rec.get("content", ""))
                url = rec.get("url", "")

                if url:
                    title_link = f'<a href="{url}" target="_blank" style="color: inherit; text-decoration: none; font-weight: 600;">{title}</a>'
                else:
                    title_link = f'<strong>{title}</strong>' if title else ""

                recs_html += f"""
                <div class="recommendation-item {rec_type}">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="text-transform: uppercase; font-size: 0.75rem; color: var(--text-secondary);">{rec_type}</span>
                    </div>
                    {f'<div style="margin-top: 0.25rem;">{title_link}</div>' if title else ''}
                    {f'<div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.25rem;">{desc}</div>' if desc else ''}
                </div>
                """
            recs_html += "</div>"

            # Build explanation
            explanation_html = f"""
            <div style="background: rgba(99, 102, 241, 0.05); border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    {explanation}
                </p>
            </div>
            """

            return (
                mood_html,
                confidence_html,
                probs_df,
                recs_html,
                explanation_html,
            )

        except Exception as e:
            logger.error(f"Prediction error in Gradio: {e}")
            return (
                f"❌ Error: {str(e)}",
                None,
                None,
                None,
                None,
            )

    def analyze_mood_details(mood: str) -> str:
        """Get detailed info about a mood."""
        if not mood:
            return "Select a mood to see details."

        recs = MOOD_RECOMMENDATIONS.get(mood, [])
        activities = engine.get_activity_suggestions(mood, 3)

        html = f"""
        <div style="padding: 1rem;">
            <h3 style="color: var(--text-primary); margin-bottom: 1rem;">
                {mood.replace('_', ' ').title()}
            </h3>
            <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                {MOOD_DESCRIPTIONS.get(mood, '')}
            </p>
            <h4 style="color: var(--text-primary); margin: 1rem 0 0.5rem;">Suggested Activities</h4>
            <ul style="color: var(--text-secondary); padding-left: 1.5rem;">
                {''.join(f'<li style="margin: 0.25rem 0;">{act}</li>' for act in activities)}
            </ul>
        </div>
        """
        return html

    # Create interface
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="MoodSense AI",
        theme=gr.themes.Soft(),
    ) as app:
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>MoodSense AI</h1>
            <p>AI-powered emotion detection & personalized recommendations</p>
        </div>
        """)

        with gr.Row():
            # Input section
            with gr.Column(scale=1):
                gr.Markdown("## ✍️ Your Message")
                input_text = gr.TextArea(
                    label="",
                    placeholder="How are you feeling today? Describe your emotions, thoughts, or current state of mind...",
                    lines=5,
                    elem_classes=["textarea-input"],
                )

                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 Analyze Mood",
                        variant="primary",
                        elem_classes=["primary"],
                    )
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")

                # Example inputs
                gr.Markdown("### 📝 Try these examples:")
                examples = gr.Examples(
                    examples=[
                        ["I just got promoted at work! I'm so excited and happy!"],
                        ["I can't sleep, my mind is racing with worries about tomorrow."],
                        ["Why does everything have to be so difficult? I'm so frustrated!"],
                        ["Just another regular day, nothing special happening."],
                        ["I don't understand what's going on. Everything is confusing."],
                    ],
                    inputs=[input_text],
                )

            # Results section
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Analysis Results")

                with gr.Column(elem_classes=["result-card"]):
                    mood_output = gr.HTML(label="Detected Mood")
                    confidence_output = gr.HTML(label="Confidence")

                with gr.Column(elem_classes=["result-card"]):
                    gr.Markdown("### 📈 All Probabilities")
                    probs_output = gr.DataFrame(
                        label="",
                        headers=["Mood", "Probability"],
                    )

        # Recommendations section
        with gr.Row():
            with gr.Column(elem_classes=["result-card"]):
                gr.Markdown("## 🎯 Personalized Recommendations")
                recs_output = gr.HTML()

        # Explanation section
        with gr.Row():
            with gr.Column(elem_classes=["result-card"]):
                explanation_output = gr.HTML()

        # Mood explorer
        with gr.Row():
            with gr.Column(elem_classes=["result-card"]):
                gr.Markdown("## 🔍 Explore Moods")
                mood_selector = gr.Dropdown(
                    choices=list(MOOD_DESCRIPTIONS.keys()),
                    label="Select a mood to learn more",
                    value="happy",
                )
                mood_details = gr.HTML()

        # Footer
        gr.HTML("""
        <div class="footer">
            <p>Built by aman179102</p>
            <p><a href="/docs" target="_blank">API Documentation</a> | 
               <a href="https://github.com" target="_blank">GitHub</a></p>
        </div>
        """)

        # Event handlers
        analyze_btn.click(
            fn=predict_mood,
            inputs=[input_text],
            outputs=[
                mood_output,
                confidence_output,
                probs_output,
                recs_output,
                explanation_output,
            ],
        )

        clear_btn.click(
            fn=lambda: ("", None, None, None, None),
            inputs=[],
            outputs=[
                mood_output,
                confidence_output,
                probs_output,
                recs_output,
                explanation_output,
            ],
        )

        mood_selector.change(
            fn=analyze_mood_details,
            inputs=[mood_selector],
            outputs=[mood_details],
        )

        # Initialize mood details
        mood_details.value = analyze_mood_details("happy")

    return app


def launch_gradio(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
) -> None:
    """Launch the Gradio interface.

    Args:
        share: Whether to create a public shareable link
        server_name: Server host
        server_port: Server port
    """
    app = create_gradio_app(share=share, server_name=server_name, server_port=server_port)
    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )
