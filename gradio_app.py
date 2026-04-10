#!/usr/bin/env python3
"""Launch Gradio UI."""

import argparse

from app.ui.gradio_app import launch_gradio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch MoodSense AI Gradio UI")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    launch_gradio(share=args.share, server_name=args.host, server_port=args.port)
