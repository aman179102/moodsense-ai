"""CLI command for serving the application."""

import argparse
import sys

import uvicorn

from app.core.config import get_settings
from app.core.logging import setup_logging


def main():
    """Main serving command."""
    parser = argparse.ArgumentParser(
        description="Serve MoodSense AI API"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    settings = get_settings()

    host = args.host or settings.api_host
    port = args.port or settings.api_port
    workers = 1 if args.reload else (args.workers or settings.api_workers)

    print(f"Starting MoodSense AI")
    print(f"Server: http://{host}:{port}")
    print(f"API Docs: http://{host}:{port}/docs")
    print(f"Metrics: http://{host}:{port}/metrics")
    print("=" * 50)

    uvicorn.run(
        "app.api.main:create_app",
        host=host,
        port=port,
        workers=workers,
        reload=args.reload,
        log_level=args.log_level,
        factory=True,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
