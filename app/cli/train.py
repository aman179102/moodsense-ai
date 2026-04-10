"""CLI command for training models."""

import argparse
import sys

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.models.trainer import ModelTrainer

logger = get_logger(__name__)


def main():
    """Main training command."""
    parser = argparse.ArgumentParser(
        description="Train MoodSense AI mood classification models"
    )
    parser.add_argument(
        "--model",
        choices=["all", "logistic", "naive_bayes", "lightgbm"],
        default="all",
        help="Model to train",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for models",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    settings = get_settings()
    output_path = args.output or settings.model_path

    logger.info("=" * 50)
    logger.info("MoodSense AI Model Training")
    logger.info("=" * 50)

    # Initialize trainer
    trainer = ModelTrainer(
        test_size=args.test_size,
        use_mlflow=not args.no_mlflow,
    )

    # Train models
    if args.model == "all":
        results = trainer.train_all_models()
    elif args.model == "logistic":
        trainer.prepare_data()
        results = [trainer.train_logistic_regression()]
    elif args.model == "naive_bayes":
        from app.core.constants import SAMPLE_TRAINING_DATA
        texts, labels = zip(*[(t, l) for t, l in SAMPLE_TRAINING_DATA[:60]])
        trainer.prepare_data(texts, labels)
        results = [trainer.train_naive_bayes()]
    elif args.model == "lightgbm":
        trainer.prepare_data()
        results = [trainer.train_lightgbm()]

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("Training Results")
    logger.info("=" * 50)

    for result in results:
        logger.info(f"\nModel: {result.model_name}")
        logger.info(f"  Accuracy: {result.metrics.accuracy:.4f}")
        logger.info(f"  Precision: {result.metrics.precision:.4f}")
        logger.info(f"  Recall: {result.metrics.recall:.4f}")
        logger.info(f"  F1 Score: {result.metrics.f1_score:.4f}")
        logger.info(f"  Best Params: {result.best_params}")

    # Select and save best model
    best = trainer.select_best_model(results)
    logger.info(f"\nBest Model: {best.model_name}")

    save_path = trainer.save_model(best, output_path)
    logger.info(f"Saved best model to: {save_path}")

    logger.info("\nTraining complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
