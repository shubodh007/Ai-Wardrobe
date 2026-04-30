"""Utility script to retrain the classifier with configurable tuning options."""

import argparse
import pickle
from pathlib import Path

import numpy as np

from train_classifier import ClothingClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain AI Wardrobe classifier")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=192, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=8e-4, help="Initial learning rate")
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable train-time augmentation for faster runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    processed_path = project_root / "data" / "processed" / "fashion_mnist_processed.pkl"

    if not processed_path.exists():
        print(f"Processed data not found at {processed_path}")
        print("Run src/data_preparation.py first.")
        return

    print(f"Loading processed data from {processed_path}")
    with processed_path.open("rb") as handle:
        data = pickle.load(handle)

    classifier = ClothingClassifier()
    classifier.build_cnn_model()
    classifier.compile_model(learning_rate=float(args.learning_rate))

    classifier.train(
        train_data=(np.asarray(data["x_train"], dtype=np.float32), np.asarray(data["y_train"], dtype=np.int64)),
        val_data=(np.asarray(data["x_val"], dtype=np.float32), np.asarray(data["y_val"], dtype=np.int64)),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        use_augmentation=not bool(args.no_augmentation),
    )
    _, test_accuracy = classifier.evaluate(
        (np.asarray(data["x_test"], dtype=np.float32), np.asarray(data["y_test"], dtype=np.int64))
    )
    classifier.plot_training_history()
    classifier.save_model()

    print(f"Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
