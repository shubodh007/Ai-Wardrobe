"""Data preparation pipeline for AI Wardrobe."""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST


class DataPreparator:
    """Prepare, augment, and persist Fashion-MNIST data splits."""

    def __init__(self) -> None:
        self.categories: List[str] = [
            "T-shirt",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
        self.project_root = Path(__file__).resolve().parent.parent
        self.processed_dir = self.project_root / "data" / "processed"
        self.raw_dir = self.project_root / "data" / "raw"

        # Real-world inspired garment colors and neutral backgrounds for synthetic RGB generation.
        self.garment_palette = np.array(
            [
                [24, 27, 33],
                [236, 236, 232],
                [34, 70, 140],
                [163, 42, 46],
                [44, 112, 64],
                [133, 89, 52],
                [94, 62, 128],
                [221, 161, 56],
                [201, 96, 140],
                [73, 78, 86],
                [58, 132, 150],
                [180, 182, 178],
            ],
            dtype=np.float32,
        )
        self.background_palette = np.array(
            [
                [241, 240, 235],
                [229, 232, 236],
                [218, 223, 228],
                [246, 243, 239],
                [222, 218, 213],
            ],
            dtype=np.float32,
        )
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)

    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load Fashion-MNIST via torchvision datasets."""
        print("📥 Loading Fashion-MNIST dataset...")
        train_dataset = FashionMNIST(root=str(self.raw_dir), train=True, download=True)
        test_dataset = FashionMNIST(root=str(self.raw_dir), train=False, download=True)

        x_train = train_dataset.data.numpy()
        y_train = train_dataset.targets.numpy()
        x_test = test_dataset.data.numpy()
        y_test = test_dataset.targets.numpy()

        print(f"✅ Loaded {x_train.shape[0]} training and {x_test.shape[0]} test images")
        return (x_train, y_train), (x_test, y_test)

    def colorize_to_rgb(
        self,
        images: np.ndarray,
        seed: int,
        training: bool,
    ) -> np.ndarray:
        """Convert grayscale Fashion-MNIST silhouettes into realistic RGB garment images."""
        grayscale = np.asarray(images, dtype=np.float32)

        if grayscale.ndim == 4 and grayscale.shape[-1] == 1:
            grayscale = grayscale[:, :, :, 0]
        if grayscale.ndim != 3:
            raise ValueError("Expected grayscale images with shape (N, H, W) or (N, H, W, 1).")

        if np.max(grayscale) > 1.0:
            grayscale = grayscale / 255.0

        print(
            "🎨 Colorizing grayscale silhouettes into RGB "
            f"({'train' if training else 'eval'}) split..."
        )
        rng = np.random.default_rng(seed)
        sample_count, height, width = grayscale.shape
        rgb_images = np.empty((sample_count, height, width, 3), dtype=np.float32)

        for index in range(sample_count):
            image = np.clip(grayscale[index], 0.0, 1.0)
            garment_mask = image > 0.06
            if not np.any(garment_mask):
                garment_mask = image > 0.02

            garment_color = self.garment_palette[int(rng.integers(0, len(self.garment_palette)))].copy()
            background_color = self.background_palette[int(rng.integers(0, len(self.background_palette)))].copy()

            if training:
                garment_color *= float(rng.uniform(0.88, 1.15))
                background_color *= float(rng.uniform(0.92, 1.08))

            garment_color = np.clip(garment_color, 0.0, 255.0)
            background_color = np.clip(background_color, 0.0, 255.0)

            brightness = 0.22 + (0.78 * image)
            garment_rgb = brightness[:, :, None] * garment_color[None, None, :]

            composed = np.broadcast_to(background_color[None, None, :], (height, width, 3)).astype(np.float32).copy()
            composed[garment_mask] = garment_rgb[garment_mask]

            if training:
                noise = rng.normal(0.0, 2.4, size=composed.shape).astype(np.float32)
                composed = np.clip(composed + noise, 0.0, 255.0)

            rgb_images[index] = composed / 255.0

        return rgb_images

    @staticmethod
    def split_train_validation(
        x_train: np.ndarray,
        y_train: np.ndarray,
        validation_split: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split training data into train/validation sets with random_state=42."""
        print("✂️ Splitting train/validation with 85/15 ratio...")
        x_train_split, x_val, y_train_split, y_val = train_test_split(
            x_train,
            y_train,
            test_size=validation_split,
            random_state=42,
            stratify=y_train,
        )
        print(
            "✅ Split complete: "
            f"train={x_train_split.shape[0]}, val={x_val.shape[0]}"
        )
        return x_train_split, y_train_split, x_val, y_val

    @staticmethod
    def create_augmentation_config() -> Dict[str, Any]:
        """Return augmentation settings used by the trainer."""
        print("🌀 Preparing augmentation configuration...")
        return {
            "rotation_range": 20,
            "width_shift_range": 0.12,
            "height_shift_range": 0.12,
            "horizontal_flip": True,
            "zoom_range": 0.15,
            "brightness_jitter": 0.2,
            "contrast_jitter": 0.2,
            "saturation_jitter": 0.2,
            "hue_jitter": 0.04,
        }

    def save_pickle(self, payload: Any, filename: str) -> Path:
        """Save a Python payload to data/processed as pickle."""
        output_path = self.processed_dir / filename
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)
        print(f"💾 Saved {output_path.name}")
        return output_path

    def print_dataset_statistics(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Print data split and class statistics."""
        print("\n📊 Dataset statistics")
        print("=" * 60)
        print(f"Train samples:      {x_train.shape[0]}")
        print(f"Validation samples: {x_val.shape[0]}")
        print(f"Test samples:       {x_test.shape[0]}")
        print(f"Image shape:        {x_train.shape[1:]}")
        print(f"Number of classes:  {len(np.unique(y_train))}")
        print("\n🏷️ Class distribution")
        for index, label in enumerate(self.categories):
            train_count = int(np.sum(y_train == index))
            val_count = int(np.sum(y_val == index))
            test_count = int(np.sum(y_test == index))
            print(
                f"{label:10s} -> "
                f"train={train_count:5d}, val={val_count:4d}, test={test_count:4d}"
            )

    def run(self) -> None:
        """Execute the full preparation flow and persist outputs."""
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = self.load_data()
        x_train_raw, y_train, x_val_raw, y_val = self.split_train_validation(x_train_raw, y_train_raw)

        x_train = self.colorize_to_rgb(x_train_raw, seed=42, training=True)
        x_val = self.colorize_to_rgb(x_val_raw, seed=4242, training=False)
        x_test = self.colorize_to_rgb(x_test_raw, seed=9001, training=False)

        augmentation_config = self.create_augmentation_config()

        processed_data: Dict[str, Any] = {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test_raw,
            "categories": self.categories,
        }

        self.save_pickle(processed_data, "fashion_mnist_processed.pkl")
        self.save_pickle({"augmentation_config": augmentation_config, "random_state": 42}, "augmentation_config.pkl")
        self.save_pickle({"categories": self.categories}, "categories.pkl")

        self.print_dataset_statistics(x_train, y_train, x_val, y_val, x_test, y_test_raw)
        print("\n🎉 Data preparation completed successfully.")


def main() -> None:
    """Run the data preparation module as a script."""
    preparator = DataPreparator()
    preparator.run()


if __name__ == "__main__":
    main()