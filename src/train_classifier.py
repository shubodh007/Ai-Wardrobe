"""Model training module for AI Wardrobe."""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset


class FashionMNISTCNN(nn.Module):
    """CNN architecture for Fashion-MNIST classification."""

    def __init__(self, input_shape: Tuple[int, int, int] = (28, 28, 3), num_classes: int = 10) -> None:
        super().__init__()
        channels = input_shape[2]

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
        )

        self.flatten = nn.Flatten()
        self.embedding_layer = nn.Linear(128 * 3 * 3, 256)
        self.embedding_bn = nn.BatchNorm1d(256)
        self.embedding_dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(256, num_classes)

    def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer embeddings for feature search."""
        x = self.features(inputs)
        x = self.flatten(x)
        x = torch.relu(self.embedding_layer(x))
        x = self.embedding_bn(x)
        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return logits for classification."""
        embeddings = self.extract_embeddings(inputs)
        logits = self.classifier(self.embedding_dropout(embeddings))
        return logits


class NumpyImageDataset(Dataset):
    """Dataset backed by numpy arrays with optional torchvision transforms."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, transform: Optional[Any] = None) -> None:
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.features[index].clone()
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class ClothingClassifier:
    """Train and evaluate a CNN for Fashion-MNIST classification."""

    def __init__(self, input_shape: Tuple[int, int, int] = (28, 28, 3), num_classes: int = 10) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model: Optional[FashionMNISTCNN] = None
        self.history: Dict[str, List[float]] = {
            "accuracy": [],
            "val_accuracy": [],
            "loss": [],
            "val_loss": [],
        }

        self.optimizer: Optional[AdamW] = None
        self.criterion: Optional[nn.CrossEntropyLoss] = None
        self.scheduler: Optional[OneCycleLR] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.project_root = Path(__file__).resolve().parent.parent
        self.models_dir = self.project_root / "models" / "saved_models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.augmentation_config = self._load_augmentation_config()

    def build_cnn_model(self) -> None:
        """Build a CNN with 3 Conv-BN-Pool-Dropout blocks."""
        print("Building CNN architecture...")
        self.model = FashionMNISTCNN(input_shape=self.input_shape, num_classes=self.num_classes).to(self.device)
        print("CNN model built.")

    def compile_model(self, learning_rate: float = 8e-4) -> None:
        """Configure optimizer, criterion, and LR scheduler."""
        if self.model is None:
            raise ValueError("Model is not built. Call build_cnn_model first.")

        print("Configuring optimizer and loss...")
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=2e-4)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.scheduler = None
        print("Training components configured.")

    @staticmethod
    def _to_channel_first(features: np.ndarray) -> np.ndarray:
        """Convert numpy arrays into channel-first normalized float arrays."""
        x_array = np.asarray(features, dtype=np.float32)

        if x_array.ndim != 4:
            raise ValueError("Expected image input with shape (N, H, W, C) or (N, C, H, W).")

        if np.max(x_array) > 1.0:
            x_array = x_array / 255.0

        if x_array.shape[-1] in (1, 3):
            x_array = np.transpose(x_array, (0, 3, 1, 2))
        elif x_array.shape[1] not in (1, 3):
            raise ValueError("Expected input channels to be 1 or 3.")

        return np.asarray(x_array, dtype=np.float32)

    def _load_augmentation_config(self) -> Dict[str, Any]:
        """Load augmentation config produced by data preparation, or defaults."""
        defaults: Dict[str, Any] = {
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

        config_path = self.project_root / "data" / "processed" / "augmentation_config.pkl"
        if not config_path.exists():
            return defaults

        try:
            with config_path.open("rb") as handle:
                payload = pickle.load(handle)

            if isinstance(payload, dict):
                loaded = payload.get("augmentation_config", payload)
                if isinstance(loaded, dict):
                    merged = dict(defaults)
                    merged.update(loaded)
                    return merged
        except Exception:
            print(f"⚠️ Failed to load augmentation config from {config_path}. Using defaults.")

        return defaults

    def _build_train_transform(self, channels: int) -> Optional[Any]:
        """Build a lightweight torch-native augmentation function."""
        config = self.augmentation_config
        rotation = float(config.get("rotation_range", 0.0))
        width_shift = float(config.get("width_shift_range", 0.0))
        height_shift = float(config.get("height_shift_range", 0.0))
        zoom = float(config.get("zoom_range", 0.0))
        brightness = float(config.get("brightness_jitter", 0.0))
        contrast = float(config.get("contrast_jitter", 0.0))
        saturation = float(config.get("saturation_jitter", 0.0)) if channels == 3 else 0.0
        horizontal_flip = bool(config.get("horizontal_flip", False))

        max_dx = int(round(width_shift * self.input_shape[1]))
        max_dy = int(round(height_shift * self.input_shape[0]))
        max_dx = max(0, min(max_dx, self.input_shape[1] // 3))
        max_dy = max(0, min(max_dy, self.input_shape[0] // 3))

        erase_p = 0.08
        erase_scale_min, erase_scale_max = 0.02, 0.10

        def _random_shift(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
            shifted = torch.roll(image, shifts=(dy, dx), dims=(1, 2))
            if dy > 0:
                shifted[:, :dy, :] = 0.0
            elif dy < 0:
                shifted[:, dy:, :] = 0.0
            if dx > 0:
                shifted[:, :, :dx] = 0.0
            elif dx < 0:
                shifted[:, :, dx:] = 0.0
            return shifted

        def _random_zoom(image: torch.Tensor, zoom_strength: float) -> torch.Tensor:
            if zoom_strength <= 0.0:
                return image

            scale = 1.0 + float((torch.rand(1).item() * 2.0 - 1.0) * zoom_strength)
            if abs(scale - 1.0) < 1e-3:
                return image

            channels_local, height, width = image.shape
            new_h = max(8, int(round(height * scale)))
            new_w = max(8, int(round(width * scale)))

            resized = F.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            if scale >= 1.0:
                y0 = (new_h - height) // 2
                x0 = (new_w - width) // 2
                return resized[:, y0 : y0 + height, x0 : x0 + width]

            output = torch.zeros((channels_local, height, width), dtype=image.dtype)
            y0 = (height - new_h) // 2
            x0 = (width - new_w) // 2
            output[:, y0 : y0 + new_h, x0 : x0 + new_w] = resized
            return output

        def _random_erasing(image: torch.Tensor) -> torch.Tensor:
            if torch.rand(1).item() >= erase_p:
                return image

            channels_local, height, width = image.shape
            area = float(height * width)
            target_area = float(torch.empty(1).uniform_(erase_scale_min, erase_scale_max).item()) * area
            aspect_ratio = float(torch.empty(1).uniform_(0.4, 2.2).item())

            erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(np.sqrt(target_area / max(aspect_ratio, 1e-6))))
            erase_h = max(1, min(erase_h, height - 1))
            erase_w = max(1, min(erase_w, width - 1))

            top = int(torch.randint(0, max(1, height - erase_h + 1), (1,)).item())
            left = int(torch.randint(0, max(1, width - erase_w + 1), (1,)).item())
            fill = torch.rand((channels_local, erase_h, erase_w), dtype=image.dtype)
            image[:, top : top + erase_h, left : left + erase_w] = fill
            return image

        def augment(image: torch.Tensor) -> torch.Tensor:
            augmented = image

            if horizontal_flip and torch.rand(1).item() < 0.5:
                augmented = torch.flip(augmented, dims=[2])

            if max_dx > 0 or max_dy > 0:
                dx = int(torch.randint(-max_dx, max_dx + 1, (1,)).item()) if max_dx > 0 else 0
                dy = int(torch.randint(-max_dy, max_dy + 1, (1,)).item()) if max_dy > 0 else 0
                if dx != 0 or dy != 0:
                    augmented = _random_shift(augmented, dx=dx, dy=dy)

            if zoom > 0.0 and torch.rand(1).item() < 0.35:
                augmented = _random_zoom(augmented, zoom_strength=zoom)

            if rotation > 0.0 and torch.rand(1).item() < 0.25:
                angle = float((torch.rand(1).item() * 2.0 - 1.0) * rotation)
                radians = np.deg2rad(angle)
                theta = torch.tensor(
                    [
                        [np.cos(radians), -np.sin(radians), 0.0],
                        [np.sin(radians), np.cos(radians), 0.0],
                    ],
                    dtype=augmented.dtype,
                ).unsqueeze(0)
                grid = F.affine_grid(theta, size=(1, channels, self.input_shape[0], self.input_shape[1]), align_corners=False)
                augmented = F.grid_sample(
                    augmented.unsqueeze(0),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).squeeze(0)

            if brightness > 0.0:
                factor = 1.0 + float((torch.rand(1).item() * 2.0 - 1.0) * brightness)
                augmented = torch.clamp(augmented * factor, 0.0, 1.0)

            if contrast > 0.0:
                factor = 1.0 + float((torch.rand(1).item() * 2.0 - 1.0) * contrast)
                mean = augmented.mean(dim=(1, 2), keepdim=True)
                augmented = torch.clamp((augmented - mean) * factor + mean, 0.0, 1.0)

            if saturation > 0.0 and channels == 3:
                factor = 1.0 + float((torch.rand(1).item() * 2.0 - 1.0) * saturation)
                gray = augmented.mean(dim=0, keepdim=True)
                augmented = torch.clamp((augmented - gray) * factor + gray, 0.0, 1.0)

            augmented = _random_erasing(augmented)
            return augmented

        return augment

    def _save_checkpoint(self, path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model weights and metadata to a .pt file."""
        if self.model is None:
            raise ValueError("No model available to save.")

        payload: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
        }
        if metadata:
            payload.update(metadata)

        torch.save(payload, path)

    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 30,
        batch_size: int = 96,
        use_augmentation: bool = True,
    ) -> Dict[str, List[float]]:
        """Train model with early stopping, checkpointing, and LR scheduling."""
        if self.model is None:
            raise ValueError("Model is not built. Call build_cnn_model first.")
        if self.optimizer is None or self.criterion is None:
            raise ValueError("Model is not compiled. Call compile_model first.")

        train_x = self._to_channel_first(train_data[0])
        train_y = np.asarray(train_data[1], dtype=np.int64)
        val_x = self._to_channel_first(val_data[0])
        val_y = np.asarray(val_data[1], dtype=np.int64)

        channels = int(train_x.shape[1])
        train_transform = self._build_train_transform(channels=channels) if use_augmentation else None

        print(f"Augmentation enabled: {use_augmentation}")

        train_loader = DataLoader(
            NumpyImageDataset(train_x, train_y, transform=train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            NumpyImageDataset(val_x, val_y, transform=None),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=2e-3,
            epochs=epochs,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=0.2,
            anneal_strategy="cos",
            div_factor=12.0,
            final_div_factor=250.0,
        )

        self.history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
        best_val_accuracy = -1.0
        epochs_without_improvement = 0
        checkpoint_path = self.models_dir / "best_model.pt"

        print(f"Training started on {self.device.type}...")
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss_total = 0.0
            train_correct = 0
            train_samples = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
                self.scheduler.step()

                train_loss_total += float(loss.item()) * int(batch_y.size(0))
                predictions = torch.argmax(logits, dim=1)
                train_correct += int((predictions == batch_y).sum().item())
                train_samples += int(batch_y.size(0))

            train_loss = train_loss_total / max(train_samples, 1)
            train_accuracy = train_correct / max(train_samples, 1)

            self.model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_samples = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)

                    val_loss_total += float(loss.item()) * int(batch_y.size(0))
                    predictions = torch.argmax(logits, dim=1)
                    val_correct += int((predictions == batch_y).sum().item())
                    val_samples += int(batch_y.size(0))

            val_loss = val_loss_total / max(val_samples, 1)
            val_accuracy = val_correct / max(val_samples, 1)

            self.history["loss"].append(float(train_loss))
            self.history["accuracy"].append(float(train_accuracy))
            self.history["val_loss"].append(float(val_loss))
            self.history["val_accuracy"].append(float(val_accuracy))

            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"loss={train_loss:.4f}, acc={train_accuracy:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}, lr={current_lr:.6f}"
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
                self._save_checkpoint(
                    checkpoint_path,
                    metadata={
                        "epoch": epoch,
                        "val_accuracy": float(val_accuracy),
                        "augmentation_config": self.augmentation_config,
                    },
                )
                print(f"💾 Saved new best checkpoint to {checkpoint_path}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= 8:
                print("⏹️ Early stopping triggered (no val_accuracy improvement for 8 epochs).")
                break

        print("✅ Training complete.")
        return self.history

    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray], batch_size: int = 256) -> Tuple[float, float]:
        """Evaluate model on the test set and print formatted metrics."""
        if self.model is None:
            raise ValueError("Model is not built. Call build_cnn_model first.")
        if self.criterion is None:
            raise ValueError("Model is not compiled. Call compile_model first.")

        test_x = self._to_channel_first(test_data[0])
        test_y = np.asarray(test_data[1], dtype=np.int64)
        loader = DataLoader(
            NumpyImageDataset(test_x, test_y, transform=None),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        self.model.eval()
        loss_total = 0.0
        correct = 0
        samples = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

                loss_total += float(loss.item()) * int(batch_y.size(0))
                predictions = torch.argmax(logits, dim=1)
                correct += int((predictions == batch_y).sum().item())
                samples += int(batch_y.size(0))

        test_loss = loss_total / max(samples, 1)
        test_accuracy = correct / max(samples, 1)

        print("🧪 Evaluating on test set...")
        print(f"📉 Test Loss     : {test_loss:.4f}")
        print(f"📈 Test Accuracy : {test_accuracy:.4f}")
        return float(test_loss), float(test_accuracy)

    def plot_training_history(self) -> Optional[Path]:
        """Plot and save accuracy/loss curves to models/saved_models."""
        if not self.history["loss"]:
            print("⚠️ No training history to plot.")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.history["accuracy"], label="train")
        ax1.plot(self.history["val_accuracy"], label="val")
        ax1.set_title("Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        ax2.plot(self.history["loss"], label="train")
        ax2.plot(self.history["val_loss"], label="val")
        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        output_path = self.models_dir / "training_history.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"💾 Saved training curves to {output_path}")
        return output_path

    def save_model(self, filepath: Optional[Path] = None) -> Path:
        """Save final trained model as clothing_classifier.pt."""
        target_path = filepath or (self.models_dir / "clothing_classifier.pt")
        self._save_checkpoint(target_path)
        print(f"💾 Saved model to {target_path}")
        return target_path


def main() -> None:
    """Run end-to-end training from prepared pickle data."""
    project_root = Path(__file__).resolve().parent.parent
    processed_path = project_root / "data" / "processed" / "fashion_mnist_processed.pkl"

    if not processed_path.exists():
        print("❌ Processed data file not found.")
        print("➡️ Run src/data_preparation.py first.")
        return

    print(f"📂 Loading processed data from {processed_path}...")
    with processed_path.open("rb") as handle:
        data = pickle.load(handle)

    classifier = ClothingClassifier()
    classifier.build_cnn_model()
    classifier.compile_model()

    if classifier.model is not None:
        print("\n📋 Model Architecture")
        print(classifier.model)

    classifier.train(
        train_data=(data["x_train"], data["y_train"]),
        val_data=(data["x_val"], data["y_val"]),
        epochs=30,
        batch_size=96,
    )
    classifier.evaluate((data["x_test"], data["y_test"]))
    classifier.plot_training_history()
    classifier.save_model()
    print("🎉 Training pipeline finished.")


if __name__ == "__main__":
    main()