"""Feature extraction and similarity search for AI Wardrobe."""

import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from train_classifier import FashionMNISTCNN


class FeatureExtractor:
    """Extract embedding vectors from the trained clothing classifier."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        default_model_path = self.project_root / "models" / "saved_models" / "best_model.pt"
        self.model_path = Path(model_path) if model_path else default_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_model: Optional[FashionMNISTCNN] = None
        self.input_shape: Tuple[int, int, int] = (28, 28, 3)

        print(f"🔧 Loading feature model from {self.model_path}...")
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                num_classes = int(checkpoint.get("num_classes", 10))
                raw_input_shape = checkpoint.get("input_shape", self.input_shape)
                if isinstance(raw_input_shape, (list, tuple)) and len(raw_input_shape) == 3:
                    self.input_shape = tuple(int(value) for value in raw_input_shape)
            else:
                state_dict = checkpoint
                num_classes = 10

            self.feature_model = FashionMNISTCNN(input_shape=self.input_shape, num_classes=num_classes).to(self.device)
            self.feature_model.load_state_dict(state_dict)
            self.feature_model.eval()
            print("✅ Feature extractor initialized.")
        else:
            print("⚠️ Model file not found. Train the classifier first.")

    def _prepare_batch(self, images: np.ndarray) -> np.ndarray:
        """Validate and convert images to channel-first float32."""
        batch = np.asarray(images, dtype=np.float32)

        if batch.ndim == 2:
            batch = np.expand_dims(batch, axis=0)
            batch = np.expand_dims(batch, axis=-1)
        elif batch.ndim == 3:
            batch = np.expand_dims(batch, axis=-1)

        if batch.ndim != 4:
            raise ValueError("Input images must have shape (N, H, W, C) or (N, H, W).")

        if np.max(batch) > 1.0:
            batch = batch / 255.0

        if batch.shape[-1] in (1, 3):
            batch = np.transpose(batch, (0, 3, 1, 2))
        elif batch.shape[1] not in (1, 3):
            raise ValueError("Expected input channels to be 1 or 3.")

        expected_channels = int(self.input_shape[2])
        if batch.shape[1] != expected_channels:
            if expected_channels == 3 and batch.shape[1] == 1:
                batch = np.repeat(batch, repeats=3, axis=1)
            elif expected_channels == 1 and batch.shape[1] == 3:
                batch = np.mean(batch, axis=1, keepdims=True)
            else:
                raise ValueError(f"Input channels {batch.shape[1]} do not match model channels {expected_channels}.")

        return batch.astype(np.float32)

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extract and L2-normalize feature vectors for a batch of images."""
        if self.feature_model is None:
            print("⚠️ Feature model is unavailable.")
            return np.empty((0, 0), dtype=np.float32)

        if images.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        batch = self._prepare_batch(images)
        batch_tensor = torch.from_numpy(batch).to(self.device)

        with torch.no_grad():
            features = self.feature_model.extract_embeddings(batch_tensor).cpu().numpy()

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = features / norms
        return normalized.astype(np.float32)

    def extract_single_image_features(self, image: np.ndarray) -> np.ndarray:
        """Extract normalized feature vector for one image."""
        single = np.asarray(image)
        if single.ndim == 2:
            single = np.expand_dims(single, axis=-1)
        if single.ndim != 3:
            print("⚠️ Single image must have shape (H, W) or (H, W, C).")
            return np.array([], dtype=np.float32)

        batch = np.expand_dims(single, axis=0)
        features = self.extract_features(batch)
        if features.size == 0:
            return np.array([], dtype=np.float32)
        return features[0]

    @staticmethod
    def compute_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity using dot product between two vectors."""
        vector1 = np.asarray(features1, dtype=np.float32).reshape(-1)
        vector2 = np.asarray(features2, dtype=np.float32).reshape(-1)
        if vector1.size == 0 or vector2.size == 0:
            return 0.0

        denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if denominator == 0.0:
            return 0.0

        return float(np.dot(vector1, vector2) / denominator)

    def find_similar_items(
        self,
        query_features: np.ndarray,
        wardrobe_features: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-k similar item indices and scores sorted by similarity."""
        wardrobe = np.asarray(wardrobe_features, dtype=np.float32)
        query = np.asarray(query_features, dtype=np.float32).reshape(-1)

        if wardrobe.size == 0 or query.size == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        if wardrobe.ndim == 1:
            wardrobe = wardrobe.reshape(1, -1)

        query_norm = np.linalg.norm(query)
        if query_norm == 0.0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        query = query / query_norm

        wardrobe_norms = np.linalg.norm(wardrobe, axis=1, keepdims=True)
        wardrobe_norms[wardrobe_norms == 0.0] = 1.0
        wardrobe = wardrobe / wardrobe_norms

        similarities = np.dot(wardrobe, query)
        k = min(max(top_k, 1), similarities.shape[0])
        indices = np.argsort(similarities)[::-1][:k]
        scores = similarities[indices]
        return indices.astype(int), scores.astype(np.float32)


def main() -> None:
    """Demonstrate feature extraction and nearest-neighbor search."""
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)

    data_path = processed_dir / "fashion_mnist_processed.pkl"
    extractor = FeatureExtractor()

    if extractor.feature_model is None:
        print("❌ Feature extraction unavailable because model loading failed.")
        return

    if not data_path.exists():
        print("❌ Processed data file is missing. Run src/data_preparation.py first.")
        return

    print(f"📂 Loading test data from {data_path}...")
    with data_path.open("rb") as handle:
        data = pickle.load(handle)

    x_test = np.asarray(data["x_test"], dtype=np.float32)
    demo_batch = x_test[:100]
    print("🧬 Extracting features for first 100 test images...")
    features = extractor.extract_features(demo_batch)

    if features.size == 0:
        print("⚠️ No features were extracted.")
        return

    query_features = features[0]
    indices, scores = extractor.find_similar_items(query_features, features, top_k=5)

    print("\n🔍 Similarity search results (query index: 0):")
    for rank, (index, score) in enumerate(zip(indices, scores), start=1):
        print(f"{rank}. index={int(index):3d} | similarity={float(score):.4f}")

    output_path = processed_dir / "test_features.pkl"
    with output_path.open("wb") as handle:
        pickle.dump({"features": features, "source_count": features.shape[0]}, handle)
    print(f"💾 Saved extracted features to {output_path}")


if __name__ == "__main__":
    main()