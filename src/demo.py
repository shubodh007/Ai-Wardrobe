"""End-to-end demonstration for AI Wardrobe."""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from color_extractor import ColorExtractor
from feature_extractor import FeatureExtractor
from recommendation_engine import OutfitRecommender
from train_classifier import FashionMNISTCNN


class AIWardrobeDemo:
    """Run classification, feature extraction, and outfit recommendation in one flow."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.processed_dir = self.project_root / "data" / "processed"
        self.model_dir = self.project_root / "models" / "saved_models"
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.categories = [
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

        self.wardrobe: List[Dict[str, Any]] = []
        self.next_id = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("🚀 Initializing AI Wardrobe Demo...")

        self.model_path = self.model_dir / "best_model.pt"
        self.model = None
        if self.model_path.exists():
            self.model = self._load_trained_model(self.model_path)
            if self.model is not None:
                print("✅ Loaded trained classifier model")
            else:
                print("⚠️ Failed to load trained model. Using heuristic category prediction.")
        else:
            print("⚠️ Trained model not found. Using heuristic category prediction.")

        self.color_extractor = ColorExtractor()
        print("✅ ColorExtractor loaded")

        self.feature_extractor = FeatureExtractor(model_path=str(self.model_path))
        print("✅ FeatureExtractor initialized")

        self.recommender = OutfitRecommender()
        print("✅ OutfitRecommender ready")

    def _load_trained_model(self, model_path: Path) -> Any:
        """Load a classifier checkpoint and return an eval-mode model."""
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            num_classes = int(checkpoint.get("num_classes", len(self.categories)))
            raw_input_shape = checkpoint.get("input_shape", (28, 28, 3))
            input_shape = tuple(int(value) for value in raw_input_shape) if isinstance(raw_input_shape, (list, tuple)) and len(raw_input_shape) == 3 else (28, 28, 3)
        else:
            state_dict = checkpoint
            num_classes = len(self.categories)
            input_shape = (28, 28, 3)

        model = FashionMNISTCNN(input_shape=input_shape, num_classes=num_classes).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def _intensity_to_color(mean_intensity: float) -> str:
        """Map grayscale intensity to a simplified color bucket."""
        if mean_intensity < 0.10:
            return "black"
        if mean_intensity < 0.20:
            return "gray"
        if mean_intensity < 0.30:
            return "blue"
        if mean_intensity < 0.40:
            return "purple"
        if mean_intensity < 0.50:
            return "green"
        if mean_intensity < 0.60:
            return "red"
        if mean_intensity < 0.70:
            return "orange"
        if mean_intensity < 0.80:
            return "yellow"
        if mean_intensity < 0.90:
            return "pink"
        return "white"

    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict category, infer color, and extract embedding from a single image."""
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.size == 0:
            raise ValueError("Input image is empty.")

        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=-1)
        if image_array.ndim != 3:
            raise ValueError("Image must have shape (H, W), (H, W, 1), or (H, W, 3).")

        if np.max(image_array) > 1.0:
            image_array = image_array / 255.0

        if image_array.shape[-1] == 1:
            model_input = np.repeat(image_array, repeats=3, axis=-1)
        elif image_array.shape[-1] == 3:
            model_input = image_array
        else:
            raise ValueError("Unsupported channel count. Expected 1 or 3 channels.")

        model_input = model_input.reshape(1, 28, 28, 3)

        if self.model is not None:
            model_tensor = torch.from_numpy(np.transpose(model_input, (0, 3, 1, 2)).astype(np.float32)).to(self.device)
            with torch.no_grad():
                logits = self.model(model_tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

            category_index = int(np.argmax(probabilities))
            confidence = float(probabilities[category_index])
        else:
            category_index = min(int(np.mean(model_input) * 10), 9)
            confidence = 0.0

        category = self.categories[category_index]
        dominant_rgb = np.mean(model_input[0], axis=(0, 1))
        color = self.color_extractor.rgb_to_color_name(tuple(int(value * 255.0) for value in dominant_rgb))

        feature_vector = self.feature_extractor.extract_single_image_features(model_input[0])
        features = feature_vector if feature_vector.size > 0 else None

        return {
            "id": self.next_id,
            "category": category,
            "color": color,
            "confidence": confidence,
            "features": features,
        }

    def add_to_wardrobe(self, item: Dict[str, Any]) -> None:
        """Add a processed item to the in-memory wardrobe."""
        item["id"] = self.next_id
        self.wardrobe.append(item)
        print(
            f"🧺 Added item #{item['id']}: "
            f"{item['category']} | color={item['color']} | confidence={item['confidence']:.3f}"
        )
        self.next_id += 1

    def display_wardrobe(self) -> None:
        """Print all wardrobe items in a clean format."""
        print("\n👗 Current Wardrobe")
        if not self.wardrobe:
            print("(empty)")
            return

        for item in self.wardrobe:
            print(
                f"- ID {item['id']:2d}: {item['category']:<10s} "
                f"| color={item['color']:<7s} | confidence={item['confidence']:.3f}"
            )

    def get_recommendations(self, occasion: str, weather: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Generate and print recommendations for the current wardrobe."""
        if not self.wardrobe:
            print("⚠️ Wardrobe is empty. No recommendations available.")
            return []

        print(f"\n🎯 Getting recommendations for occasion='{occasion}', weather='{weather}'...")
        recommendations = self.recommender.recommend_outfits(
            wardrobe=self.wardrobe,
            occasion=occasion,
            weather=weather,
            top_k=top_k,
        )

        if not recommendations:
            print("No outfits matched the current constraints.")
            return []

        for rank, outfit in enumerate(recommendations, start=1):
            print(f"\n{rank}. Recommendation")
            print(self.recommender.explain_recommendation(outfit))

        return recommendations


def main() -> None:
    """Run a complete local demo using test images from prepared data."""
    demo = AIWardrobeDemo()
    data_path = demo.processed_dir / "fashion_mnist_processed.pkl"

    if data_path.exists():
        print(f"📂 Loading test split from {data_path}...")
        with data_path.open("rb") as handle:
            payload = pickle.load(handle)
        x_test = np.asarray(payload["x_test"], dtype=np.float32)
    else:
        print("⚠️ Processed test data missing. Using synthetic sample images.")
        rng = np.random.default_rng(42)
        x_test = rng.random((100, 28, 28, 3), dtype=np.float32)

    if x_test.shape[0] == 0:
        print("❌ No test images available for demo.")
        return

    rng = np.random.default_rng(42)
    sample_count = min(10, x_test.shape[0])
    selected_indices = rng.choice(x_test.shape[0], size=sample_count, replace=False)

    print(f"🔎 Processing {sample_count} random test images...")
    for index in selected_indices:
        image = x_test[index]
        item = demo.process_image(image)
        demo.add_to_wardrobe(item)

    demo.display_wardrobe()
    demo.get_recommendations(occasion="formal", weather="mild", top_k=3)
    demo.get_recommendations(occasion="casual", weather="hot", top_k=3)


if __name__ == "__main__":
    main()
