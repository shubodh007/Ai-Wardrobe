"""Dominant color extraction utilities for AI Wardrobe."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


class ColorExtractor:
    """Extract dominant colors and map RGB values to human-friendly names."""

    def __init__(self) -> None:
        self.max_pixels_for_kmeans = 20000
        self.center_crop_ratio = 0.72
        self.random_seed = 42

    @staticmethod
    def _rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert RGB tuple into OpenCV HSV tuple."""
        rgb_array = np.uint8([[list(rgb)]])
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)[0, 0]
        return int(hsv[0]), int(hsv[1]), int(hsv[2])

    @staticmethod
    def _center_crop(image_rgb: np.ndarray, crop_ratio: float) -> np.ndarray:
        """Return the central crop to prioritize garment pixels over background."""
        height, width = image_rgb.shape[:2]
        crop_ratio = float(min(max(crop_ratio, 0.2), 1.0))

        crop_h = max(1, int(height * crop_ratio))
        crop_w = max(1, int(width * crop_ratio))
        y0 = max(0, (height - crop_h) // 2)
        x0 = max(0, (width - crop_w) // 2)
        y1 = min(height, y0 + crop_h)
        x1 = min(width, x0 + crop_w)
        return image_rgb[y0:y1, x0:x1]

    @staticmethod
    def _hsv_filter_mask(
        hsv_pixels: np.ndarray,
        min_sat: int,
        min_val: int,
        max_val: int,
    ) -> np.ndarray:
        """Create an HSV mask used for selecting informative garment pixels."""
        sat = hsv_pixels[:, 1]
        val = hsv_pixels[:, 2]
        return (sat >= min_sat) & (val >= min_val) & (val <= max_val)

    def _select_relevant_pixels(self, image_rgb: np.ndarray) -> np.ndarray:
        """Select pixels likely to represent garment color while suppressing background."""
        center_image = self._center_crop(image_rgb, self.center_crop_ratio)

        center_pixels = center_image.reshape(-1, 3)
        center_hsv = cv2.cvtColor(center_image, cv2.COLOR_RGB2HSV).reshape(-1, 3)

        strong_color_mask = self._hsv_filter_mask(center_hsv, min_sat=30, min_val=20, max_val=245)
        selected = center_pixels[strong_color_mask]

        if selected.shape[0] < 800:
            full_pixels = image_rgb.reshape(-1, 3)
            full_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)
            full_mask = self._hsv_filter_mask(full_hsv, min_sat=25, min_val=15, max_val=248)
            selected = full_pixels[full_mask]

        # Keep neutral garments (white/gray/black) detectable when saturation is low.
        if selected.shape[0] < 500:
            neutral_mask = center_hsv[:, 1] < 25
            neutral_pixels = center_pixels[neutral_mask]
            if neutral_pixels.shape[0] > 0:
                selected = (
                    np.vstack([selected, neutral_pixels])
                    if selected.shape[0] > 0
                    else neutral_pixels
                )

        if selected.shape[0] == 0:
            selected = center_pixels if center_pixels.shape[0] > 0 else image_rgb.reshape(-1, 3)

        if selected.shape[0] > self.max_pixels_for_kmeans:
            rng = np.random.default_rng(self.random_seed)
            indices = rng.choice(selected.shape[0], size=self.max_pixels_for_kmeans, replace=False)
            selected = selected[indices]

        return selected.astype(np.float32)

    def rgb_to_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to a robust color label with dedicated navy detection."""
        hue, sat, val = self._rgb_to_hsv(rgb)

        # Low saturation colors are better handled by value only.
        if sat <= 20:
            if val >= 220:
                return "white"
            if val <= 55:
                return "black"
            return "gray"

        # Very dark and low-signal colors are black.
        if val <= 28:
            return "black"

        if (0 <= hue <= 10) or (170 <= hue <= 179):
            return "red"

        if 11 <= hue <= 25:
            return "brown" if val < 125 else "orange"

        if 26 <= hue <= 35:
            return "yellow"

        if 36 <= hue <= 85:
            return "green"

        if 86 <= hue <= 130:
            return "navy" if val <= 120 else "blue"

        if 131 <= hue <= 150:
            return "purple"

        if 151 <= hue <= 169:
            return "pink"

        return "gray"

    def _extract_from_rgb_array(
        self,
        image_rgb: np.ndarray,
        n_colors: int = 3,
        mask: Optional[np.ndarray] = None,
    ) -> List[Dict[str, object]]:
        """Extract dominant colors from an RGB ndarray using KMeans clustering."""
        rgb_image = np.asarray(image_rgb)
        if rgb_image.size == 0:
            return []

        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            return []

        filtered_pixels: np.ndarray
        if mask is not None:
            mask_array = np.asarray(mask)
            if mask_array.ndim == 3:
                mask_array = mask_array[:, :, 0]

            if mask_array.shape[:2] == rgb_image.shape[:2]:
                garment_pixels = rgb_image[mask_array > 0]
                filtered_pixels = garment_pixels.astype(np.float32) if garment_pixels.size > 0 else np.empty((0, 3), dtype=np.float32)
            else:
                filtered_pixels = np.empty((0, 3), dtype=np.float32)
        else:
            filtered_pixels = np.empty((0, 3), dtype=np.float32)

        if filtered_pixels.size == 0 or filtered_pixels.shape[0] < 300:
            filtered_pixels = self._select_relevant_pixels(rgb_image)

        unique_pixels = np.unique(filtered_pixels.astype(np.uint8), axis=0)
        n_clusters = max(1, min(int(n_colors), filtered_pixels.shape[0], unique_pixels.shape[0]))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(filtered_pixels)
        centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(int)

        counts = np.bincount(labels, minlength=n_clusters)
        total = int(counts.sum())

        # Slightly boost highly saturated colors to avoid neutral background drift.
        cluster_scores: List[Tuple[int, float]] = []
        for index in range(n_clusters):
            rgb = tuple(int(value) for value in centers[index])
            _, sat, _ = self._rgb_to_hsv(rgb)
            percentage = float((counts[index] / total) * 100.0) if total > 0 else 0.0
            weighted = percentage * (1.0 + (float(sat) / 255.0) * 0.15)
            cluster_scores.append((index, weighted))

        sorted_indices = [index for index, _ in sorted(cluster_scores, key=lambda item: item[1], reverse=True)]

        dominant_colors: List[Dict[str, object]] = []
        for index in sorted_indices:
            rgb = tuple(int(value) for value in centers[index])
            percentage = float((counts[index] / total) * 100.0) if total > 0 else 0.0
            dominant_colors.append(
                {
                    "rgb": rgb,
                    "percentage": percentage,
                    "name": self.rgb_to_color_name(rgb),
                }
            )

        return dominant_colors

    def extract_dominant_colors_from_array(
        self,
        image_rgb: np.ndarray,
        n_colors: int = 3,
        mask: Optional[np.ndarray] = None,
    ) -> List[Dict[str, object]]:
        """Extract dominant colors from an in-memory RGB array."""
        print("Extracting dominant colors from in-memory image array")
        colors = self._extract_from_rgb_array(image_rgb=image_rgb, n_colors=n_colors, mask=mask)
        if not colors:
            print("⚠️ Failed to extract dominant colors from in-memory array.")
            return []
        print("Dominant color extraction complete.")
        return colors

    def extract_dominant_colors(self, image_path: str, n_colors: int = 3) -> List[Dict[str, object]]:
        """Extract dominant colors from an image using KMeans clustering."""
        print(f"Extracting dominant colors from: {image_path}")
        if not os.path.exists(image_path):
            print("⚠️ Image file not found.")
            return []

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print("⚠️ Failed to read image file.")
            return []

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        dominant_colors = self._extract_from_rgb_array(image_rgb=image_rgb, n_colors=n_colors)

        print("Dominant color extraction complete.")
        return dominant_colors

    def visualize_colors(self, colors: List[Dict[str, object]], save_path: Optional[str] = None) -> None:
        """Visualize colors in a horizontal bar with labels and percentages."""
        if not colors:
            print("⚠️ No colors available to visualize.")
            return

        fig, ax = plt.subplots(figsize=(10, 2.2))
        start = 0.0
        for color_info in colors:
            rgb = np.array(color_info["rgb"], dtype=np.float32) / 255.0
            percentage = float(color_info["percentage"])
            width = max(percentage, 0.1)
            ax.barh(y=0, width=width, left=start, color=rgb, height=0.6)

            label = f"{color_info['name']}\n{percentage:.1f}%"
            ax.text(start + width / 2.0, 0, label, ha="center", va="center", fontsize=9)
            start += width

        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Percentage")
        ax.set_title("Dominant Colors")
        plt.tight_layout()

        if save_path:
            save_target = Path(save_path)
            os.makedirs(save_target.parent, exist_ok=True)
            plt.savefig(save_target, dpi=150, bbox_inches="tight")
            print(f"💾 Saved color visualization to {save_target}")

        plt.show()


def _create_example_image(path: Path) -> None:
    """Create a simple synthetic image used in the module demo."""
    canvas = np.zeros((120, 360, 3), dtype=np.uint8)
    canvas[:, :120] = (220, 40, 40)      # Red in RGB
    canvas[:, 120:240] = (40, 80, 220)   # Blue in RGB
    canvas[:, 240:] = (245, 245, 245)    # White-like background in RGB
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    """Run a standalone color extraction demo."""
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)

    example_path = processed_dir / "color_demo_image.png"
    _create_example_image(example_path)

    extractor = ColorExtractor()
    colors = extractor.extract_dominant_colors(str(example_path), n_colors=3)

    if colors:
        print("\n🎯 Extracted colors:")
        for color in colors:
            print(
                f"- {color['name']:<7s} RGB={color['rgb']} "
                f"({float(color['percentage']):.2f}%)"
            )
        extractor.visualize_colors(colors, save_path=str(processed_dir / "color_demo_bar.png"))


if __name__ == "__main__":
    main()