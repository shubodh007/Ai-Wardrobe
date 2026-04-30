"""FastAPI server exposing AI Wardrobe ML capabilities as REST endpoints."""

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Resolve paths so we can import from ../src/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def _load_dotenv_file(dotenv_path: Path) -> None:
    """Load .env key/value pairs into process env when variables are unset."""
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_key = key.strip()
        env_value = value.strip().strip('"').strip("'")
        if env_key and env_key not in os.environ:
            os.environ[env_key] = env_value


_load_dotenv_file(PROJECT_ROOT / ".env")

from train_classifier import FashionMNISTCNN  # noqa: E402
from color_extractor import ColorExtractor  # noqa: E402
from feature_extractor import FeatureExtractor  # noqa: E402
from recommendation_engine import OutfitRecommender  # noqa: E402


logger = logging.getLogger("ai-wardrobe-api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
CATEGORIES: List[str] = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

DEFAULT_ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
DEFAULT_ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"]


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%s. Using default=%s", key, raw, default)
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%s. Using default=%s", key, raw, default)
        return default


def _env_list(key: str, default: List[str]) -> List[str]:
    raw = os.getenv(key)
    if raw is None:
        return list(default)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values if values else list(default)


ALLOWED_ORIGINS = _env_list("CORS_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
ALLOWED_IMAGE_TYPES: Set[str] = set(
    value.lower() for value in _env_list("ALLOWED_IMAGE_TYPES", DEFAULT_ALLOWED_IMAGE_TYPES)
)
MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 5 * 1024 * 1024)
RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_MAX_REQUESTS = _env_int("RATE_LIMIT_MAX_REQUESTS", 120)
RATE_LIMIT_WINDOW_SECONDS = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
API_DISABLE_MODEL_LOAD = _env_bool("API_DISABLE_MODEL_LOAD", False)
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "saved_models" / "best_model.pt"))
).resolve()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free").strip()
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions").strip()
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "AI Wardrobe").strip()
OPENROUTER_TIMEOUT_SECONDS = float(np.clip(_env_float("OPENROUTER_TIMEOUT_SECONDS", 18.0), 5.0, 60.0))
DEFAULT_OPENROUTER_EXPLAINABILITY_SYSTEM_PROMPT = (
    "You are a wardrobe explainability assistant for non-expert users. "
    "Your only goal is to explain one suggested outfit in plain language. "
    "Never mention model internals, embeddings, logits, algorithms, or raw numeric scores. "
    "Use only the facts provided by the user message. Do not invent missing details. "
    "Keep tone practical, warm, and concise. "
    "Return ONLY valid JSON with this exact schema: "
    '{"summary": string, "style": string, "why_it_works_today": string, '
    '"color_in_simple_words": string, "how_this_works": string, '
    '"simple_tip": string, "confidence_note": string, "curation_title": string, '
    '"vibe": string, "when_to_wear": string, "why_this_look": [string, string], '
    '"styling_steps": [string, string], "tradeoff_note": string}. '
    "Each string field must be 1-2 short sentences, max 24 words. "
    "List fields must include exactly 2 concise bullets each. "
    "No markdown, no code fences, no extra keys."
)
OPENROUTER_EXPLAINABILITY_SYSTEM_PROMPT = (
    os.getenv("OPENROUTER_EXPLAINABILITY_SYSTEM_PROMPT", DEFAULT_OPENROUTER_EXPLAINABILITY_SYSTEM_PROMPT).strip()
    or DEFAULT_OPENROUTER_EXPLAINABILITY_SYSTEM_PROMPT
)
CLASSIFIER_TTA_MODE = os.getenv("CLASSIFIER_TTA_MODE", "adaptive").strip().lower()
CLASSIFIER_TTA_BASE_WEIGHT = float(np.clip(_env_float("CLASSIFIER_TTA_BASE_WEIGHT", 0.82), 0.55, 0.95))
CLASSIFIER_TTA_CONFIDENCE_THRESHOLD = float(
    np.clip(_env_float("CLASSIFIER_TTA_CONFIDENCE_THRESHOLD", 0.58), 0.40, 0.90)
)

# In-memory wardrobe store  {item_id: {...}}
wardrobe_store: Dict[str, Dict[str, Any]] = {}

# In-memory recommendation feedback learning state
recommendation_feedback_profile: Dict[str, Any] = {
    "likes": 0,
    "dislikes": 0,
    "total_feedback": 0,
    "category_affinity": {},
    "color_affinity": {},
    "goal_mode_affinity": {},
    "color_strategy_affinity": {},
}
recommendation_registry: Dict[str, Dict[str, Any]] = {}
recommendation_registry_order: Deque[str] = deque()
RECOMMENDATION_REGISTRY_LIMIT = 1200

rate_limit_buckets: Dict[str, Deque[float]] = defaultdict(deque)
rate_limit_lock = threading.Lock()

# ML components loaded once at startup
model_holder: Dict[str, Any] = {
    "model": None,
    "feature_extractor": None,
    "color_extractor": None,
    "recommender": None,
    "class_prototypes": None,
    "prototype_sample_count": 0,
    "samples": None,  # pre-selected Fashion-MNIST samples
    "model_loaded": False,
}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class WardrobeAddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=64)
    color: str = Field(min_length=1, max_length=32)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str = Field(default="", max_length=100)
    features: List[float] = Field(default_factory=list, max_length=1024)
    wear_count: int = Field(default=0, ge=0, le=10000)
    last_worn_days_ago: Optional[int] = Field(default=None, ge=0, le=3650)


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    occasion: str = Field(default="casual", pattern="^(casual|formal|sport)$")
    weather: str = Field(default="mild", pattern="^(cold|hot|rainy|mild)$")
    top_k: int = Field(default=3, ge=1, le=10)
    goal_mode: str = Field(
        default="balanced",
        pattern="^(balanced|formal_precision|comfort_first|heat_survival|rain_safe|repeat_avoider|bold_experiment)$",
    )
    exploration: float = Field(default=0.35, ge=0.0, le=1.0)
    color_strategy: str = Field(
        default="auto",
        pattern="^(auto|monochrome|analogous|complementary|neutral_plus_one|high_contrast)$",
    )
    occasion_strictness: float = Field(default=0.6, ge=0.0, le=1.0)
    anti_repeat: bool = Field(default=True)
    hero_item_id: Optional[str] = Field(default=None, max_length=100)
    temperature_bias: str = Field(default="neutral", pattern="^(neutral|run_cold|run_warm)$")
    explainability: str = Field(default="rule", pattern="^(rule|llm)$")
    include_backup_pack: bool = Field(default=True)
    feedback_learning: bool = Field(default=True)


class RecommendationFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(min_length=6, max_length=120)
    signal: str = Field(pattern="^(like|dislike)$")


class ClassifySampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base64: str = Field(min_length=16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_and_open_image(image_bytes: bytes) -> Image.Image:
    """Validate image bytes and return a loaded PIL image."""
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed size is {MAX_UPLOAD_BYTES} bytes.",
        )

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        return img
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc


async def _read_upload_image(file: UploadFile) -> bytes:
    """Read and validate upload bytes with MIME and size constraints."""
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_TYPES))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    image_bytes = await file.read()
    _validate_and_open_image(image_bytes)
    return image_bytes


def _largest_component_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component from a binary mask."""
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8.astype(bool)

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_idx


def _score_mask(mask: np.ndarray) -> float:
    """Score a candidate foreground mask by area and center coverage."""
    mask_bool = mask > 0
    if mask_bool.size == 0:
        return -1.0

    fg_ratio = float(np.mean(mask_bool))
    if fg_ratio < 0.02 or fg_ratio > 0.95:
        return -1.0

    h, w = mask_bool.shape
    y0, y1 = int(h * 0.2), int(h * 0.8)
    x0, x1 = int(w * 0.2), int(w * 0.8)
    center_ratio = float(np.mean(mask_bool[y0:y1, x0:x1])) if y1 > y0 and x1 > x0 else 0.0

    border_pixels = np.concatenate(
        [mask_bool[0, :], mask_bool[-1, :], mask_bool[:, 0], mask_bool[:, -1]],
        axis=0,
    )
    border_ratio = float(np.mean(border_pixels)) if border_pixels.size else 0.0
    border_penalty = float(np.clip(1.0 - border_ratio, 0.0, 1.0))

    largest = _largest_component_mask(mask_bool)
    largest_ratio = float(np.mean(largest))

    # Favor coherent foreground with center presence but avoid border-heavy masks.
    return (0.50 * center_ratio) + (0.30 * largest_ratio) + (0.20 * border_penalty)


def _extract_garment_mask(rgb_image: np.ndarray) -> np.ndarray:
    """Extract a simple garment foreground mask using Otsu and morphology."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask_bin)

    # Color-distance candidate improves segmentation when web backgrounds are near-uniform.
    bg_color = _estimate_background_color(rgb_image).astype(np.float32)
    color_delta = np.linalg.norm(rgb_image.astype(np.float32) - bg_color[None, None, :], axis=2)
    color_delta = cv2.GaussianBlur(color_delta.astype(np.float32), (5, 5), 0)
    color_delta_u8 = cv2.normalize(color_delta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask_color = cv2.threshold(color_delta_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_color_inv = cv2.bitwise_not(mask_color)

    kernel = np.ones((5, 5), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel)
    mask_color_inv = cv2.morphologyEx(mask_color_inv, cv2.MORPH_OPEN, kernel)
    mask_color_inv = cv2.morphologyEx(mask_color_inv, cv2.MORPH_CLOSE, kernel)

    candidates = [mask_bin, mask_inv, mask_color, mask_color_inv]
    scores = [_score_mask(candidate) for candidate in candidates]
    best_idx = int(np.argmax(scores))
    selected = candidates[best_idx]

    largest = _largest_component_mask(selected)
    return largest


def _estimate_background_color(rgb_image: np.ndarray) -> np.ndarray:
    """Estimate scene background color from image borders."""
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        return np.array([228, 228, 228], dtype=np.uint8)

    border_pixels = np.concatenate(
        [
            rgb_image[0, :, :],
            rgb_image[-1, :, :],
            rgb_image[:, 0, :],
            rgb_image[:, -1, :],
        ],
        axis=0,
    )
    if border_pixels.size == 0:
        return np.array([228, 228, 228], dtype=np.uint8)

    return np.median(border_pixels, axis=0).astype(np.uint8)


def _build_focus_crop_view(rgb_image: np.ndarray, mask_bool: np.ndarray) -> Optional[np.ndarray]:
    """Create a centered, letterboxed crop around the foreground mask."""
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return None

    h, w = mask_bool.shape
    y0, y1 = int(np.min(ys)), int(np.max(ys)) + 1
    x0, x1 = int(np.min(xs)), int(np.max(xs)) + 1

    box_h = max(1, y1 - y0)
    box_w = max(1, x1 - x0)
    area_ratio = float((box_h * box_w) / max(h * w, 1))
    if area_ratio < 0.01 or area_ratio > 0.95:
        return None

    pad_y = max(2, int(round(box_h * 0.12)))
    pad_x = max(2, int(round(box_w * 0.12)))
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)

    crop = rgb_image[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    target_size = 28
    crop_h, crop_w = crop.shape[:2]
    if crop_h <= 0 or crop_w <= 0:
        return None

    scale = min(target_size / crop_h, target_size / crop_w)
    new_h = max(1, int(round(crop_h * scale)))
    new_w = max(1, int(round(crop_w * scale)))
    interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    bg_color = _estimate_background_color(rgb_image)
    canvas = np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas.astype(np.float32) / 255.0


def _prepare_classifier_input(rgb_image: np.ndarray, garment_mask: np.ndarray) -> np.ndarray:
    """Build a robust 28x28 RGB classifier input with minimal distribution shift."""
    base_resized = cv2.resize(rgb_image, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    mask_bool = np.asarray(garment_mask > 0, dtype=bool)
    if mask_bool.shape != rgb_image.shape[:2]:
        return base_resized

    fg_ratio = float(np.mean(mask_bool))
    if fg_ratio < 0.008 or fg_ratio > 0.95:
        return base_resized

    bg_color = _estimate_background_color(rgb_image)
    masked_rgb = rgb_image.copy()
    masked_rgb[~mask_bool] = bg_color
    masked_resized = cv2.resize(masked_rgb, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    focus_resized = _build_focus_crop_view(rgb_image=rgb_image, mask_bool=mask_bool)
    if focus_resized is None:
        alpha = 0.12 if fg_ratio < 0.16 else 0.06
        blended = ((1.0 - alpha) * base_resized) + (alpha * masked_resized)
        return np.clip(blended, 0.0, 1.0).astype(np.float32)

    # Increase foreground emphasis when the garment occupies a small image fraction.
    if fg_ratio < 0.06:
        alpha_focus, alpha_mask = 0.56, 0.08
    elif fg_ratio < 0.12:
        alpha_focus, alpha_mask = 0.42, 0.08
    elif fg_ratio < 0.22:
        alpha_focus, alpha_mask = 0.28, 0.06
    else:
        alpha_focus, alpha_mask = 0.12, 0.05

    alpha_total = float(min(0.74, alpha_focus + alpha_mask))
    alpha_base = 1.0 - alpha_total
    blended = (
        (alpha_base * base_resized)
        + (alpha_focus * focus_resized)
        + (alpha_mask * masked_resized)
    )
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def _decode_image_arrays(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build model-ready 28x28 RGB array, source RGB image, and garment mask from bytes."""
    pil_image = _validate_and_open_image(image_bytes).convert("RGB")
    rgb_image = np.asarray(pil_image, dtype=np.uint8)
    garment_mask = _extract_garment_mask(rgb_image)
    rgb_28 = _prepare_classifier_input(rgb_image=rgb_image, garment_mask=garment_mask)
    return rgb_28, rgb_image, garment_mask


def _build_tta_variants(rgb_28: np.ndarray) -> np.ndarray:
    """Create simple test-time augmentation variants for robust classification."""
    base = np.asarray(rgb_28, dtype=np.float32)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("Expected RGB input with shape (28, 28, 3).")

    variants: List[np.ndarray] = [base]
    variants.append(np.fliplr(base))

    center = (14.0, 14.0)
    for angle in (-8.0, 8.0):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(base, matrix, (28, 28), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        variants.append(rotated.astype(np.float32))

    return np.stack(variants, axis=0)


def _get_tta_weights(variant_count: int, base_confidence: Optional[float] = None) -> np.ndarray:
    """Return normalized TTA weights based on configured mode and certainty."""
    if variant_count <= 1:
        return np.ones((1,), dtype=np.float32)

    mode = CLASSIFIER_TTA_MODE
    if mode in {"off", "none", "disabled"}:
        weights = np.zeros((variant_count,), dtype=np.float32)
        weights[0] = 1.0
        return weights

    if mode == "equal":
        return np.full((variant_count,), fill_value=1.0 / variant_count, dtype=np.float32)

    # adaptive (default): skip augmented votes when base prediction is already confident.
    if base_confidence is not None and float(base_confidence) >= CLASSIFIER_TTA_CONFIDENCE_THRESHOLD:
        weights = np.zeros((variant_count,), dtype=np.float32)
        weights[0] = 1.0
        return weights

    base_weight = CLASSIFIER_TTA_BASE_WEIGHT
    residual = max(0.0, 1.0 - base_weight)
    tail_weight = residual / max(variant_count - 1, 1)
    weights = np.full((variant_count,), fill_value=tail_weight, dtype=np.float32)
    weights[0] = base_weight
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return np.full((variant_count,), fill_value=1.0 / variant_count, dtype=np.float32)
    return (weights / denom).astype(np.float32)


def _extract_mask_shape_features(garment_mask: np.ndarray) -> Optional[Dict[str, float]]:
    """Extract silhouette features from a binary garment mask."""
    mask = np.asarray(garment_mask > 0, dtype=bool)
    if mask.size == 0 or int(np.count_nonzero(mask)) < 40:
        return None

    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None

    y0, y1 = int(np.min(ys)), int(np.max(ys)) + 1
    x0, x1 = int(np.min(xs)), int(np.max(xs)) + 1
    crop = mask[y0:y1, x0:x1]
    h, w = crop.shape
    if h < 8 or w < 6:
        return None

    aspect_ratio = float(h / max(w, 1))
    row_density = np.mean(crop, axis=1)

    top_end = max(1, int(round(h * 0.33)))
    mid_start = max(0, int(round(h * 0.33)))
    mid_end = max(mid_start + 1, int(round(h * 0.66)))
    bottom_start = min(h - 1, int(round(h * 0.66)))

    top_density = float(np.mean(row_density[:top_end]))
    mid_density = float(np.mean(row_density[mid_start:mid_end]))
    bottom_density = float(np.mean(row_density[bottom_start:]))

    split = max(1, h // 2)
    upper_density = float(np.mean(crop[:split, :]))
    lower_density = float(np.mean(crop[split:, :])) if split < h else upper_density

    center_width = max(1, int(round(w * 0.20)))
    cx0 = max(0, (w - center_width) // 2)
    cx1 = min(w, cx0 + center_width)
    lower_start = min(h - 1, int(round(h * 0.55)))
    center_lower = crop[lower_start:, cx0:cx1]
    lower_center_gap = 1.0 - float(np.mean(center_lower)) if center_lower.size else 0.0

    return {
        "aspect_ratio": aspect_ratio,
        "top_density": top_density,
        "mid_density": mid_density,
        "bottom_density": bottom_density,
        "upper_density": upper_density,
        "lower_density": lower_density,
        "lower_center_gap": lower_center_gap,
    }


def _trouser_shape_confidence(features: Dict[str, float]) -> float:
    """Return a [0,1] score for how trouser-like a silhouette appears."""
    aspect_term = float(np.clip((features["aspect_ratio"] - 1.05) / 0.95, 0.0, 1.0))
    gap_term = float(np.clip((features["lower_center_gap"] - 0.08) / 0.32, 0.0, 1.0))
    lower_vs_upper = features["lower_density"] - features["upper_density"] + 0.08
    lower_term = float(np.clip(lower_vs_upper / 0.25, 0.0, 1.0))
    top_vs_bottom = features["top_density"] - features["bottom_density"]
    balance_term = float(np.clip((0.12 - top_vs_bottom) / 0.24, 0.0, 1.0))

    score = 0.42 * aspect_term + 0.33 * gap_term + 0.15 * lower_term + 0.10 * balance_term
    return float(np.clip(score, 0.0, 1.0))


def _refine_trouser_vs_pullover(probs: np.ndarray, garment_mask: np.ndarray) -> np.ndarray:
    """Apply a conservative shape-based correction for trouser/pullover confusion."""
    vector = np.asarray(probs, dtype=np.float32).reshape(-1).copy()
    if vector.size != len(CATEGORIES):
        return vector

    try:
        trouser_idx = CATEGORIES.index("Trouser")
        pullover_idx = CATEGORIES.index("Pullover")
    except ValueError:
        return vector

    features = _extract_mask_shape_features(garment_mask)
    if features is None:
        return vector

    trouser_prob = float(vector[trouser_idx])
    pullover_prob = float(vector[pullover_idx])
    top_idx = int(np.argmax(vector))
    trouser_like = _trouser_shape_confidence(features)

    adjusted = False
    close_confusion = (
        top_idx == pullover_idx
        and trouser_prob >= 0.06
        and (pullover_prob - trouser_prob) <= 0.35
    )

    if close_confusion and trouser_like >= 0.56:
        margin = max(0.0, pullover_prob - trouser_prob)
        boost = min(0.24, 0.08 + (trouser_like - 0.56) * 0.28 + max(0.0, 0.16 - margin))
        vector[trouser_idx] += boost
        vector[pullover_idx] = max(0.0, vector[pullover_idx] - boost * 0.92)
        adjusted = True
    elif trouser_prob >= pullover_prob * 0.80 and trouser_like >= 0.72:
        boost = min(0.10, 0.03 + (trouser_like - 0.72) * 0.20)
        vector[trouser_idx] += boost
        vector[pullover_idx] = max(0.0, vector[pullover_idx] - boost * 0.70)
        adjusted = True

    if not adjusted:
        return vector

    denom = float(np.sum(vector))
    if denom <= 0.0:
        return _np_softmax(vector)
    return (vector / denom).astype(np.float32)


def _refine_bag_overprediction(probs: np.ndarray, garment_mask: np.ndarray) -> np.ndarray:
    """Reduce bag over-prediction for clearly elongated garment silhouettes."""
    vector = np.asarray(probs, dtype=np.float32).reshape(-1).copy()
    if vector.size != len(CATEGORIES):
        return vector

    try:
        bag_idx = CATEGORIES.index("Bag")
    except ValueError:
        return vector

    if int(np.argmax(vector)) != bag_idx:
        return vector

    bag_prob = float(vector[bag_idx])
    if bag_prob < 0.12:
        return vector

    features = _extract_mask_shape_features(garment_mask)
    if features is None:
        return vector

    aspect_ratio = float(features["aspect_ratio"])
    target_names = ["Shirt", "T-shirt", "Pullover", "Coat", "Dress", "Trouser"]
    target_indices = [
        CATEGORIES.index(name)
        for name in target_names
        if name in CATEGORIES and CATEGORIES.index(name) != bag_idx
    ]
    if not target_indices:
        return vector

    topwear_best_local = int(np.argmax(vector[target_indices]))
    topwear_best_idx = int(target_indices[topwear_best_local])
    topwear_best_prob = float(vector[topwear_best_idx])
    bag_margin = bag_prob - topwear_best_prob

    # Bags are usually close to square. For near-square masks, only correct when
    # bag is uncertain and a topwear class is very close.
    if aspect_ratio < 1.06:
        if bag_prob > 0.56 or topwear_best_prob < 0.24 or bag_margin > 0.18:
            return vector

        transfer = min(
            0.12,
            0.03 + max(0.0, 0.18 - bag_margin) * 0.40,
            bag_prob - 0.02,
        )
        target_scores = vector[target_indices] + 1e-3
        target_scores[topwear_best_local] += 0.08
    else:
        if bag_prob > 0.92 and aspect_ratio < 1.40:
            return vector

        transfer = min(
            0.28,
            0.08 + (aspect_ratio - 1.06) * 0.16 + max(0.0, 0.76 - bag_prob) * 0.22,
            bag_prob - 0.02,
        )
        target_scores = vector[target_indices] + 1e-3

    target_weights = target_scores / max(float(np.sum(target_scores)), 1e-8)
    transfer = max(0.0, float(transfer))
    if transfer <= 0.0:
        return vector

    vector[bag_idx] -= transfer
    vector[target_indices] += transfer * target_weights

    denom = float(np.sum(vector))
    if denom <= 0.0:
        return _np_softmax(vector)
    return (vector / denom).astype(np.float32)


def _topwear_shape_confidence(features: Dict[str, float]) -> float:
    """Return a [0,1] score for how topwear-like a silhouette appears."""
    aspect_term = float(np.clip((features["aspect_ratio"] - 1.02) / 1.00, 0.0, 1.0))
    top_vs_bottom = features["top_density"] - features["bottom_density"]
    top_term = float(np.clip((top_vs_bottom + 0.02) / 0.26, 0.0, 1.0))
    upper_vs_lower = features["upper_density"] - features["lower_density"]
    upper_term = float(np.clip((upper_vs_lower + 0.03) / 0.24, 0.0, 1.0))
    gap_term = float(np.clip((0.36 - features["lower_center_gap"]) / 0.36, 0.0, 1.0))

    score = 0.36 * aspect_term + 0.34 * top_term + 0.22 * upper_term + 0.08 * gap_term
    return float(np.clip(score, 0.0, 1.0))


def _footwear_shape_confidence(features: Dict[str, float]) -> float:
    """Return a [0,1] score for how footwear-like a silhouette appears."""
    compact_term = float(np.clip((1.45 - features["aspect_ratio"]) / 0.80, 0.0, 1.0))
    bottom_vs_top = features["bottom_density"] - features["top_density"]
    bottom_term = float(np.clip((bottom_vs_top + 0.03) / 0.24, 0.0, 1.0))
    lower_vs_upper = features["lower_density"] - features["upper_density"]
    lower_term = float(np.clip((lower_vs_upper + 0.03) / 0.24, 0.0, 1.0))
    gap_term = float(np.clip((features["lower_center_gap"] - 0.08) / 0.35, 0.0, 1.0))

    score = 0.32 * compact_term + 0.30 * bottom_term + 0.30 * lower_term + 0.08 * gap_term
    return float(np.clip(score, 0.0, 1.0))


def _refine_shoe_overprediction(probs: np.ndarray, garment_mask: np.ndarray) -> np.ndarray:
    """Reduce shoe over-prediction when silhouette shape looks like upper-body wear."""
    vector = np.asarray(probs, dtype=np.float32).reshape(-1).copy()
    if vector.size != len(CATEGORIES):
        return vector

    try:
        shoe_indices = [CATEGORIES.index(name) for name in ("Sandal", "Sneaker", "Ankle boot")]
        topwear_indices = [
            CATEGORIES.index(name)
            for name in ("Shirt", "T-shirt", "Pullover", "Coat", "Dress")
            if name in CATEGORIES
        ]
        shirt_idx = CATEGORIES.index("Shirt")
        tshirt_idx = CATEGORIES.index("T-shirt")
    except ValueError:
        return vector

    if not topwear_indices:
        return vector

    top_idx = int(np.argmax(vector))
    if top_idx not in shoe_indices:
        return vector

    top_shoe_prob = float(vector[top_idx])
    if top_shoe_prob < 0.12:
        return vector

    features = _extract_mask_shape_features(garment_mask)
    if features is None:
        return vector

    topwear_like = _topwear_shape_confidence(features)
    footwear_like = _footwear_shape_confidence(features)
    if topwear_like < 0.52:
        return vector

    topwear_scores = vector[topwear_indices].copy()
    topwear_prior = np.full_like(topwear_scores, fill_value=0.02, dtype=np.float32)

    if shirt_idx in topwear_indices:
        topwear_prior[topwear_indices.index(shirt_idx)] = 0.40
    if tshirt_idx in topwear_indices:
        topwear_prior[topwear_indices.index(tshirt_idx)] = 0.26
    for name, weight in (("Pullover", 0.12), ("Coat", 0.12), ("Dress", 0.08)):
        if name in CATEGORIES:
            idx = CATEGORIES.index(name)
            if idx in topwear_indices:
                topwear_prior[topwear_indices.index(idx)] = weight

    topwear_scores = (0.45 * topwear_scores) + (0.55 * topwear_prior)
    if features["upper_density"] > (features["lower_density"] + 0.03):
        if shirt_idx in topwear_indices:
            topwear_scores[topwear_indices.index(shirt_idx)] += 0.05
        if tshirt_idx in topwear_indices:
            topwear_scores[topwear_indices.index(tshirt_idx)] += 0.04

    target_local_idx = int(np.argmax(topwear_scores))
    target_idx = int(topwear_indices[target_local_idx])
    target_prob = float(vector[target_idx])
    margin = top_shoe_prob - target_prob

    if top_shoe_prob > 0.72 and margin > 0.30:
        return vector
    if (topwear_like - footwear_like) < 0.08 and margin > 0.18:
        return vector

    low_topwear_bonus = 0.10 if target_prob < 0.05 else 0.0
    transfer = min(
        0.34,
        0.06
        + max(0.0, topwear_like - footwear_like) * 0.30
        + max(0.0, 0.24 - margin) * 0.32
        + low_topwear_bonus,
        top_shoe_prob - 0.04,
    )
    transfer = max(0.0, float(transfer))
    if transfer <= 0.0:
        return vector

    vector[top_idx] -= transfer
    vector[target_idx] += transfer * 0.74

    remaining = transfer * 0.26
    other_indices = [idx for idx in topwear_indices if idx != target_idx]
    if remaining > 0.0 and other_indices:
        other_scores = vector[other_indices] + 1e-3
        other_weights = other_scores / max(float(np.sum(other_scores)), 1e-8)
        vector[other_indices] += remaining * other_weights

    # If bag is also highly probable, shift some mass toward topwear priors.
    if "Bag" in CATEGORIES:
        bag_idx = CATEGORIES.index("Bag")
        bag_prob = float(vector[bag_idx])
        if bag_prob >= 0.14 and topwear_like >= 0.54 and (topwear_like - footwear_like) >= 0.12:
            bag_transfer = min(0.18, 0.04 + (topwear_like - footwear_like) * 0.20, bag_prob - 0.03)
            bag_transfer = max(0.0, float(bag_transfer))
            if bag_transfer > 0.0:
                vector[bag_idx] -= bag_transfer
                topwear_weights = topwear_scores / max(float(np.sum(topwear_scores)), 1e-8)
                vector[topwear_indices] += bag_transfer * topwear_weights

    denom = float(np.sum(vector))
    if denom <= 0.0:
        return _np_softmax(vector)
    return (vector / denom).astype(np.float32)


def _final_topwear_guardrail(probs: np.ndarray, garment_mask: np.ndarray) -> np.ndarray:
    """Force-correct low-confidence shoe/bag predictions for clearly topwear-like masks."""
    vector = np.asarray(probs, dtype=np.float32).reshape(-1).copy()
    if vector.size != len(CATEGORIES):
        return vector

    try:
        footwear_indices = [CATEGORIES.index(name) for name in ("Sandal", "Sneaker", "Ankle boot")]
        bag_idx = CATEGORIES.index("Bag")
        topwear_indices = [
            CATEGORIES.index(name)
            for name in ("Shirt", "T-shirt", "Pullover", "Coat", "Dress")
            if name in CATEGORIES
        ]
    except ValueError:
        return vector

    if not topwear_indices:
        return vector

    top_idx = int(np.argmax(vector))
    top_prob = float(vector[top_idx])

    footwear_or_bag_indices = set(footwear_indices + [bag_idx])
    topwear_set = set(topwear_indices)
    if top_idx in footwear_or_bag_indices:
        # Guardrail should only trigger on uncertain shoe/bag outputs.
        if top_prob >= 0.58:
            return vector
        target_mode = "force_topwear"
    elif top_idx in topwear_set:
        # Confidence booster for already-corrected but still uncertain topwear results.
        non_topwear_mass = float(np.sum(vector[list(footwear_or_bag_indices)]))
        if top_prob >= 0.52 or non_topwear_mass < 0.20:
            return vector
        target_mode = "boost_topwear"
    else:
        return vector

    features = _extract_mask_shape_features(garment_mask)
    if features is None:
        return vector

    topwear_like = _topwear_shape_confidence(features)
    footwear_like = _footwear_shape_confidence(features)
    aspect_ratio = float(features["aspect_ratio"])
    upper_advantage = float(features["upper_density"] - features["lower_density"])
    top_advantage = float(features["top_density"] - features["bottom_density"])

    strong_topwear_shape = (
        (topwear_like >= 0.58 and (topwear_like - footwear_like) >= 0.08)
        or (aspect_ratio >= 1.08 and upper_advantage >= 0.06 and top_advantage >= 0.05)
    )
    if not strong_topwear_shape:
        return vector

    # Keep true footwear when mask is clearly shoe-like.
    if footwear_like >= 0.66 and topwear_like <= 0.35:
        return vector

    topwear_prior_map = {
        "Shirt": 0.46,
        "T-shirt": 0.28,
        "Pullover": 0.12,
        "Coat": 0.09,
        "Dress": 0.05,
    }

    topwear_scores = vector[topwear_indices].copy()
    for idx_local, idx_global in enumerate(topwear_indices):
        name = CATEGORIES[idx_global]
        topwear_scores[idx_local] = (0.35 * topwear_scores[idx_local]) + (0.65 * topwear_prior_map.get(name, 0.02))

    target_local = int(np.argmax(topwear_scores))
    target_idx = int(topwear_indices[target_local])

    source_indices = footwear_indices + [bag_idx]
    source_mass = float(np.sum(vector[source_indices]))
    if source_mass <= 0.05:
        return vector

    target_current = float(vector[target_idx])
    evidence = max(0.0, topwear_like - footwear_like)
    desired_confidence = float(np.clip(0.52 + evidence * 0.34, 0.52, 0.72))
    needed_for_target = max(0.0, (desired_confidence - target_current) / 0.78)

    base_transfer = min(0.62, source_mass * 0.78)
    transfer = max(base_transfer, needed_for_target)
    if target_mode == "boost_topwear":
        transfer = min(0.56, transfer)

    transfer = min(transfer, max(0.0, source_mass - 0.03))
    transfer = max(0.0, transfer)
    if transfer <= 0.0:
        return vector

    source_values = vector[source_indices]
    source_weights = source_values / max(float(np.sum(source_values)), 1e-8)
    vector[source_indices] -= transfer * source_weights

    vector[target_idx] += transfer * 0.78
    residual = transfer * 0.22
    other_topwear = [idx for idx in topwear_indices if idx != target_idx]
    if residual > 0.0 and other_topwear:
        other_scores = vector[other_topwear] + 1e-3
        other_weights = other_scores / max(float(np.sum(other_scores)), 1e-8)
        vector[other_topwear] += residual * other_weights

    denom = float(np.sum(vector))
    if denom <= 0.0:
        return _np_softmax(vector)
    return (vector / denom).astype(np.float32)


def _classify_image(rgb_28: np.ndarray, rgb_image: np.ndarray, garment_mask: np.ndarray) -> Dict[str, Any]:
    """Run CNN inference, RGB color extraction, and feature extraction."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model: Optional[FashionMNISTCNN] = model_holder["model"]
    feat_ext: Optional[FeatureExtractor] = model_holder["feature_extractor"]
    color_ext: Optional[ColorExtractor] = model_holder["color_extractor"]
    class_prototypes: Optional[np.ndarray] = model_holder.get("class_prototypes")

    # Prepare input with simple TTA.
    input_4d = _build_tta_variants(rgb_28)

    expected_channels = 3
    if cnn_model is not None:
        expected_channels = int(getattr(cnn_model.features[0], "in_channels", 3))

    if expected_channels == 1:
        model_input = np.mean(input_4d, axis=-1, keepdims=True).astype(np.float32)
    else:
        model_input = input_4d.astype(np.float32)

    tensor_in = torch.from_numpy(
        np.transpose(model_input, (0, 3, 1, 2)).astype(np.float32)
    ).to(device)

    # Classification
    all_scores: Dict[str, float] = {}
    if cnn_model is not None:
        cnn_model.eval()
        with torch.no_grad():
            logits = cnn_model(tensor_in)
            probs_all = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            base_confidence = float(np.max(probs_all[0])) if probs_all.shape[0] > 0 else None
            tta_weights = _get_tta_weights(variant_count=probs_all.shape[0], base_confidence=base_confidence)
            probs = np.sum(probs_all * tta_weights[:, None], axis=0).astype(np.float32)
            probs = probs / max(float(np.sum(probs)), 1e-8)

            # Prototype similarity branch using penultimate embeddings.
            proto_probs: Optional[np.ndarray] = None
            if class_prototypes is not None and isinstance(class_prototypes, np.ndarray):
                embeddings = cnn_model.extract_embeddings(tensor_in).cpu().numpy().astype(np.float32)
                query_embedding = np.sum(embeddings * tta_weights[:, None], axis=0)
                query_norm = float(np.linalg.norm(query_embedding))
                if query_norm > 0.0:
                    query_embedding = query_embedding / query_norm
                    similarities = np.dot(class_prototypes, query_embedding)
                    # Scaling keeps prototype branch discriminative while still simple.
                    proto_probs = _np_softmax(similarities * 10.0)

            if proto_probs is not None and proto_probs.shape[0] == probs.shape[0]:
                cnn_conf = float(np.max(probs))
                # Trust prototypes more when CNN confidence is low.
                if cnn_conf < 0.35:
                    blend_alpha = 0.50
                elif cnn_conf < 0.50:
                    blend_alpha = 0.60
                else:
                    blend_alpha = 0.72

                probs = (blend_alpha * probs + (1.0 - blend_alpha) * proto_probs).astype(np.float32)
                probs = probs / max(float(np.sum(probs)), 1e-8)

            probs = _refine_trouser_vs_pullover(probs=probs, garment_mask=garment_mask)
            probs = _refine_bag_overprediction(probs=probs, garment_mask=garment_mask)
            probs = _refine_shoe_overprediction(probs=probs, garment_mask=garment_mask)
            probs = _final_topwear_guardrail(probs=probs, garment_mask=garment_mask)

        for i, cat in enumerate(CATEGORIES):
            all_scores[cat] = round(float(probs[i]), 4)
        category_idx = int(np.argmax(probs))
        confidence = float(probs[category_idx])
    else:
        # fallback – heuristic
        category_idx = min(int(np.mean(rgb_28) * 10), 9)
        confidence = 0.0
        for i, cat in enumerate(CATEGORIES):
            all_scores[cat] = 0.1 if i == category_idx else 0.0

    category = CATEGORIES[category_idx]

    # Color extraction from RGB image.
    colors: List[Dict[str, Any]] = []
    if color_ext is not None:
        extracted = color_ext.extract_dominant_colors_from_array(rgb_image, n_colors=3, mask=garment_mask)
        for color_info in extracted:
            rgb = color_info.get("rgb", (127, 127, 127))
            rgb_values = [int(rgb[0]), int(rgb[1]), int(rgb[2])]
            colors.append(
                {
                    "rgb": rgb_values,
                    "percentage": round(float(color_info.get("percentage", 0.0)), 2),
                    "name": str(color_info.get("name", "gray")).lower(),
                }
            )

    if not colors:
        mean_intensity = float(np.mean(rgb_28))
        color_name = _intensity_to_color(mean_intensity)
        rgb_approx = int(mean_intensity * 255)
        colors = [{"rgb": [rgb_approx, rgb_approx, rgb_approx], "percentage": 100.0, "name": color_name}]

    color_name = str(colors[0].get("name", "gray")).lower()

    # Feature vector
    features: List[float] = []
    if feat_ext is not None and feat_ext.feature_model is not None:
        fv = feat_ext.extract_single_image_features(input_4d[0])
        if fv.size > 0:
            features = fv.tolist()

    item_id = str(uuid.uuid4())

    return {
        "category": category,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
        "colors": colors,
        "color_name": color_name,
        "item_id": item_id,
        "features": features,
    }


def _intensity_to_color(mean: float) -> str:
    """Map grayscale mean intensity to a color name."""
    if mean < 0.10:
        return "black"
    if mean < 0.20:
        return "gray"
    if mean < 0.30:
        return "blue"
    if mean < 0.40:
        return "purple"
    if mean < 0.50:
        return "green"
    if mean < 0.60:
        return "red"
    if mean < 0.70:
        return "orange"
    if mean < 0.80:
        return "yellow"
    if mean < 0.90:
        return "pink"
    return "white"


def _load_samples() -> List[Dict[str, Any]]:
    """Load 20 pre-selected Fashion-MNIST test images as base64 strings."""
    processed_path = PROJECT_ROOT / "data" / "processed" / "fashion_mnist_processed.pkl"
    if not processed_path.exists():
        return []

    import pickle
    with processed_path.open("rb") as f:
        data = pickle.load(f)

    x_test = np.asarray(data["x_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.int64)

    if x_test.shape[0] == 0:
        return []

    # Pick 2 items per category (20 total) deterministically
    rng = np.random.default_rng(42)
    samples: List[Dict[str, Any]] = []
    for cat_idx in range(10):
        indices = np.where(y_test == cat_idx)[0]
        if len(indices) == 0:
            continue
        chosen = rng.choice(indices, size=min(2, len(indices)), replace=False)
        for idx in chosen:
            img = np.asarray(x_test[idx], dtype=np.float32)

            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))

            if np.max(img) > 1.0:
                img = img / 255.0

            img = np.clip(img, 0.0, 1.0)
            if img.ndim == 2:
                img_uint8 = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8, mode="L")
            elif img.ndim == 3 and img.shape[-1] == 1:
                img_uint8 = (img[:, :, 0] * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8, mode="L")
            elif img.ndim == 3 and img.shape[-1] == 3:
                img_uint8 = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8, mode="RGB")
            else:
                continue

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            samples.append({
                "id": f"sample-{cat_idx}-{int(idx)}",
                "base64": b64,
                "label": CATEGORIES[cat_idx],
                "label_index": int(cat_idx),
            })

    return samples


def _np_softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 1D arrays."""
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return vector
    shifted = vector - float(np.max(vector))
    exp_values = np.exp(shifted)
    denom = float(np.sum(exp_values))
    if denom <= 0.0:
        return np.full_like(vector, fill_value=1.0 / max(vector.size, 1), dtype=np.float32)
    return (exp_values / denom).astype(np.float32)


def _build_class_prototypes(
    cnn_model: FashionMNISTCNN,
    device: Any,
    samples_per_class: int = 300,
) -> Tuple[Optional[np.ndarray], int]:
    """Build per-class embedding prototypes from processed Fashion-MNIST samples."""
    processed_path = PROJECT_ROOT / "data" / "processed" / "fashion_mnist_processed.pkl"
    if not processed_path.exists():
        logger.warning("Processed dataset not found at %s. Skipping class prototypes.", processed_path)
        return None, 0

    import pickle
    import torch

    try:
        with processed_path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception:
        logger.exception("Failed to load processed dataset for prototypes")
        return None, 0

    x_train = np.asarray(data.get("x_train", []), dtype=np.float32)
    y_train = np.asarray(data.get("y_train", []), dtype=np.int64)

    if x_train.size == 0 or y_train.size == 0:
        logger.warning("Processed dataset is empty. Skipping class prototypes.")
        return None, 0

    if x_train.ndim != 4:
        logger.warning("Unexpected x_train shape=%s. Skipping class prototypes.", x_train.shape)
        return None, 0

    rng = np.random.default_rng(42)
    selected_indices: List[int] = []
    for class_idx in range(len(CATEGORIES)):
        indices = np.where(y_train == class_idx)[0]
        if len(indices) == 0:
            continue
        k = min(int(samples_per_class), len(indices))
        picked = rng.choice(indices, size=k, replace=False)
        selected_indices.extend(int(idx) for idx in picked)

    if not selected_indices:
        logger.warning("No samples selected for prototypes.")
        return None, 0

    selected_indices = sorted(selected_indices)
    selected_x = x_train[selected_indices]
    selected_y = y_train[selected_indices]

    # Convert NHWC -> NCHW
    if selected_x.shape[-1] in (1, 3):
        selected_x = np.transpose(selected_x, (0, 3, 1, 2))
    elif selected_x.shape[1] not in (1, 3):
        logger.warning("Unexpected channel layout for x_train=%s. Skipping class prototypes.", selected_x.shape)
        return None, 0

    if np.max(selected_x) > 1.0:
        selected_x = selected_x / 255.0

    expected_channels = int(getattr(cnn_model.features[0], "in_channels", 3))
    if selected_x.shape[1] != expected_channels:
        if selected_x.shape[1] == 1 and expected_channels == 3:
            selected_x = np.repeat(selected_x, repeats=3, axis=1)
        elif selected_x.shape[1] == 3 and expected_channels == 1:
            selected_x = np.mean(selected_x, axis=1, keepdims=True)
        else:
            logger.warning(
                "Channel mismatch between training data (%s) and model (%s).",
                selected_x.shape[1],
                expected_channels,
            )
            return None, 0

    tensor_x = torch.from_numpy(selected_x.astype(np.float32)).to(device)

    cnn_model.eval()
    with torch.no_grad():
        embeddings = cnn_model.extract_embeddings(tensor_x).cpu().numpy().astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    embeddings = embeddings / norms

    prototypes: List[np.ndarray] = []
    for class_idx in range(len(CATEGORIES)):
        class_embeddings = embeddings[selected_y == class_idx]
        if class_embeddings.size == 0:
            prototypes.append(np.zeros((embeddings.shape[1],), dtype=np.float32))
            continue

        prototype = np.mean(class_embeddings, axis=0)
        proto_norm = float(np.linalg.norm(prototype))
        if proto_norm > 0.0:
            prototype = prototype / proto_norm
        prototypes.append(prototype.astype(np.float32))

    prototype_matrix = np.stack(prototypes, axis=0).astype(np.float32)
    return prototype_matrix, int(len(selected_indices))


def _allow_request(client_id: str) -> Tuple[bool, int, int]:
    """Token-bucket style request limiter per client within a rolling window."""
    if not RATE_LIMIT_ENABLED:
        return True, RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS

    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    with rate_limit_lock:
        bucket = rate_limit_buckets[client_id]
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = int(max(1, RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])))
            return False, 0, retry_after

        bucket.append(now)
        remaining = max(0, RATE_LIMIT_MAX_REQUESTS - len(bucket))

    return True, remaining, RATE_LIMIT_WINDOW_SECONDS


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models once on startup."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODEL_PATH

    if API_DISABLE_MODEL_LOAD:
        logger.warning("Model loading disabled by API_DISABLE_MODEL_LOAD")
    elif model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            num_classes = int(checkpoint.get("num_classes", 10))
            raw_input_shape = checkpoint.get("input_shape", (28, 28, 3))
            if isinstance(raw_input_shape, (list, tuple)) and len(raw_input_shape) == 3:
                input_shape = tuple(int(value) for value in raw_input_shape)
            else:
                input_shape = (28, 28, 3)
        else:
            state_dict = checkpoint
            num_classes = 10
            input_shape = (28, 28, 3)

        cnn = FashionMNISTCNN(input_shape=input_shape, num_classes=num_classes).to(device)
        cnn.load_state_dict(state_dict)
        cnn.eval()
        model_holder["model"] = cnn
        model_holder["model_loaded"] = True
        logger.info("CNN model loaded from %s", model_path)

        class_prototypes, sample_count = _build_class_prototypes(cnn_model=cnn, device=device)
        model_holder["class_prototypes"] = class_prototypes
        model_holder["prototype_sample_count"] = sample_count
        if class_prototypes is not None:
            logger.info("Loaded class prototypes using %s samples", sample_count)
        else:
            logger.warning("Class prototypes unavailable. Using CNN-only prediction.")
    else:
        logger.warning("No trained model found at %s. Classification will use heuristics.", model_path)

    model_holder["feature_extractor"] = FeatureExtractor(model_path=str(model_path))
    model_holder["color_extractor"] = ColorExtractor()
    model_holder["recommender"] = OutfitRecommender()

    # Pre-load sample images
    model_holder["samples"] = _load_samples()
    logger.info("Loaded %s sample images", len(model_holder["samples"]))

    yield  # app runs

    # Cleanup
    model_holder.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Wardrobe API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply simple per-IP throttling to API routes."""
    if request.url.path.startswith("/api/"):
        client_id = request.client.host if request.client else "unknown"
        allowed, remaining, retry_after = _allow_request(client_id)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(RATE_LIMIT_MAX_REQUESTS),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_MAX_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": "Validation failed", "errors": exc.errors()})


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    prototypes = model_holder.get("class_prototypes")
    return {
        "status": "ok",
        "model_loaded": model_holder.get("model_loaded", False),
        "prototype_fusion": prototypes is not None,
        "prototype_sample_count": int(model_holder.get("prototype_sample_count", 0)),
    }


@app.get("/api/categories")
async def get_categories():
    return {"categories": CATEGORIES}


@app.get("/api/samples")
async def get_samples():
    samples = model_holder.get("samples") or []
    return {"samples": samples, "count": len(samples)}


@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    """Classify an uploaded clothing image."""
    try:
        image_bytes = await _read_upload_image(file)
        if model_holder.get("model") is None:
            raise HTTPException(
                status_code=503,
                detail="Classifier model is unavailable. Check /api/health and server startup logs.",
            )
        rgb_28, rgb_image, garment_mask = _decode_image_arrays(image_bytes)
        result = _classify_image(rgb_28, rgb_image, garment_mask)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail="Classification failed") from exc


@app.post("/api/classify-sample")
async def classify_sample(body: ClassifySampleRequest):
    """Classify a sample image by its base64 data."""
    try:
        image_bytes = base64.b64decode(body.base64, validate=True)
        if model_holder.get("model") is None:
            raise HTTPException(
                status_code=503,
                detail="Classifier model is unavailable. Check /api/health and server startup logs.",
            )
        rgb_28, rgb_image, garment_mask = _decode_image_arrays(image_bytes)
        result = _classify_image(rgb_28, rgb_image, garment_mask)
        return result
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 payload") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Sample classification failed")
        raise HTTPException(status_code=500, detail="Classification failed") from exc


@app.post("/api/wardrobe/add")
async def add_to_wardrobe(req: WardrobeAddRequest):
    """Add a classified item to the in-memory wardrobe."""
    features = [float(value) for value in req.features[:512]]
    item = {
        "item_id": req.item_id,
        "category": req.category,
        "color": req.color.lower(),
        "confidence": round(float(req.confidence), 4),
        "label": req.label or req.category,
        "features": features,
        "wear_count": int(req.wear_count),
        "last_worn_days_ago": req.last_worn_days_ago,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    wardrobe_store[req.item_id] = item
    return {"item": item, "wardrobe_size": len(wardrobe_store)}


@app.get("/api/wardrobe")
async def get_wardrobe():
    """Return all wardrobe items."""
    items = sorted(list(wardrobe_store.values()), key=lambda entry: entry.get("created_at", ""), reverse=True)
    return {"items": items, "count": len(items)}


@app.delete("/api/wardrobe/{item_id}", status_code=204)
async def delete_from_wardrobe(item_id: str):
    """Remove an item from the wardrobe."""
    if item_id not in wardrobe_store:
        raise HTTPException(status_code=404, detail="Item not found in wardrobe")
    del wardrobe_store[item_id]
    return None


def _serialize_item(slot: str, item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Serialize a wardrobe item into API response shape."""
    if not item:
        return None

    item_id = str(item.get("item_id") or item.get("id") or "").strip()
    return {
        "slot": slot,
        "item_id": item_id,
        "wardrobe_item_id": item_id,
        "category": str(item.get("category", "")),
        "color": str(item.get("color", "")),
    }


EXPLANATION_SECTION_KEYS: List[str] = [
    "style",
    "why_it_works_today",
    "color_in_simple_words",
    "how_this_works",
    "simple_tip",
    "confidence_note",
]
EXPLANATION_SECTION_TITLES: Dict[str, str] = {
    "style": "Style",
    "why_it_works_today": "Why it works today",
    "color_in_simple_words": "Color in simple words",
    "how_this_works": "How this works",
    "simple_tip": "Simple tip",
    "confidence_note": "Confidence note",
}


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _looks_like_technical_jargon(text: str) -> bool:
    lowered = _compact_text(text).lower()
    if not lowered:
        return False

    blocked_markers = [
        "feature score",
        "rotation score",
        "novelty score",
        "color_score",
        "occasion_score",
        "weather_score",
        "weight",
        "embedding",
        "logit",
    ]
    return any(marker in lowered for marker in blocked_markers)


def _score_quality_label(score: float) -> str:
    value = float(np.clip(score, 0.0, 1.0))
    if value >= 0.78:
        return "strong"
    if value >= 0.58:
        return "good"
    if value >= 0.42:
        return "mixed"
    return "limited"


def _confidence_level_from_score(score: float) -> str:
    value = float(np.clip(score, 0.0, 1.0))
    if value >= 0.76:
        return "high"
    if value >= 0.57:
        return "medium"
    return "low"


def _build_key_factors(outfit_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    factor_map = [
        ("occasion_score", "Occasion fit"),
        ("weather_score", "Weather fit"),
        ("color_score", "Color harmony"),
        ("feature_score", "Style cohesion"),
        ("rotation_score", "Wardrobe rotation"),
        ("novelty_score", "Freshness"),
    ]

    factors: List[Dict[str, Any]] = []
    for key, label in factor_map:
        score = float(np.clip(float(outfit_payload.get(key, 0.0)), 0.0, 1.0))
        if score >= 0.72:
            impact = "strong"
        elif score >= 0.52:
            impact = "moderate"
        else:
            impact = "weak"

        factors.append(
            {
                "factor": label,
                "score": round(score, 3),
                "impact": impact,
            }
        )

    return factors


def _normalize_text_list(
    value: Any,
    fallback: Optional[List[str]] = None,
    limit: int = 3,
) -> List[str]:
    """Normalize string/list content into concise sentence bullets."""
    output: List[str] = []

    if isinstance(value, list):
        for entry in value:
            candidate = _compact_text(entry)
            if not candidate or _looks_like_technical_jargon(candidate):
                continue
            output.append(candidate)
    elif isinstance(value, str):
        fragments = value.replace("\n", ".").split(".")
        for fragment in fragments:
            candidate = _compact_text(fragment)
            if not candidate or _looks_like_technical_jargon(candidate):
                continue
            output.append(candidate)

    deduped: List[str] = []
    seen: Set[str] = set()
    for candidate in output:
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(candidate)
        if len(deduped) >= max(1, int(limit)):
            break

    if deduped:
        return deduped

    if fallback:
        return _normalize_text_list(fallback, fallback=None, limit=limit)

    return []


def _build_curation_payload(
    outfit_payload: Dict[str, Any],
    req: RecommendRequest,
    key_factors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a curated, user-facing outfit narrative for each recommendation."""
    items = outfit_payload.get("items", [])
    top_item = next((entry for entry in items if entry.get("slot") == "top"), {})
    bottom_item = next((entry for entry in items if entry.get("slot") == "bottom"), {})
    shoe_item = next((entry for entry in items if entry.get("slot") == "shoes"), None)

    top_desc = f"{top_item.get('color', 'gray')} {top_item.get('category', 'top')}"
    bottom_desc = f"{bottom_item.get('color', 'gray')} {bottom_item.get('category', 'bottom')}"
    shoe_desc = ""
    if shoe_item:
        shoe_desc = f" with {shoe_item.get('color', 'gray')} {shoe_item.get('category', 'shoes')}"

    profile = str(outfit_payload.get("profile", "balanced")).strip().lower()
    color_strategy = str(outfit_payload.get("color_strategy", req.color_strategy)).strip().lower()

    title_prefix = {
        "safe": "Curated classic look",
        "bold": "Curated statement look",
        "balanced": "Curated everyday look",
    }.get(profile, "Curated outfit")

    vibe = {
        "safe": "Polished and dependable",
        "bold": "Confident and expressive",
        "balanced": "Versatile and put together",
    }.get(profile, "Clean and wearable")

    when_to_wear = {
        ("formal", "cold"): "Best for formal plans where you want structure and extra warmth.",
        ("formal", "hot"): "Best for formal plans where you still need breathable comfort.",
        ("formal", "rainy"): "Best for formal plans when weather is wet and unpredictable.",
        ("formal", "mild"): "Best for formal plans where you want a polished but easy finish.",
        ("casual", "cold"): "Best for everyday errands or meetups in cooler weather.",
        ("casual", "hot"): "Best for casual days when staying cool matters.",
        ("casual", "rainy"): "Best for casual plans when rain support is useful.",
        ("casual", "mild"): "Best for everyday wear across most daytime plans.",
        ("sport", "cold"): "Best for active moments when you still need warmth.",
        ("sport", "hot"): "Best for active moments where breathability is key.",
        ("sport", "rainy"): "Best for active moments when the weather may turn wet.",
        ("sport", "mild"): "Best for active moments in comfortable weather.",
    }.get((req.occasion, req.weather), "Best for your current setting and day context.")

    reason_templates = {
        "Occasion fit": "The outfit aligns well with the setting and expected dress code.",
        "Weather fit": "The pieces support comfort for the current weather conditions.",
        "Color harmony": "The color mix reads intentional and avoids visual clash.",
        "Style cohesion": "The items feel stylistically connected rather than random.",
        "Wardrobe rotation": "It helps rotate underused pieces so your wardrobe stays fresh.",
        "Freshness": "It introduces enough variety to keep your looks from feeling repetitive.",
    }

    sorted_factors = sorted(
        [entry for entry in key_factors if isinstance(entry, dict)],
        key=lambda entry: float(entry.get("score", 0.0)),
        reverse=True,
    )
    focus_factors = sorted_factors[:2] if sorted_factors else []

    why_this_look: List[str] = []
    for factor in focus_factors:
        factor_name = str(factor.get("factor", ""))
        sentence = reason_templates.get(factor_name)
        if sentence:
            why_this_look.append(sentence)

    if len(why_this_look) < 2:
        why_this_look.append(f"The base pairing of {top_desc} and {bottom_desc} is easy to style.")
    if len(why_this_look) < 2:
        why_this_look.append("This recommendation balances fit, comfort, and repeat value.")
    why_this_look = why_this_look[:2]

    second_step = {
        "monochrome": "Keep accessories in the same color family to maintain a minimal line.",
        "analogous": "Use one accessory tone from the same palette to keep the blend natural.",
        "complementary": "Let one contrasting piece lead and keep the rest neutral.",
        "neutral_plus_one": "Use the accent color once and keep remaining pieces neutral.",
        "high_contrast": "Preserve contrast with one light element and one dark anchor.",
        "auto": "Keep one piece simple so the overall look stays balanced.",
    }.get(color_strategy, "Keep one piece simple so the overall look stays balanced.")

    styling_steps = [
        f"Start with {top_desc} and {bottom_desc}{shoe_desc}.",
        second_step,
    ]

    lowest_factor = min(
        [entry for entry in key_factors if isinstance(entry, dict)],
        key=lambda entry: float(entry.get("score", 0.0)),
        default=None,
    )
    if lowest_factor and float(lowest_factor.get("score", 0.0)) < 0.5:
        low_name = str(lowest_factor.get("factor", "overall balance")).lower()
        tradeoff_note = f"Main trade-off: {low_name}. A small item swap can improve it if needed."
    else:
        tradeoff_note = "No major trade-off detected; this look is well balanced for the current filters."

    return {
        "curation_title": f"{title_prefix} for {req.occasion}",
        "vibe": vibe,
        "when_to_wear": when_to_wear,
        "why_this_look": why_this_look,
        "styling_steps": styling_steps,
        "tradeoff_note": tradeoff_note,
    }


def _build_rule_explanation_payload(
    outfit_payload: Dict[str, Any],
    req: RecommendRequest,
) -> Dict[str, Any]:
    """Produce deterministic user-facing explanation content and metadata."""
    items = outfit_payload.get("items", [])
    top_item = next((entry for entry in items if entry.get("slot") == "top"), {})
    bottom_item = next((entry for entry in items if entry.get("slot") == "bottom"), {})
    shoe_item = next((entry for entry in items if entry.get("slot") == "shoes"), None)

    top_desc = f"{top_item.get('color', 'gray')} {top_item.get('category', 'top')}"
    bottom_desc = f"{bottom_item.get('color', 'gray')} {bottom_item.get('category', 'bottom')}"
    shoe_desc = ""
    if shoe_item:
        shoe_desc = f", plus {shoe_item.get('color', 'gray')} {shoe_item.get('category', 'shoes')}"

    profile = str(outfit_payload.get("profile", "balanced")).strip().lower()
    if profile == "bold":
        style_prefix = "A bold, expressive outfit"
    elif profile == "safe":
        style_prefix = "A clean, low-risk outfit"
    else:
        style_prefix = "A balanced, versatile outfit"

    weather_note = {
        "cold": "for cooler weather comfort",
        "hot": "for warm weather comfort",
        "rainy": "with practical rainy-day support",
        "mild": "for balanced mild-weather comfort",
    }.get(req.weather, "for your day")

    color_strategy = str(outfit_payload.get("color_strategy", req.color_strategy)).strip().lower()
    color_note = {
        "monochrome": "Most colors stay close, so the look feels calm and polished.",
        "analogous": "The colors sit near each other, so the outfit feels smooth and natural.",
        "complementary": "The colors contrast on purpose, so each piece stands out without clashing.",
        "neutral_plus_one": "The base stays neutral and one color carries the personality.",
        "high_contrast": "The light-dark contrast gives a sharp, high-clarity look.",
        "auto": "Colors are selected to avoid clashes and keep the outfit easy to wear.",
    }.get(color_strategy, "Colors are selected to avoid clashes and keep the outfit easy to wear.")

    occasion_score = float(outfit_payload.get("occasion_score", 0.0))
    weather_score = float(outfit_payload.get("weather_score", 0.0))
    novelty_score = float(outfit_payload.get("novelty_score", 0.0))
    rotation_score = float(outfit_payload.get("rotation_score", 0.0))
    overall_score = float(outfit_payload.get("score", 0.0))

    tips: List[str] = []
    if weather_score < 0.5:
        tips.append("If conditions shift, add a layer or swap shoes to improve weather comfort.")
    if req.occasion == "formal" and occasion_score < 0.6:
        tips.append("For a sharper formal finish, switch to darker shoes or a more structured top.")
    if novelty_score > (rotation_score + 0.2):
        tips.append("If this feels too bold, keep one statement piece and switch the other piece to a neutral color.")
    elif rotation_score > (novelty_score + 0.2):
        tips.append("If you want more personality, keep the base and add one accent color.")
    if not tips:
        tips.append("Use this as a base look, then tweak one piece for mood or venue.")

    confidence_level = _confidence_level_from_score(overall_score)
    confidence_note = {
        "high": "High confidence: this is a strong match for your current settings.",
        "medium": "Medium confidence: this is a solid option with a small trade-off.",
        "low": "Lower confidence: usable, but you may get better results by adjusting filters or adding more wardrobe variety.",
    }[confidence_level]

    key_factors = _build_key_factors(outfit_payload)

    sections_by_key = {
        "style": f"{style_prefix} built from {top_desc} and {bottom_desc}{shoe_desc}.",
        "why_it_works_today": (
            f"It matches a {req.occasion} plan with {_score_quality_label(occasion_score)} occasion fit "
            f"and {_score_quality_label(weather_score)} weather fit {weather_note}."
        ),
        "color_in_simple_words": color_note,
        "how_this_works": (
            "The app checks occasion and weather first, then tests color harmony and outfit cohesion, "
            "and finally balances variety with repeat control."
        ),
        "simple_tip": tips[0],
        "confidence_note": confidence_note,
    }

    sections = [
        {
            "key": key,
            "title": EXPLANATION_SECTION_TITLES[key],
            "body": sections_by_key[key],
        }
        for key in EXPLANATION_SECTION_KEYS
    ]

    return {
        "version": "v2",
        "summary": f"{style_prefix} for {req.occasion} days, tuned {weather_note}.",
        "confidence_level": confidence_level,
        "key_factors": key_factors,
        "sections": sections,
        "curation": _build_curation_payload(outfit_payload, req, key_factors),
    }


def _format_explanation_text(explanation_payload: Dict[str, Any]) -> str:
    """Render structured explanation payload as sectioned plain text."""
    sections = explanation_payload.get("sections")
    if not isinstance(sections, list):
        summary = _compact_text(explanation_payload.get("summary"))
        return summary

    lines: List[str] = []
    for section in sections:
        if not isinstance(section, dict):
            continue

        title = _compact_text(section.get("title"))
        body = _compact_text(section.get("body"))
        if not title or not body:
            continue
        lines.append(f"{title}: {body}")

    if lines:
        return " ".join(lines)

    return _compact_text(explanation_payload.get("summary"))


def _build_rule_reasoning(outfit_payload: Dict[str, Any], req: RecommendRequest) -> str:
    """Backward-compatible text explanation output."""
    return _format_explanation_text(_build_rule_explanation_payload(outfit_payload, req))


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from model output."""
    cleaned = _compact_text(text)
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None

    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


def _normalize_llm_explanation_payload(
    llm_payload: Dict[str, Any],
    fallback_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize model output into the stable explanation schema."""
    fallback_sections = {
        str(section.get("key", "")): _compact_text(section.get("body"))
        for section in fallback_payload.get("sections", [])
        if isinstance(section, dict)
    }

    normalized_sections: List[Dict[str, Any]] = []
    for key in EXPLANATION_SECTION_KEYS:
        candidate = _compact_text(llm_payload.get(key))
        if not candidate or _looks_like_technical_jargon(candidate):
            candidate = fallback_sections.get(key, "")

        normalized_sections.append(
            {
                "key": key,
                "title": EXPLANATION_SECTION_TITLES[key],
                "body": candidate,
            }
        )

    summary = _compact_text(llm_payload.get("summary"))
    if not summary or _looks_like_technical_jargon(summary):
        summary = _compact_text(fallback_payload.get("summary"))

    fallback_curation = fallback_payload.get("curation") if isinstance(fallback_payload, dict) else {}
    if not isinstance(fallback_curation, dict):
        fallback_curation = {}

    def _safe_curation_text(key: str) -> str:
        candidate = _compact_text(llm_payload.get(key))
        if not candidate or _looks_like_technical_jargon(candidate):
            return _compact_text(fallback_curation.get(key))
        return candidate

    curation_payload = {
        "curation_title": _safe_curation_text("curation_title"),
        "vibe": _safe_curation_text("vibe"),
        "when_to_wear": _safe_curation_text("when_to_wear"),
        "why_this_look": _normalize_text_list(
            llm_payload.get("why_this_look"),
            fallback=_normalize_text_list(fallback_curation.get("why_this_look"), fallback=[], limit=2),
            limit=2,
        ),
        "styling_steps": _normalize_text_list(
            llm_payload.get("styling_steps"),
            fallback=_normalize_text_list(fallback_curation.get("styling_steps"), fallback=[], limit=2),
            limit=2,
        ),
        "tradeoff_note": _safe_curation_text("tradeoff_note"),
    }

    normalized_payload = {
        "version": "v2",
        "summary": summary,
        "confidence_level": str(fallback_payload.get("confidence_level", "medium")),
        "key_factors": fallback_payload.get("key_factors", []),
        "sections": normalized_sections,
        "curation": curation_payload,
    }
    return normalized_payload


def _extract_openrouter_text(raw_content: Any) -> str:
    """Extract message content from OpenRouter chat completion payload."""
    if isinstance(raw_content, str):
        return raw_content.strip()

    if isinstance(raw_content, list):
        fragments: List[str] = []
        for block in raw_content:
            if not isinstance(block, dict):
                continue
            candidate = block.get("text") or block.get("content")
            if isinstance(candidate, str) and candidate.strip():
                fragments.append(candidate.strip())
        return " ".join(fragments).strip()

    return ""


async def _generate_openrouter_reasoning(
    outfit_payload: Dict[str, Any],
    req: RecommendRequest,
) -> Tuple[Dict[str, Any], str]:
    """Generate structured explanation payload via OpenRouter with deterministic fallback."""
    fallback_payload = _build_rule_explanation_payload(outfit_payload, req)

    if req.explainability != "llm":
        return fallback_payload, "rule"

    if not OPENROUTER_API_KEY:
        return fallback_payload, "rule"

    items = outfit_payload.get("items", [])
    compact_items = [
        {
            "slot": str(entry.get("slot", "")),
            "category": str(entry.get("category", "")),
            "color": str(entry.get("color", "")),
        }
        for entry in items
        if isinstance(entry, dict)
    ]

    llm_context = {
        "occasion": req.occasion,
        "weather": req.weather,
        "goal_mode": outfit_payload.get("goal_mode", req.goal_mode),
        "color_strategy": outfit_payload.get("color_strategy", req.color_strategy),
        "profile": outfit_payload.get("profile", "balanced"),
        "items": compact_items,
        "confidence_level": fallback_payload.get("confidence_level", "medium"),
        "key_factors": fallback_payload.get("key_factors", []),
        "curation_seed": fallback_payload.get("curation", {}),
    }

    user_prompt = (
        "Generate one beginner-friendly explanation for the outfit below. "
        "Keep content practical and grounded in the provided facts only.\n"
        "Outfit context JSON:\n"
        f"{json.dumps(llm_context, ensure_ascii=True)}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME

    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": 0.2,
        "max_tokens": 260,
        "messages": [
            {
                "role": "system",
                "content": OPENROUTER_EXPLAINABILITY_SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        timeout = httpx.Timeout(timeout=OPENROUTER_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()

        choices = body.get("choices") if isinstance(body, dict) else None
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = _extract_openrouter_text(message.get("content"))
            parsed = _extract_json_object(content)
            if parsed:
                return _normalize_llm_explanation_payload(parsed, fallback_payload), "llm"
    except Exception as exc:
        logger.warning("OpenRouter explainability fallback activated: %s", exc)

    return fallback_payload, "rule"


def _serialize_recommendation_outfit(
    outfit: Dict[str, Any],
    rank: int,
    req: RecommendRequest,
    pack_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Serialize recommendation output with expanded score metadata."""
    top_item = outfit.get("top", {})
    bottom_item = outfit.get("bottom", {})
    shoe_item = outfit.get("shoes") or {}

    items_list: List[Dict[str, Any]] = []
    top_payload = _serialize_item("top", top_item)
    bottom_payload = _serialize_item("bottom", bottom_item)
    shoe_payload = _serialize_item("shoes", shoe_item)

    if top_payload:
        items_list.append(top_payload)
    if bottom_payload:
        items_list.append(bottom_payload)
    if shoe_payload:
        items_list.append(shoe_payload)

    payload = {
        "rank": rank,
        "score": round(float(outfit.get("score", 0.0)), 4),
        "color_score": round(float(outfit.get("color_score", 0.0)), 4),
        "occasion_score": round(float(outfit.get("occasion_score", 0.0)), 4),
        "weather_score": round(float(outfit.get("weather_score", 0.0)), 4),
        "feature_score": round(float(outfit.get("feature_score", 0.0)), 4),
        "novelty_score": round(float(outfit.get("novelty_score", 0.0)), 4),
        "rotation_score": round(float(outfit.get("rotation_score", 0.0)), 4),
        "boldness": round(float(outfit.get("boldness", 0.0)), 4),
        "profile": str(outfit.get("profile", "balanced")),
        "goal_mode": str(outfit.get("goal_mode", req.goal_mode)),
        "color_strategy": str(outfit.get("color_strategy", req.color_strategy)),
        "occasion": req.occasion,
        "weather": req.weather,
        "items": items_list,
        "score_breakdown": outfit.get("score_breakdown", {}),
        "explanation_source": "rule",
        "explanation": {},
        "reasoning": "",
    }

    payload["recommendation_id"] = _build_recommendation_id(payload, req)

    if pack_label is not None:
        payload["recommendation_pack_label"] = pack_label

    explanation_payload = _build_rule_explanation_payload(payload, req)
    payload["explanation"] = explanation_payload
    payload["reasoning"] = _format_explanation_text(explanation_payload)
    return payload


def _build_closet_gap_insights(wardrobe_items: List[Dict[str, Any]], req: RecommendRequest) -> List[str]:
    """Generate actionable closet gap insights from the current wardrobe."""
    if not wardrobe_items:
        return []

    categories = [str(item.get("category", "")) for item in wardrobe_items]
    colors = [str(item.get("color", "gray")).lower() for item in wardrobe_items]

    insights: List[str] = []

    top_count = sum(1 for category in categories if category in {"T-shirt", "Pullover", "Dress", "Coat", "Shirt"})
    bottom_count = sum(1 for category in categories if category in {"Trouser"})
    shoe_count = sum(1 for category in categories if category in {"Sandal", "Sneaker", "Ankle boot"})

    if top_count < 2:
        insights.append("Add at least one more top to increase outfit diversity.")
    if bottom_count < 2:
        insights.append("Add a second bottom option to avoid repeated pairings.")
    if shoe_count < 2:
        insights.append("Add one more shoe style to improve weather and occasion flexibility.")

    neutral_count = sum(1 for color in colors if color in {"black", "white", "gray"})
    if neutral_count < max(1, int(round(len(colors) * 0.3))):
        insights.append("Add a neutral item (black, white, or gray) to improve color compatibility.")

    if req.occasion == "formal":
        formal_ready = sum(1 for category in categories if category in {"Shirt", "Coat", "Trouser", "Ankle boot"})
        if formal_ready < 3:
            insights.append("For formal looks, add more shirts, trousers, or ankle boots.")

    if req.weather == "rainy":
        rain_ready = any(category in {"Coat", "Ankle boot"} for category in categories)
        if not rain_ready:
            insights.append("For rainy weather, add a coat or ankle boot for practical coverage.")

    return insights[:4]


def _feedback_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _build_recommendation_id(outfit_payload: Dict[str, Any], req: RecommendRequest) -> str:
    """Build deterministic recommendation id for tracking feedback signals."""
    parts = [
        _feedback_token(req.occasion),
        _feedback_token(req.weather),
        _feedback_token(outfit_payload.get("goal_mode")),
        _feedback_token(outfit_payload.get("color_strategy")),
    ]

    for item in outfit_payload.get("items", []):
        if not isinstance(item, dict):
            continue
        slot = _feedback_token(item.get("slot"))
        item_id = _feedback_token(item.get("item_id") or item.get("wardrobe_item_id"))
        category = _feedback_token(item.get("category"))
        color = _feedback_token(item.get("color"))
        parts.append(f"{slot}:{item_id}:{category}:{color}")

    raw = "|".join(parts)
    return uuid.uuid5(uuid.NAMESPACE_URL, raw).hex[:20]


def _register_recommendation_payload(recommendation_id: str, payload: Dict[str, Any]) -> None:
    """Store recommendation payload for subsequent feedback updates."""
    recommendation_registry[recommendation_id] = {
        "recommendation_id": recommendation_id,
        "goal_mode": payload.get("goal_mode"),
        "color_strategy": payload.get("color_strategy"),
        "items": payload.get("items", []),
    }

    recommendation_registry_order.append(recommendation_id)
    while len(recommendation_registry_order) > RECOMMENDATION_REGISTRY_LIMIT:
        stale = recommendation_registry_order.popleft()
        recommendation_registry.pop(stale, None)


def _update_affinity_bucket(bucket: Dict[str, float], key: str, delta: float) -> None:
    if not key:
        return

    prior = float(bucket.get(key, 0.0))
    # Small momentum keeps learning stable across repeated signals.
    next_value = (0.92 * prior) + delta
    bucket[key] = float(np.clip(next_value, -4.0, 4.0))


def _apply_feedback_signal(recommendation_payload: Dict[str, Any], signal: str) -> None:
    """Update user preference profile from explicit like/dislike signals."""
    delta = 1.0 if signal == "like" else -1.0

    recommendation_feedback_profile["total_feedback"] = int(recommendation_feedback_profile["total_feedback"]) + 1
    if delta > 0:
        recommendation_feedback_profile["likes"] = int(recommendation_feedback_profile["likes"]) + 1
    else:
        recommendation_feedback_profile["dislikes"] = int(recommendation_feedback_profile["dislikes"]) + 1

    category_affinity: Dict[str, float] = recommendation_feedback_profile["category_affinity"]
    color_affinity: Dict[str, float] = recommendation_feedback_profile["color_affinity"]
    goal_mode_affinity: Dict[str, float] = recommendation_feedback_profile["goal_mode_affinity"]
    color_strategy_affinity: Dict[str, float] = recommendation_feedback_profile["color_strategy_affinity"]

    goal_mode = _feedback_token(recommendation_payload.get("goal_mode"))
    color_strategy = _feedback_token(recommendation_payload.get("color_strategy"))
    _update_affinity_bucket(goal_mode_affinity, goal_mode, delta * 0.45)
    _update_affinity_bucket(color_strategy_affinity, color_strategy, delta * 0.35)

    for item in recommendation_payload.get("items", []):
        if not isinstance(item, dict):
            continue
        category = _feedback_token(item.get("category"))
        color = _feedback_token(item.get("color"))
        _update_affinity_bucket(category_affinity, category, delta * 0.50)
        _update_affinity_bucket(color_affinity, color, delta * 0.42)


def _feedback_adjustment(payload: Dict[str, Any]) -> float:
    """Compute score adjustment from learned user preference affinities."""
    total_feedback = int(recommendation_feedback_profile.get("total_feedback", 0))
    if total_feedback <= 0:
        return 0.0

    category_affinity: Dict[str, float] = recommendation_feedback_profile["category_affinity"]
    color_affinity: Dict[str, float] = recommendation_feedback_profile["color_affinity"]
    goal_mode_affinity: Dict[str, float] = recommendation_feedback_profile["goal_mode_affinity"]
    color_strategy_affinity: Dict[str, float] = recommendation_feedback_profile["color_strategy_affinity"]

    cat_values: List[float] = []
    color_values: List[float] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        cat_key = _feedback_token(item.get("category"))
        color_key = _feedback_token(item.get("color"))
        if cat_key:
            cat_values.append(float(category_affinity.get(cat_key, 0.0)))
        if color_key:
            color_values.append(float(color_affinity.get(color_key, 0.0)))

    category_component = float(np.mean(cat_values)) if cat_values else 0.0
    color_component = float(np.mean(color_values)) if color_values else 0.0
    goal_component = float(goal_mode_affinity.get(_feedback_token(payload.get("goal_mode")), 0.0))
    strategy_component = float(color_strategy_affinity.get(_feedback_token(payload.get("color_strategy")), 0.0))

    raw = (
        (0.40 * category_component)
        + (0.27 * color_component)
        + (0.18 * goal_component)
        + (0.15 * strategy_component)
    )

    # Convert affinity scale [-4, 4] to bounded score delta.
    return float(np.clip(raw / 16.0, -0.22, 0.22))


def _refresh_ranks(outfits: List[Dict[str, Any]]) -> None:
    """Recompute rank field after score-based reordering."""
    for index, payload in enumerate(outfits, start=1):
        payload["rank"] = index


def _feedback_profile_summary() -> Dict[str, Any]:
    """Return compact summary of learned preference state."""

    def top_entries(bucket: Dict[str, float], reverse: bool = True, limit: int = 4) -> List[Dict[str, Any]]:
        pairs = sorted(bucket.items(), key=lambda item: item[1], reverse=reverse)
        output: List[Dict[str, Any]] = []
        for key, value in pairs:
            if reverse and value <= 0:
                continue
            if not reverse and value >= 0:
                continue
            output.append({"key": key, "score": round(float(value), 3)})
            if len(output) >= limit:
                break
        return output

    return {
        "total_feedback": int(recommendation_feedback_profile.get("total_feedback", 0)),
        "likes": int(recommendation_feedback_profile.get("likes", 0)),
        "dislikes": int(recommendation_feedback_profile.get("dislikes", 0)),
        "top_liked_categories": top_entries(recommendation_feedback_profile["category_affinity"], reverse=True),
        "top_disliked_categories": top_entries(recommendation_feedback_profile["category_affinity"], reverse=False),
        "top_liked_colors": top_entries(recommendation_feedback_profile["color_affinity"], reverse=True),
        "top_disliked_colors": top_entries(recommendation_feedback_profile["color_affinity"], reverse=False),
    }


@app.post("/api/recommend")
async def recommend_outfits(req: RecommendRequest):
    """Generate outfit recommendations from current wardrobe."""
    if not wardrobe_store:
        raise HTTPException(status_code=400, detail="Wardrobe is empty. Add items first.")

    recommender: Optional[OutfitRecommender] = model_holder.get("recommender")
    if recommender is None:
        raise HTTPException(status_code=500, detail="Recommendation engine not initialized")

    # Convert wardrobe_store to list format expected by the engine
    wardrobe_list = []
    for idx, (iid, item) in enumerate(wardrobe_store.items()):
        wardrobe_list.append({
            "id": idx + 1,
            "item_id": iid,
            "category": item["category"],
            "color": item["color"],
            "confidence": item.get("confidence", 0),
            "features": item.get("features", []),
            "wear_count": item.get("wear_count", 0),
            "last_worn_days_ago": item.get("last_worn_days_ago"),
        })

    results = recommender.recommend_outfits(
        wardrobe=wardrobe_list,
        occasion=req.occasion,
        weather=req.weather,
        top_k=min(48, max(req.top_k * 6, req.top_k)),
        goal_mode=req.goal_mode,
        exploration=req.exploration,
        color_strategy=req.color_strategy,
        occasion_strictness=req.occasion_strictness,
        anti_repeat=req.anti_repeat,
        hero_item_id=req.hero_item_id,
        temperature_bias=req.temperature_bias,
    )

    candidate_outfits = [
        _serialize_recommendation_outfit(outfit=outfit, rank=rank, req=req)
        for rank, outfit in enumerate(results, start=1)
    ]

    for payload in candidate_outfits:
        _register_recommendation_payload(payload["recommendation_id"], payload)

    if req.feedback_learning:
        for payload in candidate_outfits:
            adjustment = _feedback_adjustment(payload)
            payload["feedback_adjustment"] = round(float(adjustment), 4)
            payload["score"] = round(float(payload.get("score", 0.0)) + float(adjustment), 4)

        candidate_outfits.sort(key=lambda entry: float(entry.get("score", 0.0)), reverse=True)
        _refresh_ranks(candidate_outfits)

    outfits = candidate_outfits[: req.top_k]

    backup_pack: Dict[str, Dict[str, Any]] = {}
    if req.include_backup_pack:
        raw_pack = recommender.build_backup_pack(results)
        backup_pack = {
            label: _serialize_recommendation_outfit(
                outfit=raw_outfit,
                rank=0,
                req=req,
                pack_label=label,
            )
            for label, raw_outfit in raw_pack.items()
        }

        for payload in backup_pack.values():
            _register_recommendation_payload(payload["recommendation_id"], payload)

    explain_targets: List[Dict[str, Any]] = list(outfits) + list(backup_pack.values())
    if explain_targets and req.explainability == "llm":
        explained = await asyncio.gather(
            *[_generate_openrouter_reasoning(target, req) for target in explain_targets],
            return_exceptions=True,
        )
        for target, explained_result in zip(explain_targets, explained):
            if isinstance(explained_result, Exception):
                fallback_explanation = _build_rule_explanation_payload(target, req)
                target["explanation"] = fallback_explanation
                target["reasoning"] = _format_explanation_text(fallback_explanation)
                target["explanation_source"] = "rule"
                continue

            explanation_payload, source = explained_result
            target["explanation"] = explanation_payload
            target["reasoning"] = _format_explanation_text(explanation_payload)
            target["explanation_source"] = source

    gap_insights = _build_closet_gap_insights(wardrobe_list, req)

    return {
        "outfits": outfits,
        "backup_pack": backup_pack,
        "closet_gap_insights": gap_insights,
        "options_applied": {
            "goal_mode": req.goal_mode,
            "exploration": req.exploration,
            "color_strategy": req.color_strategy,
            "occasion_strictness": req.occasion_strictness,
            "anti_repeat": req.anti_repeat,
            "hero_item_id": req.hero_item_id,
            "temperature_bias": req.temperature_bias,
            "explainability": req.explainability,
            "include_backup_pack": req.include_backup_pack,
            "feedback_learning": req.feedback_learning,
        },
        "feedback_profile": _feedback_profile_summary(),
        "total": len(outfits),
    }


@app.post("/api/recommend/feedback")
async def submit_recommendation_feedback(req: RecommendationFeedbackRequest):
    """Store recommendation feedback and update adaptive preference profile."""
    recommendation_id = str(req.recommendation_id).strip()
    payload = recommendation_registry.get(recommendation_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Unknown recommendation_id")

    _apply_feedback_signal(payload, req.signal)

    return {
        "status": "accepted",
        "recommendation_id": recommendation_id,
        "signal": req.signal,
        "feedback_profile": _feedback_profile_summary(),
    }


@app.get("/api/recommend/feedback-profile")
async def get_recommendation_feedback_profile():
    """Return current learned recommendation preference profile summary."""
    return _feedback_profile_summary()
