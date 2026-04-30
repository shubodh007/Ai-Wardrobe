"""Outfit recommendation logic for AI Wardrobe."""

from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class OutfitRecommender:
    """Recommend outfit combinations based on context and configurable style options."""

    def __init__(self) -> None:
        self.color_aliases: Dict[str, str] = {
            "navy": "blue",
            "teal": "green",
            "maroon": "red",
        }

        self.neutral_colors = {"white", "black", "gray"}

        self.color_harmony: Dict[str, List[str]] = {
            "red": ["white", "black", "blue", "gray"],
            "orange": ["white", "black", "blue", "gray"],
            "yellow": ["blue", "black", "gray", "white"],
            "green": ["white", "black", "gray", "blue"],
            "blue": ["white", "gray", "black", "yellow"],
            "purple": ["white", "gray", "black", "pink"],
            "pink": ["white", "gray", "purple", "black"],
            "white": ["red", "orange", "yellow", "green", "blue", "purple", "pink", "black", "gray"],
            "black": ["red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "gray"],
            "gray": ["red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black"],
        }

        self.analogous_map: Dict[str, List[str]] = {
            "red": ["orange", "pink"],
            "orange": ["red", "yellow"],
            "yellow": ["orange", "green"],
            "green": ["yellow", "blue"],
            "blue": ["green", "purple"],
            "purple": ["blue", "pink"],
            "pink": ["purple", "red"],
        }

        self.complementary_map: Dict[str, str] = {
            "red": "green",
            "green": "red",
            "orange": "blue",
            "blue": "orange",
            "yellow": "purple",
            "purple": "yellow",
            "pink": "green",
        }

        self.occasion_rules: Dict[str, Dict[str, List[str]]] = {
            "formal": {
                "top": ["Shirt", "Coat", "Pullover"],
                "bottom": ["Trouser"],
                "shoes": ["Ankle boot", "Sneaker"],
                "avoid_colors": ["pink", "orange", "yellow"],
            },
            "casual": {
                "top": ["T-shirt", "Shirt", "Pullover", "Dress"],
                "bottom": ["Trouser"],
                "shoes": ["Sneaker", "Sandal", "Ankle boot"],
                "avoid_colors": [],
            },
            "sport": {
                "top": ["T-shirt", "Shirt"],
                "bottom": ["Trouser"],
                "shoes": ["Sneaker", "Sandal"],
                "avoid_colors": ["white"],
            },
        }

        self.weather_rules: Dict[str, Dict[str, List[str]]] = {
            "cold": {
                "prefer": ["Coat", "Pullover", "Ankle boot", "black", "gray"],
                "avoid": ["Sandal", "white"],
            },
            "hot": {
                "prefer": ["T-shirt", "Sandal", "white", "yellow"],
                "avoid": ["Coat", "Pullover", "Ankle boot", "black"],
            },
            "rainy": {
                "prefer": ["Coat", "Ankle boot", "black", "blue"],
                "avoid": ["Sandal", "white", "yellow"],
            },
        }

        self.goal_profiles: Dict[str, Dict[str, Any]] = {
            "balanced": {
                "weights": {
                    "color": 0.22,
                    "occasion": 0.24,
                    "weather": 0.20,
                    "feature": 0.14,
                    "novelty": 0.10,
                    "rotation": 0.10,
                },
                "default_color_strategy": "auto",
            },
            "formal_precision": {
                "weights": {
                    "color": 0.22,
                    "occasion": 0.35,
                    "weather": 0.17,
                    "feature": 0.10,
                    "novelty": 0.05,
                    "rotation": 0.11,
                },
                "default_color_strategy": "neutral_plus_one",
            },
            "comfort_first": {
                "weights": {
                    "color": 0.14,
                    "occasion": 0.20,
                    "weather": 0.34,
                    "feature": 0.10,
                    "novelty": 0.08,
                    "rotation": 0.14,
                },
                "default_color_strategy": "auto",
            },
            "heat_survival": {
                "weights": {
                    "color": 0.12,
                    "occasion": 0.17,
                    "weather": 0.40,
                    "feature": 0.08,
                    "novelty": 0.08,
                    "rotation": 0.15,
                },
                "default_color_strategy": "monochrome",
            },
            "rain_safe": {
                "weights": {
                    "color": 0.14,
                    "occasion": 0.18,
                    "weather": 0.38,
                    "feature": 0.10,
                    "novelty": 0.06,
                    "rotation": 0.14,
                },
                "default_color_strategy": "auto",
            },
            "repeat_avoider": {
                "weights": {
                    "color": 0.14,
                    "occasion": 0.15,
                    "weather": 0.11,
                    "feature": 0.06,
                    "novelty": 0.20,
                    "rotation": 0.34,
                },
                "default_color_strategy": "auto",
            },
            "bold_experiment": {
                "weights": {
                    "color": 0.23,
                    "occasion": 0.14,
                    "weather": 0.10,
                    "feature": 0.12,
                    "novelty": 0.27,
                    "rotation": 0.14,
                },
                "default_color_strategy": "high_contrast",
            },
        }

        self.supported_color_strategies = {
            "auto",
            "monochrome",
            "analogous",
            "complementary",
            "neutral_plus_one",
            "high_contrast",
        }

        self.top_categories = {"T-shirt", "Pullover", "Dress", "Coat", "Shirt"}
        self.bottom_categories = {"Trouser"}
        self.shoe_categories = {"Sandal", "Sneaker", "Ankle boot"}

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = float(sum(max(0.0, value) for value in weights.values()))
        if total <= 0.0:
            count = max(1, len(weights))
            return {key: 1.0 / count for key in weights}
        return {key: float(max(0.0, value) / total) for key, value in weights.items()}

    def _normalize_color(self, color: str) -> str:
        """Map color aliases to canonical names used by recommendation rules."""
        normalized = (color or "gray").strip().lower()
        return self.color_aliases.get(normalized, normalized)

    def _is_neutral(self, color: str) -> bool:
        return self._normalize_color(color) in self.neutral_colors

    @staticmethod
    def _extract_item_id(item: Dict[str, Any]) -> str:
        return str(item.get("item_id") or item.get("id") or "").strip()

    def score_color_match(self, color_a: str, color_b: str) -> float:
        """Backward-compatible automatic color compatibility scoring."""
        return self._score_color_strategy_pair(color_a=color_a, color_b=color_b, strategy="auto")

    def _score_color_strategy_pair(self, color_a: str, color_b: str, strategy: str) -> float:
        """Score color pair compatibility for a specific strategy."""
        color_1 = self._normalize_color(color_a)
        color_2 = self._normalize_color(color_b)
        mode = (strategy or "auto").strip().lower()

        if mode not in self.supported_color_strategies:
            mode = "auto"

        if mode == "auto":
            if color_2 in self.color_harmony.get(color_1, []):
                return 1.0
            if color_1 == color_2:
                return 0.35
            if self._is_neutral(color_1) or self._is_neutral(color_2):
                return 0.7
            return 0.15

        if mode == "monochrome":
            if color_1 == color_2:
                return 1.0
            if self._is_neutral(color_1) and self._is_neutral(color_2):
                return 0.9
            if self._is_neutral(color_1) or self._is_neutral(color_2):
                return 0.75
            return 0.2

        if mode == "analogous":
            if color_2 in self.analogous_map.get(color_1, []) or color_1 in self.analogous_map.get(color_2, []):
                return 1.0
            if color_1 == color_2:
                return 0.6
            if self._is_neutral(color_1) or self._is_neutral(color_2):
                return 0.65
            if color_2 in self.color_harmony.get(color_1, []):
                return 0.5
            return 0.15

        if mode == "complementary":
            if self.complementary_map.get(color_1) == color_2:
                return 1.0
            if self._is_neutral(color_1) or self._is_neutral(color_2):
                return 0.62
            if color_2 in self.color_harmony.get(color_1, []):
                return 0.45
            return 0.12

        if mode == "neutral_plus_one":
            neutral_count = int(self._is_neutral(color_1)) + int(self._is_neutral(color_2))
            if neutral_count == 1:
                return 1.0
            if neutral_count == 2:
                return 0.72
            if color_1 == color_2:
                return 0.35
            return 0.2

        # high_contrast
        if self.complementary_map.get(color_1) == color_2:
            return 1.0
        if {color_1, color_2} == {"black", "white"}:
            return 1.0
        if (self._is_neutral(color_1) and not self._is_neutral(color_2)) or (
            self._is_neutral(color_2) and not self._is_neutral(color_1)
        ):
            return 0.82
        if color_2 in self.color_harmony.get(color_1, []):
            return 0.55
        if color_1 == color_2:
            return 0.12
        return 0.2

    def score_occasion_match(
        self,
        item: Dict[str, Any],
        occasion: str,
        item_type: str,
        strictness: float = 0.6,
    ) -> float:
        """Score how suitable an item is for a given occasion and slot."""
        rules = self.occasion_rules.get((occasion or "").lower())
        if not rules:
            return 0.55

        strictness_value = self._clamp01(float(strictness))
        category = str(item.get("category", ""))
        color = self._normalize_color(str(item.get("color", "gray")))

        allowed = rules.get(item_type, [])
        if category in allowed:
            score = 1.0
        else:
            score = 0.2 + (0.6 * (1.0 - strictness_value))

        if color in rules.get("avoid_colors", []):
            score -= (0.2 + (0.45 * strictness_value))

        return self._clamp01(float(score))

    def score_weather_match(self, item: Dict[str, Any], weather: str, temperature_bias: str = "neutral") -> float:
        """Score weather suitability based on preferred and avoided categories/colors."""
        rules = self.weather_rules.get((weather or "").lower())
        if not rules:
            score = 0.62
        else:
            category = str(item.get("category", ""))
            color = self._normalize_color(str(item.get("color", "gray")))
            score = 0.62

            if category in rules.get("prefer", []) or color in rules.get("prefer", []):
                score += 0.33
            if category in rules.get("avoid", []) or color in rules.get("avoid", []):
                score -= 0.42

        bias = (temperature_bias or "neutral").strip().lower()
        category = str(item.get("category", ""))
        if bias == "run_cold":
            if category in {"Coat", "Pullover", "Ankle boot"}:
                score += 0.08
            if category in {"Sandal"}:
                score -= 0.08
        elif bias == "run_warm":
            if category in {"Sandal", "T-shirt"}:
                score += 0.08
            if category in {"Coat", "Pullover"}:
                score -= 0.08

        return self._clamp01(float(score))

    def categorize_items(self, wardrobe: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group wardrobe entries into tops, bottoms, and shoes."""
        grouped = {"tops": [], "bottoms": [], "shoes": []}

        for item in wardrobe:
            category = str(item.get("category", ""))
            if category in self.top_categories:
                grouped["tops"].append(item)
            elif category in self.bottom_categories:
                grouped["bottoms"].append(item)
            elif category in self.shoe_categories:
                grouped["shoes"].append(item)

        return grouped

    @staticmethod
    def _feature_similarity(feature_a: Any, feature_b: Any) -> float:
        """Compute non-negative cosine similarity from optional feature vectors."""
        if feature_a is None or feature_b is None:
            return 0.0

        vec_a = np.asarray(feature_a, dtype=np.float32).reshape(-1)
        vec_b = np.asarray(feature_b, dtype=np.float32).reshape(-1)
        if vec_a.size == 0 or vec_b.size == 0:
            return 0.0

        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0.0:
            return 0.0

        cosine = float(np.dot(vec_a, vec_b) / denom)
        return max(0.0, cosine)

    def _item_rotation_score(self, item: Optional[Dict[str, Any]], anti_repeat: bool = True) -> float:
        """Score whether the item is underused and therefore worth rotating in."""
        if item is None or not anti_repeat:
            return 0.5

        wear_count = float(max(0.0, float(item.get("wear_count", 0.0))))
        wear_freshness = 1.0 - min(wear_count, 30.0) / 30.0

        days_raw = item.get("last_worn_days_ago")
        if days_raw is None:
            days_score = 0.55
        else:
            days_score = min(max(float(days_raw), 0.0), 30.0) / 30.0

        return self._clamp01((0.65 * wear_freshness) + (0.35 * days_score))

    def _outfit_rotation_score(self, items: List[Optional[Dict[str, Any]]], anti_repeat: bool = True) -> float:
        valid_items = [item for item in items if item is not None]
        if not valid_items:
            return 0.5

        scores = [self._item_rotation_score(item, anti_repeat=anti_repeat) for item in valid_items]
        return float(np.mean(scores))

    def _score_novelty(
        self,
        top_item: Dict[str, Any],
        bottom_item: Dict[str, Any],
        shoe_item: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Score how exploratory a combination is, independent of fit constraints."""
        colors = [
            self._normalize_color(str(top_item.get("color", "gray"))),
            self._normalize_color(str(bottom_item.get("color", "gray"))),
        ]
        if shoe_item is not None:
            colors.append(self._normalize_color(str(shoe_item.get("color", "gray"))))

        unique_ratio = len(set(colors)) / max(1, len(colors))
        non_neutral_ratio = float(sum(0 if color in self.neutral_colors else 1 for color in colors)) / max(1, len(colors))

        top_color, bottom_color = colors[0], colors[1]
        contrast_bonus = 0.0
        if self.complementary_map.get(top_color) == bottom_color or {top_color, bottom_color} == {"black", "white"}:
            contrast_bonus = 0.25

        base = (0.45 * unique_ratio) + (0.35 * non_neutral_ratio) + contrast_bonus

        if top_color == bottom_color and top_color not in self.neutral_colors:
            base -= 0.15
        if top_color == bottom_color and top_color in self.neutral_colors:
            base -= 0.10

        return self._clamp01(base)

    def _score_boldness(
        self,
        top_item: Dict[str, Any],
        bottom_item: Dict[str, Any],
        shoe_item: Optional[Dict[str, Any]],
        color_strategy: str,
    ) -> float:
        novelty = self._score_novelty(top_item, bottom_item, shoe_item)

        colors = [
            self._normalize_color(str(top_item.get("color", "gray"))),
            self._normalize_color(str(bottom_item.get("color", "gray"))),
        ]
        if shoe_item is not None:
            colors.append(self._normalize_color(str(shoe_item.get("color", "gray"))))

        non_neutral_ratio = float(sum(0 if color in self.neutral_colors else 1 for color in colors)) / max(1, len(colors))
        strategy_bias = 0.0
        mode = (color_strategy or "auto").lower()
        if mode in {"high_contrast", "complementary"}:
            strategy_bias = 0.16
        elif mode in {"monochrome", "neutral_plus_one"}:
            strategy_bias = -0.06

        return self._clamp01((0.58 * novelty) + (0.42 * non_neutral_ratio) + strategy_bias)

    def _resolve_recommendation_profile(
        self,
        goal_mode: str,
        exploration: float,
        occasion_strictness: float,
        anti_repeat: bool,
        color_strategy: str,
    ) -> Tuple[Dict[str, float], str]:
        """Resolve dynamic scoring weights and effective color strategy."""
        selected_goal = (goal_mode or "balanced").strip().lower()
        profile = self.goal_profiles.get(selected_goal, self.goal_profiles["balanced"])

        conservative = {
            "color": 0.20,
            "occasion": 0.30,
            "weather": 0.24,
            "feature": 0.14,
            "novelty": 0.03,
            "rotation": 0.09,
        }
        adventurous = {
            "color": 0.24,
            "occasion": 0.13,
            "weather": 0.12,
            "feature": 0.12,
            "novelty": 0.27,
            "rotation": 0.12,
        }

        alpha = self._clamp01(exploration)
        exploration_blend = {
            key: ((1.0 - alpha) * conservative[key]) + (alpha * adventurous[key]) for key in conservative
        }

        base_weights = profile["weights"]
        merged_weights = {
            key: ((0.70 * base_weights[key]) + (0.30 * exploration_blend[key])) for key in base_weights
        }

        strictness_value = self._clamp01(occasion_strictness)
        merged_weights["occasion"] *= (0.82 + (0.36 * strictness_value))

        if not anti_repeat:
            merged_weights["rotation"] *= 0.18

        normalized_weights = self._normalize_weights(merged_weights)

        requested_strategy = (color_strategy or "auto").strip().lower()
        if requested_strategy not in self.supported_color_strategies:
            requested_strategy = "auto"

        if requested_strategy == "auto":
            requested_strategy = profile.get("default_color_strategy", "auto")
            if requested_strategy not in self.supported_color_strategies:
                requested_strategy = "auto"

        return normalized_weights, requested_strategy

    @staticmethod
    def _outfit_signature(outfit: Dict[str, Any]) -> Tuple[str, str, str]:
        top = outfit.get("top") or {}
        bottom = outfit.get("bottom") or {}
        shoe = outfit.get("shoes") or {}

        def normalize(item: Dict[str, Any]) -> str:
            item_id = str(item.get("item_id") or item.get("id") or "").strip()
            if item_id:
                return item_id
            return f"{item.get('category', '')}:{item.get('color', '')}".lower()

        return (normalize(top), normalize(bottom), normalize(shoe))

    @staticmethod
    def _label_profile(boldness: float, exploration: float) -> str:
        if boldness >= 0.72 or (exploration >= 0.7 and boldness >= 0.58):
            return "bold"
        if boldness <= 0.35 and exploration <= 0.55:
            return "safe"
        return "balanced"

    def recommend_outfits(
        self,
        wardrobe: List[Dict[str, Any]],
        occasion: str,
        weather: str,
        top_k: int = 3,
        goal_mode: str = "balanced",
        exploration: float = 0.35,
        color_strategy: str = "auto",
        occasion_strictness: float = 0.6,
        anti_repeat: bool = True,
        hero_item_id: Optional[str] = None,
        temperature_bias: str = "neutral",
    ) -> List[Dict[str, Any]]:
        """Generate and rank outfit combinations for the given context and style options."""
        if not wardrobe:
            print("Wardrobe is empty. Add items before requesting recommendations.")
            return []

        exploration_value = self._clamp01(float(exploration))
        strictness_value = self._clamp01(float(occasion_strictness))
        hero_id = (hero_item_id or "").strip()

        weights, effective_color_strategy = self._resolve_recommendation_profile(
            goal_mode=goal_mode,
            exploration=exploration_value,
            occasion_strictness=strictness_value,
            anti_repeat=anti_repeat,
            color_strategy=color_strategy,
        )

        grouped = self.categorize_items(wardrobe)
        tops = grouped["tops"]
        bottoms = grouped["bottoms"]
        shoes = grouped["shoes"]

        if not tops or not bottoms:
            print("Need at least one top and one bottom to recommend outfits.")
            return []

        all_outfits: List[Dict[str, Any]] = []

        for top_item, bottom_item in product(tops, bottoms):
            top_id = self._extract_item_id(top_item)
            bottom_id = self._extract_item_id(bottom_item)
            pair_has_hero = bool(hero_id and hero_id in {top_id, bottom_id})

            top_color = str(top_item.get("color", "gray"))
            bottom_color = str(bottom_item.get("color", "gray"))

            pair_color_score = self._score_color_strategy_pair(top_color, bottom_color, effective_color_strategy)
            top_occasion_score = self.score_occasion_match(top_item, occasion, "top", strictness=strictness_value)
            bottom_occasion_score = self.score_occasion_match(
                bottom_item, occasion, "bottom", strictness=strictness_value
            )
            top_weather_score = self.score_weather_match(top_item, weather, temperature_bias=temperature_bias)
            bottom_weather_score = self.score_weather_match(bottom_item, weather, temperature_bias=temperature_bias)
            pair_feature_score = self._feature_similarity(top_item.get("features"), bottom_item.get("features"))

            best_shoe: Optional[Dict[str, Any]] = None
            best_shoe_components = {
                "score": -1.0,
                "color": 0.0,
                "occasion": 0.0,
                "weather": 0.0,
                "feature": 0.0,
                "rotation": 0.5,
            }

            for shoe_item in shoes:
                shoe_id = self._extract_item_id(shoe_item)
                if hero_id and not pair_has_hero and shoe_id != hero_id:
                    continue

                shoe_color = str(shoe_item.get("color", "gray"))
                shoe_color_score = (
                    self._score_color_strategy_pair(top_color, shoe_color, effective_color_strategy)
                    + self._score_color_strategy_pair(bottom_color, shoe_color, effective_color_strategy)
                ) / 2.0
                shoe_occasion_score = self.score_occasion_match(
                    shoe_item,
                    occasion,
                    "shoes",
                    strictness=strictness_value,
                )
                shoe_weather_score = self.score_weather_match(
                    shoe_item,
                    weather,
                    temperature_bias=temperature_bias,
                )
                shoe_feature_score = (
                    self._feature_similarity(top_item.get("features"), shoe_item.get("features"))
                    + self._feature_similarity(bottom_item.get("features"), shoe_item.get("features"))
                ) / 2.0
                shoe_rotation_score = self._item_rotation_score(shoe_item, anti_repeat=anti_repeat)

                shoe_total = (
                    (weights["color"] * shoe_color_score)
                    + (weights["occasion"] * shoe_occasion_score)
                    + (weights["weather"] * shoe_weather_score)
                    + (weights["feature"] * shoe_feature_score)
                    + (weights["rotation"] * shoe_rotation_score)
                )

                if shoe_total > best_shoe_components["score"]:
                    best_shoe = shoe_item
                    best_shoe_components = {
                        "score": float(shoe_total),
                        "color": float(shoe_color_score),
                        "occasion": float(shoe_occasion_score),
                        "weather": float(shoe_weather_score),
                        "feature": float(shoe_feature_score),
                        "rotation": float(shoe_rotation_score),
                    }

            if hero_id and not pair_has_hero:
                if best_shoe is None or self._extract_item_id(best_shoe) != hero_id:
                    continue

            occasion_components = [top_occasion_score, bottom_occasion_score]
            weather_components = [top_weather_score, bottom_weather_score]
            color_components = [pair_color_score]
            feature_components = [pair_feature_score]

            selected_items: List[Optional[Dict[str, Any]]] = [top_item, bottom_item]

            if best_shoe is not None:
                selected_items.append(best_shoe)
                occasion_components.append(best_shoe_components["occasion"])
                weather_components.append(best_shoe_components["weather"])
                color_components.append(best_shoe_components["color"])
                feature_components.append(best_shoe_components["feature"])

            color_score = float(np.mean(color_components))
            occasion_score = float(np.mean(occasion_components))
            weather_score = float(np.mean(weather_components))
            feature_score = float(np.mean(feature_components))
            novelty_score = self._score_novelty(top_item, bottom_item, best_shoe)
            rotation_score = self._outfit_rotation_score(selected_items, anti_repeat=anti_repeat)
            boldness_score = self._score_boldness(top_item, bottom_item, best_shoe, effective_color_strategy)

            component_scores = {
                "color": color_score,
                "occasion": occasion_score,
                "weather": weather_score,
                "feature": feature_score,
                "novelty": novelty_score,
                "rotation": rotation_score,
            }

            weighted_total = float(sum(weights[name] * component_scores[name] for name in weights))
            exploration_shift = float((exploration_value - 0.5) * (novelty_score - 0.5) * 0.22)
            total_score = weighted_total + exploration_shift

            all_outfits.append(
                {
                    "top": top_item,
                    "bottom": bottom_item,
                    "shoes": best_shoe,
                    "score": float(total_score),
                    "color_score": float(color_score),
                    "occasion_score": float(occasion_score),
                    "weather_score": float(weather_score),
                    "feature_score": float(feature_score),
                    "novelty_score": float(novelty_score),
                    "rotation_score": float(rotation_score),
                    "boldness": float(boldness_score),
                    "profile": self._label_profile(boldness_score, exploration_value),
                    "goal_mode": goal_mode,
                    "color_strategy": effective_color_strategy,
                    "occasion": occasion,
                    "weather": weather,
                    "score_breakdown": {
                        "weights": dict(weights),
                        "components": dict(component_scores),
                        "exploration_shift": float(exploration_shift),
                    },
                }
            )

        all_outfits.sort(key=lambda entry: entry["score"], reverse=True)
        return all_outfits[: max(1, top_k)]

    def build_backup_pack(self, outfits: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build safe, balanced, and bold alternatives from already-ranked outfits."""
        if not outfits:
            return {}

        used_signatures = set()
        packs: Dict[str, Dict[str, Any]] = {}

        safe_sorted = sorted(
            outfits,
            key=lambda entry: (
                entry.get("score", 0.0)
                + (0.35 * entry.get("occasion_score", 0.0))
                + (0.25 * entry.get("weather_score", 0.0))
                - (0.55 * entry.get("boldness", 0.0))
            ),
            reverse=True,
        )
        balanced_sorted = sorted(outfits, key=lambda entry: entry.get("score", 0.0), reverse=True)
        bold_sorted = sorted(
            outfits,
            key=lambda entry: (
                entry.get("score", 0.0)
                + (0.80 * entry.get("boldness", 0.0))
                + (0.40 * entry.get("novelty_score", 0.0))
            ),
            reverse=True,
        )

        for label, candidate_list in [
            ("safe", safe_sorted),
            ("balanced", balanced_sorted),
            ("bold", bold_sorted),
        ]:
            selected = None
            for candidate in candidate_list:
                signature = self._outfit_signature(candidate)
                if signature in used_signatures:
                    continue
                selected = dict(candidate)
                used_signatures.add(signature)
                break

            if selected is None:
                selected = dict(candidate_list[0])

            selected["recommendation_pack_label"] = label
            packs[label] = selected

        return packs

    def explain_recommendation(self, outfit: Dict[str, Any]) -> str:
        """Create a concise explanation string for a recommended outfit."""
        top_item = outfit.get("top", {})
        bottom_item = outfit.get("bottom", {})
        shoe_item = outfit.get("shoes") or {"category": "None", "color": "none"}

        profile = outfit.get("profile", "balanced")
        goal_mode = outfit.get("goal_mode", "balanced")
        color_strategy = outfit.get("color_strategy", "auto")

        return (
            f"Top: {top_item.get('category')} ({top_item.get('color')})\n"
            f"Bottom: {bottom_item.get('category')} ({bottom_item.get('color')})\n"
            f"Shoes: {shoe_item.get('category')} ({shoe_item.get('color')})\n"
            f"Score: {float(outfit.get('score', 0.0)):.3f}\n"
            f"Profile: {profile} | Goal mode: {goal_mode} | Color strategy: {color_strategy}"
        )


def main() -> None:
    """Run a standalone recommendation example."""
    sample_wardrobe = [
        {"id": 1, "item_id": "1", "category": "Shirt", "color": "white", "wear_count": 8},
        {"id": 2, "item_id": "2", "category": "Coat", "color": "black", "wear_count": 2},
        {"id": 3, "item_id": "3", "category": "Trouser", "color": "gray", "wear_count": 3},
        {"id": 4, "item_id": "4", "category": "Trouser", "color": "blue", "wear_count": 1},
        {"id": 5, "item_id": "5", "category": "Sneaker", "color": "white", "wear_count": 9},
        {"id": 6, "item_id": "6", "category": "Ankle boot", "color": "black", "wear_count": 0},
        {"id": 7, "item_id": "7", "category": "Sandal", "color": "blue", "wear_count": 6},
    ]

    recommender = OutfitRecommender()
    recommendations = recommender.recommend_outfits(
        wardrobe=sample_wardrobe,
        occasion="formal",
        weather="cold",
        top_k=3,
        goal_mode="formal_precision",
        exploration=0.25,
        color_strategy="neutral_plus_one",
        occasion_strictness=0.9,
        anti_repeat=True,
    )

    print("\nTop formal recommendations")
    if not recommendations:
        print("No recommendations available.")
        return

    for rank, outfit in enumerate(recommendations, start=1):
        print(f"\n{rank}. Recommendation")
        print(recommender.explain_recommendation(outfit))

    backup_pack = recommender.build_backup_pack(recommendations)
    if backup_pack:
        print("\nBackup pack labels:", ", ".join(sorted(backup_pack.keys())))


if __name__ == "__main__":
    main()
