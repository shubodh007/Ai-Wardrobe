import api.main as api_main
import numpy as np
import cv2
from fastapi.testclient import TestClient

from api.main import (
    CATEGORIES,
    _final_topwear_guardrail,
    _prepare_classifier_input,
    _refine_bag_overprediction,
    _refine_shoe_overprediction,
    _refine_trouser_vs_pullover,
    app,
    wardrobe_store,
)


def _synthetic_trouser_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[6:20, 20:44] = 1
    mask[20:58, 20:30] = 1
    mask[20:58, 34:44] = 1
    return mask


def _synthetic_pullover_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:42, 16:48] = 1
    mask[14:34, 8:56] = 1
    return mask


def _synthetic_tall_garment_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[6:58, 20:44] = 1
    return mask


def _synthetic_square_bag_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[18:46, 18:46] = 1
    return mask


def _synthetic_shirt_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:50, 20:44] = 1
    mask[14:30, 10:54] = 1
    return mask


def _synthetic_ankle_boot_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[22:34, 24:40] = 1
    mask[32:58, 18:46] = 1
    return mask


def test_refine_trouser_vs_pullover_promotes_trouser_for_trouser_like_shape() -> None:
    trouser_idx = CATEGORIES.index("Trouser")
    pullover_idx = CATEGORIES.index("Pullover")

    probs = np.full(len(CATEGORIES), 0.01, dtype=np.float32)
    probs[pullover_idx] = 0.36
    probs[trouser_idx] = 0.30
    probs = probs / probs.sum()

    adjusted = _refine_trouser_vs_pullover(probs=probs, garment_mask=_synthetic_trouser_mask())

    assert float(adjusted[trouser_idx]) > float(adjusted[pullover_idx])


def test_refine_trouser_vs_pullover_keeps_pullover_for_pullover_shape() -> None:
    trouser_idx = CATEGORIES.index("Trouser")
    pullover_idx = CATEGORIES.index("Pullover")

    probs = np.full(len(CATEGORIES), 0.01, dtype=np.float32)
    probs[pullover_idx] = 0.42
    probs[trouser_idx] = 0.22
    probs = probs / probs.sum()

    adjusted = _refine_trouser_vs_pullover(probs=probs, garment_mask=_synthetic_pullover_mask())

    assert float(adjusted[pullover_idx]) > float(adjusted[trouser_idx])


def test_refine_bag_overprediction_demotes_bag_for_tall_shape() -> None:
    bag_idx = CATEGORIES.index("Bag")
    shirt_idx = CATEGORIES.index("Shirt")
    dress_idx = CATEGORIES.index("Dress")

    probs = np.full(len(CATEGORIES), 0.01, dtype=np.float32)
    probs[bag_idx] = 0.56
    probs[shirt_idx] = 0.18
    probs[dress_idx] = 0.16
    probs = probs / probs.sum()

    adjusted = _refine_bag_overprediction(probs=probs, garment_mask=_synthetic_tall_garment_mask())

    assert float(adjusted[bag_idx]) < float(probs[bag_idx])


def test_refine_bag_overprediction_keeps_bag_for_square_shape() -> None:
    bag_idx = CATEGORIES.index("Bag")
    shirt_idx = CATEGORIES.index("Shirt")

    probs = np.full(len(CATEGORIES), 0.01, dtype=np.float32)
    probs[bag_idx] = 0.74
    probs[shirt_idx] = 0.10
    probs = probs / probs.sum()

    adjusted = _refine_bag_overprediction(probs=probs, garment_mask=_synthetic_square_bag_mask())

    assert np.allclose(adjusted, probs)


def test_refine_bag_overprediction_handles_uncertain_square_bag_vs_shirt() -> None:
    bag_idx = CATEGORIES.index("Bag")
    shirt_idx = CATEGORIES.index("Shirt")

    probs = np.full(len(CATEGORIES), 0.01, dtype=np.float32)
    probs[bag_idx] = 0.34
    probs[shirt_idx] = 0.30
    probs = probs / probs.sum()

    adjusted = _refine_bag_overprediction(probs=probs, garment_mask=_synthetic_square_bag_mask())

    assert float(adjusted[bag_idx]) < float(probs[bag_idx])


def test_refine_shoe_overprediction_promotes_shirt_for_topwear_shape() -> None:
    ankle_idx = CATEGORIES.index("Ankle boot")
    shirt_idx = CATEGORIES.index("Shirt")
    sneaker_idx = CATEGORIES.index("Sneaker")

    probs = np.full(len(CATEGORIES), 0.01, dtype=np.float32)
    probs[ankle_idx] = 0.33
    probs[shirt_idx] = 0.20
    probs[sneaker_idx] = 0.12
    probs = probs / probs.sum()

    adjusted = _refine_shoe_overprediction(probs=probs, garment_mask=_synthetic_shirt_mask())

    assert float(adjusted[shirt_idx]) > float(probs[shirt_idx])
    assert float(adjusted[ankle_idx]) < float(probs[ankle_idx])


def test_refine_shoe_overprediction_keeps_ankle_boot_for_boot_shape() -> None:
    ankle_idx = CATEGORIES.index("Ankle boot")
    shirt_idx = CATEGORIES.index("Shirt")

    probs = np.full(len(CATEGORIES), 0.01, dtype=np.float32)
    probs[ankle_idx] = 0.64
    probs[shirt_idx] = 0.08
    probs = probs / probs.sum()

    adjusted = _refine_shoe_overprediction(probs=probs, garment_mask=_synthetic_ankle_boot_mask())

    assert np.allclose(adjusted, probs)


def test_refine_shoe_overprediction_handles_low_shirt_prob_with_ankle_and_bag_confusion() -> None:
    ankle_idx = CATEGORIES.index("Ankle boot")
    bag_idx = CATEGORIES.index("Bag")
    shirt_idx = CATEGORIES.index("Shirt")

    probs = np.full(len(CATEGORIES), 0.01, dtype=np.float32)
    probs[ankle_idx] = 0.313
    probs[bag_idx] = 0.242
    probs[shirt_idx] = 0.021
    probs = probs / probs.sum()

    adjusted = _refine_shoe_overprediction(probs=probs, garment_mask=_synthetic_shirt_mask())

    assert float(adjusted[shirt_idx]) > float(probs[shirt_idx])
    assert float(adjusted[ankle_idx]) < float(probs[ankle_idx])


def test_final_topwear_guardrail_forces_topwear_on_uncertain_shoe_bag_mix() -> None:
    ankle_idx = CATEGORIES.index("Ankle boot")
    bag_idx = CATEGORIES.index("Bag")
    shirt_idx = CATEGORIES.index("Shirt")

    probs = np.full(len(CATEGORIES), 0.01, dtype=np.float32)
    probs[ankle_idx] = 0.313
    probs[bag_idx] = 0.242
    probs[shirt_idx] = 0.021
    probs = probs / probs.sum()

    adjusted = _final_topwear_guardrail(probs=probs, garment_mask=_synthetic_shirt_mask())

    assert int(np.argmax(adjusted)) == shirt_idx
    assert float(adjusted[shirt_idx]) > float(probs[shirt_idx])
    assert float(adjusted[shirt_idx]) >= 0.52


def test_final_topwear_guardrail_boosts_uncertain_topwear_confidence() -> None:
    ankle_idx = CATEGORIES.index("Ankle boot")
    bag_idx = CATEGORIES.index("Bag")
    shirt_idx = CATEGORIES.index("Shirt")

    probs = np.full(len(CATEGORIES), 0.01, dtype=np.float32)
    probs[shirt_idx] = 0.30
    probs[ankle_idx] = 0.29
    probs[bag_idx] = 0.23
    probs = probs / probs.sum()

    adjusted = _final_topwear_guardrail(probs=probs, garment_mask=_synthetic_shirt_mask())

    assert int(np.argmax(adjusted)) == shirt_idx
    assert float(adjusted[shirt_idx]) > float(probs[shirt_idx])
    assert float(adjusted[shirt_idx]) >= 0.52
    assert float(adjusted[ankle_idx]) < float(probs[ankle_idx])


def test_final_topwear_guardrail_does_not_override_boot_like_shape() -> None:
    ankle_idx = CATEGORIES.index("Ankle boot")
    shirt_idx = CATEGORIES.index("Shirt")

    probs = np.full(len(CATEGORIES), 0.01, dtype=np.float32)
    probs[ankle_idx] = 0.48
    probs[shirt_idx] = 0.05
    probs = probs / probs.sum()

    adjusted = _final_topwear_guardrail(probs=probs, garment_mask=_synthetic_ankle_boot_mask())

    assert int(np.argmax(adjusted)) == ankle_idx


def test_prepare_classifier_input_recenters_small_offcenter_foreground() -> None:
    image = np.full((256, 256, 3), 230, dtype=np.uint8)
    image[28:94, 36:102] = np.array([38, 42, 48], dtype=np.uint8)

    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[28:94, 36:102] = 1

    prepared = _prepare_classifier_input(image, mask)
    base = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    prepared_center_darkness = float(1.0 - np.mean(prepared[10:18, 10:18]))
    base_center_darkness = float(1.0 - np.mean(base[10:18, 10:18]))

    assert prepared_center_darkness > (base_center_darkness + 0.08)


def test_health_endpoint_returns_status() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_loaded" in payload
    assert "prototype_fusion" in payload
    assert "prototype_sample_count" in payload


def test_classify_rejects_non_image_upload() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/classify",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_classify_sample_rejects_invalid_base64() -> None:
    with TestClient(app) as client:
        response = client.post("/api/classify-sample", json={"base64": "not-valid-base64"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid base64 payload"


def test_classify_sample_returns_503_when_model_unavailable() -> None:
    # 1x1 PNG
    tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6fVdYAAAAASUVORK5CYII="

    with TestClient(app) as client:
        response = client.post("/api/classify-sample", json={"base64": tiny_png})

    assert response.status_code == 503
    assert "Classifier model is unavailable" in response.json()["detail"]


def test_recommendation_includes_feature_score() -> None:
    wardrobe_store.clear()

    with TestClient(app) as client:
        add_payloads = [
            {
                "item_id": "top-1",
                "category": "Shirt",
                "color": "white",
                "confidence": 0.92,
                "label": "Shirt",
                "features": [1.0, 0.0, 0.0],
            },
            {
                "item_id": "bottom-1",
                "category": "Trouser",
                "color": "black",
                "confidence": 0.88,
                "label": "Trouser",
                "features": [1.0, 0.0, 0.0],
            },
            {
                "item_id": "shoe-1",
                "category": "Sneaker",
                "color": "white",
                "confidence": 0.80,
                "label": "Sneaker",
                "features": [1.0, 0.0, 0.0],
            },
        ]

        for payload in add_payloads:
            response = client.post("/api/wardrobe/add", json=payload)
            assert response.status_code == 200

        recommend_response = client.post(
            "/api/recommend",
            json={"occasion": "formal", "weather": "mild", "top_k": 2},
        )

    assert recommend_response.status_code == 200
    body = recommend_response.json()
    assert body["total"] >= 1
    assert "feature_score" in body["outfits"][0]
    assert "occasion_score" in body["outfits"][0]
    assert "weather_score" in body["outfits"][0]
    assert "profile" in body["outfits"][0]
    assert "explanation" in body["outfits"][0]
    assert isinstance(body["outfits"][0]["explanation"], dict)
    assert body["outfits"][0]["explanation"].get("summary")
    assert isinstance(body["outfits"][0]["explanation"].get("sections"), list)
    curation = body["outfits"][0]["explanation"].get("curation")
    assert isinstance(curation, dict)
    assert curation.get("curation_title")
    assert isinstance(curation.get("why_this_look"), list)
    assert isinstance(curation.get("styling_steps"), list)
    assert "backup_pack" in body
    assert "options_applied" in body


def test_wardrobe_persists_features() -> None:
    wardrobe_store.clear()

    with TestClient(app) as client:
        add_response = client.post(
            "/api/wardrobe/add",
            json={
                "item_id": "persist-1",
                "category": "Coat",
                "color": "gray",
                "confidence": 0.7,
                "label": "Coat",
                "features": [0.1, 0.2, 0.3],
            },
        )
        assert add_response.status_code == 200

        wardrobe_response = client.get("/api/wardrobe")

    assert wardrobe_response.status_code == 200
    items = wardrobe_response.json()["items"]
    assert len(items) == 1
    assert items[0]["features"] == [0.1, 0.2, 0.3]


def test_recommendation_explainability_falls_back_to_rule_without_api_key(monkeypatch) -> None:
    wardrobe_store.clear()
    monkeypatch.setattr(api_main, "OPENROUTER_API_KEY", "")

    with TestClient(app) as client:
        add_payloads = [
            {
                "item_id": "top-llm",
                "category": "Shirt",
                "color": "white",
                "confidence": 0.9,
                "label": "Shirt",
                "features": [1.0, 0.0],
            },
            {
                "item_id": "bottom-llm",
                "category": "Trouser",
                "color": "black",
                "confidence": 0.9,
                "label": "Trouser",
                "features": [1.0, 0.0],
            },
        ]
        for payload in add_payloads:
            response = client.post("/api/wardrobe/add", json=payload)
            assert response.status_code == 200

        recommend_response = client.post(
            "/api/recommend",
            json={
                "occasion": "formal",
                "weather": "mild",
                "top_k": 1,
                "explainability": "llm",
            },
        )

    assert recommend_response.status_code == 200
    body = recommend_response.json()
    assert body["outfits"]
    assert body["outfits"][0]["explanation_source"] == "rule"
    assert isinstance(body["outfits"][0]["reasoning"], str)
    assert isinstance(body["outfits"][0].get("explanation"), dict)
    assert body["outfits"][0]["explanation"].get("confidence_level") in {"high", "medium", "low"}
    curation = body["outfits"][0]["explanation"].get("curation")
    assert isinstance(curation, dict)
    assert curation.get("curation_title")
    assert isinstance(curation.get("why_this_look"), list)
    reasoning = body["outfits"][0]["reasoning"]
    assert reasoning
    assert "style:" in reasoning.lower()
    assert "how this works:" in reasoning.lower()
    assert "confidence note:" in reasoning.lower()
    assert "scores -" not in reasoning.lower()


def test_recommendation_options_are_echoed_and_backup_pack_is_structured() -> None:
    wardrobe_store.clear()

    with TestClient(app) as client:
        add_payloads = [
            {
                "item_id": "top-1",
                "category": "Shirt",
                "color": "white",
                "confidence": 0.92,
                "label": "Shirt",
                "features": [1.0, 0.0, 0.0],
                "wear_count": 8,
            },
            {
                "item_id": "top-2",
                "category": "T-shirt",
                "color": "red",
                "confidence": 0.8,
                "label": "T-shirt",
                "features": [0.4, 0.8, 0.2],
            },
            {
                "item_id": "bottom-1",
                "category": "Trouser",
                "color": "black",
                "confidence": 0.88,
                "label": "Trouser",
                "features": [1.0, 0.0, 0.0],
            },
            {
                "item_id": "shoe-1",
                "category": "Sneaker",
                "color": "white",
                "confidence": 0.8,
                "label": "Sneaker",
                "features": [1.0, 0.0, 0.0],
            },
        ]

        for payload in add_payloads:
            response = client.post("/api/wardrobe/add", json=payload)
            assert response.status_code == 200

        recommend_response = client.post(
            "/api/recommend",
            json={
                "occasion": "casual",
                "weather": "mild",
                "top_k": 3,
                "goal_mode": "repeat_avoider",
                "exploration": 0.75,
                "color_strategy": "complementary",
                "occasion_strictness": 0.4,
                "anti_repeat": True,
                "temperature_bias": "run_warm",
                "include_backup_pack": True,
            },
        )

    assert recommend_response.status_code == 200
    body = recommend_response.json()
    assert body["options_applied"]["goal_mode"] == "repeat_avoider"
    assert body["options_applied"]["color_strategy"] == "complementary"
    assert body["options_applied"]["temperature_bias"] == "run_warm"
    assert isinstance(body["backup_pack"], dict)


def test_recommendation_contains_recommendation_id_and_feedback_learning_flag() -> None:
    wardrobe_store.clear()

    with TestClient(app) as client:
        for payload in [
            {
                "item_id": "top-rid",
                "category": "Shirt",
                "color": "white",
                "confidence": 0.9,
                "label": "Shirt",
                "features": [1.0, 0.0],
            },
            {
                "item_id": "bottom-rid",
                "category": "Trouser",
                "color": "black",
                "confidence": 0.85,
                "label": "Trouser",
                "features": [1.0, 0.0],
            },
        ]:
            assert client.post("/api/wardrobe/add", json=payload).status_code == 200

        response = client.post(
            "/api/recommend",
            json={
                "occasion": "formal",
                "weather": "mild",
                "top_k": 1,
                "feedback_learning": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["outfits"]
    assert isinstance(body["outfits"][0]["recommendation_id"], str)
    assert body["options_applied"]["feedback_learning"] is True


def test_hero_item_id_forces_selected_item_in_outfit() -> None:
    wardrobe_store.clear()

    with TestClient(app) as client:
        add_payloads = [
            {
                "item_id": "hero-top",
                "category": "Shirt",
                "color": "white",
                "confidence": 0.9,
                "label": "Shirt",
                "features": [1.0, 0.0],
            },
            {
                "item_id": "other-top",
                "category": "T-shirt",
                "color": "red",
                "confidence": 0.9,
                "label": "T-shirt",
                "features": [0.0, 1.0],
            },
            {
                "item_id": "bottom-hero",
                "category": "Trouser",
                "color": "black",
                "confidence": 0.9,
                "label": "Trouser",
                "features": [1.0, 0.0],
            },
            {
                "item_id": "shoe-hero",
                "category": "Sneaker",
                "color": "white",
                "confidence": 0.8,
                "label": "Sneaker",
                "features": [1.0, 0.0],
            },
        ]

        for payload in add_payloads:
            assert client.post("/api/wardrobe/add", json=payload).status_code == 200

        response = client.post(
            "/api/recommend",
            json={
                "occasion": "casual",
                "weather": "mild",
                "top_k": 2,
                "hero_item_id": "hero-top",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["outfits"]
    for outfit in body["outfits"]:
        item_ids = {str(item.get("item_id") or "") for item in outfit.get("items", [])}
        assert "hero-top" in item_ids


def test_feedback_endpoint_updates_profile_counts() -> None:
    wardrobe_store.clear()
    api_main.recommendation_registry.clear()
    api_main.recommendation_registry_order.clear()
    api_main.recommendation_feedback_profile["likes"] = 0
    api_main.recommendation_feedback_profile["dislikes"] = 0
    api_main.recommendation_feedback_profile["total_feedback"] = 0
    api_main.recommendation_feedback_profile["category_affinity"].clear()
    api_main.recommendation_feedback_profile["color_affinity"].clear()
    api_main.recommendation_feedback_profile["goal_mode_affinity"].clear()
    api_main.recommendation_feedback_profile["color_strategy_affinity"].clear()

    with TestClient(app) as client:
        for payload in [
            {
                "item_id": "top-fb",
                "category": "Shirt",
                "color": "white",
                "confidence": 0.9,
                "label": "Shirt",
                "features": [1.0, 0.0],
            },
            {
                "item_id": "bottom-fb",
                "category": "Trouser",
                "color": "black",
                "confidence": 0.9,
                "label": "Trouser",
                "features": [1.0, 0.0],
            },
        ]:
            assert client.post("/api/wardrobe/add", json=payload).status_code == 200

        recommend_response = client.post(
            "/api/recommend",
            json={"occasion": "formal", "weather": "mild", "top_k": 1},
        )
        assert recommend_response.status_code == 200
        recommendation_id = recommend_response.json()["outfits"][0]["recommendation_id"]

        feedback_response = client.post(
            "/api/recommend/feedback",
            json={"recommendation_id": recommendation_id, "signal": "like"},
        )

        profile_response = client.get("/api/recommend/feedback-profile")

    assert feedback_response.status_code == 200
    feedback_body = feedback_response.json()
    assert feedback_body["status"] == "accepted"
    assert feedback_body["signal"] == "like"
    assert profile_response.status_code == 200
    profile_body = profile_response.json()
    assert profile_body["likes"] >= 1
    assert profile_body["total_feedback"] >= 1
