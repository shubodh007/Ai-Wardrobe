from src.recommendation_engine import OutfitRecommender


def test_feature_similarity_influences_ranking() -> None:
    recommender = OutfitRecommender()

    wardrobe = [
        {
            "item_id": "top-similar",
            "category": "T-shirt",
            "color": "white",
            "features": [1.0, 0.0],
        },
        {
            "item_id": "top-dissimilar",
            "category": "T-shirt",
            "color": "white",
            "features": [0.0, 1.0],
        },
        {
            "item_id": "bottom-reference",
            "category": "Trouser",
            "color": "black",
            "features": [1.0, 0.0],
        },
    ]

    outfits = recommender.recommend_outfits(
        wardrobe=wardrobe,
        occasion="casual",
        weather="mild",
        top_k=2,
    )

    assert len(outfits) >= 2
    assert outfits[0]["top"]["item_id"] == "top-similar"
    assert outfits[0]["feature_score"] >= outfits[1]["feature_score"]


def test_empty_wardrobe_returns_no_recommendations() -> None:
    recommender = OutfitRecommender()
    outfits = recommender.recommend_outfits(wardrobe=[], occasion="casual", weather="mild", top_k=3)
    assert outfits == []


def test_navy_alias_uses_blue_harmony_rules() -> None:
    recommender = OutfitRecommender()
    score = recommender.score_color_match("navy", "white")
    assert score == 1.0


def test_monochrome_color_strategy_prefers_same_color_pairing() -> None:
    recommender = OutfitRecommender()

    wardrobe = [
        {"item_id": "top-white", "category": "Shirt", "color": "white"},
        {"item_id": "top-blue", "category": "Shirt", "color": "blue"},
        {"item_id": "bottom-white", "category": "Trouser", "color": "white"},
        {"item_id": "bottom-black", "category": "Trouser", "color": "black"},
    ]

    outfits = recommender.recommend_outfits(
        wardrobe=wardrobe,
        occasion="casual",
        weather="mild",
        top_k=2,
        color_strategy="monochrome",
    )

    assert outfits
    assert outfits[0]["color_strategy"] == "monochrome"
    assert outfits[0]["color_score"] >= 0.8


def test_anti_repeat_prioritizes_less_worn_items() -> None:
    recommender = OutfitRecommender()

    wardrobe = [
        {
            "item_id": "top-worn",
            "category": "Shirt",
            "color": "white",
            "wear_count": 25,
            "features": [1.0, 0.0],
        },
        {
            "item_id": "top-fresh",
            "category": "Shirt",
            "color": "gray",
            "wear_count": 0,
            "features": [1.0, 0.0],
        },
        {
            "item_id": "bottom-1",
            "category": "Trouser",
            "color": "black",
            "features": [1.0, 0.0],
        },
    ]

    outfits = recommender.recommend_outfits(
        wardrobe=wardrobe,
        occasion="casual",
        weather="mild",
        top_k=2,
        goal_mode="repeat_avoider",
        anti_repeat=True,
    )

    assert outfits
    assert outfits[0]["top"]["item_id"] == "top-fresh"


def test_backup_pack_contains_safe_balanced_and_bold() -> None:
    recommender = OutfitRecommender()

    wardrobe = [
        {"item_id": "top-1", "category": "Shirt", "color": "white"},
        {"item_id": "top-2", "category": "T-shirt", "color": "red"},
        {"item_id": "bottom-1", "category": "Trouser", "color": "black"},
        {"item_id": "bottom-2", "category": "Trouser", "color": "blue"},
        {"item_id": "shoe-1", "category": "Sneaker", "color": "white"},
        {"item_id": "shoe-2", "category": "Ankle boot", "color": "black"},
    ]

    outfits = recommender.recommend_outfits(
        wardrobe=wardrobe,
        occasion="casual",
        weather="mild",
        top_k=5,
        exploration=0.7,
    )
    backup_pack = recommender.build_backup_pack(outfits)

    assert set(backup_pack.keys()) == {"safe", "balanced", "bold"}
