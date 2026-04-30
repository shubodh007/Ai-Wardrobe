import numpy as np

from src.color_extractor import ColorExtractor


def test_rgb_to_color_name_detects_navy() -> None:
    extractor = ColorExtractor()
    assert extractor.rgb_to_color_name((20, 40, 110)) == "navy"


def test_dominant_color_prefers_center_garment() -> None:
    extractor = ColorExtractor()

    # White background with a large navy garment region in the center.
    image = np.full((300, 300, 3), 245, dtype=np.uint8)
    image[55:245, 55:245] = (20, 40, 110)

    colors = extractor.extract_dominant_colors_from_array(image, n_colors=3)

    assert len(colors) > 0
    assert colors[0]["name"] in {"navy", "blue"}
    top_rgb = colors[0]["rgb"]
    assert int(top_rgb[2]) > int(top_rgb[0])


def test_mask_aware_extraction_ignores_orange_background() -> None:
    extractor = ColorExtractor()

    image = np.zeros((240, 240, 3), dtype=np.uint8)
    image[:] = (210, 120, 30)  # warm/orange background
    image[60:180, 60:180] = (20, 40, 110)  # navy garment

    mask = np.zeros((240, 240), dtype=np.uint8)
    mask[60:180, 60:180] = 255

    colors = extractor.extract_dominant_colors_from_array(image, n_colors=3, mask=mask)

    assert len(colors) > 0
    assert colors[0]["name"] in {"navy", "blue"}
