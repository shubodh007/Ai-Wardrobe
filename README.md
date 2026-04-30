# AI Wardrobe - Smart Outfit Recommendation System

A complete pure-ML Python project that prepares Fashion-MNIST data, trains a clothing classifier, extracts visual/color features, and recommends outfits using rule-based scoring.

The project now uses PyTorch (not TensorFlow) and is suitable for Python 3.14 environments with compatible PyTorch wheels.

## Project Structure

```text
ai-wardrobe-ml/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── saved_models/
├── src/
│   ├── data_preparation.py
│   ├── train_classifier.py
│   ├── color_extractor.py
│   ├── feature_extractor.py
│   ├── recommendation_engine.py
│   └── demo.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+ (Python 3.14 supported)
- Packages listed in requirements.txt

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional: create a `.env` file in the project root to enable LLM explainability for recommendations.

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
OPENROUTER_API_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_TIMEOUT_SECONDS=18
OPENROUTER_EXPLAINABILITY_SYSTEM_PROMPT=
```

## Run Frontend and Backend

Open two terminals from the `ai-wardrobe-ml` root folder.

1. Backend (FastAPI on port 8000):

```bash
pip install -r requirements.txt -r api/requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

2. Frontend (Vite on port 5173):

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser.

## Run Order

1. Prepare data:

```bash
python src/data_preparation.py
```

2. Train classifier:

```bash
python src/train_classifier.py
```

3. Optional modules:

```bash
python src/color_extractor.py
python src/feature_extractor.py
python src/recommendation_engine.py
```

4. End-to-end demo:

```bash
python src/demo.py
```

## Module Summary

- data_preparation.py: loads Fashion-MNIST via torchvision, splits 85/15, colorizes silhouettes into realistic RGB garments/backgrounds, stores augmentation settings, saves pickle outputs.
- train_classifier.py: builds and trains an RGB-aware PyTorch CNN with AdamW, label smoothing, gradient clipping, early stopping, and LR scheduling; evaluates test performance and saves best/final models plus training plot.
- color_extractor.py: extracts dominant colors with KMeans and visualizes color bars.
- feature_extractor.py: uses penultimate-layer PyTorch embeddings for feature vectors and similarity search.
- recommendation_engine.py: scores outfits using weighted goal modes, exploration tuning, configurable color strategies, occasion/weather constraints, anti-repeat rotation scoring, optional hero item focus, and safe/balanced/bold backup packs.
- demo.py: processes sample images, runs RGB PyTorch category inference, builds wardrobe entries, and prints recommendations for formal+mild and casual+hot contexts.

## Output Artifacts

Generated files include:

- data/processed/fashion_mnist_processed.pkl
- data/processed/augmentation_config.pkl
- data/processed/categories.pkl
- data/processed/test_features.pkl (after feature extractor demo)
- models/saved_models/best_model.pt
- models/saved_models/clothing_classifier.pt
- models/saved_models/training_history.png

## Notes

- All scripts are independently runnable with if __name__ == "__main__" blocks.
- Randomness-sensitive steps use random_state=42 where applicable.
- The recommendation engine gracefully handles empty wardrobes and missing category groups.
- Recommendation API now supports advanced controls: `goal_mode`, `exploration`, `color_strategy`, `occasion_strictness`, `anti_repeat`, `temperature_bias`, `explainability`, and `include_backup_pack`.
- Recommendation API also supports `hero_item_id` (build outfits around a chosen piece) and `feedback_learning` (adaptive reranking from likes/dislikes).
- When `explainability=llm`, the backend uses OpenRouter and falls back to deterministic rule-based explanations if the API key is missing or a provider call fails.
- Recommendation responses now include both `reasoning` (legacy text) and `explanation` (structured sections + confidence level + key factors + `curation` guidance) for richer UI explainability.
- Each outfit now carries a curated explainability block (`curation_title`, `vibe`, `when_to_wear`, `why_this_look`, `styling_steps`, `tradeoff_note`) so users get practical, outfit-specific guidance.
- `OPENROUTER_EXPLAINABILITY_SYSTEM_PROMPT` can override the built-in prompt if you want to customize LLM explanation tone/format.
- New feedback endpoints:
	- `POST /api/recommend/feedback` with `{ recommendation_id, signal }` where signal is `like` or `dislike`.
	- `GET /api/recommend/feedback-profile` to inspect learned preference summaries.
