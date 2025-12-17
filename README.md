# GNN Recipe Recommender (Food.com subset, Apple Silicon friendly)

This is a minimal, **local** runnable starter to train a heterogenous GNN (users–recipes–ingredients) recommender on a small **Food.com** subset. Designed for **Apple Silicon (M1–M4)** with optional **MPS** acceleration.

## Quick Start

```bash
# 1) Create venv
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 2) Install deps (CPU wheels work on Apple Silicon; MPS is in torch)
pip install -r requirements.txt

# 3) Put Food.com CSVs into ./data/
#    Required minimally: recipes.csv, ingredients.csv, interactions.csv
#    (You can export from Kaggle "Food.com Recipes and Interactions" dataset.)

# 4) Build graph edges and meta
python scripts/build_graph_foodcom.py

# 5) Train GNN with BPR loss (GraphSAGE over hetero graph)
python scripts/train_sage_recommender.py

# 6) Evaluate Recall@K / NDCG@K (leave-one-out by user)
python scripts/eval_metrics.py
```

> **Tip (MPS):** If you see `mps` available in logs, PyTorch will use the Apple GPU automatically. If not, it falls back to CPU.

## Data Layout (expected in `./data/`)

- `recipes.csv` — columns: `id`, `name`, ... (we use `id`)
- `ingredients.csv` — columns: `recipe_id`, `ingredient`
- `interactions.csv` — columns: `user_id`, `recipe_id`, `rating`, `date` (rating >= 4 counted as positive; you can adjust)

You can rename the files in the scripts if your column names differ slightly.

## What this does

- Build a hetero graph:
  - **(user) —rates→ (recipe)**
  - **(recipe) —has→ (ing)**

- Learn embeddings with a 2-layer GraphSAGE (converted to hetero) and **BPR** objective on (user, positive recipe, sampled negative recipe).

- Evaluate with **leave-one-out per user** (last interaction as test, rest for train) and report **Recall@K / NDCG@K**.

## Notes

- Start small: filter to *active users* (>=5 interactions) and *popular recipes* (>=5 interactions) in `build_graph_foodcom.py` to keep RAM small.
- Improve later: temporal negative sampling, meta-path aware message passing (U→R→Ing→R), GAT-based relation attention, etc.
- Use `--epochs`, `--hidden_dim`, etc., to experiment (you can extend the scripts).

## License & Data

This repo is code-only. Make sure you respect the dataset license/terms of use.


## Using RAW files only (no pre-made ingredients.csv)

If you only place **`RAW_recipes.csv`** and **`RAW_interactions.csv`** under `./data/`, you're good.
`build_graph_foodcom.py` will automatically parse `ingredients` from `RAW_recipes.csv` and create `ingredients.csv` for you.

Expected columns:
- `RAW_recipes.csv`: `id`, `name`, `ingredients` (stringified list), ...
- `RAW_interactions.csv`: `user_id`, `recipe_id`, `rating`, `date`, ...

