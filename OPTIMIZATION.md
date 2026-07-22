# Performance Optimization

This project reduces repeated work in two complementary layers:

1. **Offline teaching packs** (`project/scripts/precompute_data.py`) — bake market data, features, and labels into Parquet once so the Streamlit app and grid runs load instantly.
2. **Shared + per-run disk cache** (`project/src/utils/cache.py`) — reuse expensive `build_features` outputs across training runs, and persist each experiment’s artifacts under a stable directory layout.

Together they avoid re-downloading OHLCV data, re-running the full feature pipeline, and re-deriving labels whenever the same configuration appears again.

---

## 1. Why these steps dominate runtime

A typical training or teaching session spends most of its time before model fitting:

| Stage | Cost | Notes |
| --- | --- | --- |
| Market data fetch | Network + I/O | yfinance (or synthetic fallback) |
| Feature engineering | CPU-heavy | Full default feature step list |
| Label construction | Medium | Forward RV + expanding / rolling median thresholds |
| Feature selection + Optuna | CPU-heavy | Intentionally **not** precomputed into teaching packs |

Optimization therefore targets the first three stages: compute once, store on disk, reload by key.

---

## 2. Offline teaching packs (`precompute_data.py`)

### Purpose

Build **wide** Parquet frames for a fixed teaching universe (default: `SPY`, `QQQ` × horizons `5d`, `10d`) covering the full timeline with **all features and dual labels**, and **no feature selection**. Downstream UI and `precompute_grid.py` slice columns as needed instead of rebuilding features live.

### Pipeline

For each symbol:

1. Load OHLCV via `get_data_with_fallback(symbol, start, end)`.
2. Run `apply_features(raw)` once (shared across horizons).
3. For each horizon, attach dual labels:
   - `rv_fwd_{horizon}d` from `compute_forward_rv`
   - `y_expanding` — above expanding median of forward RV
   - `y_median` — above rolling median (`window = horizon * 10`)
4. Write `cache/precomputed/<SYMBOL>_<HORIZON>d.parquet`.
5. Update `cache/precomputed/manifest.json` with paths, row/column counts, and date range.

Horizon labeling is cheap relative to feature engineering; reusing the enriched frame across horizons is an intentional micro-optimization inside the script.

### How to run

```bash
cd project
uv run python scripts/precompute_data.py
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--start` / `--end` | Date window (defaults from teaching pack constants) |
| `--symbols` | Override symbol list |
| `--horizons` | Override horizons (integers, days) |
| `--synthetic` | Use synthetic data instead of live downloads |

### What this buys you

- Streamlit teaching pages load Parquet instead of downloading and featurizing on every session.
- Grid search (`precompute_grid.py`) starts from the same packs, so experiment matrix cost is mostly model/CV work.
- Artifacts are portable (see `migrate_precomputed.py`) for Docker / demos.

### Contract (outputs)

```
project/cache/precomputed/
├── SPY_5d.parquet
├── SPY_10d.parquet
├── QQQ_5d.parquet
├── QQQ_10d.parquet
└── manifest.json
```

Each frame is wide: raw OHLCV + macros + all feature columns + dual label columns. Feature selection happens later at experiment time.

---

## 3. Cache layout (`cache.py`)

Root directory (env override: `CACHE_PATH`, default `./cache`):

```
cache/
├── data/              # Shared market-data CSVs (yfinance)
├── datasets/          # Content-hashed pre-selection feature sets
│   └── <sha256>/
│       ├── raw.csv
│       ├── X.csv
│       ├── y.csv
│       └── meta.csv
├── metadata.json      # Index: cache_key → dataset entry
└── runs/
    └── {model}-{feature_selection}-{YYYYMMDD_HHMMSS}/
        ├── config/    # pipeline, cv, params, metrics JSON
        ├── data/      # raw / X / y / meta / OOF CSVs
        ├── model/     # best_model.joblib (model + scaler + names)
        └── plots/     # diagnostics
```

Helpers:

| API | Role |
| --- | --- |
| `get_cache_root` | Resolve/create root; memoized with `lru_cache(maxsize=1)` |
| `get_data_dir` | Shared price cache |
| `get_datasets_dir` | Shared feature-dataset cache |
| `get_runs_dir` | Parent of per-run folders |
| `RunCache` | Create/write artifacts for one training run |

---

## 4. Shared dataset cache (pre feature-selection)

### Cache key

`dataset_cache_key(...)` hashes a canonical JSON payload:

- `symbol`, `start_date`, `end_date`
- `horizon`, `label_method`
- `use_synthetic`
- sorted `feature_names`

Same inputs → same SHA-256 → same `datasets/<hash>/` directory. Changing any field (including the feature step list) forces a miss and a rebuild.

### Hit / miss path (`RVPipeline.build_features`)

1. Compute key from pipeline config.
2. Unless `--force-rebuild` (or equivalent), call `load_cached_dataset(key)`.
3. **Hit**: restore `raw`, `X`, `y`, `meta` from CSV; skip download + feature build.
4. **Miss**: load data, `build_dataset(...)`, then `save_cached_dataset(...)` and update `metadata.json`.

Integrity check on lookup: all of `raw.csv`, `X.csv`, `y.csv`, `meta.csv` must exist; otherwise the entry is treated as incomplete and ignored.

### Why share across runs

Feature selection, model choice, and Optuna trials differ between runs, but the **full** feature matrix for a given symbol/window/horizon/label is often identical. Caching it once means repeated experiments only pay for selection + training.

---

## 5. Per-run artifact store (`RunCache`)

Each training run gets an isolated folder named:

```text
{model}-{feature_selection}-{timestamp}
```

(with unsafe characters sanitized). `RunCache` provides:

- `save_json` / `save_csv` for config and tabular outputs
- `save_model_bundle` / `load_model_bundle` via joblib for the best model, scaler, and feature names

This does not speed up a single run’s compute, but it:

- Avoids re-fitting when evaluation or the LLM briefing only needs stored predictions and weights
- Makes runs comparable and reproducible without re-executing the pipeline
- Keeps shared `datasets/` free of run-specific selection or OOF noise

---

## 6. How the two layers relate

```text
                    ┌─────────────────────────────────────┐
                    │  precompute_data.py (offline, once) │
                    │  → cache/precomputed/*.parquet      │
                    └──────────────┬──────────────────────┘
                                   │ teaching / grid UI
                                   ▼
┌──────────────────────────────────────────────────────────┐
│  Interactive / CLI training (RVPipeline)                 │
│                                                          │
│  datasets/ + metadata.json  ←── shared feature cache     │
│  runs/{id}/                 ←── per-run model artifacts  │
└──────────────────────────────────────────────────────────┘
```

| Concern | Teaching / Streamlit | Research CLI training |
| --- | --- | --- |
| Primary speedup | Precomputed Parquets | Hashed dataset cache |
| Feature selection | Applied at use time | After cache load |
| Run artifacts | Under `cache/precomputed/runs/` (grid) | Under `cache/runs/` |

They are independent: you can use teaching packs without the dataset hash cache, and vice versa.

---

## 7. Operational tips

1. **Rebuild teaching packs** after changing default feature steps, date range, or symbols:
   `uv run python scripts/precompute_data.py`
2. **Bypass dataset cache** when debugging feature code (`train.py --force-rebuild` or pipeline `force_rebuild_features=True`).
3. **Invalidate by key**: delete `cache/datasets/<hash>/` and the matching entry in `metadata.json`, or clear `datasets/` and `metadata.json` entirely for a full wipe.
4. **Disk vs CPU**: CSV dataset storage is simple and portable but larger/slower than Parquet; teaching packs already use Parquet for the hottest UI path.
5. **Set `CACHE_PATH`** if the default `./cache` is not writable or should live on a faster volume.

---

## 8. Summary

| Mechanism | File | Avoids repeating |
| --- | --- | --- |
| Teaching Parquets + manifest | `scripts/precompute_data.py` | Download + full feature build + dual labels for fixed symbols/horizons |
| Content-hashed `datasets/` | `utils/cache.py` | `load_data` + `build_features` across CLI runs with the same inputs |
| `RunCache` under `runs/` | `utils/cache.py` | Re-training / re-plotting when only artifacts are needed |

The design principle is **compute once at the right granularity**: pack-level for demos, config-hash level for research features, run-level for models and diagnostics.
