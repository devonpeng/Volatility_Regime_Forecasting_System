#!/usr/bin/env python3
"""Build teaching-pack Parquets: full timeline + all features (NO feature selection)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "project").is_dir()),
    None,
)
if _REPO_ROOT is not None and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.src.data.load_data import get_data_with_fallback
from project.src.features.core import apply_features
from project.src.features.rv import compute_forward_rv
from project.src.streamlit_app.precomputed import (
    DEFAULT_PACK_END,
    DEFAULT_PACK_START,
    MANIFEST_PATH,
    PRECOMPUTED_DIR,
    TEACHING_HORIZONS,
    TEACHING_SYMBOLS,
    feature_column_names,
    parquet_path,
)


def _add_dual_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if "ret_1d" not in df.columns:
        raise ValueError("ret_1d must exist before labels")
    data = df.copy()
    target_col = f"rv_fwd_{horizon}d"
    data[target_col] = compute_forward_rv(data["ret_1d"], horizon=horizon)
    thresh_exp = data[target_col].expanding(min_periods=horizon).median()
    thresh_med = data[target_col].rolling(window=horizon * 10).median()
    data["y_expanding"] = (data[target_col] > thresh_exp).astype("Int64")
    data["y_median"] = (data[target_col] > thresh_med).astype("Int64")
    return data


def build_wide_frame(
    symbol: str,
    horizon: int,
    start: str,
    end: str,
    *,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    raw = get_data_with_fallback(symbol, start, end, use_synthetic=use_synthetic)
    enriched, _ = apply_features(raw)
    wide = _add_dual_labels(enriched, horizon=horizon)
    wide.attrs["source"] = "precomputed"
    wide.attrs["symbol"] = symbol.upper()
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_PACK_START)
    parser.add_argument("--end", default=DEFAULT_PACK_END)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(TEACHING_SYMBOLS),
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(TEACHING_HORIZONS),
    )
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "start": args.start,
        "end": args.end,
        "synthetic": bool(args.synthetic),
        "feature_names": feature_column_names(),
        "files": [],
    }

    for symbol in args.symbols:
        print(f"==> Loading + features for {symbol.upper()} ...")
        raw = get_data_with_fallback(
            symbol.upper(),
            args.start,
            args.end,
            use_synthetic=args.synthetic,
        )
        enriched, _ = apply_features(raw)
        for horizon in args.horizons:
            print(f"    labeling horizon={horizon}d ...")
            wide = _add_dual_labels(enriched, horizon=horizon)
            wide.attrs["source"] = "precomputed"
            wide.attrs["symbol"] = symbol.upper()
            out = parquet_path(symbol, horizon)
            wide.to_parquet(out, index=False)
            entry = {
                "symbol": symbol.upper(),
                "horizon": horizon,
                "path": str(out.relative_to(PRECOMPUTED_DIR.parent.parent)),
                "n_rows": int(len(wide)),
                "n_cols": int(wide.shape[1]),
                "date_min": str(pd.to_datetime(wide["Date"]).min().date()),
                "date_max": str(pd.to_datetime(wide["Date"]).max().date()),
            }
            manifest["files"].append(entry)
            print(f"    wrote {out} ({entry['n_rows']} rows)")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
