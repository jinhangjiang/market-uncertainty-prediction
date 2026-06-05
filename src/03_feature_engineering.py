"""
03_feature_engineering.py
Aggregates GDCM concept weights to daily concept volumes,
builds NeuralForecast-compatible time series feature matrices,
and saves three variants (30/60/90-day context windows).

Usage:
    python src/03_feature_engineering.py --dataset all --mode smoke
    python src/03_feature_engineering.py --dataset reddit --mode full
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config, get_processed_path, get_dataset_output_dir


def load_sentences_with_concepts(gdcm_dir: str) -> pd.DataFrame:
    """Load merged sentences+concepts parquet. Falls back to separate files."""
    merged_path = os.path.join(gdcm_dir, "sentences_with_concepts.parquet")
    if os.path.exists(merged_path):
        return pd.read_parquet(merged_path)

    weights_path = os.path.join(gdcm_dir, "doc_concept_weights.parquet")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(gdcm_dir, "final_model", "doc_concept_weights.parquet")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Cannot find concept weights in {gdcm_dir}")

    raise FileNotFoundError(
        f"sentences_with_concepts.parquet not found. "
        f"Concept weights exist at {weights_path} but cannot be merged without sentence metadata."
    )


def get_concept_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("concept_")]


def aggregate_daily_features(df: pd.DataFrame, eui_col: str = "eui_t1") -> pd.DataFrame:
    """
    Aggregate concept weights to daily totals.
    Also computes daily post count.
    Returns long-format DataFrame with columns:
        unique_id, ds, y, concept_0..N, daily_post_count
    """
    concept_cols = get_concept_cols(df)
    if not concept_cols:
        raise ValueError("No concept columns found. Check GDCM output.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    agg = df.groupby("date").agg(
        **{c: (c, "sum") for c in concept_cols},
        daily_post_count=("text", "count") if "text" in df.columns else ("date", "count"),
    ).reset_index()

    eui_by_date = df.groupby("date")[eui_col].first().reset_index().rename(columns={eui_col: "y"})
    daily = agg.merge(eui_by_date, on="date", how="inner")
    daily = daily.rename(columns={"date": "ds"})
    daily = daily.sort_values("ds").reset_index(drop=True)
    daily["unique_id"] = "EUI"

    cols = ["unique_id", "ds", "y"] + concept_cols + ["daily_post_count"]
    daily = daily[[c for c in cols if c in daily.columns]]
    return daily


def build_window_variants(daily_df: pd.DataFrame, windows: list) -> dict:
    """
    For each window size, build a DataFrame that only includes rows where
    we have at least `window` days of history (i.e., drop first `window` rows).
    NeuralForecast handles windowing internally; this just records metadata.
    """
    variants = {}
    for w in windows:
        df_w = daily_df.copy()
        df_w.attrs["train_window"] = w
        variants[w] = df_w
    return variants


def process_dataset(dataset: str, cfg: dict, mode: str):
    print(f"\n=== Feature Engineering: {dataset} ===")

    gdcm_dir = get_dataset_output_dir(cfg, dataset, "gdcm")

    try:
        df = load_sentences_with_concepts(gdcm_dir)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return

    print(f"  Loaded {len(df)} sentences with concepts")
    concept_cols = get_concept_cols(df)
    print(f"  Found {len(concept_cols)} concept columns")

    daily = aggregate_daily_features(df, eui_col="eui_t1")
    print(f"  Daily feature matrix: {daily.shape} | dates: {daily['ds'].min().date()} – {daily['ds'].max().date()}")
    print(f"  EUI (y) stats: mean={daily['y'].mean():.2f}, std={daily['y'].std():.2f}")

    train_windows = cfg.get("train_windows", [30, 60, 90])
    if isinstance(train_windows, int):
        train_windows = [train_windows]

    for window in train_windows:
        out_path = get_processed_path(cfg, dataset, f"daily_features_{window}d.parquet")
        daily.to_parquet(out_path, index=False)
        print(f"  Saved {window}d features: {out_path} ({len(daily)} rows)")

    concept_stats = daily[concept_cols].describe().T
    stats_path = os.path.join(get_dataset_output_dir(cfg, dataset, "features"), "concept_stats.csv")
    concept_stats.to_csv(stats_path)
    print(f"  Concept stats saved: {stats_path}")

    print(f"  Top concepts by mean volume:")
    mean_vols = daily[concept_cols].mean().sort_values(ascending=False)
    print(mean_vols.head(10).to_string())


def main():
    parser = argparse.ArgumentParser(description="Daily concept volume feature engineering")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "reddit", "stackexchange", "bogleheads"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, args.mode)
    datasets = ["reddit", "stackexchange", "bogleheads"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        process_dataset(dataset, cfg, args.mode)

    print("\n[03_feature_engineering] DONE")


if __name__ == "__main__":
    main()
