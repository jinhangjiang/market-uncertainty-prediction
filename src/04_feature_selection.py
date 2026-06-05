"""
04_feature_selection.py
Feature selection via Lasso (CV) followed by SHAP-based ranking
using LinearExplainer on Ridge. No Random Forest, no permutation importance.

Usage:
    python src/04_feature_selection.py --dataset all --mode smoke
    python src/04_feature_selection.py --dataset reddit --mode full
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config, get_processed_path, get_dataset_output_dir
from src.utils.plotting import plot_shap_bar, plot_correlation_heatmap


def load_daily_features(cfg: dict, dataset: str) -> pd.DataFrame:
    """Load the 90-day (or longest available) feature file for feature selection."""
    windows = sorted(cfg.get("train_windows", [30, 60, 90]), reverse=True)
    for w in windows:
        path = get_processed_path(cfg, dataset, f"daily_features_{w}d.parquet")
        if os.path.exists(path):
            print(f"  Loading features from: {path}")
            return pd.read_parquet(path)
    raise FileNotFoundError(f"No daily feature files found for {dataset}. Run 03_feature_engineering.py first.")


def get_concept_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("concept_")]


def lasso_selection(X: pd.DataFrame, y: pd.Series) -> list:
    """LassoCV feature selection. Returns list of selected feature names."""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(cv=min(10, len(X) // 5), max_iter=5000, random_state=42, n_jobs=-1)
    lasso.fit(X_scaled, y)

    selected_mask = lasso.coef_ != 0
    selected = X.columns[selected_mask].tolist()
    n_total = X.shape[1]
    n_selected = len(selected)
    pct_filtered = round((1 - n_selected / n_total) * 100, 1)

    print(f"  LassoCV alpha={lasso.alpha_:.5f}")
    print(f"  Features: {n_total} total → {n_selected} selected ({pct_filtered}% filtered out)")

    return selected, lasso.alpha_, pct_filtered


def shap_ranking(X: pd.DataFrame, y: pd.Series, top_k: int = 10) -> pd.DataFrame:
    """
    Fit Ridge on Lasso-selected features, compute SHAP via LinearExplainer.
    Returns DataFrame sorted by mean |SHAP|.
    """
    import shap
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    r2 = ridge.score(X_scaled, y)
    print(f"  Ridge R² on training data: {r2:.4f}")

    explainer = shap.LinearExplainer(ridge, X_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_df["rank"] = range(1, len(shap_df) + 1)
    top_k_df = shap_df.head(top_k)
    print(f"  Top-{top_k} features by mean |SHAP|:")
    print(top_k_df[["rank", "feature", "mean_abs_shap"]].to_string(index=False))

    return shap_df, shap_values, X.columns.tolist()


def correlation_analysis(daily_df: pd.DataFrame, features: list, eui_lags: list = [1, 3, 7]) -> pd.DataFrame:
    """Spearman correlation between each feature and EUI at multiple lags."""
    corr_data = {}
    for lag in eui_lags:
        eui_col = f"eui_t{lag}"
        if eui_col not in daily_df.columns:
            continue
        corrs = {}
        for feat in features:
            if feat not in daily_df.columns:
                continue
            valid = daily_df[[feat, eui_col]].dropna()
            if len(valid) < 5:
                corrs[feat] = np.nan
                continue
            corr, _ = spearmanr(valid[feat], valid[eui_col])
            corrs[feat] = round(corr, 3)
        corr_data[f"EUI_t+{lag}"] = corrs

    corr_df = pd.DataFrame(corr_data)
    return corr_df


def process_dataset(dataset: str, cfg: dict, mode: str):
    print(f"\n=== Feature Selection: {dataset} ===")

    try:
        daily_df = load_daily_features(cfg, dataset)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return

    concept_cols = get_concept_cols(daily_df)
    if not concept_cols:
        print("  SKIP: No concept columns found.")
        return

    print(f"  {len(concept_cols)} concept features, {len(daily_df)} daily observations")

    target_col = "y"
    valid = daily_df[concept_cols + [target_col]].dropna()
    X = valid[concept_cols]
    y = valid[target_col]

    if len(X) < 20:
        print(f"  WARNING: Only {len(X)} valid rows. Skipping Lasso (need ≥ 20).")
        selected_features = concept_cols[:cfg.get("top_k_features", 10)]
        lasso_alpha = None
        pct_filtered = 0.0
    else:
        selected_features, lasso_alpha, pct_filtered = lasso_selection(X, y)

    if not selected_features:
        print("  WARNING: Lasso selected 0 features. Using all concept features.")
        selected_features = concept_cols

    top_k = cfg.get("top_k_features", 10)
    X_selected = valid[selected_features]

    if len(X_selected) >= 5 and len(selected_features) >= 2:
        shap_df, shap_values, feat_names = shap_ranking(X_selected, y, top_k=top_k)
    else:
        print("  Insufficient data for SHAP. Using all selected features.")
        shap_df = pd.DataFrame({
            "feature": selected_features,
            "mean_abs_shap": [1.0 / len(selected_features)] * len(selected_features),
            "rank": range(1, len(selected_features) + 1),
        })

    top_k_features = shap_df.head(top_k)["feature"].tolist()
    print(f"  Final top-{top_k} features: {top_k_features}")

    feat_dir = get_dataset_output_dir(cfg, dataset, "features")

    selected_info = {
        "dataset": dataset,
        "n_concept_features": len(concept_cols),
        "n_lasso_selected": len(selected_features),
        "pct_filtered_by_lasso": pct_filtered,
        "lasso_alpha": lasso_alpha,
        "top_k": top_k,
        "top_k_features": top_k_features,
    }
    with open(os.path.join(feat_dir, "selected_features.json"), "w") as f:
        json.dump(selected_info, f, indent=2, default=str)

    shap_df.to_csv(os.path.join(feat_dir, "shap_ranking.csv"), index=False)

    plot_shap_bar(
        shap_df, feature_col="feature", shap_col="mean_abs_shap",
        title=f"SHAP Feature Importance — {dataset.title()}",
        save_path=os.path.join(feat_dir, "shap_importance.png"),
        top_n=min(15, len(shap_df)),
    )

    eui_cols_present = [c for c in daily_df.columns if c.startswith("eui_t")]
    if eui_cols_present:
        corr_df = correlation_analysis(daily_df, top_k_features, eui_lags=[1, 3, 7])
        corr_df.to_csv(os.path.join(feat_dir, "correlation_eui.csv"))
        if not corr_df.empty:
            plot_correlation_heatmap(
                corr_df,
                title=f"Spearman Correlation: Top Concepts vs EUI — {dataset.title()}",
                save_path=os.path.join(feat_dir, "correlation_heatmap.png"),
            )

    print(f"  Selection complete. Outputs in {feat_dir}")


def main():
    parser = argparse.ArgumentParser(description="Lasso + SHAP feature selection")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "reddit", "stackexchange", "bogleheads"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, args.mode)
    datasets = ["reddit", "stackexchange", "bogleheads"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        process_dataset(dataset, cfg, args.mode)

    print("\n[04_feature_selection] DONE")


if __name__ == "__main__":
    main()
