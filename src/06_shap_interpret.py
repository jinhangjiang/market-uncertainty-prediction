"""
06_shap_interpret.py
SHAP interpretation for NeuralForecast models.
Extracts SHAP values for GDCM concept volume features (hist_exog)
using GradientExplainer on the underlying PyTorch modules.
Also builds concept-to-interpretation bridge table.

Usage:
    python src/06_shap_interpret.py --dataset all --mode smoke
    python src/06_shap_interpret.py --dataset reddit --mode full
"""
import argparse
import glob
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config, get_processed_path, get_dataset_output_dir
from src.utils.plotting import plot_shap_bar, save_fig


def load_concept_words(gdcm_dir: str) -> dict:
    """Load concept word lists from GDCM output. Returns {concept_id: [words]}."""
    concepts = {}
    search_dirs = [
        os.path.join(gdcm_dir, "final_model", "concept"),
        os.path.join(gdcm_dir, "final_model", "concepts"),
        os.path.join(gdcm_dir, "concepts"),
    ]
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for fpath in sorted(glob.glob(os.path.join(search_dir, "*.txt"))):
            cname = os.path.splitext(os.path.basename(fpath))[0]
            with open(fpath) as f:
                words = [line.strip() for line in f if line.strip()]
            concepts[cname] = words[:20]
        if concepts:
            break

    if not concepts:
        print("  WARNING: No concept word files found. Concept names will be IDs only.")
    return concepts


def load_cv_predictions(forecast_dir: str, window: int) -> dict:
    """Load cross-validation prediction parquets per model."""
    pattern = os.path.join(forecast_dir, f"*_{window}d_cv.parquet")
    files = glob.glob(pattern)
    cv_data = {}
    for f in files:
        model_name = os.path.basename(f).replace(f"_{window}d_cv.parquet", "")
        cv_data[model_name] = pd.read_parquet(f)
    return cv_data


def shap_on_neuralforecast_model(model_obj, X_tensor, feature_names: list,
                                  n_background: int = 50) -> np.ndarray:
    """
    Compute SHAP values using GradientExplainer on a NeuralForecast model's
    underlying PyTorch network. Returns array of shape (n_samples, n_features).
    """
    import torch
    import shap

    try:
        if hasattr(model_obj, "model"):
            net = model_obj.model
        elif hasattr(model_obj, "_model"):
            net = model_obj._model
        else:
            net = model_obj

        net.eval()

        background = X_tensor[:min(n_background, len(X_tensor))]
        explainer = shap.GradientExplainer(net, background)
        shap_values = explainer.shap_values(X_tensor)

        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=-1).mean(axis=-1)
        shap_values = np.asarray(shap_values)

        if shap_values.ndim > 2:
            shap_values = shap_values.mean(axis=tuple(range(1, shap_values.ndim - 1)))

        return shap_values

    except Exception as e:
        print(f"  GradientExplainer failed: {e}. Falling back to permutation-based SHAP estimate.")
        return permutation_shap_fallback(net, X_tensor, feature_names)


def permutation_shap_fallback(net, X_tensor, feature_names: list) -> np.ndarray:
    """Simple permutation-based feature importance as SHAP fallback."""
    import torch
    net.eval()
    with torch.no_grad():
        base_pred = net(X_tensor)
        if isinstance(base_pred, (list, tuple)):
            base_pred = base_pred[0]
        base_pred = base_pred.mean().item()

    n_samples, n_features = X_tensor.shape[0], X_tensor.shape[-1]
    importances = np.zeros((n_samples, n_features))

    for feat_idx in range(n_features):
        X_perm = X_tensor.clone()
        perm_idx = torch.randperm(n_samples)
        X_perm[..., feat_idx] = X_tensor[perm_idx, ..., feat_idx]
        with torch.no_grad():
            perm_pred = net(X_perm)
            if isinstance(perm_pred, (list, tuple)):
                perm_pred = perm_pred[0]
        diff = (base_pred - perm_pred.mean().item())
        importances[:, feat_idx] = diff

    return importances


def build_concept_shap_table(shap_df: pd.DataFrame, concept_words: dict) -> pd.DataFrame:
    """
    Merge SHAP rankings with concept word lists to produce interpretable table.
    Columns: feature, top_words, mean_abs_shap, direction, rank
    """
    rows = []
    for _, row in shap_df.iterrows():
        feat = row["feature"]
        cname = feat if feat in concept_words else feat.replace("_", " ")
        words = concept_words.get(feat, concept_words.get(cname, []))
        rows.append({
            "feature": feat,
            "top_words": ", ".join(words[:10]) if words else "N/A",
            "mean_abs_shap": round(row.get("mean_abs_shap", 0), 6),
            "direction": row.get("direction", "unknown"),
            "rank": row.get("rank", 0),
        })
    return pd.DataFrame(rows)


def compute_shap_from_cv_data(cv_df: pd.DataFrame, hist_exog: list,
                               model_name: str, mode: str) -> pd.DataFrame:
    """
    Compute approximate SHAP feature importance from CV predictions
    using the concept volume features directly (model-agnostic approach).
    Uses LinearExplainer on a Ridge fitted to predictions.
    """
    import shap
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    feat_cols = [c for c in hist_exog if c in cv_df.columns]
    if not feat_cols:
        print(f"  WARNING: No hist_exog columns found in CV df for {model_name}")
        return pd.DataFrame()

    pred_col = [c for c in cv_df.columns if c not in ("unique_id", "ds", "y", "cutoff") + hist_exog]
    if not pred_col:
        return pd.DataFrame()
    pred_col = pred_col[0]

    valid = cv_df[feat_cols + [pred_col]].dropna()
    if len(valid) < 5:
        return pd.DataFrame()

    X = valid[feat_cols].values
    y_pred = valid[pred_col].values

    n_bg = min(50 if mode == "full" else 10, len(X))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y_pred)

    try:
        explainer = shap.LinearExplainer(ridge, X_scaled[:n_bg], feature_perturbation="interventional")
        shap_vals = explainer.shap_values(X_scaled)
    except Exception as e:
        print(f"  LinearExplainer failed: {e}")
        shap_vals = np.abs(X_scaled * ridge.coef_)

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    mean_shap = shap_vals.mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap,
        "direction": ["positive" if v >= 0 else "negative" for v in mean_shap],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["rank"] = range(1, len(shap_df) + 1)

    return shap_df


def process_dataset(dataset: str, cfg: dict, mode: str):
    print(f"\n=== SHAP Interpretation: {dataset} ===")

    forecast_dir = get_dataset_output_dir(cfg, dataset, "forecasts")
    shap_dir = get_dataset_output_dir(cfg, dataset, "shap")
    gdcm_dir = get_dataset_output_dir(cfg, dataset, "gdcm")
    feat_dir = get_dataset_output_dir(cfg, dataset, "features")

    concept_words = load_concept_words(gdcm_dir)
    print(f"  Loaded concept words for {len(concept_words)} concepts")

    selection_path = os.path.join(feat_dir, "selected_features.json")
    if os.path.exists(selection_path):
        with open(selection_path) as f:
            sel = json.load(f)
        hist_exog = sel.get("top_k_features", [])
    else:
        hist_exog = []
        print("  WARNING: No feature selection file found.")

    train_windows = cfg.get("train_windows", [30, 60, 90])
    if isinstance(train_windows, int):
        train_windows = [train_windows]

    all_shap = []

    for window in train_windows:
        cv_data = load_cv_predictions(forecast_dir, window)
        if not cv_data:
            print(f"  No CV data found for window={window}d")
            continue

        for model_name, cv_df in cv_data.items():
            print(f"  Computing SHAP: {model_name}, window={window}d")

            if not hist_exog:
                print(f"  Skipping {model_name}: no hist_exog features.")
                continue

            feat_cols_in_cv = [c for c in hist_exog if c in cv_df.columns]
            if not feat_cols_in_cv:
                print(f"  No concept columns in CV df for {model_name}. Skipping.")
                continue

            shap_df = compute_shap_from_cv_data(cv_df, hist_exog, model_name, mode)
            if shap_df.empty:
                print(f"  Empty SHAP df for {model_name}")
                continue

            shap_df["model"] = model_name
            shap_df["train_window"] = window
            shap_df["dataset"] = dataset
            all_shap.append(shap_df)

            csv_path = os.path.join(shap_dir, f"shap_{model_name}_{window}d.csv")
            shap_df.to_csv(csv_path, index=False)

            plot_shap_bar(
                shap_df, feature_col="feature", shap_col="mean_abs_shap",
                title=f"SHAP Importance — {dataset.title()} | {model_name} | {window}d",
                save_path=os.path.join(shap_dir, f"shap_{model_name}_{window}d.png"),
                top_n=min(15, len(shap_df)),
            )

    if not all_shap:
        print("  No SHAP results computed.")
        return

    combined_shap = pd.concat(all_shap, ignore_index=True)

    agg_shap = combined_shap.groupby("feature").agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        mean_shap=("mean_shap", "mean"),
    ).reset_index().sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    agg_shap["rank"] = range(1, len(agg_shap) + 1)
    agg_shap["direction"] = agg_shap["mean_shap"].apply(lambda v: "positive" if v >= 0 else "negative")

    interp_table = build_concept_shap_table(agg_shap, concept_words)
    interp_path = os.path.join(shap_dir, "concept_shap_summary.csv")
    interp_table.to_csv(interp_path, index=False)
    print(f"  Concept SHAP summary saved: {interp_path}")
    print(interp_table.head(10).to_string(index=False))

    plot_shap_bar(
        agg_shap, feature_col="feature", shap_col="mean_abs_shap",
        title=f"Aggregated SHAP Importance — {dataset.title()} (all models & windows)",
        save_path=os.path.join(shap_dir, "shap_aggregated.png"),
        top_n=min(15, len(agg_shap)),
    )

    combined_shap.to_csv(os.path.join(shap_dir, "all_shap_values.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="SHAP interpretation for forecasting models")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "reddit", "stackexchange", "bogleheads"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, args.mode)
    datasets = ["reddit", "stackexchange", "bogleheads"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        process_dataset(dataset, cfg, args.mode)

    print("\n[06_shap_interpret] DONE")


if __name__ == "__main__":
    main()
