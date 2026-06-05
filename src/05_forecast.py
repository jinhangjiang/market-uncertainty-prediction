"""
05_forecast.py
NeuralForecast walk-forward evaluation with LSTM, TFT, and TimeXer.
Runs base (no features) and topic (with GDCM concept volumes) variants.
Sensitivity analysis across 30/60/90-day training windows.

Usage:
    python src/05_forecast.py --dataset all --mode smoke
    python src/05_forecast.py --dataset reddit --mode full
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config, get_processed_path, get_dataset_output_dir
from src.utils.metrics import compute_all_metrics
from src.utils.device import get_accelerator
from src.utils.plotting import plot_nfa_over_time, plot_sc_over_time, plot_sensitivity


def load_features_and_selection(cfg: dict, dataset: str, window: int):
    """Load daily features and selected top-K concept feature names."""
    feat_path = get_processed_path(cfg, dataset, f"daily_features_{window}d.parquet")
    if not os.path.exists(feat_path):
        available_windows = sorted(cfg.get("train_windows", [30, 60, 90]), reverse=True)
        for w in available_windows:
            alt = get_processed_path(cfg, dataset, f"daily_features_{w}d.parquet")
            if os.path.exists(alt):
                print(f"  Using {w}d features (requested {window}d not found)")
                feat_path = alt
                break
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"No feature file found for {dataset}")

    daily_df = pd.read_parquet(feat_path)

    selection_path = os.path.join(
        get_dataset_output_dir(cfg, dataset, "features"), "selected_features.json"
    )
    if os.path.exists(selection_path):
        with open(selection_path) as f:
            sel = json.load(f)
        top_k_features = sel.get("top_k_features", [])
    else:
        concept_cols = [c for c in daily_df.columns if c.startswith("concept_")]
        top_k = cfg.get("top_k_features", 10)
        top_k_features = concept_cols[:top_k]
        print(f"  WARNING: No selection file found. Using first {top_k} concept columns.")

    return daily_df, [f for f in top_k_features if f in daily_df.columns]


def prepare_neuralforecast_df(daily_df: pd.DataFrame, hist_exog: list) -> pd.DataFrame:
    """Ensure NeuralForecast long-format with required columns."""
    df = daily_df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)
    if "unique_id" not in df.columns:
        df["unique_id"] = "EUI"
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    df = df.dropna(subset=["y"] + [c for c in hist_exog if c in df.columns])
    return df


def build_models(h: int, input_size: int, hist_exog: list, max_steps: int,
                 accelerator: str, mode: str) -> dict:
    """Build all 6 NeuralForecast models (base + topic × LSTM/TFT/TimeXer)."""
    from neuralforecast.models import LSTM, TFT, TimeXer
    from neuralforecast.losses.pytorch import MAE

    trainer_kw = {
        "accelerator": accelerator,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }
    if accelerator == "gpu":
        trainer_kw["devices"] = 1

    lstm_hidden = 32 if mode == "smoke" else 128
    tft_hidden = 16 if mode == "smoke" else 64
    timexer_hidden = 64 if mode == "smoke" else 256

    models = {}

    models["LSTM_base"] = LSTM(
        h=h,
        input_size=input_size,
        encoder_hidden_size=lstm_hidden,
        encoder_n_layers=2,
        dropout_prob_theta=0.1,
        loss=MAE(),
        max_steps=max_steps,
        early_stop_patience_steps=5,
        scaler_type="standard",
        trainer_kwargs=trainer_kw,
        alias="LSTM_base",
    )

    if hist_exog:
        models["LSTM_topic"] = LSTM(
            h=h,
            input_size=input_size,
            encoder_hidden_size=lstm_hidden,
            encoder_n_layers=2,
            hist_exog_list=hist_exog,
            dropout_prob_theta=0.1,
            loss=MAE(),
            max_steps=max_steps,
            early_stop_patience_steps=5,
            scaler_type="standard",
            trainer_kwargs=trainer_kw,
            alias="LSTM_topic",
        )

        models["LSTM_topic_single"] = LSTM(
            h=1,
            input_size=input_size,
            encoder_hidden_size=lstm_hidden,
            encoder_n_layers=2,
            hist_exog_list=hist_exog,
            dropout_prob_theta=0.1,
            loss=MAE(),
            max_steps=max_steps,
            early_stop_patience_steps=5,
            scaler_type="standard",
            trainer_kwargs=trainer_kw,
            alias="LSTM_topic_single",
        )

    models["TFT_base"] = TFT(
        h=h,
        input_size=input_size,
        hidden_size=tft_hidden,
        n_head=min(4, tft_hidden // 8),
        attn_dropout=0.1,
        dropout=0.1,
        loss=MAE(),
        max_steps=max_steps,
        early_stop_patience_steps=5,
        scaler_type="standard",
        trainer_kwargs=trainer_kw,
        alias="TFT_base",
    )

    if hist_exog:
        models["TFT_topic"] = TFT(
            h=h,
            input_size=input_size,
            hidden_size=tft_hidden,
            n_head=min(4, tft_hidden // 8),
            attn_dropout=0.1,
            dropout=0.1,
            hist_exog_list=hist_exog,
            loss=MAE(),
            max_steps=max_steps,
            early_stop_patience_steps=5,
            scaler_type="standard",
            trainer_kwargs=trainer_kw,
            alias="TFT_topic",
        )

    try:
        models["TimeXer_base"] = TimeXer(
            h=h,
            input_size=input_size,
            n_series=1,
            hidden_size=timexer_hidden,
            n_heads=min(8, timexer_hidden // 32),
            e_layers=2,
            patch_len=min(16, input_size // 2),
            dropout=0.1,
            loss=MAE(),
            max_steps=max_steps,
            early_stop_patience_steps=5,
            scaler_type="standard",
            trainer_kwargs=trainer_kw,
            alias="TimeXer_base",
        )

        if hist_exog:
            models["TimeXer_topic"] = TimeXer(
                h=h,
                input_size=input_size,
                n_series=1,
                hist_exog_list=hist_exog,
                hidden_size=timexer_hidden,
                n_heads=min(8, timexer_hidden // 32),
                e_layers=2,
                patch_len=min(16, input_size // 2),
                dropout=0.1,
                loss=MAE(),
                max_steps=max_steps,
                early_stop_patience_steps=5,
                scaler_type="standard",
                trainer_kwargs=trainer_kw,
                alias="TimeXer_topic",
            )
    except Exception as e:
        print(f"  WARNING: TimeXer not available ({e}). Skipping TimeXer models.")

    return models


def run_walk_forward(df: pd.DataFrame, models_dict: dict, h: int, window: int,
                     hist_exog: list, out_dir: str, dataset: str, mode: str) -> pd.DataFrame:
    """Run NeuralForecast cross_validation for each model, collect results."""
    from neuralforecast import NeuralForecast

    n_obs = len(df["ds"].unique())
    min_train = window
    n_windows = max(1, (n_obs - min_train - h) // 1)
    if mode == "smoke":
        n_windows = min(n_windows, 3)

    print(f"  Walk-forward: {n_obs} days, window={window}, h={h}, n_windows={n_windows}")

    all_results = []

    for model_name, model in models_dict.items():
        print(f"  Running model: {model_name}")
        try:
            nf = NeuralForecast(models=[model], freq="D")
            cv_df = nf.cross_validation(
                df=df,
                n_windows=n_windows,
                step_size=1,
                refit=False,
            )

            pred_col = [c for c in cv_df.columns if c not in ("unique_id", "ds", "y", "cutoff")]
            if not pred_col:
                print(f"  WARNING: No prediction column found for {model_name}")
                continue
            pred_col = pred_col[0]

            for cutoff, grp in cv_df.groupby("cutoff"):
                actual = grp["y"].values
                predicted = grp[pred_col].values
                metrics = compute_all_metrics(actual, predicted)
                all_results.append({
                    "dataset": dataset,
                    "model": model_name,
                    "train_window": window,
                    "cutoff": cutoff,
                    **metrics,
                })

            model_path = os.path.join(out_dir, f"{model_name}_{window}d_cv.parquet")
            cv_df.to_parquet(model_path, index=False)
            print(f"    Saved CV results: {model_path}")

        except Exception as e:
            print(f"  ERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return pd.DataFrame(all_results)


def process_dataset(dataset: str, cfg: dict, mode: str):
    print(f"\n=== Forecasting: {dataset} ===")

    train_windows = cfg.get("train_windows", [30, 60, 90])
    if isinstance(train_windows, int):
        train_windows = [train_windows]
    h = cfg.get("forecast_horizon", 7)
    max_steps = cfg.get("max_steps", 500) if mode == "full" else cfg.get("max_steps", 10)
    accelerator = get_accelerator()
    print(f"  Device: {accelerator} | h={h} | max_steps={max_steps}")

    forecast_dir = get_dataset_output_dir(cfg, dataset, "forecasts")
    all_results = []

    for window in train_windows:
        print(f"\n  --- Training window: {window}d ---")
        try:
            daily_df, hist_exog = load_features_and_selection(cfg, dataset, window)
        except FileNotFoundError as e:
            print(f"  SKIP window {window}: {e}")
            continue

        print(f"  hist_exog features ({len(hist_exog)}): {hist_exog[:5]}{'...' if len(hist_exog) > 5 else ''}")

        df_nf = prepare_neuralforecast_df(daily_df, hist_exog)
        input_size = window

        models_dict = build_models(
            h=h,
            input_size=input_size,
            hist_exog=hist_exog,
            max_steps=max_steps,
            accelerator=accelerator,
            mode=mode,
        )
        print(f"  Models: {list(models_dict.keys())}")

        window_results = run_walk_forward(
            df=df_nf,
            models_dict=models_dict,
            h=h,
            window=window,
            hist_exog=hist_exog,
            out_dir=forecast_dir,
            dataset=dataset,
            mode=mode,
        )
        all_results.append(window_results)

    if not all_results:
        print(f"  No results for {dataset}.")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    results_path = os.path.join(forecast_dir, "all_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n  All results saved: {results_path}")

    summary = results_df.groupby(["model", "train_window"]).agg(
        NFA_mean=("NFA", "mean"),
        NFA_std=("NFA", "std"),
        SC_mean=("SC", "mean"),
        MAE_mean=("MAE", "mean"),
        RMSE_mean=("RMSE", "mean"),
        n_windows=("NFA", "count"),
    ).reset_index()
    summary_path = os.path.join(forecast_dir, "summary_table.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Summary table saved: {summary_path}")
    print(summary.to_string(index=False))

    if not results_df.empty:
        results_df["ds"] = pd.to_datetime(results_df["cutoff"])
        plot_nfa_over_time(
            results_df, model_col="model", nfa_col="NFA", date_col="ds",
            title=f"NFA Over Time — {dataset.title()}",
            save_path=os.path.join(forecast_dir, "nfa_over_time.png"),
        )
        plot_sc_over_time(
            results_df, model_col="model", sc_col="SC", date_col="ds",
            title=f"Spearman Correlation Over Time — {dataset.title()}",
            save_path=os.path.join(forecast_dir, "sc_over_time.png"),
        )
        if len(train_windows) > 1:
            plot_sensitivity(
                summary, model_col="model", window_col="train_window", nfa_col="NFA_mean",
                title=f"NFA Sensitivity: Training Window — {dataset.title()}",
                save_path=os.path.join(forecast_dir, "sensitivity_window.png"),
            )


def main():
    parser = argparse.ArgumentParser(description="NeuralForecast walk-forward evaluation")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "reddit", "stackexchange", "bogleheads"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, args.mode)
    datasets = ["reddit", "stackexchange", "bogleheads"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        process_dataset(dataset, cfg, args.mode)

    print("\n[05_forecast] DONE")


if __name__ == "__main__":
    main()
