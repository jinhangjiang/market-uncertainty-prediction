"""
02_gdcm_train.py
Runs GDCM (Guided Diverse Concept Miner) to replace BERTopic.
GDCM is guided by same-day EUI (eui_t0) — no future data leakage.
Number of concepts is empirically determined via grid search.

Usage:
    python src/02_gdcm_train.py --dataset all --mode smoke
    python src/02_gdcm_train.py --dataset reddit --mode full
"""
import argparse
import json
import os
import subprocess
import sys
import shutil
import tempfile
import itertools
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config, get_processed_path, get_dataset_output_dir


GDCM_REPO_URL = "https://github.com/cygit/gdcm.git"
GDCM_DIR = "gdcm"


def ensure_gdcm_installed():
    """Clone GDCM repo and install as editable package if not present."""
    if not os.path.exists(GDCM_DIR):
        print(f"Cloning GDCM from {GDCM_REPO_URL} ...")
        subprocess.run(["git", "clone", GDCM_REPO_URL, GDCM_DIR], check=True)
    else:
        print(f"GDCM directory already exists: {GDCM_DIR}")

    src_path = os.path.join(GDCM_DIR, "src")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"GDCM src not found at {src_path}")

    try:
        import gdcm as _gdcm_check  # noqa: F401
        print("GDCM already importable.")
    except ImportError:
        print("Installing GDCM as editable package ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", src_path], check=True)
        print("GDCM installed.")


def write_gdcm_csv(df: pd.DataFrame, out_path: str, text_col: str = "text", label_col: str = "eui_t0"):
    """Write a CSV file that GDCM can consume with --csv-path."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    df[[text_col, label_col]].dropna().rename(
        columns={text_col: "docs", label_col: "labels"}
    ).to_csv(out_path, index=False)
    print(f"  GDCM input CSV: {out_path} ({len(df)} rows)")


def normalize_labels(df: pd.DataFrame, label_col: str = "eui_t0") -> pd.DataFrame:
    """Normalize EUI labels to [0, 1] range for GDCM training."""
    df = df.copy()
    lo, hi = df[label_col].min(), df[label_col].max()
    if hi - lo > 0:
        df[label_col] = (df[label_col] - lo) / (hi - lo)
    else:
        df[label_col] = 0.0
    return df


def run_gdcm_grid_search(csv_path: str, config_path: str, out_dir: str, mode: str) -> dict:
    """
    Run GDCM grid search via CLI. Returns best hyperparameter dict.
    Falls back to direct Python training if CLI unavailable.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path) as f:
        grid_cfg = json.load(f)

    grid_cfg["csv-path"] = csv_path
    grid_cfg["csv-text"] = "docs"
    grid_cfg["csv-label"] = "labels"
    grid_cfg["out_dir"] = out_dir

    if mode == "smoke":
        grid_cfg["gpus"] = []

    tmp_cfg = os.path.join(out_dir, "gdcm_run_config.json")
    with open(tmp_cfg, "w") as f:
        json.dump(grid_cfg, f, indent=2)

    try:
        result = subprocess.run(
            ["gdcm", "grid-search", tmp_cfg],
            capture_output=True, text=True, timeout=3600
        )
        print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        if result.returncode != 0:
            print(f"GDCM grid-search stderr: {result.stderr[-2000:]}")
            raise RuntimeError("GDCM grid-search CLI failed")

        results_csv = os.path.join(out_dir, "results.csv")
        if os.path.exists(results_csv):
            results_df = pd.read_csv(results_csv)
            best_row = results_df.sort_values("best_val_loss").iloc[0]
            best_params = best_row.to_dict()
            print(f"  Best GDCM config: nconcepts={best_params.get('nconcepts')}, "
                  f"val_loss={best_params.get('best_val_loss', 'N/A'):.4f}")
            return best_params
        else:
            print("WARNING: results.csv not found, using first config from grid")
            return extract_first_config(grid_cfg)

    except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError) as e:
        print(f"GDCM CLI unavailable or failed ({e}). Running Python fallback grid search.")
        return python_gdcm_grid_search(csv_path, grid_cfg, out_dir)


def extract_first_config(grid_cfg: dict) -> dict:
    """Extract first combination from grid config."""
    params = {}
    for k, v in grid_cfg.get("gdcm_params", {}).items():
        params[k] = v[0] if isinstance(v, list) else v
    for k, v in grid_cfg.get("fit_params", {}).items():
        params[k] = v[0] if isinstance(v, list) else v
    for k, v in grid_cfg.get("dataset_params", {}).items():
        params[k] = v[0] if isinstance(v, list) else v
    return params


def python_gdcm_grid_search(csv_path: str, grid_cfg: dict, out_dir: str) -> dict:
    """
    Python-level grid search over GDCM hyperparameters.
    Trains each config, records validation loss, returns best.
    """
    try:
        from gdcm.model import GuidedDiverseConceptMiner
        from gdcm.dataset.csv_dataset import CSVDataset
    except ImportError:
        print("ERROR: Cannot import GDCM. Ensure `pip install -e gdcm/src` succeeded.")
        return extract_first_config(grid_cfg)

    gdcm_keys = list(grid_cfg["gdcm_params"].keys())
    gdcm_vals = [grid_cfg["gdcm_params"][k] for k in gdcm_keys]
    fit_keys = list(grid_cfg["fit_params"].keys())
    fit_vals = [grid_cfg["fit_params"][k] for k in fit_keys]
    ds_keys = list(grid_cfg["dataset_params"].keys())
    ds_vals = [grid_cfg["dataset_params"][k] for k in ds_keys]

    all_combos = list(itertools.product(
        itertools.product(*gdcm_vals),
        itertools.product(*fit_vals),
        itertools.product(*ds_vals),
    ))

    print(f"  Python GDCM grid search: {len(all_combos)} combinations")
    best_loss = float("inf")
    best_params = {}

    results_rows = []
    for (gv, fv, dv) in tqdm(all_combos, desc="  GDCM grid search"):
        gp = dict(zip(gdcm_keys, gv))
        fp = dict(zip(fit_keys, fv))
        dp = dict(zip(ds_keys, dv))

        try:
            dataset = CSVDataset(
                csv_path=csv_path,
                text_col="docs",
                label_col="labels",
                window_size=dp.get("window_size", 4),
                min_df=dp.get("min_df", 0.01),
                max_df=dp.get("max_df", 0.8),
            )
            data = dataset.load_data()

            model = GuidedDiverseConceptMiner(
                vocab_size=data["vocab_size"],
                embed_dim=gp.get("embed_dim", 50),
                nconcepts=gp.get("nconcepts", 10),
                nnegs=gp.get("nnegs", 15),
                lam=gp.get("lam", 10),
                rho=gp.get("rho", 100),
                eta=gp.get("eta", 10),
                inductive=gp.get("inductive", True),
                inductive_dropout=gp.get("inductive_dropout", 0.0),
                hidden_size=gp.get("hidden_size", 100),
                num_layers=gp.get("num_layers", 1),
            )

            val_loss = model.fit(
                data,
                lr=fp.get("lr", 0.01),
                nepochs=fp.get("nepochs", 30),
                pred_only_epochs=fp.get("pred_only_epochs", 15),
                batch_size=fp.get("batch_size", 1024),
                grad_clip=fp.get("grad_clip", 1024),
                val_fraction=0.1,
            )

            row = {**gp, **fp, **dp, "best_val_loss": val_loss}
            results_rows.append(row)

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = row.copy()
                print(f"    New best: nconcepts={gp['nconcepts']}, val_loss={val_loss:.4f}")

        except Exception as e:
            print(f"    Config failed: {gp} | {e}")
            continue

    if results_rows:
        pd.DataFrame(results_rows).to_csv(os.path.join(out_dir, "results.csv"), index=False)

    if not best_params:
        print("WARNING: All configs failed. Using first config.")
        best_params = extract_first_config(grid_cfg)

    return best_params


def run_gdcm_final(csv_path: str, best_params: dict, out_dir: str, mode: str) -> str:
    """
    Train final GDCM model with best hyperparameters.
    Returns path to output directory containing concept weights.
    """
    os.makedirs(out_dir, exist_ok=True)
    nconcepts = int(best_params.get("nconcepts", 10))
    print(f"  Training final GDCM: nconcepts={nconcepts}")

    try:
        result = subprocess.run(
            [
                "gdcm", "train", "csv", out_dir,
                "--csv-path", csv_path,
                "--csv-text", "docs",
                "--csv-label", "labels",
                "--nconcepts", str(nconcepts),
                "--embed-dim", str(int(best_params.get("embed_dim", 50))),
                "--nnegs", str(int(best_params.get("nnegs", 15))),
                "--lam", str(int(best_params.get("lam", 10))),
                "--rho", str(int(best_params.get("rho", 100))),
                "--eta", str(int(best_params.get("eta", 10))),
                "--lr", str(float(best_params.get("lr", 0.01))),
                "--batch", str(int(best_params.get("batch_size", 1024))),
                "--nepochs", str(int(best_params.get("nepochs", 30))),
                "--gpu", "0" if mode == "full" else "-1",
            ],
            capture_output=True, text=True, timeout=7200
        )
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            print(f"GDCM train stderr: {result.stderr[-2000:]}")
            raise RuntimeError("GDCM train CLI failed")
        return out_dir

    except (FileNotFoundError, RuntimeError) as e:
        print(f"GDCM CLI failed ({e}). Running Python fallback training.")
        return python_gdcm_final(csv_path, best_params, out_dir)


def python_gdcm_final(csv_path: str, best_params: dict, out_dir: str) -> str:
    """Python fallback for final GDCM training."""
    try:
        from gdcm.model import GuidedDiverseConceptMiner
        from gdcm.dataset.csv_dataset import CSVDataset
    except ImportError:
        print("ERROR: GDCM not importable. Skipping final training.")
        return out_dir

    dataset = CSVDataset(
        csv_path=csv_path,
        text_col="docs",
        label_col="labels",
        window_size=int(best_params.get("window_size", 4)),
        min_df=float(best_params.get("min_df", 0.01)),
        max_df=float(best_params.get("max_df", 0.8)),
    )
    data = dataset.load_data()

    model = GuidedDiverseConceptMiner(
        vocab_size=data["vocab_size"],
        embed_dim=int(best_params.get("embed_dim", 50)),
        nconcepts=int(best_params.get("nconcepts", 10)),
        nnegs=int(best_params.get("nnegs", 15)),
        lam=int(best_params.get("lam", 10)),
        rho=int(best_params.get("rho", 100)),
        eta=int(best_params.get("eta", 10)),
        inductive=bool(best_params.get("inductive", True)),
        inductive_dropout=float(best_params.get("inductive_dropout", 0.0)),
        hidden_size=int(best_params.get("hidden_size", 100)),
        num_layers=int(best_params.get("num_layers", 1)),
    )

    model.fit(
        data,
        lr=float(best_params.get("lr", 0.01)),
        nepochs=int(best_params.get("nepochs", 30)),
        pred_only_epochs=int(best_params.get("pred_only_epochs", 15)),
        batch_size=int(best_params.get("batch_size", 1024)),
        grad_clip=int(best_params.get("grad_clip", 1024)),
    )

    concept_weights = model.get_document_concept_weights(data)
    weights_df = pd.DataFrame(
        concept_weights,
        columns=[f"concept_{i}" for i in range(concept_weights.shape[1])]
    )

    weights_path = os.path.join(out_dir, "doc_concept_weights.parquet")
    weights_df.to_parquet(weights_path, index=False)
    print(f"  Concept weights saved: {weights_path}")

    concept_words_dir = os.path.join(out_dir, "concepts")
    os.makedirs(concept_words_dir, exist_ok=True)
    top_words = model.get_concept_top_words(data["vocab"], top_n=20)
    for i, words in enumerate(top_words):
        with open(os.path.join(concept_words_dir, f"concept_{i}.txt"), "w") as f:
            f.write("\n".join(words))

    return out_dir


def extract_concept_weights_from_output(gdcm_out_dir: str, n_docs: int) -> pd.DataFrame:
    """
    Parse concept weights from GDCM CLI output directory.
    GDCM CLI saves concept weights in model/ subdirectory.
    Returns DataFrame with shape (n_docs, nconcepts).
    """
    import glob
    weight_files = sorted(glob.glob(os.path.join(gdcm_out_dir, "model", "*.pytorch")))
    if not weight_files:
        parquet_path = os.path.join(gdcm_out_dir, "doc_concept_weights.parquet")
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        raise FileNotFoundError(f"No GDCM model outputs found in {gdcm_out_dir}")

    import torch
    latest_model = weight_files[-1]
    print(f"  Loading GDCM model: {latest_model}")
    state = torch.load(latest_model, map_location="cpu")

    if "concept_weights" in state:
        weights = state["concept_weights"].numpy()
    elif "theta" in state:
        weights = state["theta"].numpy()
    else:
        print(f"  WARNING: Unknown GDCM state dict keys: {list(state.keys())}")
        return pd.DataFrame()

    cols = [f"concept_{i}" for i in range(weights.shape[1])]
    return pd.DataFrame(weights, columns=cols)


def process_dataset(dataset: str, cfg: dict, mode: str):
    print(f"\n=== GDCM: {dataset} ===")

    sentences_path = get_processed_path(cfg, dataset, "sentences_labeled.parquet")
    if not os.path.exists(sentences_path):
        print(f"  SKIP: {sentences_path} not found. Run 01_preprocess.py first.")
        return

    df = pd.read_parquet(sentences_path)
    print(f"  Loaded {len(df)} sentences")

    df = normalize_labels(df, label_col="eui_t0")

    gdcm_dir = get_dataset_output_dir(cfg, dataset, "gdcm")
    csv_path = os.path.join(gdcm_dir, "gdcm_input.csv")
    write_gdcm_csv(df, csv_path, text_col="text", label_col="eui_t0")

    config_path = cfg["gdcm_config"]
    gridsearch_out = os.path.join(gdcm_dir, "gridsearch")
    best_params = run_gdcm_grid_search(csv_path, config_path, gridsearch_out, mode)

    best_params_path = os.path.join(gdcm_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"  Best params saved: {best_params_path}")

    final_out = os.path.join(gdcm_dir, "final_model")
    run_gdcm_final(csv_path, best_params, final_out, mode)

    weights_path = os.path.join(final_out, "doc_concept_weights.parquet")
    if os.path.exists(weights_path):
        weights_df = pd.read_parquet(weights_path)
        print(f"  Concept weights shape: {weights_df.shape}")

        merged_path = os.path.join(gdcm_dir, "sentences_with_concepts.parquet")
        if len(weights_df) == len(df):
            merged = pd.concat([df.reset_index(drop=True), weights_df.reset_index(drop=True)], axis=1)
            merged.to_parquet(merged_path, index=False)
            print(f"  Merged sentences+concepts: {merged_path}")
        else:
            print(f"  WARNING: Row count mismatch (sentences={len(df)}, weights={len(weights_df)})")
            weights_df.to_parquet(os.path.join(gdcm_dir, "doc_concept_weights.parquet"), index=False)
    else:
        print(f"  WARNING: No concept weights parquet found at {weights_path}")

    concept_dir = os.path.join(final_out, "concepts")
    if os.path.exists(concept_dir):
        import glob
        concept_files = glob.glob(os.path.join(concept_dir, "*.txt"))
        print(f"  Concepts word lists: {len(concept_files)} files in {concept_dir}")

    nconcepts = int(best_params.get("nconcepts", "?"))
    print(f"  GDCM complete: dataset={dataset}, nconcepts={nconcepts}")


def main():
    parser = argparse.ArgumentParser(description="GDCM Topic Modeling")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "reddit", "stackexchange", "bogleheads"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, args.mode)

    ensure_gdcm_installed()

    datasets = ["reddit", "stackexchange", "bogleheads"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        process_dataset(dataset, cfg, args.mode)

    print("\n[02_gdcm_train] DONE")


if __name__ == "__main__":
    main()
