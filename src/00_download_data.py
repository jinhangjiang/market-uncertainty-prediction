"""
00_download_data.py
Downloads Reddit data from Google Drive, copies StackExchange/Bogleheads to raw/,
and downloads the EUI time series from policyuncertainty.com.

Usage:
    python src/00_download_data.py --dataset all --mode smoke
    python src/00_download_data.py --dataset reddit --mode full
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.utils.eui_loader import download_eui, load_eui


REDDIT_RAW_ID = "1tTs80B110WrfIYwKNRS0S7G0kn3dEZQW"
REDDIT_TRANSFORMED_ID = "1y2L85Igjk6tU94lfqtImzKfVllc5WHsy"


def download_reddit(cfg: dict, mode: str):
    out_dir = "data/raw/reddit"
    os.makedirs(out_dir, exist_ok=True)

    raw_path = cfg["paths"]["raw_reddit_raw"]
    transformed_path = cfg["paths"]["raw_reddit_transformed"]

    if mode == "smoke":
        print("[SMOKE] Skipping Google Drive download. Expecting Reddit files already present.")
        for p in [raw_path, transformed_path]:
            if not os.path.exists(p):
                print(f"  WARNING: {p} not found. Smoke mode will use StackExchange as proxy.")
        return

    try:
        import gdown
    except ImportError:
        raise ImportError("Please install gdown: pip install gdown")

    if not os.path.exists(raw_path):
        print(f"Downloading Reddit raw data → {raw_path}")
        gdown.download(id=REDDIT_RAW_ID, output=raw_path, quiet=False)
    else:
        print(f"Reddit raw already exists: {raw_path}")

    if not os.path.exists(transformed_path):
        print(f"Downloading Reddit transformed data → {transformed_path}")
        gdown.download(id=REDDIT_TRANSFORMED_ID, output=transformed_path, quiet=False)
    else:
        print(f"Reddit transformed already exists: {transformed_path}")


def copy_existing_datasets(cfg: dict):
    """Copy StackExchange and Bogleheads to data/raw/ for consistency.
    Checks both data/<file> and data/raw/<file> as source locations.
    """
    file_names = {
        cfg["paths"]["raw_stackexchange"]: "StackExchange_.csv",
        cfg["paths"]["raw_bogleheads"]: "bogleheads_.csv",
    }
    for dst, fname in file_names.items():
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            print(f"Already exists: {dst}")
            continue
        # Try multiple source locations
        src_candidates = [
            os.path.join("data", fname),
            os.path.join("data", "raw", fname),
            fname,
        ]
        copied = False
        for src in src_candidates:
            if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
                print(f"Copied {src} → {dst}")
                copied = True
                break
        if not copied:
            print(f"WARNING: Source not found for {fname}. Tried: {src_candidates}")


def download_eui_data(cfg: dict):
    eui_path = cfg["paths"]["eui"]
    if os.path.exists(eui_path):
        print(f"EUI already exists: {eui_path}")
        eui_df = load_eui(eui_path)
    else:
        eui_df = download_eui(save_path=eui_path)
    print(f"EUI coverage: {eui_df['ds'].min().date()} to {eui_df['ds'].max().date()}")
    return eui_df


def validate_eui_coverage(eui_df, cfg: dict):
    import pandas as pd
    start = pd.Timestamp(cfg["common_window"]["start"])
    end = pd.Timestamp(cfg["common_window"]["end"])
    eui_start = eui_df["ds"].min()
    eui_end = eui_df["ds"].max()
    if eui_start > start or eui_end < end:
        print(f"WARNING: EUI coverage ({eui_start.date()} to {eui_end.date()}) "
              f"does not fully cover common window ({start.date()} to {end.date()})")
    else:
        print(f"EUI coverage validated for common window [{start.date()} – {end.date()}]")


def main():
    parser = argparse.ArgumentParser(description="Download all data sources")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "reddit", "stackexchange", "bogleheads"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, args.mode)

    if args.dataset in ("all", "reddit"):
        download_reddit(cfg, args.mode)

    if args.dataset in ("all", "stackexchange", "bogleheads"):
        copy_existing_datasets(cfg)

    eui_df = download_eui_data(cfg)
    validate_eui_coverage(eui_df, cfg)

    print("\n[00_download_data] DONE")


if __name__ == "__main__":
    main()
