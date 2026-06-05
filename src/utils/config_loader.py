import yaml
import os


def load_config(config_path: str = "configs/pipeline_config.yaml", mode: str = "full") -> dict:
    """
    Load pipeline config. If mode == 'smoke', merge smoke overrides on top of full config.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if mode == "smoke":
        smoke = cfg.pop("smoke", {})
        cfg.update(smoke)
        print(f"[SMOKE MODE] Config overrides applied: {smoke}")
    else:
        cfg.pop("smoke", None)
        print("[FULL MODE] Running with full config.")

    return cfg


def get_dataset_output_dir(cfg: dict, dataset: str, subdir: str = "") -> str:
    base = os.path.join(cfg["paths"]["outputs"], dataset, subdir)
    os.makedirs(base, exist_ok=True)
    return base


def get_processed_path(cfg: dict, dataset: str, filename: str) -> str:
    base = os.path.join(cfg["paths"]["processed"], dataset)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)
