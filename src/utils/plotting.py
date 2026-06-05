import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


PALETTE = sns.color_palette("tab10")
FIG_DPI = 150


def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def plot_nfa_over_time(results_df: pd.DataFrame, model_col: str, nfa_col: str,
                       date_col: str, title: str, save_path: str):
    """Line plot of NFA over test dates for each model."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(results_df[model_col].unique()):
        sub = results_df[results_df[model_col] == model].sort_values(date_col)
        ax.plot(sub[date_col], sub[nfa_col], label=model, color=PALETTE[i % 10])
    ax.set_xlabel("Date")
    ax.set_ylabel("NFA")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    save_fig(fig, save_path)


def plot_sc_over_time(results_df: pd.DataFrame, model_col: str, sc_col: str,
                      date_col: str, title: str, save_path: str):
    """Line plot of Spearman Correlation over test dates for each model."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(results_df[model_col].unique()):
        sub = results_df[results_df[model_col] == model].sort_values(date_col)
        ax.plot(sub[date_col], sub[sc_col], label=model, color=PALETTE[i % 10])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spearman Correlation")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    save_fig(fig, save_path)


def plot_shap_bar(shap_df: pd.DataFrame, feature_col: str, shap_col: str,
                  title: str, save_path: str, top_n: int = 15):
    """Horizontal bar chart of mean |SHAP| values."""
    df = shap_df.nlargest(top_n, shap_col)
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.45)))
    ax.barh(df[feature_col], df[shap_col], color=PALETTE[0])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    save_fig(fig, save_path)


def plot_correlation_heatmap(corr_df: pd.DataFrame, title: str, save_path: str):
    """Seaborn heatmap of feature-EUI correlations."""
    fig, ax = plt.subplots(figsize=(max(6, corr_df.shape[1] * 1.2),
                                     max(4, corr_df.shape[0] * 0.5)))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title(title)
    save_fig(fig, save_path)


def plot_sensitivity(sensitivity_df: pd.DataFrame, model_col: str, window_col: str,
                     nfa_col: str, title: str, save_path: str):
    """Bar chart: NFA by model and training window for sensitivity analysis."""
    fig, ax = plt.subplots(figsize=(10, 5))
    models = sensitivity_df[model_col].unique()
    windows = sorted(sensitivity_df[window_col].unique())
    x = np.arange(len(models))
    width = 0.8 / len(windows)
    for i, w in enumerate(windows):
        vals = [
            sensitivity_df[(sensitivity_df[model_col] == m) & (sensitivity_df[window_col] == w)][nfa_col].mean()
            for m in models
        ]
        ax.bar(x + i * width, vals, width, label=f"{w}-day window", color=PALETTE[i % 10])
    ax.set_xticks(x + width * (len(windows) - 1) / 2)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean NFA")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, save_path)


def plot_eui_series(eui_df: pd.DataFrame, title: str, save_path: str):
    """Plot the EUI time series."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(eui_df["ds"], eui_df["eui"], color=PALETTE[0], linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("EUI")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    save_fig(fig, save_path)
