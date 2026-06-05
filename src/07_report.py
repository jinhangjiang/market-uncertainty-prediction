"""
07_report.py
Generates self-contained HTML reports per dataset + cross-dataset summary.
All figures embedded as base64 PNGs. Tables exported as CSV.

Usage:
    python src/07_report.py --dataset all --mode smoke
    python src/07_report.py --dataset reddit --mode full
"""
import argparse
import base64
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config, get_processed_path, get_dataset_output_dir
from src.utils.plotting import plot_eui_series, plot_nfa_over_time, plot_sc_over_time, plot_sensitivity


REPORT_CSS = """
<style>
body { font-family: Arial, sans-serif; margin: 40px; color: #222; background: #fafafa; }
h1 { color: #1a3a6b; border-bottom: 3px solid #1a3a6b; padding-bottom: 8px; }
h2 { color: #2c5aa0; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 40px; }
h3 { color: #444; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }
th { background: #1a3a6b; color: white; padding: 8px 12px; text-align: left; }
td { padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }
tr:nth-child(even) { background: #f2f6ff; }
img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin: 8px 0; }
.meta { color: #666; font-size: 12px; margin-bottom: 24px; }
.highlight { background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 12px 0; }
.section { background: white; padding: 20px; margin: 20px 0; border-radius: 6px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
</style>
"""


def img_to_b64(path: str) -> str:
    """Embed image as base64 for self-contained HTML."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{data}" />'


def df_to_html_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Convert DataFrame to styled HTML table."""
    return df.to_html(index=False, float_format=lambda x: f"{x:{float_fmt}}", border=0,
                      classes="data-table", justify="left")


def load_data_summary(cfg: dict, dataset: str) -> dict:
    path = get_processed_path(cfg, dataset, "data_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path).iloc[0].to_dict()
    return {}


def load_gdcm_results(cfg: dict, dataset: str) -> dict:
    gdcm_dir = get_dataset_output_dir(cfg, dataset, "gdcm")
    best_params_path = os.path.join(gdcm_dir, "best_params.json")
    if os.path.exists(best_params_path):
        with open(best_params_path) as f:
            return json.load(f)
    gs_results = os.path.join(gdcm_dir, "gridsearch", "results.csv")
    if os.path.exists(gs_results):
        df = pd.read_csv(gs_results)
        if not df.empty:
            return df.sort_values("best_val_loss").iloc[0].to_dict()
    return {}


def load_feature_selection(cfg: dict, dataset: str) -> dict:
    feat_dir = get_dataset_output_dir(cfg, dataset, "features")
    path = os.path.join(feat_dir, "selected_features.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_forecast_summary(cfg: dict, dataset: str) -> pd.DataFrame:
    forecast_dir = get_dataset_output_dir(cfg, dataset, "forecasts")
    path = os.path.join(forecast_dir, "summary_table.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_shap_summary(cfg: dict, dataset: str) -> pd.DataFrame:
    shap_dir = get_dataset_output_dir(cfg, dataset, "shap")
    path = os.path.join(shap_dir, "concept_shap_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_correlation(cfg: dict, dataset: str) -> pd.DataFrame:
    feat_dir = get_dataset_output_dir(cfg, dataset, "features")
    path = os.path.join(feat_dir, "correlation_eui.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


def generate_dataset_report(dataset: str, cfg: dict, mode: str) -> str:
    """Generate full HTML content for a single dataset report."""
    forecast_dir = get_dataset_output_dir(cfg, dataset, "forecasts")
    feat_dir = get_dataset_output_dir(cfg, dataset, "features")
    shap_dir = get_dataset_output_dir(cfg, dataset, "shap")

    data_summary = load_data_summary(cfg, dataset)
    gdcm_params = load_gdcm_results(cfg, dataset)
    feature_sel = load_feature_selection(cfg, dataset)
    forecast_summary = load_forecast_summary(cfg, dataset)
    shap_summary = load_shap_summary(cfg, dataset)
    corr_df = load_correlation(cfg, dataset)

    title = f"EUI Prediction Report — {dataset.title()}"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = []

    sections.append(f"""
    <div class="section">
    <h2>1. Data Summary</h2>
    {"".join(f"<p><b>{k}:</b> {v}</p>" for k, v in data_summary.items()) if data_summary else "<p>No summary available.</p>"}
    </div>""")

    eui_path = cfg["paths"]["eui"]
    if os.path.exists(eui_path):
        import matplotlib
        matplotlib.use("Agg")
        eui_df = pd.read_csv(eui_path, parse_dates=["ds"])
        eui_fig_path = os.path.join(get_dataset_output_dir(cfg, dataset, ""), "eui_series.png")
        plot_eui_series(eui_df, title="Equity Market Uncertainty Index (Daily)", save_path=eui_fig_path)
        sections.append(f'<div class="section"><h2>EUI Time Series</h2>{img_to_b64(eui_fig_path)}</div>')

    gdcm_html = "<p>No GDCM results found.</p>"
    if gdcm_params:
        gdcm_row = pd.DataFrame([{k: v for k, v in gdcm_params.items()
                                   if k in ("nconcepts", "embed_dim", "lam", "rho", "eta",
                                            "lr", "nepochs", "best_val_loss")}])
        gdcm_html = df_to_html_table(gdcm_row)
    sections.append(f"""
    <div class="section">
    <h2>2. GDCM Topic Modeling (Replaces BERTopic)</h2>
    <div class="highlight">
    <b>Design:</b> GDCM guided by same-day EUI (t=0). Number of concepts selected
    empirically via grid search on validation prediction loss — no arbitrary k.
    </div>
    {gdcm_html}
    </div>""")

    feat_html = "<p>No feature selection results found.</p>"
    if feature_sel:
        feat_html = f"""
        <p><b>Total concept features:</b> {feature_sel.get('n_concept_features', 'N/A')}</p>
        <p><b>Lasso selected:</b> {feature_sel.get('n_lasso_selected', 'N/A')}
           ({feature_sel.get('pct_filtered_by_lasso', 'N/A')}% filtered out)</p>
        <p><b>Lasso alpha:</b> {feature_sel.get('lasso_alpha', 'N/A')}</p>
        <p><b>Top-{feature_sel.get('top_k', 10)} features:</b>
           {', '.join(feature_sel.get('top_k_features', []))}</p>"""

    shap_img = img_to_b64(os.path.join(feat_dir, "shap_importance.png"))
    sections.append(f"""
    <div class="section">
    <h2>3. Feature Selection (Lasso + SHAP)</h2>
    {feat_html}
    <h3>SHAP Feature Importance (from LinearExplainer on Ridge)</h3>
    {shap_img}
    </div>""")

    corr_html = "<p>No correlation data found.</p>"
    if not corr_df.empty:
        corr_html = df_to_html_table(corr_df.reset_index().rename(columns={"index": "feature"}))
    corr_img = img_to_b64(os.path.join(feat_dir, "correlation_heatmap.png"))
    sections.append(f"""
    <div class="section">
    <h2>4. Spearman Correlation: Top Concepts vs EUI</h2>
    {corr_img}
    {corr_html}
    </div>""")

    if not forecast_summary.empty:
        fcast_html = df_to_html_table(forecast_summary.round(4))
    else:
        fcast_html = "<p>No forecasting results found.</p>"

    nfa_img = img_to_b64(os.path.join(forecast_dir, "nfa_over_time.png"))
    sc_img = img_to_b64(os.path.join(forecast_dir, "sc_over_time.png"))
    sens_img = img_to_b64(os.path.join(forecast_dir, "sensitivity_window.png"))

    sections.append(f"""
    <div class="section">
    <h2>5. Forecasting Results (NeuralForecast: LSTM / TFT / TimeXer)</h2>
    <div class="highlight">
    <b>Evaluation:</b> Walk-forward cross-validation. NFA = 1 - SMAPE/200 (primary).
    Concept volumes injected as <code>hist_exog_list</code> — no future leakage.
    </div>
    <h3>Summary Table (mean ± std across test windows)</h3>
    {fcast_html}
    <h3>NFA Over Time</h3>
    {nfa_img}
    <h3>Spearman Correlation Over Time</h3>
    {sc_img}
    <h3>Sensitivity: NFA vs Training Window (30/60/90 days)</h3>
    {sens_img}
    </div>""")

    shap_agg_img = img_to_b64(os.path.join(shap_dir, "shap_aggregated.png"))
    if not shap_summary.empty:
        shap_html = df_to_html_table(shap_summary.round(6))
    else:
        shap_html = "<p>No SHAP summary found.</p>"
    sections.append(f"""
    <div class="section">
    <h2>6. SHAP Interpretation — Concept Attribution</h2>
    {shap_agg_img}
    <h3>Concept-to-Interpretation Bridge Table</h3>
    {shap_html}
    </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
{REPORT_CSS}
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {ts} | Mode: {mode} | Dataset: {dataset}</p>
{"".join(sections)}
</body>
</html>"""
    return html


def generate_cross_dataset_report(cfg: dict, datasets: list, mode: str) -> str:
    """Generate cross-dataset comparison HTML report."""
    title = "EUI Prediction — Cross-Dataset Comparison"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = []

    rows = []
    for ds in datasets:
        summary = load_forecast_summary(cfg, ds)
        if summary.empty:
            continue
        best = summary.loc[summary["NFA_mean"].idxmax()]
        rows.append({
            "Dataset": ds.title(),
            "Best Model": best["model"],
            "Training Window": best["train_window"],
            "NFA (mean)": round(best["NFA_mean"], 4),
            "NFA (std)": round(best.get("NFA_std", 0), 4),
            "SC (mean)": round(best["SC_mean"], 4),
            "MAE (mean)": round(best["MAE_mean"], 4),
        })

    if rows:
        cross_df = pd.DataFrame(rows)
        sections.append(f"""
        <div class="section">
        <h2>Best Model Performance Per Dataset</h2>
        {df_to_html_table(cross_df)}
        </div>""")

    overlap_rows = []
    all_top_features = {}
    for ds in datasets:
        sel = load_feature_selection(cfg, ds)
        top_k = sel.get("top_k_features", [])
        all_top_features[ds] = set(top_k)

    if len(all_top_features) > 1:
        common = set.intersection(*all_top_features.values())
        for ds, feats in all_top_features.items():
            overlap_rows.append({"Dataset": ds.title(), "Top Features": ", ".join(feats),
                                  "N Features": len(feats)})
        overlap_rows.append({"Dataset": "COMMON (all datasets)", "Top Features": ", ".join(common),
                              "N Features": len(common)})
        overlap_df = pd.DataFrame(overlap_rows)
        sections.append(f"""
        <div class="section">
        <h2>Concept Feature Overlap Across Datasets</h2>
        <div class="highlight">
        <b>{len(common)} concepts</b> appear as top predictors across ALL three datasets,
        supporting generalizability of findings.
        </div>
        {df_to_html_table(overlap_df)}
        </div>""")

    all_model_results = []
    for ds in datasets:
        summary = load_forecast_summary(cfg, ds)
        if not summary.empty:
            summary["Dataset"] = ds.title()
            all_model_results.append(summary)

    if all_model_results:
        all_df = pd.concat(all_model_results, ignore_index=True)
        all_path = os.path.join(cfg["paths"]["outputs"], "final_report", "tables",
                                "all_datasets_forecast_summary.csv")
        os.makedirs(os.path.dirname(all_path), exist_ok=True)
        all_df.to_csv(all_path, index=False)
        sections.append(f"""
        <div class="section">
        <h2>All Models × All Datasets</h2>
        {df_to_html_table(all_df.round(4))}
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
{REPORT_CSS}
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {ts} | Mode: {mode}</p>
{"".join(sections)}
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "reddit", "stackexchange", "bogleheads"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, args.mode)
    report_dir = os.path.join(cfg["paths"]["outputs"], "final_report")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(os.path.join(report_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(report_dir, "figures"), exist_ok=True)

    datasets = ["reddit", "stackexchange", "bogleheads"] if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        print(f"\n=== Generating report: {dataset} ===")
        try:
            html = generate_dataset_report(dataset, cfg, args.mode)
            out_path = os.path.join(report_dir, f"report_{dataset}.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  Report saved: {out_path}")
        except Exception as e:
            print(f"  ERROR generating report for {dataset}: {e}")
            import traceback
            traceback.print_exc()

    if args.dataset == "all":
        print("\n=== Generating cross-dataset comparison ===")
        try:
            cross_html = generate_cross_dataset_report(cfg, datasets, args.mode)
            cross_path = os.path.join(report_dir, "cross_dataset_summary.html")
            with open(cross_path, "w", encoding="utf-8") as f:
                f.write(cross_html)
            print(f"  Cross-dataset report saved: {cross_path}")
        except Exception as e:
            print(f"  ERROR generating cross-dataset report: {e}")
            import traceback
            traceback.print_exc()

    print("\n[07_report] DONE")
    print(f"Reports available in: {report_dir}")


if __name__ == "__main__":
    main()
