# EUI Prediction Pipeline — Project Overview

**Paper:** Explainable Deep Learning for Equity Market Uncertainty Index (EUI) Forecasting Using Social Media Data  
**Venue:** JAIS (Major Revision, R1)  
**Status:** Revision in progress — implementing reviewer feedback

---

## What This Repo Does

This pipeline predicts the **Equity Market-related Economic Uncertainty Index (EUI)** from social media text (Reddit, StackExchange, Bogleheads). It addresses three reviewer concerns from the JAIS submission:

| Reviewer Concern | Our Response |
|---|---|
| Single Reddit source, selection bias | Extended to **3 datasets**: Reddit, StackExchange Personal Finance, Bogleheads |
| COVID-only time window (2021–2022) | Bogleheads spans **2018–2022**, pre-COVID baseline included |
| BERTopic k is arbitrary ("black box") | Replaced with **GDCM** — concept count chosen empirically via grid search on validation loss |
| No SHAP / interpretability | **SHAP** at both feature selection (LinearExplainer) and forecasting (GradientExplainer) stages |
| Potential data leakage | GDCM guided by **same-day EUI (t=0)**, not future values; strict temporal walk-forward for forecasting |
| Training window unjustified | **Sensitivity analysis** across 30/60/90-day windows |

---

## Datasets

| Dataset | Source | Columns Used | Date Range |
|---|---|---|---|
| Reddit Personal Finance | Google Drive (gdown) | `Text`, `Date` | 2021-11-23 – 2022-06-25 |
| StackExchange Personal Finance | `data/StackExchange_.csv` | `Content` (HTML) + `Title`, `CreationDate` | 2021-11-23 – 2022-06-25 |
| Bogleheads Forum | `data/bogleheads_.csv` | `content` + `topic_title`, `time` (ISO-8601 TZ) | **2018–2022** (full) + common window |

All three datasets are filtered to a **common window (2021-11-23 to 2022-06-25)** for cross-dataset comparison. Bogleheads additionally runs over its full 2018–2022 range.

---

## Pipeline Architecture

```
run_pipeline.sh [dataset] [mode]
        │
        ├── 00_download_data.py     Download Reddit (GDrive), copy SE/BH, fetch EUI series
        ├── 01_preprocess.py        Clean text, sentence-split, join EUI labels
        ├── 02_gdcm_train.py        GDCM grid search → final topic model training
        ├── 03_feature_engineering.py   Daily concept volumes → NeuralForecast feature matrix
        ├── 04_feature_selection.py     LassoCV → SHAP ranking → top-K features
        ├── 05_forecast.py          Walk-forward eval: LSTM / TFT / TimeXer (base + topic)
        ├── 06_shap_interpret.py    SHAP on forecasting models + concept attribution table
        └── 07_report.py            Self-contained HTML reports per dataset + cross-dataset
```

---

## Models

All forecasting via **NeuralForecast** only (no Prophet, no Random Forest):

| Model | Features | Horizon |
|---|---|---|
| `LSTM_base` | EUI history only | 7-day multi-step |
| `LSTM_topic` | + GDCM concept volumes (`hist_exog_list`) | 7-day multi-step |
| `TFT_base` | EUI history only | 7-day multi-step |
| `TFT_topic` | + GDCM concept volumes | 7-day multi-step |
| `TimeXer_base` | EUI history only | 7-day multi-step |
| `TimeXer_topic` | + GDCM concept volumes | 7-day multi-step |
| `LSTM_topic_single` | + GDCM concept volumes | 1-day (h=1) comparison |

**Evaluation metrics:** NFA = 1 − SMAPE/2 (primary), Spearman Correlation (SC), MAE, RMSE

---

## Execution Modes

| Mode | Where | Data | Speed |
|---|---|---|---|
| `smoke` (default) | Local Mac / CPU | 500 rows, 5 concepts, 2 epochs, 10 steps | < 5 min |
| `full` | Colab A100 / AzureML H100 | Full datasets, 500 steps | 3–6 hours |

GPU is auto-detected: `cuda` → `mps` (Apple Silicon) → `cpu`.

### Run locally (smoke test)
```bash
conda activate eui-revision
./run_pipeline.sh all smoke
```

### Run on Google Colab (full)
Open `notebooks/colab_runner.ipynb`, set Runtime to A100, run all cells.

### Run on AzureML (full)
```bash
cp cloud/azure_config.json.template cloud/azure_config.json
# Fill in subscription_id, resource_group, workspace_name
python cloud/submit_job.py --dataset all
```

---

## Repo Structure

```
market-uncertainty-prediction/
├── data/
│   ├── StackExchange_.csv          # StackExchange Personal Finance posts + comments
│   ├── bogleheads_.csv             # Bogleheads forum posts + replies
│   ├── raw/                        # copies used by pipeline (gitignored for Reddit)
│   ├── processed/                  # cleaned sentences + EUI labels (gitignored)
│   └── eui/                        # EUI daily time series (gitignored, downloaded at runtime)
├── src/
│   ├── 00_download_data.py
│   ├── 01_preprocess.py
│   ├── 02_gdcm_train.py
│   ├── 03_feature_engineering.py
│   ├── 04_feature_selection.py
│   ├── 05_forecast.py
│   ├── 06_shap_interpret.py
│   ├── 07_report.py
│   └── utils/
│       ├── config_loader.py        # load YAML config + apply smoke/full overrides
│       ├── device.py               # GPU auto-detection
│       ├── eui_loader.py           # download + join EUI time series
│       ├── metrics.py              # NFA, SC, SMAPE, MAE, RMSE
│       └── plotting.py             # NFA/SC over time, SHAP bar charts, heatmaps
├── configs/
│   ├── pipeline_config.yaml        # datasets, windows, top-K, paths, smoke overrides
│   ├── gdcm_config.json            # GDCM grid search space (full run)
│   └── gdcm_config_smoke.json      # GDCM reduced grid (smoke run)
├── notebooks/
│   └── colab_runner.ipynb          # Colab entry point
├── cloud/
│   ├── azureml_job.yml             # AzureML H100 job definition
│   ├── submit_job.py               # Azure ML SDK v2 submission script
│   ├── environment.dockerfile      # CUDA Docker image for AzureML
│   └── azure_config.json.template  # fill in your workspace credentials
├── tests/
│   └── test_smoke_preprocess.py    # 37-check smoke test on real SE + BH data
├── environment.yml                 # conda env: eui-revision
├── run_pipeline.sh                 # single shell entry point
└── project.md                      # this file
```

---

## Setup

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate eui-revision

# 2. Install GDCM (cloned at runtime by 02_gdcm_train.py, or manually)
git clone https://github.com/cygit/gdcm.git gdcm
pip install -e gdcm/src

# 3. Run smoke test to verify setup
./run_pipeline.sh all smoke
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| GDCM label = same-day EUI (t=0) | Post reacts to current economic climate; no leakage of future EUI into topic model |
| `hist_exog_list` in NeuralForecast | Correct API for past-observed exogenous variables; concept volumes at time t cannot leak t+h |
| LassoCV → SHAP (LinearExplainer) | Addresses R1 SHAP request; more defensible than permutation importance / MDA |
| `--mode smoke` default | Prevents accidental full GPU runs locally; all scripts are independently re-runnable |
| No Prophet / No Random Forest | Explicitly excluded per paper scope — NeuralForecast LSTM serves as simpler DL baseline |
| Bogleheads full 2018–2022 range | Covers pre-COVID baseline; directly addresses R2 concern about COVID-only data bias |
