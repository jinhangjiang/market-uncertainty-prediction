import os
import pandas as pd
import numpy as np


EUI_URL = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx"
EUI_COLUMN = "Equity_Mkt_Uncertainty"


def download_eui(save_path: str = "data/eui/eui_daily.csv") -> pd.DataFrame:
    """
    Download the US Policy Uncertainty Excel file and extract the
    Equity Market Uncertainty Index at monthly frequency, then forward-fill
    to daily. Saves result to save_path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Downloading EUI data from {EUI_URL} ...")
    df = pd.read_excel(EUI_URL, sheet_name=0)

    df.columns = [str(c).strip() for c in df.columns]

    year_col = [c for c in df.columns if "year" in c.lower()][0]
    month_col = [c for c in df.columns if "month" in c.lower()][0]
    eui_col = [c for c in df.columns if "equity" in c.lower() or "stock" in c.lower()]

    if not eui_col:
        raise ValueError(
            f"Cannot find Equity Market Uncertainty column. Available: {df.columns.tolist()}"
        )
    eui_col = eui_col[0]

    df = df[[year_col, month_col, eui_col]].dropna()
    df.columns = ["year", "month", "eui"]
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["ds"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
    )
    df = df[["ds", "eui"]].sort_values("ds").reset_index(drop=True)

    date_range = pd.date_range(df["ds"].min(), df["ds"].max() + pd.offsets.MonthEnd(0), freq="D")
    daily = pd.DataFrame({"ds": date_range})
    daily = daily.merge(df, on="ds", how="left")
    daily["eui"] = daily["eui"].ffill()

    daily.to_csv(save_path, index=False)
    print(f"EUI saved to {save_path} ({len(daily)} daily rows)")
    return daily


def load_eui(path: str = "data/eui/eui_daily.csv") -> pd.DataFrame:
    """Load EUI from CSV. Returns df with columns [ds, eui]."""
    df = pd.read_csv(path, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def join_eui_to_posts(
    posts_df: pd.DataFrame,
    eui_df: pd.DataFrame,
    date_col: str = "date",
    lookaheads: list = [0, 1, 3, 7],
) -> pd.DataFrame:
    """
    Join EUI values to posts_df by date.
    For each lookahead n in lookaheads, creates column eui_t{n}
    with the EUI value n days after the post date.
    """
    eui_map = eui_df.set_index("ds")["eui"].to_dict()
    posts_df = posts_df.copy()
    posts_df[date_col] = pd.to_datetime(posts_df[date_col]).dt.normalize()

    for n in lookaheads:
        col = f"eui_t{n}"
        posts_df[col] = posts_df[date_col].apply(
            lambda d: eui_map.get(d + pd.Timedelta(days=n), np.nan)
        )

    before = len(posts_df)
    posts_df = posts_df.dropna(subset=[f"eui_t{n}" for n in lookaheads])
    after = len(posts_df)
    print(f"EUI join: kept {after}/{before} rows (dropped {before - after} with missing EUI)")
    return posts_df
