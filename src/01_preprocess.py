"""
01_preprocess.py
Per-dataset text cleaning, sentence splitting, and EUI label joining.

Usage:
    python src/01_preprocess.py --dataset all --mode smoke
    python src/01_preprocess.py --dataset stackexchange --mode full
"""
import argparse
import os
import re
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config, get_processed_path
from src.utils.eui_loader import load_eui, join_eui_to_posts


def _try_nltk_download(resource: str):
    """Download NLTK resource, patching SSL for macOS if needed. Silent on failure."""
    import io, ssl, contextlib
    import nltk

    sink = io.StringIO()
    with contextlib.redirect_stderr(sink):
        try:
            nltk.download(resource, quiet=True)
            return
        except Exception:
            pass
        # macOS SSL cert workaround
        try:
            _orig = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            nltk.download(resource, quiet=True)
            ssl._create_default_https_context = _orig
        except Exception:
            pass


try:
    _try_nltk_download("punkt")
    _try_nltk_download("punkt_tab")
    _try_nltk_download("stopwords")
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
    _SENT_TOKENIZER = "nltk"
except Exception:
    STOP_WORDS = set()
    _SENT_TOKENIZER = "regex"
    def sent_tokenize(text):  # noqa: F811
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


URL_RE = re.compile(r"http\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPECIAL_RE = re.compile(r"[^\w\s.,!?'-]")
WHITESPACE_RE = re.compile(r"\s+")
QUOTE_RE = re.compile(r"^.*wrote:\n.*\n", re.MULTILINE)


def clean_text(text: str) -> str:
    text = str(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = QUOTE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = SPECIAL_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize_simple(text: str) -> list:
    return text.lower().split()


def count_tokens(text: str) -> int:
    return len(tokenize_simple(text))


def split_into_sentences(text: str, max_tokens: int = 500) -> list:
    """Split document into sentence-level chunks, capping at max_tokens."""
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        tokens = tokenize_simple(sent)
        if not tokens:
            continue
        if current_len + len(tokens) > max_tokens and current:
            chunks.append(" ".join(current))
            current = tokens
            current_len = len(tokens)
        else:
            current.extend(tokens)
            current_len += len(tokens)
    if current:
        chunks.append(" ".join(current))
    return chunks


def deduplicate(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    print(f"  Dedup: {before} → {len(df)} rows")
    return df


def load_reddit(cfg: dict, mode: str) -> pd.DataFrame:
    path = cfg["paths"]["raw_reddit_transformed"]
    if not os.path.exists(path):
        path = cfg["paths"]["raw_reddit_raw"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reddit data not found at {path}. Run 00_download_data.py first.")
    print(f"Loading Reddit from {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Reddit columns: {df.columns.tolist()}")
    if mode == "smoke":
        df = df.head(cfg.get("sample_rows", 500))

    # Probe actual column names — Reddit transformed has 'Text'/'Date',
    # raw may have 'selftext'/'created_utc' (PRAW format)
    text_candidates = ["Text", "text", "selftext", "body", "content"]
    date_candidates = ["Date", "date", "created_utc", "created", "timestamp"]
    text_col = next((c for c in text_candidates if c in df.columns), None)
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if text_col is None:
        raise ValueError(f"Cannot find text column in Reddit data. Columns: {df.columns.tolist()}")
    if date_col is None:
        raise ValueError(f"Cannot find date column in Reddit data. Columns: {df.columns.tolist()}")
    print(f"  Using text='{text_col}', date='{date_col}'")

    df = df.rename(columns={text_col: "raw_text", date_col: "date"})
    # Handle Unix timestamps (PRAW format) vs string dates
    if pd.api.types.is_numeric_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce").dt.normalize()
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["raw_text", "date"])
    df = df[df["raw_text"].astype(str).str.strip() != ""]
    df = df[df["raw_text"].astype(str) != "[deleted]"]
    df = df[df["raw_text"].astype(str) != "[removed]"]
    return df[["raw_text", "date"]]


def load_stackexchange(cfg: dict, mode: str) -> pd.DataFrame:
    # Real columns: SourceType, Id, PostTypeId, ParentId, CreationDate,
    #               Score, Content, Title, Tags
    # Posts have Content (HTML) + possibly Title
    # Comments have Content only (plain text), Title is always null
    path = cfg["paths"]["raw_stackexchange"]
    print(f"Loading StackExchange from {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  StackExchange shape: {df.shape}, SourceType counts: {df['SourceType'].value_counts().to_dict()}")
    if mode == "smoke":
        df = df.head(cfg.get("sample_rows", 500))

    # Combine Content (HTML for posts, plain for comments) + Title where present
    df["raw_text"] = df["Content"].fillna("") + " " + df["Title"].fillna("")
    df["raw_text"] = df["raw_text"].str.strip()
    # Drop rows where both Content and Title are null
    df = df[df["raw_text"].str.len() > 0]
    df["date"] = pd.to_datetime(df["CreationDate"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    return df[["raw_text", "date"]]


def load_bogleheads(cfg: dict, mode: str) -> pd.DataFrame:
    # Real columns: topic_title, row_type, username, content, time, url
    # row_type ∈ {"Original Post", "Reply"}
    # time is ISO-8601 with timezone e.g. "2022-08-12T22:50:09+00:00"
    # Replies often start with "username wrote:\n..." quote block — strip those
    path = cfg["paths"]["raw_bogleheads"]
    print(f"Loading Bogleheads from {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Bogleheads shape: {df.shape}, row_type counts: {df['row_type'].value_counts().to_dict()}")
    if mode == "smoke":
        df = df.head(cfg.get("sample_rows", 500))

    # For Original Posts: use content as-is
    # For Replies: strip the quoted block at top ("X wrote:\n<quote>\n")
    def strip_bogleheads_quote(text: str) -> str:
        """Remove leading forum quote block: 'Username wrote:\nFri ... pm\n<quoted text>\n'"""
        text = str(text)
        # Pattern: one line ending in 'wrote:' then 1-2 context lines then quoted text
        text = re.sub(r'^.*wrote:\n.*?\n', '', text, flags=re.MULTILINE).strip()
        return text

    df["content_clean"] = df["content"].fillna("").apply(strip_bogleheads_quote)
    # Prepend topic_title to give context — helps GDCM distinguish topic domains
    df["raw_text"] = df["topic_title"].fillna("") + " " + df["content_clean"]
    df["raw_text"] = df["raw_text"].str.strip()
    df = df[df["raw_text"].str.len() > 0]
    # time has timezone — normalize to UTC date
    df["date"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date"])
    return df[["raw_text", "date"]]


def apply_window_filter(df: pd.DataFrame, cfg: dict, dataset: str) -> pd.DataFrame:
    """
    Filter to appropriate time window and tag each row with a 'window' label.
    Bogleheads: keep full range (2018-2022), tag rows in common sub-window as 'common'.
    All other datasets: filter to common window only.
    No row duplication — a single row gets a single window label.
    """
    if dataset == "bogleheads":
        start_full = pd.Timestamp(cfg["bogleheads_full_range"]["start"])
        end_full = pd.Timestamp(cfg["bogleheads_full_range"]["end"])
        start_c = pd.Timestamp(cfg["common_window"]["start"])
        end_c = pd.Timestamp(cfg["common_window"]["end"])

        df = df[(df["date"] >= start_full) & (df["date"] <= end_full)].copy()
        # Tag rows in the common sub-window; rest are full-only
        in_common = (df["date"] >= start_c) & (df["date"] <= end_c)
        df["window"] = "full"
        df.loc[in_common, "window"] = "common"
        n_common = in_common.sum()
        print(f"  Bogleheads: {len(df)} total (full range), {n_common} in common window")
        return df
    else:
        start = pd.Timestamp(cfg["common_window"]["start"])
        end = pd.Timestamp(cfg["common_window"]["end"])
        df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        df["window"] = "common"
        return df


def preprocess_dataset(dataset: str, cfg: dict, mode: str, eui_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n--- Preprocessing: {dataset} (tokenizer: {_SENT_TOKENIZER}) ---")

    loaders = {
        "reddit": load_reddit,
        "stackexchange": load_stackexchange,
        "bogleheads": load_bogleheads,
    }
    raw_df = loaders[dataset](cfg, mode)
    print(f"  Loaded {len(raw_df)} raw records")

    raw_df = apply_window_filter(raw_df, cfg, dataset)
    print(f"  After window filter: {len(raw_df)} records")

    rows = []
    min_tokens = 10
    max_tokens = 500

    for _, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc=f"  Splitting {dataset}"):
        cleaned = clean_text(row["raw_text"])
        chunks = split_into_sentences(cleaned, max_tokens=max_tokens)
        for chunk in chunks:
            if count_tokens(chunk) >= min_tokens:
                rows.append({
                    "text": chunk,
                    "date": row["date"],
                    "window": row.get("window", "common"),
                })

    df = pd.DataFrame(rows)
    print(f"  After sentence split + token filter: {len(df)} sentences")

    df = deduplicate(df, text_col="text")

    df = join_eui_to_posts(df, eui_df, date_col="date", lookaheads=[0, 1, 3, 7])

    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess text data")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "reddit", "stackexchange", "bogleheads"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, args.mode)
    eui_path = cfg["paths"]["eui"]
    if not os.path.exists(eui_path):
        raise FileNotFoundError(f"EUI file not found at {eui_path}. Run 00_download_data.py first.")
    eui_df = load_eui(eui_path)

    datasets = ["reddit", "stackexchange", "bogleheads"] if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        try:
            df = preprocess_dataset(dataset, cfg, args.mode, eui_df)
            out_path = get_processed_path(cfg, dataset, "sentences_labeled.parquet")
            df.to_parquet(out_path, index=False)
            print(f"  Saved {len(df)} rows → {out_path}")

            summary = {
                "dataset": dataset,
                "n_sentences": len(df),
                "date_min": str(df["date"].min().date()),
                "date_max": str(df["date"].max().date()),
                "eui_t0_mean": round(df["eui_t0"].mean(), 2),
                "eui_t0_std": round(df["eui_t0"].std(), 2),
                "avg_tokens": round(df["text"].apply(count_tokens).mean(), 1),
            }
            print(f"  Summary: {summary}")
            summary_df = pd.DataFrame([summary])
            summary_path = get_processed_path(cfg, dataset, "data_summary.csv")
            summary_df.to_csv(summary_path, index=False)

        except FileNotFoundError as e:
            print(f"  SKIP {dataset}: {e}")

    print("\n[01_preprocess] DONE")


if __name__ == "__main__":
    main()
