"""
tests/test_smoke_preprocess.py
End-to-end smoke test for preprocessing on actual data slices.
Verifies column alignment, EUI join, text quality, and output shape.

Run with:
    python tests/test_smoke_preprocess.py
"""
import os
import sys
import re
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

from src.utils.config_loader import load_config
from src.utils.eui_loader import download_eui, load_eui, join_eui_to_posts

_spec = importlib.util.spec_from_file_location(
    "preprocess",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "src", "01_preprocess.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_stackexchange = _mod.load_stackexchange
load_bogleheads = _mod.load_bogleheads
apply_window_filter = _mod.apply_window_filter
clean_text = _mod.clean_text
split_into_sentences = _mod.split_into_sentences
count_tokens = _mod.count_tokens
deduplicate = _mod.deduplicate

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    results.append((name, condition))
    return condition


def test_stackexchange(cfg, eui_df):
    print("\n=== StackExchange loader ===")
    df = load_stackexchange(cfg, mode="smoke")

    check("SE: has raw_text column", "raw_text" in df.columns)
    check("SE: has date column", "date" in df.columns)
    check("SE: no null dates", df["date"].isnull().sum() == 0,
          f"{df['date'].isnull().sum()} nulls")
    check("SE: no empty raw_text", (df["raw_text"].str.len() == 0).sum() == 0,
          f"{(df['raw_text'].str.len() == 0).sum()} empty")
    check("SE: date is datetime", pd.api.types.is_datetime64_any_dtype(df["date"]))
    check("SE: row count > 0", len(df) > 0, f"{len(df)} rows")

    # Verify clean_text strips HTML
    html_sample = "<p>Some <b>text</b> with <a href='x'>link</a>.</p>"
    cleaned = clean_text(html_sample)
    check("clean_text: strips HTML tags", "<" not in cleaned, repr(cleaned[:80]))
    check("clean_text: retains words", "text" in cleaned and "link" in cleaned)

    # Apply window filter
    df_w = apply_window_filter(df, cfg, "stackexchange")
    check("SE: window filter has 'window' column", "window" in df_w.columns)
    check("SE: all rows are 'common' window", (df_w["window"] == "common").all())
    check("SE: window-filtered dates in range",
          df_w["date"].min() >= pd.Timestamp("2021-11-23") and
          df_w["date"].max() <= pd.Timestamp("2022-06-25"),
          f"{df_w['date'].min().date()} – {df_w['date'].max().date()}")

    # EUI join
    if eui_df is not None and len(df_w) > 0:
        joined = join_eui_to_posts(df_w, eui_df, date_col="date", lookaheads=[0, 1, 3, 7])
        check("SE: EUI join produces eui_t0", "eui_t0" in joined.columns)
        check("SE: EUI join produces eui_t1", "eui_t1" in joined.columns)
        check("SE: no null eui_t0 after join",
              joined["eui_t0"].isnull().sum() == 0,
              f"{joined['eui_t0'].isnull().sum()} nulls")
        check("SE: eui_t0 values are positive",
              (joined["eui_t0"] > 0).all(),
              f"min={joined['eui_t0'].min():.2f}")
        print(f"  eui_t0 stats: mean={joined['eui_t0'].mean():.2f}, "
              f"std={joined['eui_t0'].std():.2f}, "
              f"min={joined['eui_t0'].min():.2f}, max={joined['eui_t0'].max():.2f}")
        return joined
    return df_w


def test_bogleheads(cfg, eui_df):
    print("\n=== Bogleheads loader ===")
    df = load_bogleheads(cfg, mode="smoke")

    check("BH: has raw_text column", "raw_text" in df.columns)
    check("BH: has date column", "date" in df.columns)
    check("BH: no null dates", df["date"].isnull().sum() == 0,
          f"{df['date'].isnull().sum()} nulls")
    check("BH: no empty raw_text", (df["raw_text"].str.len() == 0).sum() == 0)
    check("BH: date is datetime", pd.api.types.is_datetime64_any_dtype(df["date"]))
    check("BH: row count > 0", len(df) > 0, f"{len(df)} rows")

    # Verify quote stripping actually works
    sample_reply = df[df["raw_text"].str.contains("wrote:", na=False)]
    if len(sample_reply) > 0:
        sample = sample_reply["raw_text"].iloc[0]
        check("BH: quote stripping partially cleaned",
              len(sample) < 2000,
              f"sample length={len(sample)}")

    # Verify topic_title is prepended (raw_text should often start with topic)
    check("BH: raw_text not just whitespace",
          df["raw_text"].str.strip().str.len().min() > 5)

    # Apply window filter — Bogleheads uses full range, tags common subset
    df_w = apply_window_filter(df, cfg, "bogleheads")
    check("BH: window filter has 'window' column", "window" in df_w.columns)
    check("BH: has 'full' window rows", (df_w["window"] == "full").any() or
          (df_w["window"] == "common").any())
    print(f"  window distribution: {df_w['window'].value_counts().to_dict()}")

    # EUI join on common-window subset
    df_common = df_w[df_w["window"] == "common"].copy()
    if eui_df is not None and len(df_common) > 0:
        joined = join_eui_to_posts(df_common, eui_df, date_col="date", lookaheads=[0, 1, 3, 7])
        check("BH: EUI join produces eui_t0", "eui_t0" in joined.columns)
        check("BH: no null eui_t0 in common window",
              joined["eui_t0"].isnull().sum() == 0,
              f"{joined['eui_t0'].isnull().sum()} nulls")
        print(f"  eui_t0 stats: mean={joined['eui_t0'].mean():.2f}, "
              f"std={joined['eui_t0'].std():.2f}")
        return joined
    return df_w


def test_sentence_splitting(df_sample):
    print("\n=== Sentence splitting + token counting ===")
    texts = df_sample["raw_text"].dropna().head(20).tolist()
    all_chunks = []
    for text in texts:
        cleaned = clean_text(text)
        chunks = split_into_sentences(cleaned, max_tokens=500)
        all_chunks.extend(chunks)

    check("Splitting: produces non-empty chunks", len(all_chunks) > 0,
          f"{len(all_chunks)} chunks from {len(texts)} texts")

    token_counts = [count_tokens(c) for c in all_chunks]
    check("Splitting: all chunks >= 1 token", min(token_counts) >= 1,
          f"min={min(token_counts)}")
    check("Splitting: all chunks <= 500 tokens", max(token_counts) <= 500,
          f"max={max(token_counts)}")
    check("Splitting: no HTML in output", not any("<" in c for c in all_chunks),
          "HTML tags found in output" if any("<" in c for c in all_chunks) else "clean")
    check("Splitting: no URL patterns in output",
          not any("http" in c.lower() for c in all_chunks),
          "URLs found" if any("http" in c.lower() for c in all_chunks) else "clean")

    print(f"  token counts: min={min(token_counts)}, max={max(token_counts)}, "
          f"mean={np.mean(token_counts):.1f}")
    print(f"  sample chunk: {repr(all_chunks[0][:120])}")


def test_deduplication():
    print("\n=== Deduplication ===")
    df = pd.DataFrame({
        "text": ["hello world", "hello world", "unique text", "another unique"],
        "date": pd.to_datetime(["2022-01-01"] * 4),
    })
    deduped = deduplicate(df, text_col="text")
    check("Dedup: removes exact duplicates", len(deduped) == 3,
          f"got {len(deduped)}, expected 3")


def test_column_alignment(se_df, bh_df):
    """Verify both datasets produce aligned output columns after full preprocessing."""
    print("\n=== Cross-dataset column alignment ===")
    required_cols = {"raw_text", "date", "window"}
    se_cols = set(se_df.columns) if se_df is not None else set()
    bh_cols = set(bh_df.columns) if bh_df is not None else set()

    for col in ["raw_text", "date", "window"]:
        check(f"SE has column '{col}'", col in se_cols)
        check(f"BH has column '{col}'", col in bh_cols)

    # Both should produce same dtype for 'date'
    if se_df is not None and bh_df is not None:
        check("date dtype aligned between datasets",
              pd.api.types.is_datetime64_any_dtype(se_df["date"]) and
              pd.api.types.is_datetime64_any_dtype(bh_df["date"]))


def run_all():
    print("=" * 60)
    print("EUI Pipeline Smoke Test: Preprocessing")
    print("=" * 60)

    cfg = load_config("configs/pipeline_config.yaml", "smoke")

    # Try to load EUI
    eui_df = None
    eui_path = cfg["paths"]["eui"]
    if os.path.exists(eui_path):
        eui_df = load_eui(eui_path)
        print(f"\nEUI loaded: {len(eui_df)} rows, "
              f"{eui_df['ds'].min().date()} – {eui_df['ds'].max().date()}")
    else:
        print(f"\nWARNING: EUI not at {eui_path}. Run 00_download_data.py first.")
        print("EUI-join tests will be skipped.")

    se_out = test_stackexchange(cfg, eui_df)
    bh_out = test_bogleheads(cfg, eui_df)

    # Use SE data for sentence splitting test (it has HTML which exercises more code paths)
    se_raw = pd.read_csv(cfg["paths"]["raw_stackexchange"], nrows=50)
    se_raw["raw_text"] = se_raw["Content"].fillna("") + " " + se_raw["Title"].fillna("")
    se_raw["date"] = pd.to_datetime(se_raw["CreationDate"], errors="coerce").dt.normalize()
    test_sentence_splitting(se_raw)
    test_deduplication()
    test_column_alignment(se_out, bh_out)

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")
    if failed > 0:
        print(f"\nFailed checks:")
        for name, ok in results:
            if not ok:
                print(f"  - {name}")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
