"""
Extract recent Personal-Finance Stack Exchange (money.stackexchange.com) posts+comments
for the EUI revision, matching the original StackExchange_.csv schema.

Source: Internet Archive community Stack Exchange Data Dump.
We use the 2025-03-31 release on purpose: it is the LAST release BEFORE Stack Exchange
began watermarking / "poisoning" dump text (which started 2025-06-30). So the text is clean.
This covers any window up to 2025-03-31; our default window (2023-11-23..2024-06-25) is fully inside it.

Output schema (identical to data/StackExchange_.csv):
    SourceType, Id, PostTypeId, ParentId, CreationDate, Score, Content, Title, Tags

Run:
    python extract_stackexchange_recent.py
Requires: requests, py7zr  (auto-checked below)
"""
import os, sys, csv, html, subprocess, tempfile
from datetime import datetime
import xml.etree.ElementTree as ET

# ---- config ----
ARCHIVE_ITEM = "stackexchange_20250331"          # last clean (un-poisoned) community dump
SITE_FILE    = "money.stackexchange.com.7z"
DUMP_URL     = f"https://archive.org/download/{ARCHIVE_ITEM}/{ARCHIVE_ITEM}/{SITE_FILE}"
START = datetime(2023, 11, 23)
END   = datetime(2024, 6, 25)
HERE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.normpath(os.path.join(HERE, "..", "..", "data"))
OUT    = os.path.join(DATA, "StackExchange_recent.csv")
WORK   = os.path.join(tempfile.gettempdir(), "eui_se_work")

def ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg], check=True)

def download(url, dest):
    import requests
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  already downloaded: {dest} ({os.path.getsize(dest)/1e6:.0f} MB)"); return
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=120,
                      headers={"User-Agent": "academic-research (yetanotherdatascientist@gmail.com)"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"  saved {os.path.getsize(dest)/1e6:.0f} MB")

def extract_members(archive, members, outdir):
    import py7zr
    with py7zr.SevenZipFile(archive, "r") as z:
        names = z.getnames()
        want = [n for n in names if os.path.basename(n) in members]
        z.extract(path=outdir, targets=want)
    return {m: os.path.join(outdir, m) for m in members}

def parse_dt(s):
    # SE dump format: 2024-01-05T12:34:56.000
    return datetime.fromisoformat(s.split(".")[0])

def main():
    os.makedirs(WORK, exist_ok=True)
    ensure("requests"); ensure("py7zr")
    archive = os.path.join(WORK, SITE_FILE)
    download(DUMP_URL, archive)
    print("  extracting Posts.xml, Comments.xml ...")
    paths = extract_members(archive, ["Posts.xml", "Comments.xml"], WORK)

    rows = []
    # --- Posts ---
    kept = total = 0
    for _, el in ET.iterparse(paths["Posts.xml"], events=("end",)):
        if el.tag != "row":
            continue
        total += 1
        cd = el.get("CreationDate")
        if cd:
            d = parse_dt(cd)
            if START <= d <= END:
                rows.append(["Post", el.get("Id"), el.get("PostTypeId"), el.get("ParentId", ""),
                             cd, el.get("Score", ""), el.get("Body", ""),
                             el.get("Title", ""), el.get("Tags", "")])
                kept += 1
        el.clear()
    print(f"  Posts: kept {kept:,} / {total:,}")
    # --- Comments ---
    kc = tc = 0
    for _, el in ET.iterparse(paths["Comments.xml"], events=("end",)):
        if el.tag != "row":
            continue
        tc += 1
        cd = el.get("CreationDate")
        if cd:
            d = parse_dt(cd)
            if START <= d <= END:
                rows.append(["Comment", el.get("Id"), "", el.get("PostId", ""),
                             cd, el.get("Score", ""), el.get("Text", ""), "", ""])
                kc += 1
        el.clear()
    print(f"  Comments: kept {kc:,} / {tc:,}")

    rows.sort(key=lambda r: r[4])  # by CreationDate
    os.makedirs(DATA, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SourceType", "Id", "PostTypeId", "ParentId", "CreationDate",
                    "Score", "Content", "Title", "Tags"])
        w.writerows(rows)
    print(f"DONE -> {OUT}  ({len(rows):,} rows, {START.date()}..{END.date()})")

if __name__ == "__main__":
    main()
