"""
Extract recent r/personalfinance posts + comments for the EUI revision via the
Arctic Shift API (https://github.com/ArthurHeitmann/arctic_shift), matching the
original data/reddit2022.csv schema.

Why this and not PRAW: PRAW/the official Reddit API cannot retrieve a complete
historical date range retrospectively (it returns only recent/limited listings).
Arctic Shift serves the pushshift-style archive, so historical pulls are possible.

Output schema (identical to data/reddit2022.csv):
    Submission_Id, Reply_Id, Submission_title, Author, Date, Vote, Text
  - A submission row:  Submission_Id == Reply_Id == post id; Text = selftext.
  - A comment row:     Submission_Id = parent post id (link_id w/o 't3_'),
                       Reply_Id = comment id; Submission_title filled by join
                       to the collected posts when available, else "".
  - Date stored as YYYY-MM-DD (matches original, which is date-only).

Resumable: writes incrementally and records a cursor in <out>.state.json, so a
re-run continues where it left off. Safe to Ctrl-C / re-launch.

Run:
    python extract_reddit_recent.py
"""
import os, csv, json, time
from datetime import datetime, timezone
import requests

SUBREDDIT = "personalfinance"
# Long recent span for the PRIMARY corpus (2023-2025). The matched 215-day slice
# (2023-11-23..2024-06-25), comparable to the COVID window, is a subset of this and
# can be filtered out anytime for the strict COVID-vs-normal comparison.
START = datetime(2023, 1, 1, tzinfo=timezone.utc)
END   = datetime(2025, 12, 31, tzinfo=timezone.utc)
HERE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.normpath(os.path.join(HERE, "..", "..", "data"))
OUT   = os.path.join(DATA, "reddit_recent.csv")
STATE = OUT + ".state.json"
API   = "https://arctic-shift.photon-reddit.com/api"
UA    = {"User-Agent": "academic-research (yetanotherdatascientist@gmail.com)"}
HEADER = ["Submission_Id", "Reply_Id", "Submission_title", "Author", "Date", "Vote", "Text"]

def epoch(dt): return int(dt.timestamp())
def to_date(u): return datetime.fromtimestamp(int(u), tz=timezone.utc).strftime("%Y-%m-%d")

def get(kind, after, before):
    """One page of <=100 items, ascending, with retry + rate-limit handling."""
    url = f"{API}/{kind}/search"
    params = {"subreddit": SUBREDDIT, "after": after, "before": before, "limit": 100, "sort": "asc"}
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=45)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("X-RateLimit-Reset", 5)) + 1); continue
            r.raise_for_status()
            rem = int(r.headers.get("X-RateLimit-Remaining", 999))
            if rem < 25:
                time.sleep(int(r.headers.get("X-RateLimit-Reset", 2)) + 1)
            return r.json().get("data", [])
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))
    return []

def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"phase": "posts", "cursor": epoch(START), "boundary": [], "posts": 0, "comments": 0}

def save_state(s): json.dump(s, open(STATE, "w"))

def paginate(kind, state, writer, fout, title_map=None, collect_titles=False):
    after = state["cursor"]; before = epoch(END)
    boundary_seen = set(state.get("boundary", []))
    n = state["posts"] if kind == "posts" else state["comments"]
    last_flush = time.time()
    while True:
        data = get(kind, after, before)
        if not data:
            break
        fresh = [d for d in data if d.get("id") not in boundary_seen]
        if not fresh and len(data) < 100:
            break
        maxc = after
        for d in data:
            cu = int(d.get("created_utc", 0))
            if cu > maxc: maxc = cu
            if d.get("id") in boundary_seen:
                continue
            if kind == "posts":
                pid = d.get("id")
                if collect_titles:
                    title_map[pid] = d.get("title", "")
                writer.writerow([pid, pid, d.get("title", ""), d.get("author", ""),
                                 to_date(cu), d.get("score", ""), d.get("selftext", "")])
            else:
                link = (d.get("link_id", "") or "").replace("t3_", "")
                writer.writerow([link, d.get("id"), (title_map or {}).get(link, ""),
                                 d.get("author", ""), to_date(cu), d.get("score", ""),
                                 d.get("body", "")])
            n += 1
        # advance cursor; remember ids at the boundary second to dedup next page
        boundary_seen = {d.get("id") for d in data if int(d.get("created_utc", 0)) == maxc}
        after = maxc
        state["cursor"] = after
        state["boundary"] = list(boundary_seen)
        state["posts" if kind == "posts" else "comments"] = n
        if time.time() - last_flush > 10:
            fout.flush(); save_state(state)
            print(f"  {kind}: {n:,} rows  (cursor {to_date(after)})", flush=True)
            last_flush = time.time()
        if len(data) < 100 and maxc >= before:
            break
    return n

def main():
    os.makedirs(DATA, exist_ok=True)
    state = load_state()
    title_map = {}
    new = not os.path.exists(OUT)
    fout = open(OUT, "a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if new:
        writer.writerow(HEADER)

    if state["phase"] == "posts":
        print(f"PHASE posts  {START.date()}..{END.date()}", flush=True)
        paginate("posts", state, writer, fout, title_map, collect_titles=True)
        json.dump(title_map, open(OUT + ".titles.json", "w"))
        state["phase"] = "comments"; state["cursor"] = epoch(START); state["boundary"] = []
        save_state(state); fout.flush()
        print(f"  posts done: {state['posts']:,}", flush=True)

    if state["phase"] == "comments":
        if os.path.exists(OUT + ".titles.json"):
            title_map = json.load(open(OUT + ".titles.json"))
        print("PHASE comments", flush=True)
        paginate("comments", state, writer, fout, title_map)
        state["phase"] = "done"; save_state(state)

    fout.flush(); fout.close()
    print(f"DONE -> {OUT}  posts={state['posts']:,} comments={state['comments']:,}", flush=True)

if __name__ == "__main__":
    main()
