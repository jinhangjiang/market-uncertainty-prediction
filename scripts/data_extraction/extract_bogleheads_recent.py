r"""
Extract recent Bogleheads "Personal Finance (Not Investing)" forum (f=2) posts
for the EUI revision, matching the original data/bogleheads_.csv schema:
    topic_title, row_type, username, content, time, url

WHY SELENIUM: bogleheads.org sits behind Cloudflare. Plain requests / cloudscraper
get HTTP 403 (verified). A real browser via Selenium clears the challenge — this is
the same approach that worked in ScrapeBogleheads2025.ipynb. This script therefore
must be run on a machine WITH Google Chrome installed (the sandbox has chromedriver
but no Chrome browser, so it cannot run there).

The forum index (viewforum.php?f=2) is sorted by LAST-POST descending, so start=0 is
the most recent. We page forward (start += 50), enter every thread whose last activity
is on/after START, scrape that thread's pages, and keep posts whose timestamp is inside
[START, END]. We stop once the index reaches threads last active before START.

Resumable: appends to the CSV and records visited thread URLs in <out>.seen.txt.

Run (Windows, your chromedriver is at C:\chromedriver-win64\...):
    python extract_bogleheads_recent.py
Requires: selenium  (pip install selenium)  + Google Chrome installed.
"""
import os, csv, time, random
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException

# ---- config ----
FORUM_ID = 2                                   # Personal Finance (Not Investing)
INDEX = f"https://www.bogleheads.org/forum/viewforum.php?f={FORUM_ID}"
BASE  = "https://www.bogleheads.org/forum/"
START = datetime(2023, 11, 23)
END   = datetime(2024, 6, 25)
HERE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.normpath(os.path.join(HERE, "..", "..", "data"))
OUT   = os.path.join(DATA, "bogleheads_recent.csv")
SEEN  = OUT + ".seen.txt"
HEADER = ["topic_title", "row_type", "username", "content", "time", "url"]
# Leave as None so Selenium Manager (built into Selenium 4.6+) auto-downloads the
# chromedriver that matches your installed Chrome. Only set an explicit path if you
# have a driver whose major version EXACTLY matches your Chrome (e.g. 148 with 148).
# A mismatch raises: "This version of ChromeDriver only supports Chrome version N".
CHROMEDRIVER = None
HEADLESS = False          # KEEP False — headless is far easier for Cloudflare to block
MAX_INDEX_PAGES = 4000    # safety cap
START_OFFSET = None       # None = auto-locate window via binary search.
                          # If auto-locate misbehaves, set an integer offset to start
                          # paging from (e.g. 15000) — like the original notebook did.
USE_UNDETECTED = True     # use undetected-chromedriver to evade Cloudflare bot detection
                          #   pip install undetected-chromedriver setuptools
CHROME_MAJOR = None       # None = auto-detect your installed Chrome major version.
                          # uc otherwise grabs the LATEST driver, which may not match
                          # your Chrome ("only supports Chrome version N"). Set e.g. 148
                          # to force it if auto-detect fails.

def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00").split("+")[0])
    except Exception:
        return None

def detect_chrome_major():
    if CHROME_MAJOR:
        return CHROME_MAJOR
    import subprocess
    candidates = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
    for p in candidates:
        if os.path.exists(p):
            try:
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     f"(Get-Item '{p}').VersionInfo.ProductVersion"],
                    capture_output=True, text=True, timeout=15).stdout.strip()
                if out:
                    return int(out.split(".")[0])
            except Exception:
                pass
    return None

def make_driver():
    if USE_UNDETECTED:
        import undetected_chromedriver as uc
        o = uc.ChromeOptions()
        if HEADLESS:
            o.add_argument("--headless=new")
        o.add_argument("--window-size=1280,900")
        major = detect_chrome_major()
        print(f"  launching undetected-chromedriver (Chrome major = {major})", flush=True)
        # version_main pins the driver to YOUR Chrome version (else uc grabs latest).
        return uc.Chrome(options=o, version_main=major) if major else uc.Chrome(options=o)
    o = Options()
    o.add_argument("--disable-blink-features=AutomationControlled")
    o.add_experimental_option("excludeSwitches", ["enable-automation"])
    o.add_experimental_option("useAutomationExtension", False)
    if HEADLESS:
        o.add_argument("--headless=new")
    o.add_argument("--window-size=1280,900")
    svc = Service(CHROMEDRIVER) if CHROMEDRIVER and os.path.exists(CHROMEDRIVER) else Service()
    d = webdriver.Chrome(service=svc, options=o)
    d.execute_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    return d

def is_hard_blocked(driver):
    """Detect a Cloudflare hard block (not the transient 'just a moment' challenge)."""
    s = (driver.page_source or "").lower()
    return any(x in s for x in ["you have been blocked", "sorry, you have been blocked",
                                "error 1020", "access denied", "attention required"])

def load_seen():
    return set(open(SEEN, encoding="utf-8").read().split()) if os.path.exists(SEEN) else set()

def parse_topic_row(r):
    """(last_active, title, href) for a REAL topic row, or None for stickies/
    announcements/non-topic rows. Bogleheads pins old global-announcement threads
    at the top of every page; those must be ignored or they corrupt the date logic."""
    cls = (r.get_attribute("class") or "").lower()
    if any(k in cls for k in ("sticky", "announce", "global")):
        return None
    links = r.find_elements(By.CSS_SELECTOR, "a.topictitle")
    if not links:                      # not a topic row (stats/footer/etc.)
        return None
    tt = r.find_elements(By.CSS_SELECTOR, "dd.lastpost time") or r.find_elements(By.TAG_NAME, "time")
    la = parse_dt(tt[-1].get_attribute("datetime")) if tt else None
    return (la, links[0].text, links[0].get_attribute("href"))

def real_topic_rows(driver):
    out = []
    for r in driver.find_elements(By.CSS_SELECTOR, "li.row"):
        try:
            t = parse_topic_row(r)
            if t:
                out.append(t)
        except Exception:
            continue
    return out

_NET_ERRS = ("ERR_INTERNET_DISCONNECTED", "ERR_NAME_NOT_RESOLVED", "ERR_CONNECTION",
             "ERR_NETWORK_CHANGED", "ERR_TIMED_OUT", "ERR_PROXY_CONNECTION",
             "ERR_ADDRESS_UNREACHABLE", "ERR_NETWORK_IO_SUSPENDED")

def net_get(driver, url, tries=12):
    """driver.get that survives transient network drops: waits + retries instead of crashing."""
    for i in range(tries):
        try:
            driver.get(url)
            return True
        except WebDriverException as e:
            if any(x in str(e) for x in _NET_ERRS):
                wait = min(300, 20 * (i + 1))     # 20s,40s,... capped at 5min — rides out outages
                print(f"    [net] offline; waiting {wait}s then retry ({i+1}/{tries}) — {url[:55]}", flush=True)
                time.sleep(wait)
                continue
            raise
    raise WebDriverException(f"net_get: gave up after {tries} tries: {url}")

def page_top_last_active(driver, offset):
    """Last-activity datetime of the first NON-sticky topic on the index page (or None)."""
    net_get(driver, f"{INDEX}&start={offset}")
    time.sleep(random.uniform(2.0, 3.5))
    if is_hard_blocked(driver):
        raise RuntimeError("Cloudflare block during index probe")
    for la, _, _ in real_topic_rows(driver):
        if la:
            return la
    return None

def find_window_start(driver, max_pages=1400):
    """Binary-search the smallest index page whose top thread is last-active <= END.
    The index is sorted by last-activity DESC, so this jumps us to the window top
    instead of crawling from page 0."""
    lo, hi, ans = 0, max_pages, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        la = page_top_last_active(driver, mid * 50)
        if la is None:          # past the end of the forum -> go shallower
            hi = mid - 1
        elif la > END:          # still too recent -> go deeper
            lo = mid + 1
        else:                   # at/older than END -> candidate, try shallower
            ans = mid; hi = mid - 1
    return max(0, (ans - 1)) * 50   # back up one page for safety

def scrape_thread(driver, url, title, writer, fout):
    """Walk every page of a thread; emit posts whose time is in [START, END]."""
    cur = url
    n = 0
    while cur:
        net_get(driver, cur)
        try:
            WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.CLASS_NAME, "post")))
        except Exception:
            break
        posts = driver.find_elements(By.CLASS_NAME, "post")
        for i, p in enumerate(posts):
            try:
                tt = p.find_element(By.TAG_NAME, "time").get_attribute("datetime")
                d = parse_dt(tt)
                if not d or not (START <= d <= END):
                    continue
                user = p.find_element(By.CSS_SELECTOR, "p.author strong").text
                content = p.find_element(By.CLASS_NAME, "content").text
                writer.writerow([title,
                                 "Original Post" if (i == 0 and "start=" not in cur) else "Reply",
                                 user, content, tt, cur])
                n += 1
            except Exception:
                continue
        nxt = driver.find_elements(By.CSS_SELECTOR, "li.arrow.next a")
        cur = nxt[0].get_attribute("href") if nxt else None
        if cur:
            time.sleep(random.uniform(1.0, 2.2))
    fout.flush()
    return n

def main():
    os.makedirs(DATA, exist_ok=True)
    seen = load_seen()
    new = not os.path.exists(OUT)
    fout = open(OUT, "a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if new:
        writer.writerow(HEADER)
    seenf = open(SEEN, "a", encoding="utf-8")
    driver = make_driver()
    total = 0
    try:
        net_get(driver, INDEX)
        time.sleep(random.uniform(10, 14))  # let Cloudflare's challenge clear
        if is_hard_blocked(driver):
            print("\nCloudflare HARD BLOCK detected. Stopping so repeated hits don't worsen it.\n"
                  "Options: (1) wait several hours and retry (IP blocks are usually temporary),\n"
                  "         (2) retry from a different network/IP,\n"
                  "         (3) confirm undetected-chromedriver is installed and HEADLESS=False.")
            return

        if START_OFFSET is not None:
            start_offset = START_OFFSET
            print(f"  using manual START_OFFSET={start_offset}", flush=True)
        else:
            print(f"  locating window {START.date()}..{END.date()} in the index ...", flush=True)
            start_offset = find_window_start(driver)
        print(f"  starting at offset {start_offset}", flush=True)

        done = False
        for page in range(MAX_INDEX_PAGES):
            offset = start_offset + page * 50
            net_get(driver, f"{INDEX}&start={offset}")
            time.sleep(random.uniform(2.5, 4.5))
            if is_hard_blocked(driver):
                print(f"Cloudflare block hit at offset {offset}; stopping gracefully. "
                      f"Progress saved ({total} rows). Re-run later to resume.")
                break
            threads = real_topic_rows(driver)        # excludes stickies/announcements
            if not threads:
                print("No more topic rows; stopping."); break
            top = next((la for la, _, _ in threads if la), None)
            print(f"  [offset {offset}] {len(threads)} topics, top last-active="
                  f"{top.date() if top else '?'}  total saved={total}", flush=True)
            for last_active, title, href in threads:
                if last_active and last_active > END:
                    continue                      # newer than window -> skip WITHOUT opening
                if last_active and last_active < START:
                    done = True; continue         # older than window -> we're past it
                if not href or href in seen:
                    continue
                try:
                    got = scrape_thread(driver, href, title, writer, fout)
                except WebDriverException as e:
                    print(f"      [skip] {title[:40]!r} failed after retries: {str(e).splitlines()[0][:50]}", flush=True)
                    seen.add(href); seenf.write(href + "\n"); seenf.flush()
                    continue
                total += got
                seen.add(href); seenf.write(href + "\n"); seenf.flush()
                if got:
                    print(f"      + {got:3d}  {title[:55]!r}  (total {total})", flush=True)
                time.sleep(random.uniform(1.2, 2.5))
            if done:
                print("Reached threads older than window; stopping."); break
    finally:
        driver.quit()
        fout.flush(); fout.close(); seenf.close()
    print(f"DONE -> {OUT}  ({total} in-window post rows, {START.date()}..{END.date()})")

if __name__ == "__main__":
    main()
