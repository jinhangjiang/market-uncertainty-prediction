r"""
SCOUT — Sentence-level COncept-guided Uncertainty Tracker  (Stage-1 prototype)

Frozen sentence-transformer -> sentence spans -> soft assignment to K learnable
concept prototypes -> document = mixture of its spans -> daily concept-intensity
features -> forecast-guided quantile head predicting future Equity-Market EPU.
Trained with:  L = L_fcst (pinball) + lambda*L_spr (span entropy) + eta*L_div (prototype diversity).
Encoder is FROZEN (Stage 1): embeddings are cached once; each step only recomputes
a = softmax(E C^T / tau) and re-aggregates by day -> cheap.

This script runs a SMALL SMOKE TEST by default (sample of reddit2022.csv) to validate
the pipeline end to end on CPU. Scale up via the Config (encoder, sample size, K, window)
on a GPU machine for the real runs.

Equations: see EUI_Revision_Design.md section 3.
"""
import os, re, sys, json, math, hashlib
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# ----------------------------- config -----------------------------
@dataclass
class Config:
    here: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = ""
    reddit_csv: str = "reddit2022.csv"
    epu_csv: str = "EquityMarketEPU_daily.csv"
    encoder: str = "all-MiniLM-L6-v2"     # fast 384-d for prototype; use all-mpnet-base-v2 for full
    sample_docs: int = 8000               # SMOKE: sample size; set None for full corpus (GPU)
    min_sent_tokens: int = 4
    max_sent_per_doc: int = 8
    K: int = 12                           # number of concepts
    tau: float = 0.10                     # assignment temperature
    window: int = 20                      # look-back days (90 for full)
    horizons: tuple = (1, 5, 10)          # forecast horizons (use up to 30 for full)
    quantiles: tuple = (0.1, 0.5, 0.9)
    hidden: int = 64
    lam_spr: float = 0.05                 # per-span sparsity weight (each sentence -> one concept)
    eta_div: float = 0.10                 # prototype-diversity weight
    gamma_bal: float = 1.0                # load-balance weight (even concept usage; prevents collapse)
    epochs: int = 300
    lr: float = 1e-2
    test_frac: float = 0.25               # last 25% of days = test (temporal split)
    seed: int = 0

    def __post_init__(self):
        if not self.data_dir:
            self.data_dir = os.path.normpath(os.path.join(self.here, "..", "..", "data"))

# ----------------------------- data -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_URL = re.compile(r"http\S+|www\.\S+")
_WS = re.compile(r"\s+")

def split_sentences(text):
    text = _URL.sub(" ", str(text))
    out = []
    for s in _SENT_SPLIT.split(text):
        s = _WS.sub(" ", s).strip()
        if len(s.split()) >= 4:
            out.append(s)
    return out

def load_spans(cfg):
    """Return DataFrame of spans: columns [doc_id, date, sent]."""
    path = os.path.join(cfg.data_dir, cfg.reddit_csv)
    df = pd.read_csv(path, usecols=["Date", "Text"]).dropna(subset=["Text"])
    if cfg.sample_docs:
        df = df.sample(min(cfg.sample_docs, len(df)), random_state=cfg.seed)
    rows = []
    for did, (_, r) in enumerate(df.iterrows()):
        d = pd.to_datetime(r["Date"], errors="coerce")
        if pd.isna(d):
            continue
        for sent in split_sentences(r["Text"])[: cfg.max_sent_per_doc]:
            rows.append((did, d.normalize(), sent))
    spans = pd.DataFrame(rows, columns=["doc_id", "date", "sent"])
    return spans

def load_epu(cfg):
    e = pd.read_csv(os.path.join(cfg.data_dir, cfg.epu_csv))
    e["date"] = pd.to_datetime(dict(year=e.year, month=e.month, day=e.day))
    return e.set_index("date")["daily_equity_index"].sort_index()

def embed(cfg, sents):
    """Frozen encoder; cache embeddings to .npy keyed by (encoder, sents-hash)."""
    from sentence_transformers import SentenceTransformer
    h = hashlib.md5(("||".join(sents)).encode()).hexdigest()[:10]
    cache = os.path.join(cfg.data_dir, f".scout_emb_{cfg.encoder.split('/')[-1]}_{len(sents)}_{h}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    model = SentenceTransformer(cfg.encoder)
    E = model.encode(sents, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    E = np.asarray(E, dtype=np.float32)
    np.save(cache, E)
    return E

# ----------------------------- model -----------------------------
def build_model(cfg):
    import torch, torch.nn as nn

    class SCOUT(nn.Module):
        def __init__(self, m, K, w, nH, nQ, hidden, C_init):
            super().__init__()
            self.C = nn.Parameter(torch.tensor(C_init, dtype=torch.float32))  # K x m prototypes
            self.head = nn.Sequential(
                nn.Linear(w * K, hidden), nn.ReLU(),
                nn.Linear(hidden, nH * nQ))
            self.nH, self.nQ, self.w, self.K = nH, nQ, w, K

        def assign(self, E, tau):                    # E: N x m  ->  A: N x K
            C = torch.nn.functional.normalize(self.C, dim=1)
            return torch.softmax(E @ C.t() / tau, dim=1)

        def daily_X(self, A, day_idx, T):            # volume aggregation -> T x K
            X = torch.zeros(T, self.K, device=A.device).index_add_(0, day_idx, A)
            return torch.log1p(X)

        def forecast(self, Xw):                      # Xw: B x (w*K) -> B x nH x nQ
            return self.head(Xw).view(-1, self.nH, self.nQ)

    return SCOUT

def pinball(y_true, y_pred, quantiles):              # y_true: B x nH ; y_pred: B x nH x nQ
    import torch
    q = torch.tensor(quantiles, device=y_pred.device).view(1, 1, -1)
    u = y_true.unsqueeze(-1) - y_pred
    return torch.maximum(q * u, (q - 1) * u).mean()

# ----------------------------- train -----------------------------
def run(cfg):
    import torch
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    print(f"[SCOUT] encoder={cfg.encoder} K={cfg.K} window={cfg.window} horizons={cfg.horizons}")

    spans = load_spans(cfg)
    print(f"  spans: {len(spans):,} from {spans.doc_id.nunique():,} docs")
    E = embed(cfg, spans["sent"].tolist())
    print(f"  embeddings: {E.shape}")

    # ----- daily axis -----
    epu = load_epu(cfg)
    days = pd.date_range(spans["date"].min(), spans["date"].max(), freq="D")
    day_pos = {d: i for i, d in enumerate(days)}
    T = len(days)
    span_day = spans["date"].map(day_pos).to_numpy()
    y_full = epu.reindex(days).interpolate().to_numpy()           # daily target aligned to span days

    import torch
    E_t = torch.tensor(E)
    day_idx = torch.tensor(span_day, dtype=torch.long)

    # ----- prototype init via k-means++ on a sample of embeddings -----
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=cfg.K, n_init=4, random_state=cfg.seed).fit(E[np.random.choice(len(E), min(4000, len(E)), replace=False)])
    C_init = km.cluster_centers_ / (np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True) + 1e-9)

    SCOUT = build_model(cfg)
    nH, nQ = len(cfg.horizons), len(cfg.quantiles)
    model = SCOUT(E.shape[1], cfg.K, cfg.window, nH, nQ, cfg.hidden, C_init)

    # ----- windowed supervised samples (temporal) -----
    last = T - max(cfg.horizons)
    t_idx = np.arange(cfg.window - 1, last)                       # window-end days with valid future
    split = int(len(t_idx) * (1 - cfg.test_frac))
    train_t, test_t = t_idx[:split], t_idx[split:]
    y_tr_days = [tt + h for tt in train_t for h in cfg.horizons]
    ymean, ystd = np.nanmean(y_full[:train_t.max() + 1]), np.nanstd(y_full[:train_t.max() + 1]) + 1e-9
    yz = torch.tensor((y_full - ymean) / ystd, dtype=torch.float32)

    def make_targets(ts):
        return torch.tensor(np.stack([[ (y_full[t + h] - ymean) / ystd for h in cfg.horizons] for t in ts]),
                            dtype=torch.float32)
    Y_tr, Y_te = make_targets(train_t), make_targets(test_t)
    train_t_t = torch.tensor(train_t); test_t_t = torch.tensor(test_t)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def window_features(X, ts):                                  # X: T x K -> B x (w*K)
        Xs = (X - X[: train_t.max() + 1].mean(0)) / (X[: train_t.max() + 1].std(0) + 1e-6)  # train-day stats
        return torch.stack([Xs[t - cfg.window + 1: t + 1].reshape(-1) for t in ts])

    for ep in range(cfg.epochs):
        model.train(); opt.zero_grad()
        A = model.assign(E_t, cfg.tau)
        X = model.daily_X(A, day_idx, T)
        pred = model.forecast(window_features(X, train_t_t))
        l_fcst = pinball(Y_tr, pred, cfg.quantiles)
        l_spr = (-(A * (A + 1e-9).log()).sum(1)).mean()                          # per-span entropy (sparsity)
        pbar = A.mean(0)                                                          # marginal concept usage
        l_bal = (pbar * (pbar + 1e-9).log()).sum()                               # = -H(pbar); min -> even usage
        Cn = torch.nn.functional.normalize(model.C, dim=1)
        S = Cn @ Cn.t()
        iu = torch.triu_indices(cfg.K, cfg.K, 1)
        l_div = torch.log(torch.sigmoid(S[iu[0], iu[1]])).mean()                 # diversity (minimize sim)
        loss = l_fcst + cfg.lam_spr * l_spr + cfg.eta_div * l_div + cfg.gamma_bal * l_bal
        loss.backward(); opt.step()
        if ep % 50 == 0 or ep == cfg.epochs - 1:
            print(f"  ep{ep:4d}  L={loss.item():.4f}  fcst={l_fcst.item():.4f}  spr={l_spr.item():.3f}"
                  f"  div={l_div.item():.3f}  bal={l_bal.item():.3f}")

    # ----- evaluation (median forecast NFA) -----
    model.eval()
    with torch.no_grad():
        A = model.assign(E_t, cfg.tau); X = model.daily_X(A, day_idx, T)
        pred_te = model.forecast(window_features(X, test_t_t))
        qmid = cfg.quantiles.index(0.5)
        yhat = pred_te[:, :, qmid].numpy() * ystd + ymean
        ytrue = np.stack([[y_full[t + h] for h in cfg.horizons] for t in test_t])
        smape = np.mean(np.abs(yhat - ytrue) / ((np.abs(yhat) + np.abs(ytrue)) / 2 + 1e-9)) * 100
        nfa = (1 - smape / 200) * 100
    print(f"  [test] NFA={nfa:.2f}%  (h={cfg.horizons})")

    # ----- interpretation: exemplars + c-TF-IDF keywords + forecast importance -----
    with torch.no_grad():
        A = model.assign(E_t, cfg.tau).numpy()
    hard = A.argmax(1)
    interpret(cfg, spans, A, hard)
    concept_importance(cfg, model, E_t, day_idx, T, window_features, test_t_t)
    print("[SCOUT] done.")

def interpret(cfg, spans, A, hard, topn=4, topk_words=8):
    from sklearn.feature_extraction.text import CountVectorizer
    print("\n  ===== concepts (exemplars + c-TF-IDF keywords) =====")
    docs_by_k = [" ".join(spans["sent"].to_numpy()[hard == k]) for k in range(cfg.K)]
    cv = CountVectorizer(stop_words="english", min_df=3, max_features=4000)
    counts = cv.fit_transform(docs_by_k).toarray().astype(float)
    tf = counts / (counts.sum(1, keepdims=True) + 1e-9)
    idf = np.log((cfg.K + 1) / (1 + (counts > 0).sum(0))) + 1
    ctfidf = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    for k in range(cfg.K):
        n = int((hard == k).sum())
        kw = vocab[ctfidf[k].argsort()[::-1][:topk_words]] if counts[k].sum() else []
        ex = spans["sent"].to_numpy()[A[:, k].argsort()[::-1][:topn]]
        print(f"\n  concept {k:2d}  (n={n})  keywords: {', '.join(kw)}")
        for e in ex[:2]:
            print(f"      ex: {e[:90]}")

def concept_importance(cfg, model, E_t, day_idx, T, window_features, test_t_t):
    import torch
    model.zero_grad()
    A = model.assign(E_t, cfg.tau); X = model.daily_X(A, day_idx, T)
    Xs = X.detach().clone().requires_grad_(True)
    feats = torch.stack([((Xs - Xs[:T].mean(0)) / (Xs[:T].std(0) + 1e-6))[t - cfg.window + 1:t + 1].reshape(-1)
                         for t in test_t_t])
    pred = model.forecast(feats)
    pred[:, :, cfg.quantiles.index(0.5)].abs().mean().backward()
    imp = Xs.grad.abs().mean(0).numpy()
    order = imp.argsort()[::-1]
    print("\n  ===== concept forecast-importance (|d yhat / d X_k|) =====")
    for k in order[:6]:
        print(f"      concept {k:2d}: {imp[k]:.4f}")

if __name__ == "__main__":
    run(Config())
