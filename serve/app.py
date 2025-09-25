from __future__ import annotations
import os, numpy as np
import pandas as pd, re, difflib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from joblib import load
# standard-library fuzzy via difflib (no extra deps)

MODEL_DIR = os.getenv("MODEL_DIR", "artifacts")
TOPK_DEFAULT = int(os.getenv("TOPK_DEFAULT", "20"))
SIM_METRIC = os.getenv("SIM_METRIC", "cosine")  # or "dot"

UF = os.path.join(MODEL_DIR, "user_factors.npy")
IF = os.path.join(MODEL_DIR, "item_factors.npy")
UIDS = os.path.join(MODEL_DIR, "user_ids.npy")
IIDS = os.path.join(MODEL_DIR, "item_ids.npy")
UINDEX = os.path.join(MODEL_DIR, "user_index.joblib")
IINDEX = os.path.join(MODEL_DIR, "item_index.joblib")

MOVIES_CSV = os.getenv("MOVIES_CSV", "data/ml-25m/movies.csv")
FUZZY_THRESHOLD = int(os.getenv("FUZZY_THRESHOLD", "70"))

app = FastAPI(title="Duku Recs (Local)")

class ScoredItemWithTitle(BaseModel):
    item_id: int
    score: float
    title: str
    genres: str | None = None

class RecommendRequest(BaseModel):
    user_id: int = Field(...)
    topk: int = Field(TOPK_DEFAULT, ge=1, le=200)

class ScoredItem(BaseModel):
    item_id: int
    score: float

# --- movie title normalization and lookup helpers ---
def _strip_year(title: str) -> str:
    # remove patterns like "(1995)" or trailing years
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()

def _normalize(s: str) -> str:
    # lowercase, remove non-alphanumerics
    s = _strip_year(s)
    return re.sub(r"[^a-z0-9]+", "", s.lower())

_movies_df = None
_norm_to_movieid = {}
_norm_titles_list = []
_movie_meta = {}

_user_f = _item_f = _user_ids = _item_ids = _uindex = _iindex = None
_inv_index = {}
def _load():
    global _user_f, _item_f, _user_ids, _item_ids, _uindex, _iindex, _inv_index
    global _movies_df, _norm_to_movieid, _norm_titles_list, _movie_meta
    if _user_f is None:
        _user_f = np.load(UF)
        _item_f = np.load(IF)
        _user_ids = np.load(UIDS)
        _item_ids = np.load(IIDS)
        _uindex = load(UINDEX)
        _iindex = load(IINDEX)

        _inv_index = {idx: int(iid) for iid, idx in _iindex.items()}

        if not isinstance(_item_ids, np.ndarray) or _item_ids.shape[0] != _item_f.shape[0]:
            # Rebuild id list from item_index mapping
            n = _item_f.shape[0]
            inv = np.empty(n, dtype=np.int64)
            # _iindex is item_id -> row_index; invert it
            for iid, idx in _iindex.items():
                if 0 <= idx < n:
                    inv[idx] = int(iid)
            _item_ids = inv  # overwrite with the corrected mapping

        # Load movies CSV for title lookups (MovieLens format)
        if os.path.exists(MOVIES_CSV):
            _movies_df = pd.read_csv(MOVIES_CSV)
            # Build normalization columns
            _movies_df["bare_title"] = _movies_df["title"].apply(_strip_year)
            _movies_df["norm_title"] = _movies_df["bare_title"].apply(_normalize)
            # Map normalized title -> movieId (first occurrence wins)
            _norm_to_movieid = {}
            _movie_meta = {}
            for _, row in _movies_df.iterrows():
                mid = int(row["movieId"])
                _movie_meta[mid] = {
                    "title": str(row["title"]),
                    "genres": (None if "genres" not in row or pd.isna(row["genres"]) else str(row["genres"]))
                }
                nt = row["norm_title"]
                if nt and nt not in _norm_to_movieid:
                    _norm_to_movieid[nt] = mid
            _norm_titles_list = list(_norm_to_movieid.keys())
        else:
            _movies_df = None
            _norm_to_movieid = {}
            _norm_titles_list = []
            _movie_meta = {}

def _find_movie_id(query: str) -> int | None:
    """Return movieId for a free-text query (case/year/spacing agnostic, fuzzy)."""
    _load()
    if not query:
        return None
    qn = _normalize(query)
    # Exact normalized match
    if qn in _norm_to_movieid:
        return _norm_to_movieid[qn]
    # Substring search on normalized titles
    subs = [nt for nt in _norm_titles_list if qn in nt]
    if subs:
        # choose the shortest (most specific) match
        best = min(subs, key=len)
        return _norm_to_movieid.get(best)
    # Fuzzy via difflib
    if _norm_titles_list:
        best = difflib.get_close_matches(qn, _norm_titles_list, n=1, cutoff=FUZZY_THRESHOLD/100.0)
        if best:
            return _norm_to_movieid.get(best[0])
    return None

def _recommend_similar_items(movie_id: int, topk: int = 10) -> list[tuple[int, float]]:
    _load()
    if movie_id not in _iindex:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not in trained model")

    idx = _iindex[movie_id]
    vec = _item_f[idx]

    if SIM_METRIC == "dot":
        sims = _item_f @ vec
    else:  # default: cosine
        denom = (np.linalg.norm(vec) * np.linalg.norm(_item_f, axis=1))
        denom[denom == 0] = 1e-12
        sims = (_item_f @ vec) / denom

    # oversample to compensate for invalid/missing IDs, then filter
    want = max(1, topk)
    oversample = min(_item_f.shape[0] - 1, want * 10)

    # indices of top 'oversample+1' (include self, we'll drop it)
    k = oversample + 1
    top_idx = np.argpartition(-sims, k-1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    out: list[tuple[int, float]] = []
    for j in top_idx:
        if j == idx:
            continue  # drop self
        iid = _inv_index.get(int(j))  # map row -> movieId
        if iid is None:
            continue  # row has no ID mapping — skip
        # only keep if we actually have title metadata
        if iid not in _movie_meta:
            continue
        out.append((iid, float(sims[j])))
        if len(out) >= want:
            break

    return out

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=list[ScoredItem])
def recommend(req: RecommendRequest):
    _load()
    uid = req.user_id
    if uid not in _uindex:
        raise HTTPException(status_code=404, detail=f"user_id {uid} not found in model")
    uvec = _user_f[_uindex[uid]]
    scores = _item_f @ uvec
    k = min(req.topk, scores.shape[0])
    idx = np.argpartition(-scores, k-1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [ScoredItem(item_id=int(_item_ids[i]), score=float(scores[i])) for i in idx]

@app.get("/search_movie")
def search_movie(query: str, topk: int = 10):
    """Return best-matching titles for a query (debug helper)."""
    _load()
    if _movies_df is None:
        raise HTTPException(status_code=500, detail="Movies catalog not available")
    qn = _normalize(query)
    # simple scoring: exact contains on norm_title, then difflib fallback
    contains = _movies_df[_movies_df["norm_title"].str.contains(qn, na=False)]
    results = []
    for _, row in contains.head(topk).iterrows():
        results.append({
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "genres": (None if "genres" not in row or pd.isna(row["genres"]) else str(row["genres"]))
        })
    if len(results) < topk and not results and len(_norm_titles_list) > 0:
        # fuzzy fallback
        matches = difflib.get_close_matches(qn, _norm_titles_list, n=topk, cutoff=FUZZY_THRESHOLD/100.0)
        for nt in matches:
            mid = _norm_to_movieid[nt]
            meta = _movie_meta.get(mid, {})
            results.append({"movieId": mid, "title": meta.get("title", str(mid)), "genres": meta.get("genres")})
    return results[:topk]

@app.get("/recommend_by_movie", response_model=List[str])
def recommend_by_movie(query: str, topk: int = 10):
    _load()
    mid = _find_movie_id(query)
    if mid is None:
        raise HTTPException(404, f"No movie found for query: {query}")
    recs = _recommend_similar_items(mid, topk=topk)
    return [_movie_meta[iid]["title"] for iid, _ in recs]

@app.get("/debug_neighbors")
def debug_neighbors(query: str = "Toy Story", topk: int = 10):
    """
    Debug helper: return raw neighbor row indices, item_ids, and similarity scores.
    """
    _load()
    mid = _find_movie_id(query)
    if mid is None:
        raise HTTPException(404, f"No movie found for query: {query}")

    recs = _recommend_similar_items(mid, topk=topk)

    debug_out = []
    for iid, score in recs:
        debug_out.append({
            "movieId": iid,
            "title": _movie_meta.get(iid, {}).get("title", "<missing>"),
            "score": score
        })
    return debug_out