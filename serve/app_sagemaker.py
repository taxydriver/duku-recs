# serve/app_sagemaker.py
import os, re, json
import numpy as np, pandas as pd, joblib
from fastapi import FastAPI, Response, Request
from pydantic import BaseModel, Field
from typing import List, Union
from lightfm import LightFM

SM_MODEL_DIR = os.getenv("SM_MODEL_DIR", "/opt/ml/model")
MOVIES_CSV = os.getenv("MOVIES_CSV", os.path.join(SM_MODEL_DIR, "movies.csv"))
SIM_METRIC = os.getenv("SIM_METRIC", "cosine")

MIN_LIKES_FOR_EMBED = int(os.getenv("MIN_LIKES_FOR_EMBED", "3"))
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "1000"))
BETA_BIAS = float(os.getenv("BETA_BIAS", "0.25"))
CANDIDATE_POOL_POP = int(os.getenv("CANDIDATE_POOL_POP", "600"))
CANDIDATE_POOL_SIM_PER_LIKE = int(os.getenv("CANDIDATE_POOL_SIM_PER_LIKE", "200"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.25"))
PER_GENRE_CAP = int(os.getenv("PER_GENRE_CAP", "3"))

app = FastAPI(title="duku-recs (SageMaker)")

# ---- lazy globals
_model: LightFM | None = None
_user_ids = _item_ids = None
_user_map = _item_map = None
_item_embeddings = _item_unit_embeddings = _item_biases = _item_popularity = None
_popularity_idx_sorted = None
_movies = None
_genres_by_idx = None

def _load_array(path, fallback=None, **np_kwargs):
    return np.load(path, **np_kwargs) if os.path.exists(path) else fallback

def _strip_year(t): return re.sub(r"\s*\(\d{4}\)\s*$", "", t).strip()

def _load_once():
    global _model,_user_ids,_item_ids,_user_map,_item_map
    global _item_embeddings,_item_unit_embeddings,_item_biases,_item_popularity
    global _popularity_idx_sorted,_movies,_genres_by_idx

    if _model is not None:
        return

    _model = joblib.load(os.path.join(SM_MODEL_DIR, "lightfm_model.pkl"))
    _user_ids = _load_array(os.path.join(SM_MODEL_DIR, "user_ids.npy"), allow_pickle=True)
    _item_ids = _load_array(os.path.join(SM_MODEL_DIR, "item_ids.npy"), allow_pickle=True)
    _user_map = joblib.load(os.path.join(SM_MODEL_DIR, "user_map.joblib")) if os.path.exists(os.path.join(SM_MODEL_DIR,"user_map.joblib")) else {}
    _item_map = joblib.load(os.path.join(SM_MODEL_DIR, "item_map.joblib")) if os.path.exists(os.path.join(SM_MODEL_DIR,"item_map.joblib")) else {}

    _item_embeddings = _load_array(os.path.join(SM_MODEL_DIR, "item_embeddings.npy"))
    if _item_embeddings is None:
        emb, _ = _model.get_item_representations()
        _item_embeddings = np.asarray(emb, dtype=np.float32)

    _item_unit_embeddings = _load_array(os.path.join(SM_MODEL_DIR, "item_embeddings_unit.npy"))
    if _item_unit_embeddings is None:
        norms = np.linalg.norm(_item_embeddings, axis=1, keepdims=True); norms[norms==0]=1.0
        _item_unit_embeddings = (_item_embeddings / norms).astype(np.float32)

    _item_biases = _load_array(os.path.join(SM_MODEL_DIR, "item_biases.npy"))
    if _item_biases is None:
        _, b = _model.get_item_representations()
        _item_biases = np.asarray(b, dtype=np.float32)

    _item_popularity = _load_array(os.path.join(SM_MODEL_DIR, "item_popularity.npy"))
    if _item_popularity is None:
        _item_popularity = np.ones(_item_ids.shape[0], dtype=np.float32)
    _popularity_idx_sorted = np.argsort(-_item_popularity)

    _movies = pd.read_csv(MOVIES_CSV) if os.path.exists(MOVIES_CSV) else pd.DataFrame({"movieId":_item_ids,"title":[str(i) for i in _item_ids]})
    if "genres" in _movies.columns:
        mid2genres = {int(r.movieId): tuple([] if r.genres == "(no genres listed)" else r.genres.split("|")) for r in _movies.itertuples(index=False)}
    else:
        mid2genres = {int(m):() for m in _movies["movieId"].tolist()}
    _genres_by_idx = {idx: mid2genres.get(int(mid), ()) for mid, idx in _item_map.items()}

def _idx_to_title(item_idx: int) -> str:
    mid = int(_item_ids[item_idx])
    row = _movies[_movies["movieId"] == mid]
    return row["title"].values[0] if not row.empty else str(mid)

def _resolve_movie_ids(liked_ids: List[Union[int, str]]) -> List[int]:
    out = []
    for raw in liked_ids:
        try:
            mid = int(raw)
            if mid in _item_map: out.append(mid)
        except Exception:
            continue
    return out

def _pseudo_user(liked_idx: List[int]) -> np.ndarray | None:
    if not liked_idx: return None
    vec = _item_embeddings[liked_idx].mean(axis=0)
    n = np.linalg.norm(vec)
    if not np.isfinite(n) or n == 0: return None
    return vec / n

def _like_neighbors_pool(liked_idx, per_like: int) -> np.ndarray:
    if not liked_idx or per_like <= 0: return np.array([], dtype=int)
    neigh = []
    for i in liked_idx:
        sims = _item_unit_embeddings @ _item_unit_embeddings[i]
        sims[i] = -np.inf
        k = min(per_like, sims.shape[0])
        neigh.append(np.argpartition(-sims, k-1)[:k])
    return np.unique(np.concatenate(neigh)) if neigh else np.array([], dtype=int)

def _top_popular(topk: int, exclude: set[int]) -> list[int]:
    res = []
    for idx in _popularity_idx_sorted:
        if idx not in exclude:
            res.append(int(idx))
            if len(res) >= topk: break
    return res

def _mmr(candidate_idx: np.ndarray, scores_map: dict[int, float], k: int, lam: float, cap: int | None) -> list[int]:
    picked, caps = [], {}
    C = candidate_idx
    V = _item_unit_embeddings[C]
    row_of = {int(i): r for r, i in enumerate(C)}
    for _ in range(min(k, len(C))):
        best, best_val = None, -1e18
        for idx in C:
            if idx in picked: continue
            if cap and cap > 0:
                gset = _genres_by_idx.get(int(idx), ())
                if any(caps.get(g,0) >= cap for g in gset): continue
            rel = scores_map[int(idx)]
            if not picked:
                mmr = rel
            else:
                r_idx = row_of[int(idx)]
                v = V[r_idx]
                max_sim = max(float(v @ V[row_of[int(p)]]) for p in picked)
                mmr = lam * rel - (1.0 - lam) * max_sim
            if mmr > best_val:
                best, best_val = int(idx), mmr
        if best is None: break
        picked.append(best)
        if cap and cap > 0:
            for g in _genres_by_idx.get(int(best), ()):
                caps[g] = caps.get(g, 0) + 1
    return picked

class NewUserRequest(BaseModel):
    liked_ids: List[Union[int, str]] = Field(default_factory=list)
    topk: int = Field(10, ge=1, le=200)

@app.get("/ping")
def ping():
    try:
        _load_once()
        return {"status": "ok"}
    except Exception as e:
        return Response(content=json.dumps({"status":"error","detail":str(e)}),
                        status_code=500, media_type="application/json")

@app.post("/invocations")
async def invocations(request: Request):
    _load_once()
    try:
        payload = await request.json()
    except Exception as e:
        return Response(content=json.dumps({"error": f"invalid json: {str(e)}"}),
                        status_code=400, media_type="application/json")

    op = payload.get("op", "recommend_new_user")
    if op != "recommend_new_user":
        return Response(json.dumps({"error":"unsupported op"}),
                        status_code=400, media_type="application/json")

    req = NewUserRequest(
        liked_ids=payload.get("liked_ids", []),
        topk=payload.get("topk", 10)
    )

    mids = _resolve_movie_ids(req.liked_ids)
    liked_idx = [ _item_map[m] for m in mids if m in _item_map ]
    liked_set = set(liked_idx)

    if not liked_idx or len(liked_idx) < MIN_LIKES_FOR_EMBED:
        idx = _top_popular(req.topk, exclude=liked_set)
        return {"titles": [_idx_to_title(i) for i in idx]}

    uvec = _pseudo_user(liked_idx)
    if uvec is None:
        idx = _top_popular(req.topk, exclude=liked_set)
        return {"titles": [_idx_to_title(i) for i in idx]}

    pop_part = [int(i) for i in _popularity_idx_sorted if i not in liked_set][:CANDIDATE_POOL_POP]
    sim_part = _like_neighbors_pool(liked_idx, CANDIDATE_POOL_SIM_PER_LIKE)
    candidates = np.unique(np.concatenate([np.array(pop_part, dtype=int), sim_part])) if sim_part.size or pop_part else np.array([], dtype=int)
    candidates = np.array([int(i) for i in candidates if int(i) not in liked_set], dtype=int)

    if candidates.size == 0:
        idx = _top_popular(req.topk, exclude=liked_set)
        return {"titles": [_idx_to_title(i) for i in idx]}

    C = candidates[:max(CANDIDATE_POOL, req.topk)].astype(np.int32)

    if SIM_METRIC == "dot":
        s = _item_embeddings[C] @ uvec
    else:
        uhat = uvec / (np.linalg.norm(uvec) + 1e-9)
        s = _item_unit_embeddings[C] @ uhat
    s = s + BETA_BIAS * _item_biases[C]
    s = np.where(np.isfinite(s), s, -np.inf)

    scores_map = {int(i): float(si) for i, si in zip(C, s)}
    picked = _mmr(C, scores_map, req.topk, MMR_LAMBDA, PER_GENRE_CAP)
    if len(picked) < req.topk:
        picked += [i for i in C if i not in picked][:req.topk - len(picked)]

    return {"titles": [_idx_to_title(i) for i in picked[:req.topk]]}