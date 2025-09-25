# serve/app_lightfm.py
import os, re, difflib, numpy as np, pandas as pd, joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union
from lightfm import LightFM

# --------------------
# Config / knobs
# --------------------
MODEL_DIR = os.getenv("MODEL_DIR", "artifacts/lightfm")
MOVIES_CSV = os.getenv("MOVIES_CSV", "data/ml-25m/movies.csv")
FUZZY_THRESHOLD = int(os.getenv("FUZZY_THRESHOLD", "70"))
SIM_METRIC = os.getenv("SIM_METRIC", "cosine")  # "cosine" or "dot"

# New-user behavior
MIN_LIKES_FOR_EMBED = int(os.getenv("MIN_LIKES_FOR_EMBED", "3"))
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "1000"))       # final pool size (>= topk)
# Additional knobs to make cold-start sensitive to likes
BETA_BIAS = float(os.getenv("BETA_BIAS", "0.25"))               # 0..1 weight on item_bias for new users
CANDIDATE_POOL_POP = int(os.getenv("CANDIDATE_POOL_POP", "600"))  # popularity portion of the pool
CANDIDATE_POOL_SIM_PER_LIKE = int(os.getenv("CANDIDATE_POOL_SIM_PER_LIKE", "200"))  # neighbors per liked item
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.25"))             # lower = more diversity influence
PER_GENRE_CAP = int(os.getenv("PER_GENRE_CAP", "3"))            # 0 to disable

# --------------------
# Load artifacts
# --------------------
model: LightFM = joblib.load(os.path.join(MODEL_DIR, "lightfm_model.pkl"))
user_ids = np.load(os.path.join(MODEL_DIR, "user_ids.npy"))
item_ids = np.load(os.path.join(MODEL_DIR, "item_ids.npy"))
user_map = joblib.load(os.path.join(MODEL_DIR, "user_map.joblib"))
item_map = joblib.load(os.path.join(MODEL_DIR, "item_map.joblib"))

item_embeddings_path = os.path.join(MODEL_DIR, os.getenv("ITEM_EMBED_FILE", "item_embeddings.npy"))
if os.path.exists(item_embeddings_path):
    item_embeddings = np.load(item_embeddings_path).astype(np.float32)
else:
    # embeddings, biases
    emb, _ = model.get_item_representations()
    item_embeddings = np.asarray(emb, dtype=np.float32)

item_unit_embeddings_path = os.path.join(MODEL_DIR, os.getenv("ITEM_UNIT_EMBED_FILE", "item_embeddings_unit.npy"))
if os.path.exists(item_unit_embeddings_path):
    item_unit_embeddings = np.load(item_unit_embeddings_path).astype(np.float32)
else:
    norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    item_unit_embeddings = (item_embeddings / norms).astype(np.float32)

item_biases_path = os.path.join(MODEL_DIR, os.getenv("ITEM_BIAS_FILE", "item_biases.npy"))
if os.path.exists(item_biases_path):
    item_biases = np.load(item_biases_path).astype(np.float32)
else:
    emb, biases = model.get_item_representations()  # correct order
    item_biases = np.asarray(biases, dtype=np.float32)

item_popularity_path = os.path.join(MODEL_DIR, os.getenv("ITEM_POP_FILE", "item_popularity.npy"))
if os.path.exists(item_popularity_path):
    item_popularity = np.load(item_popularity_path).astype(np.float32)
else:
    # safe fallback: uniform popularity
    item_popularity = np.ones(item_ids.shape[0], dtype=np.float32)

# Precompute popularity order (descending)
popularity_idx_sorted = np.argsort(-item_popularity)

# --------------------
# Movies metadata
# --------------------
movies = pd.read_csv(MOVIES_CSV)  # columns: movieId,title,genres

def _strip_year(t: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", t).strip()

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _strip_year(s).lower())

movies["norm_title"] = movies["title"].apply(_norm)
norm2mid = dict(zip(movies["norm_title"], movies["movieId"]))

# Build a per-index genres mapping for (optional) genre caps
# movies.genres like "Action|Adventure|Sci-Fi" or "(no genres listed)"
mid2genres = {int(r.movieId): tuple([] if r.genres == "(no genres listed)" else r.genres.split("|"))
              for r in movies.itertuples(index=False)}
genres_by_idx = {}
for mid, idx in item_map.items():
    genres_by_idx[idx] = mid2genres.get(int(mid), ())

# --------------------
# FastAPI app & schema
# --------------------
app = FastAPI(title="Duku LightFM Recs")

class NewUserRequest(BaseModel):
    liked_ids: List[Union[int, str]] = Field(default_factory=list, description="MovieLens movieIds the user liked")
    topk: int = Field(10, ge=1, le=200)

# --------------------
# Helpers
# --------------------
def _find_movie_id(query: str) -> int | None:
    qn = _norm(query)
    if qn in norm2mid:
        return int(norm2mid[qn])
    hit = difflib.get_close_matches(qn, norm2mid.keys(), n=1, cutoff=FUZZY_THRESHOLD/100.0)
    return int(norm2mid[hit[0]]) if hit else None

def _idx_to_title(item_idx: int) -> str:
    mid = int(item_ids[item_idx])
    row = movies[movies["movieId"] == mid]
    return row["title"].values[0] if not row.empty else str(mid)

def _resolve_movie_ids(liked_ids: List[Union[int, str]]) -> List[int]:
    resolved: set[int] = set()
    for raw in liked_ids:
        try:
            mid = int(raw)
        except (TypeError, ValueError):
            continue
        if mid in item_map:
            resolved.add(mid)
    return list(resolved)

def _pseudo_user_from_likes(liked_idx: List[int]) -> np.ndarray | None:
    if not liked_idx:
        return None
    vec = item_embeddings[liked_idx].mean(axis=0)
    n = np.linalg.norm(vec)
    if not np.isfinite(n) or n == 0.0:
        return None
    return vec / n

def _like_neighbors_pool(liked_idx: List[int], per_like: int) -> np.ndarray:
    """
    For each liked item index, take top-N cosine neighbors from unit embeddings.
    Returns unique pooled candidate indices.
    """
    if not liked_idx or per_like <= 0:
        return np.array([], dtype=int)
    neigh = []
    for i in liked_idx:
        sims = item_unit_embeddings @ item_unit_embeddings[i]
        sims[i] = -np.inf  # drop self
        k = min(per_like, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        neigh.append(top)
    return np.unique(np.concatenate(neigh)) if neigh else np.array([], dtype=int)

def _mmr_rerank(candidate_idx: np.ndarray,
                scores_map: dict[int, float],
                k: int,
                lambda_: float,
                per_genre_cap: int | None) -> List[int]:
    """
    MMR with cosine similarity between items (via unit embeddings).
    """
    picked: List[int] = []
    caps: dict[str, int] = {}

    # Pre-extract unit vectors for candidates
    C = candidate_idx
    V = item_unit_embeddings[C]  # [Nc, d]
    # Build index lookup: item idx -> row in V
    row_of = {int(i): r for r, i in enumerate(C)}

    for _ in range(min(k, len(C))):
        best = None
        best_val = -1e18

        for idx in C:
            if idx in picked:
                continue
            if per_genre_cap and per_genre_cap > 0:
                gset = genres_by_idx.get(int(idx), ())
                if any(caps.get(g, 0) >= per_genre_cap for g in gset):
                    continue

            rel = scores_map[int(idx)]
            if not picked:
                mmr = rel
            else:
                # diversity term = max cosine similarity to already picked
                r_idx = row_of[int(idx)]
                v = V[r_idx]
                max_sim = max(float(v @ V[row_of[int(p)]]) for p in picked)
                mmr = lambda_ * rel - (1.0 - lambda_) * max_sim

            if mmr > best_val:
                best, best_val = int(idx), mmr

        if best is None:
            break
        picked.append(best)
        if per_genre_cap and per_genre_cap > 0:
            for g in genres_by_idx.get(int(best), ()):
                caps[g] = caps.get(g, 0) + 1

    return picked

def _top_popular(topk: int, exclude_idx_set: set[int]) -> List[int]:
    res = []
    for idx in popularity_idx_sorted:
        if idx not in exclude_idx_set:
            res.append(int(idx))
            if len(res) >= topk:
                break
    return res

# --------------------
# Routes
# --------------------
@app.get("/recommend_user", response_model=List[str])
def recommend_user(user_id: int, topk: int = 10):
    if user_id not in user_map:
        raise HTTPException(404, f"user_id {user_id} not found")
    uidx = user_map[user_id]

    # Predict all items for this known user
    items = np.arange(len(item_ids), dtype=np.int32)
    users = np.full(len(item_ids), int(uidx), dtype=np.int32)
    scores = model.predict(users, items)

    # Mask seen items in training for this user (optional if you want unseen-only)
    # NOTE: if you saved train matrix, you could mask. Skipping here.

    top = np.argpartition(-scores, min(topk, scores.shape[0]) - 1)[:topk]
    top = top[np.argsort(-scores[top])]
    return [_idx_to_title(int(i)) for i in top]

@app.get("/recommend_by_movie", response_model=List[str])
def recommend_by_movie(query: str, topk: int = 10):
    mid = _find_movie_id(query)
    if mid is None or mid not in item_map:
        raise HTTPException(404, f"No movie found or not in model: {query}")
    i = item_map[mid]

    if SIM_METRIC == "dot":
        sims = item_embeddings @ item_embeddings[i]
    else:
        sims = item_unit_embeddings @ item_unit_embeddings[i]

    # drop self; take topk
    sims[i] = -np.inf
    idx = np.argpartition(-sims, min(topk, sims.shape[0]) - 1)[:topk]
    idx = idx[np.argsort(-sims[idx])]
    return [_idx_to_title(int(j)) for j in idx]

@app.post("/recommend_new_user", response_model=List[str])
def recommend_new_user(req: NewUserRequest):
    """
    New visitor clicked Like on some movies (MovieLens IDs).
    Behavior:
      - 0 likes → popularity shelf
      - 1–2 likes → popularity excluding liked (light personalization)
      - ≥3 likes → pseudo-user from likes → score candidate pool → MMR + genre caps
    """
    mids = _resolve_movie_ids(req.liked_ids)
    liked_idx = [item_map[m] for m in mids]
    liked_set = set(liked_idx)

    # 0 likes → popularity
    if not liked_idx:
        top = _top_popular(req.topk, exclude_idx_set=set())
        return [_idx_to_title(i) for i in top]

    # 1–2 likes → popularity minus liked
    if len(liked_idx) < MIN_LIKES_FOR_EMBED:
        top = _top_popular(req.topk, exclude_idx_set=liked_set)
        return [_idx_to_title(i) for i in top]

    # ≥3 likes → build pseudo-user
    uvec = _pseudo_user_from_likes(liked_idx)
    if uvec is None:
        # Fallback to popularity
        top = _top_popular(req.topk, exclude_idx_set=liked_set)
        return [_idx_to_title(i) for i in top]

    # --- Candidate pool = popularity + neighbors of likes ---
    pop_part = [int(i) for i in popularity_idx_sorted if i not in liked_set][:CANDIDATE_POOL_POP]
    sim_part = _like_neighbors_pool(liked_idx, CANDIDATE_POOL_SIM_PER_LIKE)
    candidates = np.unique(np.concatenate([np.array(pop_part, dtype=int), sim_part]))
    # Exclude any liked items that might have slipped in from neighbors
    if len(liked_set) > 0 and candidates.size > 0:
        candidates = np.array([int(i) for i in candidates if int(i) not in liked_set], dtype=int)

    if candidates.size == 0:
        top = _top_popular(req.topk, exclude_idx_set=liked_set)
        return [_idx_to_title(i) for i in top]

    pool_n = max(CANDIDATE_POOL, req.topk)
    C = candidates[:pool_n].astype(np.int32)

    # --- Score with reduced bias influence for cold-start ---
    if SIM_METRIC == "dot":
        s = item_embeddings[C] @ uvec
    else:
        uhat = uvec / (np.linalg.norm(uvec) + 1e-9)
        s = item_unit_embeddings[C] @ uhat

    # bias correlates with popularity; down-weight so likes affect the list
    s = s + BETA_BIAS * item_biases[C]

    # Guard against non-finite scores
    s = np.where(np.isfinite(s), s, -np.inf)

    # Build a scores map for MMR
    scores_map = {int(i): float(si) for i, si in zip(C, s)}

    # MMR rerank with genre caps
    picked = _mmr_rerank(candidate_idx=C,
                         scores_map=scores_map,
                         k=req.topk,
                         lambda_=MMR_LAMBDA,
                         per_genre_cap=PER_GENRE_CAP)

    # Fill if short
    if len(picked) < req.topk:
        extra = [i for i in C if i not in picked][:req.topk - len(picked)]
        picked.extend(int(x) for x in extra)

    print({
            "liked_ids_in": req.liked_ids,
            "resolved_mids": mids,
            "liked_idx": liked_idx,
            "path": (
                "popularity_no_likes" if not liked_idx else
                "popularity_lt_threshold" if len(liked_idx) < MIN_LIKES_FOR_EMBED else
                "pseudo_user"
            ),
            "pool_counts": {
                "pop": CANDIDATE_POOL_POP,
                "sim_per_like": CANDIDATE_POOL_SIM_PER_LIKE,
                "candidates_post_exclude": int(candidates.size),
                "final_pool": int(C.size)
            }
       })

    return [_idx_to_title(i) for i in picked[:req.topk]]

@app.get("/health")
def health():
    return {"status": "ok"}