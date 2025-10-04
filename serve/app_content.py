# serve/app_content.py
from fastapi import FastAPI, Query
import os, json, numpy as np, re
from collections import defaultdict
from .shared import META, ITEM_IDS, VECS, IDX
from .app_search import router as search_router


LANG_BOOST = float(os.getenv("LANG_BOOST", "0.08"))
GENRE_BOOST = float(os.getenv("GENRE_BOOST", "0.05"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.75"))
TOPK       = int(os.getenv("TOPK_DEFAULT", "40"))

app = FastAPI(title="Content Recs")
app.include_router(search_router)


# ---------- THEN build title index ----------
def _norm_title(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s

TITLE_TO_IDS = defaultdict(list)
for mid_str, m in META.items():
    t = m.get("title") or ""
    nt = _norm_title(t)
    if nt:
        TITLE_TO_IDS[nt].append(int(mid_str))

def _pick_best_id(candidates):
    if not candidates:
        return None
    best, best_year = None, -1
    for mid in candidates:
        y = (META.get(str(mid), {}) or {}).get("year") or -1
        if isinstance(y, int) and y > best_year:
            best, best_year = mid, y
    return best or candidates[0]

def _parse_liked_tokens(raw: str):
    liked_ids = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            liked_ids.append(int(tok))
        else:
            mid = _pick_best_id(TITLE_TO_IDS.get(_norm_title(tok), []))
            if mid is not None:
                liked_ids.append(mid)
    # de-dup preserve order
    seen, uniq = set(), []
    for mid in liked_ids:
        if mid not in seen:
            uniq.append(mid); seen.add(mid)
    return uniq

# ---------- rest of your code (user_vec, recommend, etc.) ----------
def user_vec(liked_ids):
    mats = []
    for mid in liked_ids:
        ix = IDX.get(mid)
        if ix is not None:
            mats.append(VECS[ix])
    if not mats:
        return None
    u = np.mean(mats, axis=0)
    n = np.linalg.norm(u) + 1e-8
    return (u / n).astype("float32")

@app.get("/health")
def health():
    return {"items": int(ITEM_IDS.shape[0]), "dims": int(VECS.shape[1])}

@app.get("/api/recommend")
def recommend(liked: str, k: int = TOPK):
    liked_ids = _parse_liked_tokens(liked)
    u = user_vec(liked_ids)
    if u is None:
        return {"results": []}

    base = VECS @ u  # cosine because vectors are L2-normalised

    # boosts
    liked_langs = []
    liked_genres = set()
    for mid in liked_ids:
        m = META.get(str(mid))
        if not m: continue
        if m.get("lang"): liked_langs.append(m["lang"])
        for g in m.get("genres", []): liked_genres.add(g)

    scores = base.copy()
    for i, mid in enumerate(ITEM_IDS):
        m = META.get(str(mid))
        if not m: continue
        if m.get("lang") and m["lang"] in liked_langs:
            scores[i] += LANG_BOOST
        g = set(m.get("genres", []))
        if g:
            inter = len(g & liked_genres)
            union = len(g | liked_genres) or 1
            scores[i] += GENRE_BOOST * (inter/union)

    # exclude liked
    mask = np.ones_like(scores, dtype=bool)
    for mid in liked_ids:
        ix = IDX.get(mid)
        if ix is not None: mask[ix] = False

    pool = max(200, k*5)
    cand_local = np.argpartition(scores[mask], -pool)[-pool:]
    full_idx = np.where(mask)[0][cand_local]
    order = np.argsort(scores[full_idx])[::-1]
    ranked = full_idx[order]

    # simple MMR
    lam = MMR_LAMBDA
    picked, picked_vecs = [], []
    for idx in ranked:
        v = VECS[idx]
        div_pen = (1-lam)*(np.max(np.dot(np.vstack(picked_vecs), v)) if picked_vecs else 0.0)
        mmr = lam*scores[idx] - div_pen
        picked.append((mmr, idx))
        picked = sorted(picked, key=lambda x: x[0], reverse=True)[:k]
        picked_vecs = [VECS[i] for _, i in picked]

    final = [int(ITEM_IDS[i]) for _, i in sorted(picked, key=lambda x: x[0], reverse=True)]
    out=[]
    for mid in final:
        m = META.get(str(mid))
        if not m: continue
        out.append({
            "id": mid,
            "title": m.get("title"),
            "year": m.get("year"),
            "posterPath": m.get("poster"),
            "language": m.get("lang"),
            "genres": m.get("genres", []),
            "overview": m.get("overview"),
        })
    return {"results": out}