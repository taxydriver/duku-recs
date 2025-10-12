from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from .shared import META, ITEM_IDS, VECS

app = FastAPI(title="Duku Semantic Search")

# allow calls from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModel.from_pretrained(MODEL_DIR)
_model.eval()

def encode_query(text: str) -> np.ndarray:
    """Encode a query into an L2-normalized embedding vector."""
    with torch.no_grad():
        toks = _tokenizer([text], truncation=True, padding=True, return_tensors="pt")
        out = _model(**toks).last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1)
        emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        v = emb[0].cpu().numpy().astype("float32")
    v /= (np.linalg.norm(v) + 1e-8)
    return v

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/api/search")
def search(q: str = Query(..., description="search query"), k: int = 20):
    """Return top-k most similar movies for query q."""
    q_vec = encode_query(q)
    sims = VECS @ q_vec
    topk = np.argpartition(sims, -k)[-k:]
    order = np.argsort(sims[topk])[::-1]
    ranked = topk[order]

    results = []
    for idx in ranked:
        mid = int(ITEM_IDS[idx])
        m = META.get(str(mid), {})
        results.append({
            "id": mid,
            "title": m.get("title"),
            "year": m.get("year"),
            "posterPath": m.get("poster") or m.get("posterUrl"),
            "language": m.get("lang"),
            "genres": m.get("genres", []),
            "overview": m.get("overview"),
            "score": float(sims[idx]),
        })
    return {"query": q, "results": results}