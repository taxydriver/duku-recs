from fastapi import APIRouter, Query
import os, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from .shared import META, ITEM_IDS, VECS, IDX

router = APIRouter()

MODEL_DIR = os.getenv("EMBED_MODEL", "/app/minilm")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModel.from_pretrained(MODEL_DIR)
_model.eval()

def encode_query(text: str) -> np.ndarray:
    with torch.no_grad():
        toks = _tokenizer([text], truncation=True, padding=True, return_tensors="pt")
        out = _model(**toks).last_hidden_state  # [B,T,H]
        mask = toks["attention_mask"].unsqueeze(-1)  # [B,T,1]
        emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)  # mean-pool
        v = emb[0].cpu().numpy().astype("float32")
    v /= (np.linalg.norm(v) + 1e-8)
    return v

@router.get("/api/search")
def search(q: str, k: int = 20):
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
            "posterPath": m.get("poster"),
            "language": m.get("lang"),
            "genres": m.get("genres", []),
            "overview": m.get("overview"),
            "score": float(sims[idx]),
        })
    return {"query": q, "results": results}