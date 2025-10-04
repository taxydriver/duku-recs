#!/usr/bin/env python
# tools/embed_movies.py
import sys, os, json, numpy as np
from pathlib import Path
from tqdm import tqdm

# choose a model: multilingual recommended
MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def load_model():
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(MODEL_NAME, device=device)

def build_text(m):
    title = m.get("title","")
    lang = m.get("original_language") or m.get("lang","")
    genres = " ".join(m.get("genres", []))
    overview = m.get("overview","")
    # compact text for embedding
    return f"{title}\nLANG:{lang}\nGENRES:{genres}\n{overview}".strip()

def stream_ndjson(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield json.loads(line)

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/embed_movies.py <in_ndjson> <out_dir>", file=sys.stderr)
        sys.exit(2)

    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)

    # load items
    ids, texts = [], []
    meta = {}
    print(f"Reading: {in_path}")
    for m in stream_ndjson(in_path):
        ov = (m.get("overview") or "").strip()
        if not ov:
            continue
        mid = int(m["id"])
        ids.append(mid)
        texts.append(build_text(m))
        meta[str(mid)] = {
            "title": m.get("title"),
            "year": m.get("year"),
            "poster": m.get("poster_path") or m.get("poster"),
            "lang": m.get("original_language") or m.get("lang"),
            "genres": m.get("genres", []),
            "overview": ov,
        }

    if not ids:
        print("No items with non-empty overviews found.", file=sys.stderr)
        sys.exit(1)

    print(f"Encoding {len(ids)} items with {MODEL_NAME} ...")
    model = load_model()

    # batch encode
    B = int(os.environ.get("EMBED_BATCH", "256"))
    vecs = []
    for i in tqdm(range(0, len(texts), B)):
        batch = texts[i:i+B]
        emb = model.encode(batch, batch_size=B, show_progress_bar=False, normalize_embeddings=False)
        vecs.append(emb.astype("float32"))
    X = np.vstack(vecs)

    # L2-normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X = X / norms

    # save artifacts
    ids_arr = np.array(ids, dtype=np.int64)
    np.save(out_dir / "item_ids.npy", ids_arr)
    np.save(out_dir / "item_vecs_norm.npy", X)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print("Wrote:", out_dir / "item_ids.npy", out_dir / "item_vecs_norm.npy", out_dir / "meta.json")

if __name__ == "__main__":
    main()