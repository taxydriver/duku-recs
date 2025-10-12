import faiss
import numpy as np
import os

# Path setup
art_dir = "artifacts/semantics"
faiss_path = os.path.join(art_dir, "movies.faiss")
npy_path   = os.path.join(art_dir, "movie_vecs.npy")

# Load FAISS index
index = faiss.read_index(faiss_path)
print(f"[INFO] Loaded FAISS index with {index.ntotal} vectors")

# Extract vectors
if hasattr(index, "reconstruct_n"):        # safest way
    vecs = index.reconstruct_n(0, index.ntotal)
else:
    # fallback: some simple flat indexes expose .xb directly
    vecs = faiss.vector_to_array(index.xb).reshape(index.ntotal, -1)

# Save to .npy
np.save(npy_path, vecs.astype("float32"))
print(f"[INFO] Saved {vecs.shape} matrix to {npy_path}")