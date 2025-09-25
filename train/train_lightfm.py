import os, time, json, joblib, numpy as np, pandas as pd
from scipy import sparse
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

RATINGS = os.getenv("RATINGS_CSV", "data/ml-25m/ratings.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "artifacts/lightfm")
EPOCHS = int(os.getenv("EPOCHS", "20"))
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))
MIN_RATING_POS = float(os.getenv("MIN_RATING_POS", "3.0"))  # implicit: rating>=3 => positive
ITEM_EMBED_FILE = os.getenv("ITEM_EMBED_FILE", "item_embeddings.npy")
ITEM_UNIT_EMBED_FILE = os.getenv("ITEM_UNIT_EMBED_FILE", "item_embeddings_unit.npy")
ITEM_BIAS_FILE = os.getenv("ITEM_BIAS_FILE", "item_biases.npy")
ITEM_POP_FILE = os.getenv("ITEM_POP_FILE", "item_popularity.npy")
VAL_PERCENT = float(os.getenv("VAL_PERCENT", "0.2"))  # 20% interactions hidden for validation
TOPK_METRIC = int(os.getenv("TOPK_METRIC", "20"))


def safe_get_item_representations(model):
    """
    Get item embeddings and biases with sanity checks.
    Guarantees:
      - embeddings: 2D (n_items, k)
      - biases: 1D (n_items,)
    """
    emb, bias = model.get_item_representations()
    emb = np.asarray(emb, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)

    # If swapped by mistake, fix automatically
    if emb.ndim == 1 and bias.ndim == 2:
        print("[warn] Swapped unpack detected, correcting order...")
        emb, bias = bias, emb

    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")
    if bias.ndim != 1:
        raise ValueError(f"Expected 1D biases, got shape {bias.shape}")

    return emb, bias


os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading ratings from {RATINGS} ...")
df = pd.read_csv(RATINGS, usecols=["userId", "movieId", "rating"])

# Implicit feedback
df = df[df["rating"] >= MIN_RATING_POS].copy()
df["interaction"] = 1.0

# Build contiguous id maps
unique_users = df["userId"].unique()
unique_items = df["movieId"].unique()
user_map = {u: i for i, u in enumerate(unique_users)}
item_map = {m: i for i, m in enumerate(unique_items)}

df["u"] = df["userId"].map(user_map)
df["i"] = df["movieId"].map(item_map)

n_users = len(user_map)
n_items = len(item_map)
print(f"Users: {n_users} | Items: {n_items} | Interactions: {len(df)}")

# Sparse interactions
X = sparse.coo_matrix(
    (df["interaction"].astype(np.float32), (df["u"].values, df["i"].values)),
    shape=(n_users, n_items),
).tocsr()

# Train/validation split (hide a percentage of interactions uniformly at random)
# Train/validation split
train, test = random_train_test_split(X, test_percentage=VAL_PERCENT, random_state=42)

# Convert to CSR for fast row access
train = train.tocsr()
test = test.tocsr()

# Train LightFM
model = LightFM(loss="warp")
start = time.time()
model.fit(train, epochs=EPOCHS, num_threads=NUM_THREADS)
train_time_s = time.time() - start

# Extract representations
item_embeddings, item_biases = safe_get_item_representations(model)


item_embeddings = np.asarray(item_embeddings, dtype=np.float32)
item_biases = np.asarray(item_biases, dtype=np.float32)

# Sanity checks (fail fast if wrong shapes)
if item_embeddings.ndim != 2:
    raise ValueError(f"Expected 2D item_embeddings, got shape {item_embeddings.shape}")
if item_biases.ndim != 1:
    raise ValueError(f"Expected 1D item_biases, got shape {item_biases.shape}")

# Normalize embeddings (for cosine similarity)
with np.errstate(divide="ignore", invalid="ignore"):
    norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    item_unit_embeddings = item_embeddings / norms

# Popularity (count of positive interactions per item in training order)
pop_counts = df.groupby("movieId")["interaction"].sum()
item_popularity = np.array([pop_counts.get(mid, 0.0) for mid in unique_items], dtype=np.float32)

# ---------- Metrics ----------
print("Computing validation metrics ...")
# LightFM metrics (computed per-user; we take means)
prec = precision_at_k(model, test_interactions=test, train_interactions=train,
                      k=TOPK_METRIC, num_threads=NUM_THREADS).mean()
rec = recall_at_k(model, test_interactions=test, train_interactions=train,
                  k=TOPK_METRIC, num_threads=NUM_THREADS).mean()
auc = auc_score(model, test_interactions=test, train_interactions=train,
                num_threads=NUM_THREADS).mean()

# Coverage@K and Diversity@K (sampled for speed if needed)
def _topk_indices_for_user(uidx: int, k: int) -> np.ndarray:
    # score all items for user uidx
    items = np.arange(n_items, dtype=np.int32)
    users = np.full(n_items, int(uidx), dtype=np.int32)
    scores = model.predict(users, items, num_threads=NUM_THREADS)

    # mask already-seen items in train
    seen = train.getrow(int(uidx)).indices  # ✅ works with CSR
    if seen.size:
        scores[seen] = -np.inf

    k = min(k, n_items)
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return top

# Optionally subsample users for speed on large data
USER_SAMPLE = min(n_users, int(os.getenv("USER_SAMPLE_FOR_METRICS", "100")))
rng = np.random.default_rng(123)
sample_users = rng.choice(n_users, size=USER_SAMPLE, replace=False) if n_users > USER_SAMPLE else np.arange(n_users)

topk_lists = [ _topk_indices_for_user(u, TOPK_METRIC) for u in sample_users ]
flat_items = np.concatenate(topk_lists) if topk_lists else np.array([], dtype=int)
unique_rec_items = np.unique(flat_items)
coverage_at_k = float(len(unique_rec_items)) / float(n_items) if n_items > 0 else 0.0

# Diversity: 1 - average cosine similarity among items in each user's top-K
def _avg_intralist_similarity(idx_list: np.ndarray) -> float:
    if len(idx_list) < 2:
        return 0.0
    # use unit embeddings to get cosine quickly
    V = item_unit_embeddings[idx_list]  # [K, d]
    sims = V @ V.T                      # [K, K]
    # take upper triangle (excluding diag)
    K = sims.shape[0]
    triu = sims[np.triu_indices(K, k=1)]
    return float(triu.mean())

diversities = [1.0 - _avg_intralist_similarity(idx) for idx in topk_lists] if topk_lists else [0.0]
diversity_at_k = float(np.mean(diversities))

metrics = {
    "dataset": {
        "users": int(n_users),
        "items": int(n_items),
        "interactions": int(X.nnz),
        "val_percent": VAL_PERCENT,
        "min_rating_pos": MIN_RATING_POS
    },
    "training": {
        "epochs": EPOCHS,
        "num_threads": NUM_THREADS,
        "loss": "warp",
        "train_time_sec": round(train_time_s, 3)
    },
    "model": {
        "embedding_dim": int(item_embeddings.shape[1]) if item_embeddings.ndim == 2 else None
    },
    "metrics@{}".format(TOPK_METRIC): {
        "precision": float(prec),
        "recall": float(rec),
        "auc": float(auc),
        "coverage": float(coverage_at_k),
        "diversity": float(diversity_at_k)
    }
}

# Persist model + mappings + arrays
joblib.dump(model, os.path.join(MODEL_DIR, "lightfm_model.pkl"))
np.save(os.path.join(MODEL_DIR, "user_ids.npy"), unique_users)
np.save(os.path.join(MODEL_DIR, "item_ids.npy"), unique_items)
joblib.dump(user_map, os.path.join(MODEL_DIR, "user_map.joblib"))
joblib.dump(item_map, os.path.join(MODEL_DIR, "item_map.joblib"))
np.save(os.path.join(MODEL_DIR, ITEM_EMBED_FILE), item_embeddings)
np.save(os.path.join(MODEL_DIR, ITEM_UNIT_EMBED_FILE), item_unit_embeddings)
np.save(os.path.join(MODEL_DIR, ITEM_BIAS_FILE), item_biases)
np.save(os.path.join(MODEL_DIR, ITEM_POP_FILE), item_popularity)

# Save metrics.json
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved artifacts to {MODEL_DIR}")
print("Metrics:", json.dumps(metrics, indent=2))


