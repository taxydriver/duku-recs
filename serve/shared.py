import os, json, numpy as np

ITEM_IDS_PATH = os.getenv("CONTENT_ITEM_IDS", "/app/artifacts/item_ids.npy")
VECS_PATH     = os.getenv("CONTENT_VECS",     "/app/artifacts/item_vecs_norm.npy")
META_PATH     = os.getenv("CONTENT_META",     "/app/artifacts/meta.json")

ITEM_IDS = np.load(ITEM_IDS_PATH)
VECS     = np.load(VECS_PATH).astype("float32")

with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

IDX = {int(i): ix for ix, i in enumerate(ITEM_IDS.tolist())}