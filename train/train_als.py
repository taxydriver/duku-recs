from __future__ import annotations
import argparse, os, yaml, numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from joblib import dump

def load_cfg(path:str)->dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_matrix(df: pd.DataFrame):
    uids = df["user_id"].astype("int64").to_numpy()
    iids = df["item_id"].astype("int64").to_numpy()
    r = df["rating"].astype("float32").to_numpy()

    unique_users = np.unique(uids)
    unique_items = np.unique(iids)

    u2idx = {u:i for i,u in enumerate(unique_users)}
    i2idx = {i:j for j,i in enumerate(unique_items)}

    rows = np.array([u2idx[u] for u in uids], dtype=np.int32)
    cols = np.array([i2idx[i] for i in iids], dtype=np.int32)

    mat = coo_matrix((r, (rows, cols)), shape=(len(unique_users), len(unique_items))).tocsr()
    return mat, unique_users, unique_items

def main(cfgpath:str):
    cfg = load_cfg(cfgpath)
    df = pd.read_csv(cfg["data"]["ratings"])

# rename columns to what the script expects
    df = df.rename(columns={"userId": "user_id", "movieId": "item_id"})

    mu = cfg["data"]["min_user_interactions"]
    mi = cfg["data"]["min_item_interactions"]
    if mu > 1:
        counts = df.groupby("user_id")["item_id"].count()
        keep = counts[counts >= mu].index
        df = df[df["user_id"].isin(keep)]
    if mi > 1:
        counts = df.groupby("item_id")["user_id"].count()
        keep = counts[counts >= mi].index
        df = df[df["item_id"].isin(keep)]

    X, user_ids, item_ids = build_matrix(df)

    model_cfg = cfg["model"]
    als = AlternatingLeastSquares(
        factors=model_cfg["factors"],
        regularization=model_cfg["regularization"],
        iterations=model_cfg["iterations"],
        use_cg=model_cfg.get("use_cg", True),
        random_state=42,
    )
    als.fit(X.T, show_progress=True)

    user_factors = als.user_factors.astype("float32")
    item_factors = als.item_factors.astype("float32")

    out = cfg["output"]
    dir = out["artifacts_dir"]
    os.makedirs(dir, exist_ok=True)
    np.save(os.path.join(dir, out["user_factors"]), user_factors)
    np.save(os.path.join(dir, out["item_factors"]), item_factors)
    np.save(os.path.join(dir, out["user_ids"]), user_ids)
    np.save(os.path.join(dir, out["item_ids"]), item_ids)

    dump({int(u):i for i,u in enumerate(user_ids)}, os.path.join(dir, "user_index.joblib"))
    dump({int(it):j for j,it in enumerate(item_ids)}, os.path.join(dir, "item_index.joblib"))
    print("Saved artifacts to", dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
