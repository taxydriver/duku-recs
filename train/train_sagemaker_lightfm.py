# train/train_sagemaker_lightfm.py
import os, json, sys, logging
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM

log = logging.getLogger("train")
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_CHANNEL_TRAIN = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
HP_JSON = "/opt/ml/input/config/hyperparameters.json"

def read_hparams():
    hp = {"epochs": 20, "no_components": 64, "loss": "warp"}
    if os.path.exists(HP_JSON):
        try:
            raw = json.load(open(HP_JSON))
            hp.update({
                "epochs": int(raw.get("epochs", hp["epochs"])),
                "no_components": int(raw.get("no_components", hp["no_components"])),
                "loss": str(raw.get("loss", hp["loss"])),
            })
        except Exception as e:
            log.warning(f"Could not parse hyperparameters: {e}")
    else:
        hp["epochs"] = int(os.environ.get("EPOCHS", hp["epochs"]))
        hp["no_components"] = int(os.environ.get("NO_COMPONENTS", hp["no_components"]))
        hp["loss"] = os.environ.get("LOSS", hp["loss"])
    log.info(f"Hyperparameters: {hp}")
    return hp

def load_from_train_channel():
    npz = Path(SM_CHANNEL_TRAIN) / "interactions.npz"      # sparse matrix
    csv = Path(SM_CHANNEL_TRAIN) / "ratings.csv"           # columns: user_id,item_id[,rating]
    if npz.exists():
        log.info(f"Loading {npz}")
        X = sp.load_npz(npz).tocsr()
        return X, np.arange(X.shape[0]), np.arange(X.shape[1])
    if csv.exists():
        import pandas as pd
        log.info(f"Building matrix from {csv}")
        df = pd.read_csv(csv)
        assert {"user_id","item_id"}.issubset(df.columns), "ratings.csv must have user_id,item_id[,rating]"
        users = df["user_id"].astype("category")
        items = df["item_id"].astype("category")
        ui = users.cat.codes.to_numpy()
        ii = items.cat.codes.to_numpy()
        vals = df["rating"].to_numpy() if "rating" in df.columns else np.ones(len(df))
        X = sp.coo_matrix((vals, (ui, ii)),
                          shape=(users.cat.categories.size, items.cat.categories.size)).tocsr()
        return X, users.cat.categories.to_numpy(), items.cat.categories.to_numpy()
    return None, None, None

def load_data():
    X, uids, iids = load_from_train_channel()
    if X is not None:
        return X, uids, iids
    log.warning("No data in train channel; generating synthetic COO matrix for smoke test.")
    users, items = 500, 1000
    X = sp.random(users, items, density=0.01, format="coo", random_state=42).tocsr()
    return X, np.arange(users), np.arange(items)

def main():
    hp = read_hparams()
    X, user_ids, item_ids = load_data()
    log.info(f"Interactions: shape={X.shape}, nnz={X.nnz}")

    model = LightFM(no_components=hp["no_components"], loss=hp["loss"])
    model.fit(X, epochs=hp["epochs"], num_threads=4)
    log.info("Training complete.")

    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(SM_MODEL_DIR, "lightfm_model.pkl"))
    np.save(os.path.join(SM_MODEL_DIR, "user_ids.npy"), user_ids)
    np.save(os.path.join(SM_MODEL_DIR, "item_ids.npy"), item_ids)
    log.info(f"Saved artifacts to {SM_MODEL_DIR}")

if __name__ == "__main__":
    main()