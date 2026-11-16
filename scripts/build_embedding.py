"""Build the 3D coordinates for the Buyer Galaxy view.

Reads the frozen segmentation model and writes results/embedding_3d_v1.csv.
It does not refit or change the model or any published result.
Run:  python scripts/build_embedding.py
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C, data, features  # noqa: E402


def main():
    bundle = joblib.load(C.RESULTS / "segmentation_model.joblib")
    clients, _ = data.clean_clients(data.load_raw()[0])
    props, _ = data.clean_properties(data.load_raw()[1])
    df = features.build_client_features(clients, props)
    X, cols, _ = features.model_matrix(df, C.NUMERIC_BRIEF + C.NUMERIC_BEHAVIOUR,
                                       C.BINARY_FEATURES, scaler=bundle["scaler"])
    pca = PCA(n_components=3, random_state=C.SEED).fit(X)
    pcs = pca.transform(X)
    seg = pd.Series(bundle["kmeans"].predict(X)).map(bundle["names"])
    out = pd.DataFrame({"client_id": df["client_id"], "segment": seg,
                        "pc1": pcs[:, 0], "pc2": pcs[:, 1], "pc3": pcs[:, 2]})
    out.to_csv(C.RESULTS / "embedding_3d_v1.csv", index=False)
    load = pd.DataFrame(pca.components_.T, index=cols, columns=["pc1", "pc2", "pc3"])
    top = {c: load[c].abs().sort_values(ascending=False).head(3).index.tolist() for c in load}
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_sha256": hashlib.sha256((C.RESULTS / "segmentation_model.joblib").read_bytes()).hexdigest(),
        "method": "PCA (3 components) on the frozen model's scaled feature matrix, seed 42",
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "top_loadings": top,
        "records": int(len(out)),
    }
    (C.RESULTS / "embedding_3d_v1.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
