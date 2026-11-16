"""Clustering, model selection and validation.

Besides the brief's K-Means / Hierarchical / Elbow / Silhouette, this module
adds three checks that answer "are these segments real?":
  * gap statistic (Tibshirani, Walther & Hastie, 2001) - can return k = 1
  * permutation null - shuffle each column independently and re-cluster
  * bootstrap cluster-wise stability (Hennig, 2007) - mean Jaccard per cluster
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

from .config import K_RANGE, SEED


def fit_kmeans(X: np.ndarray, k: int, seed: int = SEED) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)


def kmeans_scan(X: np.ndarray, k_range=K_RANGE, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = fit_kmeans(X, k, seed)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_),
            "davies_bouldin": davies_bouldin_score(X, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
        })
    return pd.DataFrame(rows)


def ward_linkage(X: np.ndarray):
    return linkage(X, method="ward")


def hierarchical_labels(Z, k: int) -> np.ndarray:
    return fcluster(Z, t=k, criterion="maxclust") - 1


def hierarchical_scan(X: np.ndarray, Z, k_range=K_RANGE) -> pd.DataFrame:
    rows = []
    for k in k_range:
        lab = hierarchical_labels(Z, k)
        rows.append({"k": k, "silhouette": silhouette_score(X, lab)})
    return pd.DataFrame(rows)


def gap_statistic(X: np.ndarray, k_max: int = 10, B: int = 20, seed: int = SEED) -> pd.DataFrame:
    """Gap(k) = E*[log W_k] - log W_k with a uniform reference over the data's
    bounding box. Selected k = smallest k with Gap(k) >= Gap(k+1) - s_{k+1}."""
    rng = np.random.default_rng(seed)
    lo, hi = X.min(axis=0), X.max(axis=0)

    def log_wk(data, k):
        if k == 1:
            return np.log(((data - data.mean(axis=0)) ** 2).sum())
        return np.log(KMeans(n_clusters=k, n_init=5, random_state=seed).fit(data).inertia_)

    rows = []
    for k in range(1, k_max + 1):
        obs = log_wk(X, k)
        ref = np.array([log_wk(rng.uniform(lo, hi, size=X.shape), k) for _ in range(B)])
        rows.append({"k": k, "log_wk": obs, "gap": ref.mean() - obs,
                     "s_k": ref.std(ddof=0) * np.sqrt(1 + 1 / B)})
    out = pd.DataFrame(rows)
    sel = k_max
    for i in range(len(out) - 1):
        if out.gap[i] >= out.gap[i + 1] - out.s_k[i + 1]:
            sel = int(out.k[i])
            break
    out.attrs["selected_k"] = sel
    return out


def permutation_null(X: np.ndarray, k: int, n_perm: int = 100, seed: int = SEED) -> dict:
    """Shuffle each column independently (keeps every variable's distribution,
    destroys relationships between variables), re-cluster, record silhouette."""
    rng = np.random.default_rng(seed)
    observed = silhouette_score(X, fit_kmeans(X, k, seed).labels_)
    null = []
    for _ in range(n_perm):
        Xp = np.column_stack([rng.permutation(col) for col in X.T])
        lab = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(Xp)
        null.append(silhouette_score(Xp, lab))
    null = np.array(null)
    p = (1 + (null >= observed).sum()) / (1 + n_perm)
    return {"k": k, "observed": float(observed), "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "null_p95": float(np.percentile(null, 95)),
            "p_value": float(p), "null_values": null.tolist()}


def bootstrap_stability(X: np.ndarray, labels: np.ndarray, k: int, B: int = 100,
                        seed: int = SEED) -> pd.DataFrame:
    """Hennig (2007) cluster-wise stability. For each bootstrap sample, re-cluster
    and record the best Jaccard match of each original cluster.
    >= 0.75 stable, 0.6-0.75 pattern but doubtful membership, < 0.6 unstable."""
    rng = np.random.default_rng(seed)
    n = len(X)
    jacc = np.zeros((B, k))
    for b in range(B):
        idx = np.unique(rng.integers(0, n, n))
        lab_b = KMeans(n_clusters=k, n_init=5, random_state=seed + b).fit_predict(X[idx])
        for c in range(k):
            orig = set(idx[labels[idx] == c])
            best = 0.0
            for cb in range(k):
                new = set(idx[lab_b == cb])
                union = len(orig | new)
                if union:
                    best = max(best, len(orig & new) / union)
            jacc[b, c] = best
    out = pd.DataFrame({
        "cluster": range(k),
        "mean_jaccard": jacc.mean(axis=0),
        "dissolved_pct": (jacc <= 0.5).mean(axis=0) * 100,
    })
    out["verdict"] = pd.cut(out["mean_jaccard"], [0, 0.6, 0.75, 1.01],
                            labels=["Unstable", "Pattern (doubtful)", "Stable"])
    return out


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(adjusted_rand_score(a, b))
