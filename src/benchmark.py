"""Measurement skeleton for the AI-shorts child-feed audit (RQ1-RQ3).

Runs on the micro-sample in data/sample/ and writes results/result.json.
Replace the data paths in configs/config.yaml for the full collection (Week 7).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def rq1(cfg, alpha):
    """One-sided two-proportion z-test: p_child > p_adult."""
    c = cfg["rq1"]
    df = pd.read_csv(c["path"])
    g = df.groupby("profile_type")[["n_ai_generated", "n_videos"]].sum()
    n_sessions = df.groupby("profile_type").size()
    assert n_sessions.min() >= c["min_sessions_per_group"], "Guardrail: too few sessions"
    x1, n1 = g.loc["child"]
    x2, n2 = g.loc["adult"]
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    pval = float(stats.norm.sf(z))
    return {"p_child": float(p1), "p_adult": float(p2), "z": float(z),
            "p_value": pval, "reject_H0": bool(pval < alpha)}


def rq2(cfg, alpha):
    """One-sided Welch t-test on change in AI share (baseline -> session 5)."""
    c = cfg["rq2"]
    df = pd.read_csv(c["path"])
    df["delta"] = df["ai_share_session5"] - df["ai_share_baseline"]
    t = df[df.arm == c["treatment_arm"]]["delta"]
    k = df[df.arm == c["control_arm"]]["delta"]
    assert min(len(t), len(k)) >= c["min_accounts_per_arm"], "Guardrail: too few accounts"
    res = stats.ttest_ind(t, k, equal_var=False, alternative="greater")
    return {"mean_delta_ai_dwell": float(t.mean()), "mean_delta_human_dwell": float(k.mean()),
            "t": float(res.statistic), "p_value": float(res.pvalue),
            "reject_H0": bool(res.pvalue < alpha)}


def rq3(cfg, seed):
    """Macro-F1: metadata-only baseline vs. multimodal; H1 needs gain >= min_improvement."""
    c = cfg["rq3"]
    df = pd.read_csv(c["path"])
    y = df[c["label_column"]]
    sets = {
        "metadata_only": c["metadata_features"],
        "multimodal": c["metadata_features"] + c["visual_features"] + c["audio_features"],
    }
    idx_tr, idx_te = train_test_split(df.index, test_size=c["test_size"],
                                      stratify=y, random_state=seed)
    out = {}
    for name, cols in sets.items():
        mu, sd = df.loc[idx_tr, cols].mean(), df.loc[idx_tr, cols].std(ddof=0) + 1e-9
        Xtr = (df.loc[idx_tr, cols] - mu) / sd
        Xte = (df.loc[idx_te, cols] - mu) / sd
        clf = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr, y[idx_tr])
        out[name] = float(f1_score(y[idx_te], clf.predict(Xte), average="macro"))
    out["gain"] = out["multimodal"] - out["metadata_only"]
    out["reject_H0"] = bool(out["gain"] >= c["min_improvement"])
    return out


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    seed, alpha = cfg["experiment"]["seed"], cfg["experiment"]["alpha"]
    np.random.seed(seed)
    t0 = time.perf_counter()
    result = {"RQ1": rq1(cfg, alpha), "RQ2": rq2(cfg, alpha), "RQ3": rq3(cfg, seed)}
    result["runtime_sec"] = round(time.perf_counter() - t0, 4)
    assert result["runtime_sec"] < cfg["timeouts"]["run_seconds"], "Guardrail: timeout"
    out = cfg["output"]["results_path"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    main(ap.parse_args().config)
