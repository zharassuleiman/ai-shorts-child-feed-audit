"""Data processing and statistical aggregation (Task 3).

Reads the raw log  data/raw/test_run.jsonl, drops the cold-start warm-up sessions,
computes medians / percentiles / bootstrap 95% CIs for RQ1-RQ3, checks the guardrail
metrics and writes data/processed/metrics_summary.csv.

Run:  python src/process_data.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (cohen_kappa_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "test_run.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "metrics_summary.csv"

SEED = 42            # fixed seed for the split and for every bootstrap
N_BOOT = 10_000      # bootstrap resamples
TEST_SIZE = 0.4
MIN_IMPROVEMENT = 0.05          # RQ3: required macro-F1 gain
MIN_KAPPA = 0.70                # guardrail: annotator agreement
MAX_ERROR_RATE = 0.05           # guardrail: share of unavailable videos
MIN_UNITS_PER_GROUP = 3         # guardrail: sessions / accounts per group

FEATURE_SETS = {
    "metadata_only": ["meta_title_len", "meta_has_ai_tag"],
    "visual": ["visual_artifact_score"],
    "audio": ["audio_synthetic_score"],
    "multimodal": ["meta_title_len", "meta_has_ai_tag",
                   "visual_artifact_score", "audio_synthetic_score"],
}
ARMS = ["ai_dwell", "human_dwell", "no_dwell"]


# ----------------------------------------------------------------- 1. loading
def load_raw(path: Path = RAW_PATH):
    """Return (feeds, videos) DataFrames from the raw .jsonl file."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    feeds = df[df["record_type"] == "feed"].dropna(axis=1, how="all").reset_index(drop=True)
    videos = df[df["record_type"] == "video"].dropna(axis=1, how="all").reset_index(drop=True)
    for col in ["n_requested", "n_unavailable", "n_ai", "session_idx"]:
        feeds[col] = feeds[col].astype(int)
    for col in ["label_a", "label_b", "label_final", "meta_has_ai_tag", "meta_title_len"]:
        videos[col] = videos[col].astype(int)
    return feeds, videos


def add_share(feeds: pd.DataFrame) -> pd.DataFrame:
    """AI share among labelled videos and the error rate (unavailable videos)."""
    f = feeds.copy()
    f["n_labeled"] = f["n_requested"] - f["n_unavailable"]
    f["ai_share"] = f["n_ai"] / f["n_labeled"]
    f["error_rate"] = f["n_unavailable"] / f["n_requested"]
    return f


def drop_warmup(df: pd.DataFrame):
    """Discard the cold-start warm-up runs; return (evaluation rows, number dropped)."""
    keep = df[df["phase"] == "eval"].copy()
    return keep, len(df) - len(keep)


# ----------------------------------------------------------------- 2. statistics
def describe(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "n": len(x), "mean": x.mean(),
        "sd": x.std(ddof=1) if len(x) > 1 else np.nan,
        "median": np.median(x),
        "p25": np.percentile(x, 25), "p75": np.percentile(x, 75),
        "p95": np.percentile(x, 95), "p99": np.percentile(x, 99),
    }


def bootstrap_ci(x, stat=np.mean, b: int = N_BOOT, seed: int = SEED):
    """95% percentile bootstrap CI of a statistic (resampling the units)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    lo, hi = np.percentile(stat(x[idx], axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_diff_ci(a, c, stat=np.mean, b: int = N_BOOT, seed: int = SEED):
    """95% bootstrap CI of stat(a) - stat(c) for two independent groups."""
    a, c = np.asarray(a, dtype=float), np.asarray(c, dtype=float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(b, len(a)))
    ic = rng.integers(0, len(c), size=(b, len(c)))
    lo, hi = np.percentile(stat(a[ia], axis=1) - stat(c[ic], axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def rq1(sessions: pd.DataFrame) -> dict:
    """Child vs adult AI share in the baseline session. Unit of analysis = session."""
    base = sessions[sessions["session_idx"] == 0]
    child = base[base["profile_type"] == "child"]["ai_share"].to_numpy()
    adult = base[base["profile_type"] == "adult"]["ai_share"].to_numpy()
    t = stats.ttest_ind(child, adult, equal_var=False, alternative="greater")
    return {"child": child, "adult": adult,
            "diff": child.mean() - adult.mean(),
            "ci": bootstrap_diff_ci(child, adult),
            "t": float(t.statistic), "p": float(t.pvalue)}


def account_deltas(sessions: pd.DataFrame) -> pd.DataFrame:
    """Change in AI share (session 5 minus baseline) for each child account."""
    child = sessions[sessions["profile_type"] == "child"]
    wide = child.pivot(index=["account_id", "arm"], columns="session_idx", values="ai_share")
    wide = wide.reset_index()
    wide["delta"] = wide[5] - wide[0]
    return wide[["account_id", "arm", "delta"]]


def rq2(sessions: pd.DataFrame) -> dict:
    """AI-dwell vs human-dwell change in AI share. Unit of analysis = account."""
    d = account_deltas(sessions)
    groups = {a: d[d["arm"] == a]["delta"].to_numpy() for a in ARMS}
    t = stats.ttest_ind(groups["ai_dwell"], groups["human_dwell"],
                        equal_var=False, alternative="greater")
    return {"groups": groups,
            "diff": groups["ai_dwell"].mean() - groups["human_dwell"].mean(),
            "ci": bootstrap_diff_ci(groups["ai_dwell"], groups["human_dwell"]),
            "t": float(t.statistic), "p": float(t.pvalue)}


def trajectory(sessions: pd.DataFrame) -> pd.DataFrame:
    """Mean AI share per arm and session with a 95% bootstrap CI over accounts."""
    child = sessions[sessions["profile_type"] == "child"]
    rows = []
    for arm, g in child.groupby("arm"):
        for idx, gg in g.groupby("session_idx"):
            x = gg["ai_share"].to_numpy()
            lo, hi = bootstrap_ci(x)
            rows.append({"arm": arm, "session_idx": int(idx), "n": len(x),
                         "mean": x.mean(), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def _macro_f1_boot(y_true, y_pred, boot_idx):
    """Vectorised macro-F1 for every bootstrap resample (binary labels 0/1)."""
    yt, yp = y_true[boot_idx], y_pred[boot_idx]
    f1s = []
    for c in (0, 1):
        tp = ((yt == c) & (yp == c)).sum(axis=1)
        fp = ((yt != c) & (yp == c)).sum(axis=1)
        fn = ((yt == c) & (yp != c)).sum(axis=1)
        denom = 2 * tp + fp + fn
        f1s.append(np.where(denom == 0, 0.0, 2 * tp / np.maximum(denom, 1)))
    return np.mean(f1s, axis=0)


def _fit_predict(x_tr, y_tr, x_te):
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0, ddof=0) + 1e-9
    clf = LogisticRegression(max_iter=1000, random_state=SEED).fit((x_tr - mu) / sd, y_tr)
    return clf.predict((x_te - mu) / sd)


def rq3(videos: pd.DataFrame) -> dict:
    """Macro-F1 of four feature sets on a fixed stratified hold-out split."""
    y = videos["label_final"].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(videos)), test_size=TEST_SIZE,
                                      stratify=y, random_state=SEED)
    y_te = y[idx_te]
    preds = {}
    for name, cols in FEATURE_SETS.items():
        x = videos[cols].to_numpy(dtype=float)
        preds[name] = _fit_predict(x[idx_tr], y[idx_tr], x[idx_te])

    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, len(y_te), size=(N_BOOT, len(y_te)))
    boot = {k: _macro_f1_boot(y_te, p, boot_idx) for k, p in preds.items()}

    out = {"n_train": len(idx_tr), "n_test": len(idx_te), "sets": {}}
    for name, p in preds.items():
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        out["sets"][name] = {
            "macro_f1": f1_score(y_te, p, average="macro"),
            "macro_precision": precision_score(y_te, p, average="macro", zero_division=0),
            "macro_recall": recall_score(y_te, p, average="macro", zero_division=0),
            "ci": (float(lo), float(hi)),
        }
    out["gain"] = out["sets"]["multimodal"]["macro_f1"] - out["sets"]["metadata_only"]["macro_f1"]
    out["gain_ci"] = tuple(float(v) for v in
                           np.percentile(boot["multimodal"] - boot["metadata_only"], [2.5, 97.5]))
    out["gain_reaches_threshold"] = bool(out["gain"] >= MIN_IMPROVEMENT)
    return out


# ----------------------------------------------------------------- 3. guardrails
def guardrails(sessions: pd.DataFrame, videos: pd.DataFrame) -> list[dict]:
    """Guardrail metrics: error rate, annotator agreement, minimum group sizes."""
    rows = []

    def add(metric, group, n, value, rule, ok):
        rows.append({"metric": metric, "group": group, "n": n, "value": value,
                     "note": f"{'PASS' if ok else 'FAIL'} ({rule})"})

    for grp, g in sessions.groupby("profile_type"):
        er = g["n_unavailable"].sum() / g["n_requested"].sum()
        add("error_rate_unavailable_pct", grp, len(g), 100 * er,
            f"<= {100 * MAX_ERROR_RATE:.0f}%", er <= MAX_ERROR_RATE)
    kappa = cohen_kappa_score(videos["label_a"], videos["label_b"])
    add("cohen_kappa", "double-coded videos", len(videos), kappa,
        f">= {MIN_KAPPA}", kappa >= MIN_KAPPA)
    base = sessions[sessions["session_idx"] == 0]
    for grp, g in base.groupby("profile_type"):
        add("baseline_sessions", grp, len(g), len(g),
            f">= {MIN_UNITS_PER_GROUP}", len(g) >= MIN_UNITS_PER_GROUP)
    for arm, g in base[base["profile_type"] == "child"].groupby("arm"):
        add("accounts_per_arm", arm, len(g), len(g),
            f">= {MIN_UNITS_PER_GROUP}", len(g) >= MIN_UNITS_PER_GROUP)
    return rows


# ----------------------------------------------------------------- 4. summary table
def build_summary(raw_path: Path = RAW_PATH):
    """Run the whole analysis; return (summary DataFrame, dict with detailed results)."""
    feeds, videos = load_raw(raw_path)
    sessions, n_dropped = drop_warmup(add_share(feeds))
    r1, r2, r3 = rq1(sessions), rq2(sessions), rq3(videos)
    guards = guardrails(sessions, videos)

    rows = []

    def add(rq, metric, group, n, value, lo=None, hi=None, note=""):
        rows.append({"rq": rq, "metric": metric, "group": group, "n": n,
                     "value": value, "ci_low": lo, "ci_high": hi, "note": note})

    def add_describe(rq, name, group, x, unit):
        d = describe(x)
        lo, hi = bootstrap_ci(x)
        add(rq, f"mean_{unit}", group, d["n"], d["mean"], lo, hi, "95% bootstrap CI of the mean")
        for key in ["sd", "median", "p25", "p75", "p95", "p99"]:
            add(rq, f"{key}_{unit}", group, d["n"], d[key])

    add("data", "warmup_sessions_dropped", "feed", n_dropped, n_dropped,
        note="cold-start sessions discarded")
    for grp in ["adult", "child"]:
        add_describe("RQ1", "ai_share", grp, r1[grp] * 100, "pct")
    add("RQ1", "diff_child_minus_adult_pp", "child-adult", len(r1["child"]) + len(r1["adult"]),
        r1["diff"] * 100, r1["ci"][0] * 100, r1["ci"][1] * 100,
        f"Welch t={r1['t']:.2f}, one-sided p={r1['p']:.2g} (session level)")
    for arm in ARMS:
        add_describe("RQ2", "delta_ai_share", arm, r2["groups"][arm] * 100, "pp")
    add("RQ2", "diff_aidwell_minus_humandwell_pp", "ai_dwell-human_dwell",
        len(r2["groups"]["ai_dwell"]) + len(r2["groups"]["human_dwell"]),
        r2["diff"] * 100, r2["ci"][0] * 100, r2["ci"][1] * 100,
        f"Welch t={r2['t']:.2f}, one-sided p={r2['p']:.2g} (account level)")
    for name, m in r3["sets"].items():
        add("RQ3", "macro_f1", name, r3["n_test"], m["macro_f1"], *m["ci"],
            "95% bootstrap CI over test videos")
        add("RQ3", "macro_precision", name, r3["n_test"], m["macro_precision"])
        add("RQ3", "macro_recall", name, r3["n_test"], m["macro_recall"])
    add("RQ3", "gain_multimodal_minus_metadata", "multimodal-metadata_only", r3["n_test"],
        r3["gain"], *r3["gain_ci"],
        f"threshold {MIN_IMPROVEMENT}: {'reached' if r3['gain_reaches_threshold'] else 'not reached'} (point estimate)")
    for g in guards:
        add("guardrail", g["metric"], g["group"], g["n"], g["value"], note=g["note"])

    summary = pd.DataFrame(rows)
    results = {"rq1": r1, "rq2": r2, "rq3": r3, "guardrails": guards,
               "trajectory": trajectory(sessions), "n_warmup_dropped": n_dropped,
               "sessions": sessions, "videos": videos}
    return summary, results


def save_summary(summary: pd.DataFrame, out_path: Path = OUT_PATH) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, float_format="%.4f")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(RAW_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()
    summary, res = build_summary(Path(args.raw))
    path = save_summary(summary, Path(args.out))
    r1, r2, r3 = res["rq1"], res["rq2"], res["rq3"]
    print(f"warm-up sessions dropped: {res['n_warmup_dropped']}")
    print(f"RQ1  child - adult = {r1['diff'] * 100:.1f} pp  95% CI [{r1['ci'][0] * 100:.1f}, {r1['ci'][1] * 100:.1f}]  p={r1['p']:.2g}")
    print(f"RQ2  ai_dwell - human_dwell = {r2['diff'] * 100:.1f} pp  95% CI [{r2['ci'][0] * 100:.1f}, {r2['ci'][1] * 100:.1f}]  p={r2['p']:.2g}")
    print(f"RQ3  F1 gain = {r3['gain']:.3f}  95% CI [{r3['gain_ci'][0]:.3f}, {r3['gain_ci'][1]:.3f}]")
    for g in res["guardrails"]:
        print(f"guardrail {g['metric']} [{g['group']}] = {g['value']:.3g}  {g['note']}")
    print(f"summary written -> {path}")


if __name__ == "__main__":
    main()
