"""Generate a SMALL SYNTHETIC demo log: data/raw/test_run.jsonl

!!! These are NOT real measurements. !!!
They only let the pipeline (process_data.py -> visualize.py -> notebook) run end to end.
Before drawing any conclusions, replace data/raw/test_run.jsonl with the real logs
(same schema, see below).

Schema: one JSON object per line, every demo record carries "synthetic": true.

record_type = "feed"   one recorded Shorts session (RQ1, RQ2)
    session_id, account_id, profile_type (child | adult),
    arm (ai_dwell | human_dwell | no_dwell | null for adults),
    session_idx (-1 = cold-start warm-up session, 0 = baseline, 1..5 = after watching),
    phase (warmup | eval), n_requested (=100), n_unavailable, n_ai

record_type = "video"  one hand-labelled video (RQ3)
    video_id, phase, label_a, label_b (two annotators), label_final (adjudicated),
    meta_title_len, meta_has_ai_tag, visual_artifact_score, audio_synthetic_score
"""
import json
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "test_run.jsonl"

ARMS = ["ai_dwell", "human_dwell", "no_dwell"]
DRIFT_PER_SESSION = {"ai_dwell": 0.030, "human_dwell": 0.003, "no_dwell": 0.0}


def feed_records(rng):
    accounts = [(f"c{i:02d}", "child", ARMS[(i - 1) % 3], 0.30) for i in range(1, 13)]
    accounts += [(f"a{i:02d}", "adult", None, 0.12) for i in range(1, 9)]
    rows = []
    for acc, profile, arm, base_mean in accounts:
        base = float(np.clip(rng.normal(base_mean, 0.03), 0.02, 0.90))
        last_session = 5 if profile == "child" else 0
        for idx in range(-1, last_session + 1):
            if idx == -1:  # cold start: unstable, inflated feed (discarded later)
                share = base + rng.uniform(0.05, 0.20)
            else:
                share = base + DRIFT_PER_SESSION.get(arm, 0.0) * idx
            share = float(np.clip(share, 0.0, 0.95))
            n_unavailable = int(rng.binomial(100, 0.02))
            n_ai = int(rng.binomial(100 - n_unavailable, share))
            rows.append({
                "record_type": "feed", "synthetic": True,
                "session_id": f"{acc}-{'w' if idx < 0 else idx}",
                "account_id": acc, "profile_type": profile, "arm": arm,
                "session_idx": idx, "phase": "warmup" if idx < 0 else "eval",
                "n_requested": 100, "n_unavailable": n_unavailable, "n_ai": n_ai,
            })
    return rows


def video_records(rng, n_per_class=50):
    labels = rng.permutation([1] * n_per_class + [0] * n_per_class)
    rows = []
    for k, y in enumerate(labels, start=1):
        y = int(y)
        label_a = y if rng.random() > 0.04 else 1 - y
        label_b = y if rng.random() > 0.06 else 1 - y
        has_tag = int(rng.random() < (0.40 if y else 0.08))
        visual = float(np.clip(rng.normal(0.62 if y else 0.38, 0.17), 0, 1))
        audio = float(np.clip(rng.normal(0.60 if y else 0.40, 0.20), 0, 1))
        rows.append({
            "record_type": "video", "synthetic": True, "video_id": f"v{k:03d}",
            "phase": "eval", "label_a": label_a, "label_b": label_b, "label_final": y,
            "meta_title_len": int(max(10, rng.normal(45, 10))),
            "meta_has_ai_tag": has_tag,
            "visual_artifact_score": round(visual, 3),
            "audio_synthetic_score": round(audio, 3),
        })
    return rows


def main():
    rng = np.random.default_rng(SEED)
    rows = feed_records(rng) + video_records(rng)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} SYNTHETIC records -> {OUT}")


if __name__ == "__main__":
    main()
