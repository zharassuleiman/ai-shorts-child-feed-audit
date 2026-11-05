"""Final visualization module (Task 3).

Builds three publication-ready figures and saves each one as vector .pdf and
raster .png (300 DPI) in reports/figures/.

Rules followed: labelled axes with units, 95% bootstrap CI shown on every figure,
colour-blind-safe palette + different markers / line styles / hatches.

Run:  python src/visualize.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import process_data as pdm

FIG_DIR = pdm.ROOT / "reports" / "figures"
PAL = sns.color_palette("colorblind")

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,  # embed TrueType fonts in the PDF
})

ARM_STYLE = {
    "ai_dwell":    {"color": PAL[0], "marker": "o", "ls": "-",  "label": "AI-dwell"},
    "human_dwell": {"color": PAL[1], "marker": "s", "ls": "--", "label": "Human-dwell"},
    "no_dwell":    {"color": PAL[2], "marker": "^", "ls": ":",  "label": "No-dwell (control)"},
}
SET_LABELS = {"metadata_only": "Metadata\nonly (baseline)", "visual": "Visual\nonly",
              "audio": "Audio\nonly", "multimodal": "Multimodal\n(all features)"}


def save_figure(fig, stem: str):
    """Save fig as <stem>.pdf (vector) and <stem>.png (300 DPI)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")


def make_fig1(traj):
    """AI share over sessions per watch policy (mean, shaded 95% bootstrap CI)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for arm, st in ARM_STYLE.items():
        g = traj[traj["arm"] == arm].sort_values("session_idx")
        n = int(g["n"].max())
        ax.plot(g["session_idx"], g["mean"] * 100, color=st["color"], marker=st["marker"],
                linestyle=st["ls"], lw=2, ms=6, label=f"{st['label']} (n={n} accounts)")
        ax.fill_between(g["session_idx"], g["lo"] * 100, g["hi"] * 100,
                        color=st["color"], alpha=0.2, linewidth=0)
    ax.set_xlabel("Session index (0 = baseline)")
    ax.set_ylabel("AI-generated share of feed (%)")
    ax.set_xticks(range(0, 6))
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 8)  # room for the legend
    ax.set_title("AI share after watching: mean and 95% bootstrap CI", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    return fig


def make_fig2(r1):
    """Distribution of the baseline AI share: adult vs child (box + points + mean CI)."""
    data = {"Adult": r1["adult"] * 100, "Child": r1["child"] * 100}
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    rng = np.random.default_rng(pdm.SEED)
    hatches, markers = ["//", ".."], ["o", "s"]
    for i, (name, x) in enumerate(data.items(), start=1):
        bp = ax.boxplot(x, positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "lw": 2})
        patch = bp["boxes"][0]
        patch.set_facecolor(PAL[i - 1])
        patch.set_alpha(0.45)
        patch.set_hatch(hatches[i - 1])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(x)), x, color=PAL[i - 1],
                   marker=markers[i - 1], edgecolor="black", s=36, zorder=3)
        lo, hi = pdm.bootstrap_ci(x)
        ax.errorbar(i + 0.38, x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]], fmt="D",
                    color="black", capsize=4, ms=6, zorder=4,
                    label="Mean and 95% bootstrap CI" if i == 1 else None)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Adult\n(n={len(data['Adult'])} sessions)",
                        f"Child\n(n={len(data['Child'])} sessions)"])
    ax.set_xlabel("Profile type")
    ax.set_ylabel("AI-generated share of the first 100 Shorts (%)")
    ax.legend(frameon=False, loc="upper left")
    return fig


def make_fig3(r3):
    """Macro-F1 per feature set with 95% bootstrap CI over the test videos."""
    names = list(pdm.FEATURE_SETS)
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for i, name in enumerate(names):
        m = r3["sets"][name]
        ax.errorbar(i, m["macro_f1"], yerr=[[m["macro_f1"] - m["ci"][0]],
                                            [m["ci"][1] - m["macro_f1"]]],
                    fmt=markers[i], color=PAL[i], capsize=5, ms=9, lw=2)
    base = r3["sets"]["metadata_only"]["macro_f1"]
    ax.axhline(base, color="gray", linestyle="--", lw=1, label="Baseline (metadata-only)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([SET_LABELS[n] for n in names])
    ax.set_xlabel(f"Feature set (hold-out test set, n={r3['n_test']} videos)")
    ax.set_ylabel("Macro-F1 (dimensionless, 0-1)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    return fig


def main():
    _, res = pdm.build_summary()
    figs = {
        "fig1_ai_share_vs_session": make_fig1(res["trajectory"]),
        "fig2_distribution_boxplot": make_fig2(res["rq1"]),
        "fig3_f1_by_feature_set": make_fig3(res["rq3"]),
    }
    for stem, fig in figs.items():
        save_figure(fig, stem)
        plt.close(fig)
        print(f"saved {stem}.pdf and {stem}.png (300 DPI) -> {FIG_DIR}")


if __name__ == "__main__":
    main()
