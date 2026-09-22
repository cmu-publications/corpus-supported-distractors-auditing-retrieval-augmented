"""Publication figures for the corpus-supported distractor paper.

Every number is read from the archived results file
    results/measurements.json
or from the archived run log
    results/run-log.json  (stdout)
No value is typed in by hand except the arm display names.

Palette: Okabe-Ito subset #0072B2 / #D55E00 / #009E73 plus a neutral gray for
reference rows. Validated with the dataviz skill's validate_palette.js
(all six checks PASS in light mode).
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "measurements.json"
RUNLOG = ROOT / "run-log.json"
OUT = ROOT

BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#7F7F7F"
INK, MUTED = "#222222", "#666666"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.edgecolor": "#BBBBBB",
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ARMS = [
    ("ungrounded_topic_only_control", "Ungrounded control", "control"),
    ("overgenerate_rerank_by_contradiction", "Rerank by contradiction", "rerank"),
    ("unfiltered_grounded_pool", "Unfiltered grounded pool", "reference"),
    ("llm_judge_faithfulness_filter", "LLM-judge faithfulness", "filter"),
    ("overgenerate_rerank_by_key_similarity", "Rerank by key similarity", "rerank"),
    ("null_random_drop", "Random-drop null", "control"),
    ("per_option_entailment_filter", "Per-option entailment", "filter"),
    ("contradiction_polarity_item_filter", "Contradiction polarity", "filter"),
    ("not_entail_polarity_item_filter", "Not-entail polarity", "filter"),
    ("positive_control_planted", "Planted positive control", "control"),
]
ROLE_COLOR = {"filter": VERM, "rerank": GREEN, "reference": BLUE, "control": GRAY}


def load():
    metrics = json.loads(RESULTS.read_text(encoding="utf-8"))["metrics"]
    stdout = json.loads(RUNLOG.read_text(encoding="utf-8"))["stdout"]
    pooled = {}
    pat = (r"condition=(\S+) distractor_caused_csd_rate_mean: ([\d.]+) "
           r"distractor_caused_csd_rate_std: ([\d.]+) pooled=([\d.]+) "
           r"pooled_ci95=\[([\d.]+),([\d.]+)\] n_distractors=(\d+)")
    for m in re.finditer(pat, stdout):
        pooled[m.group(1)] = {
            "mean": float(m.group(2)), "std": float(m.group(3)),
            "pooled": float(m.group(4)), "lo": float(m.group(5)),
            "hi": float(m.group(6)), "n": int(m.group(7)),
        }
    return metrics, pooled, stdout


def fig_main(pooled):
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    names, vals, los, his, cols = [], [], [], [], []
    for key, label, role in ARMS:
        p = pooled[key]
        names.append(label)
        vals.append(p["pooled"] * 100)
        los.append((p["pooled"] - p["lo"]) * 100)
        his.append((p["hi"] - p["pooled"]) * 100)
        cols.append(ROLE_COLOR[role])
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.62, color=cols, zorder=3)
    ax.errorbar(vals, y, xerr=[los, his], fmt="none", ecolor=INK,
                elinewidth=1.0, capsize=2.5, zorder=4)
    for i, (key, _, _) in enumerate(ARMS):
        p = pooled[key]
        events = int(round(p["pooled"] * p["n"]))
        ax.text(p["hi"] * 100 + 0.12, i, f"{events}/{p['n']}",
                va="center", ha="left", fontsize=7.6, color=MUTED)
    ax.set_yticks(y, names)
    ax.set_xlabel("Distractor-caused CSD rate (% of distractor slots), pooled over 3 seeds")
    ax.set_xlim(0, 7.2)
    ax.xaxis.grid(True, color="#E8E8E8", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLOR[r]) for r in
               ("reference", "filter", "rerank", "control")]
    ax.legend(handles, ["Reference pool", "Grounding filter", "Overgenerate and rerank",
                        "Control arm"], loc="lower right", frameon=False, ncol=1)
    fig.savefig(OUT / "main_results.pdf")
    fig.savefig(OUT / "main_results.png")
    plt.close(fig)


def fig_denominator(pooled):
    """Retention against invalidity: events held roughly constant while the
    denominator halves, so the conditional rate rises arithmetically."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    order = ["unfiltered_grounded_pool", "null_random_drop", "per_option_entailment_filter",
             "contradiction_polarity_item_filter", "not_entail_polarity_item_filter",
             "llm_judge_faithfulness_filter"]
    labels = ["Unfiltered pool", "Random-drop null", "Per-option entailment",
              "Contradiction polarity", "Not-entail polarity", "LLM-judge faithfulness"]
    cols = [BLUE, GRAY, VERM, VERM, VERM, VERM]
    y = np.arange(len(order))[::-1]

    events = [int(round(pooled[k]["pooled"] * pooled[k]["n"])) for k in order]
    slots = [pooled[k]["n"] for k in order]

    # One axis only. Retention is carried as a printed denominator, not a second scale.
    a1.barh(y, events, height=0.6, color=cols, zorder=3)
    a1.set_xlabel("Flagged distractor slots (count)")
    a1.set_xlim(0, 13.5)
    for i, (e, s) in enumerate(zip(events, slots)):
        a1.text(e + 0.3, y[i], f"{e}  of {s} retained", va="center", fontsize=7.8, color=INK)
    a1.set_yticks(y, labels, fontsize=7.8)
    a1.set_title("Flagged slots stay put as the pool halves", loc="left")
    a1.xaxis.grid(True, color="#E8E8E8", zorder=0)
    a1.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a1.spines[s].set_visible(False)

    rates = [pooled[k]["pooled"] * 100 for k in order]
    a2.barh(y, rates, height=0.6, color=cols, zorder=3)
    for i, v in enumerate(rates):
        a2.text(v + 0.05, y[i], f"{v:.2f}%", va="center", fontsize=7.8, color=INK)
    a2.set_xlabel("Conditional CSD rate (% of retained slots)")
    a2.set_xlim(0, 2.9)
    a2.set_yticks(y, ["" for _ in labels])
    a2.set_title("So the rate rises arithmetically", loc="left")
    a2.xaxis.grid(True, color="#E8E8E8", zorder=0)
    a2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "denominator.pdf")
    fig.savefig(OUT / "denominator.png")
    plt.close(fig)


def fig_evidence(metrics, stdout):
    """Seed-level spread plus the judge-validation and key-hallucination
    diagnostics that qualify how far the control comparison can be pushed."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.42})

    order = [k for k, _, _ in ARMS]
    labels = [lab for _, lab, _ in ARMS]
    cols = [ROLE_COLOR[r] for _, _, r in ARMS]
    for i, key in enumerate(order):
        seeds = [metrics[f"{key}/{s}/distractor_caused_csd_rate"] * 100 for s in (0, 1, 2)]
        a1.scatter(seeds, [i] * 3, s=26, color=cols[i], alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.6)
        mean = metrics[f"{key}/distractor_caused_csd_rate_mean"] * 100
        a1.plot([mean], [i], marker="|", ms=13, color=INK, zorder=4)
    a1.set_yticks(np.arange(len(labels)), labels, fontsize=7.6)
    a1.set_xlabel("Per-seed CSD rate (%)")
    a1.set_title("Per-seed spread (tick = seed mean)", loc="left")
    a1.xaxis.grid(True, color="#E8E8E8", zorder=0)
    a1.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a1.spines[s].set_visible(False)

    kh = {}
    for m in re.finditer(r"condition=(\S+) seed=(\d) distractor_caused_csd_rate: [\d.]+ "
                         r"item_yield=[\d.]+ key_halluc=([\d.]+)", stdout):
        kh.setdefault(m.group(1), []).append(float(m.group(3)) * 100)
    pair = ["unfiltered_grounded_pool", "ungrounded_topic_only_control"]
    plabels = ["Grounded\npool", "Ungrounded\ncontrol"]
    xs = np.arange(2)
    means = [np.mean(kh[k]) for k in pair]
    a2.bar(xs, means, width=0.5, color=[BLUE, GRAY], zorder=3)
    for i, k in enumerate(pair):
        a2.scatter([i + 0.34] * 3, kh[k], s=20, color=INK, zorder=4)
        a2.text(i - 0.02, means[i] + 3.5, f"{means[i]:.1f}%", ha="center", fontsize=8.5, color=INK)
    a2.set_xticks(xs, plabels, fontsize=8.2)
    a2.set_ylabel("Keys the corpus does not support (%)")
    a2.set_ylim(0, 108)
    a2.set_title("Why the control has a floor", loc="left")
    a2.set_xlim(-0.6, 1.7)
    a2.yaxis.grid(True, color="#E8E8E8", zorder=0)
    a2.set_axisbelow(True)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "evidence.pdf")
    fig.savefig(OUT / "evidence.png")
    plt.close(fig)


if __name__ == "__main__":
    metrics, pooled, stdout = load()
    fig_main(pooled)
    fig_denominator(pooled)
    fig_evidence(metrics, stdout)
    print("wrote main_results, denominator, evidence (pdf + png)")
