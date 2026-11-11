"""Figures for Stage 3: recorded agent episodes next to the scripted baselines.

    python experiments/stage3_plots.py ~/sdl-runs-v2
    python experiments/stage3_plots.py ~/sdl-runs-v2 --baseline-episodes 200

Writes to results/ (or --out):
  stage3_overview.png            how many mixes each agent needs, success, and calibration
  stage3_tokens.png              mixes used and correct answers against tokens per episode
  stage3_timeline_<agent>.png    every episode of each recorded agent, mix by mix
  stage3_recorded_episodes.csv   the recorded episodes as a table

Only episodes played under the current rules (at most two samples per mix) are used.
Add --include-pilot to also load the earlier pilot runs.

Needs: pip install matplotlib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from _common import write_csv
from sdl.agent.harness import run_grid
from sdl.agent.metrics import aggregate, calibration, episode_metrics, group, load_episodes
from sdl.agent.policies import POLICIES
from sdl.agent.task import FEEDBACK, MODES, RULES_VERSION

OUTCOME_COLOURS = {"target": "#c0392b", "distractor": "#a9743a", "null": "#d9d9d9"}
OUTCOME_NAMES = {"target": "showed the red complex", "distractor": "precipitate only", "null": "no reaction"}
MODE_TITLES = {"labelled": "Labelled\n(which chemical is in which bottle)",
               "listed": "Listed\n(which chemicals, not which bottle)",
               "unknown": "Unknown\n(nothing about the contents)"}
PALETTE = ["#1f77b4", "#d95f02", "#7570b3", "#1b9e77", "#e7298a"]
MARKERS = ["o", "s", "^", "D", "v"]


def conditions_in(episodes: list) -> list:
    present = {(e["task"]["mode"], e["task"]["feedback"]) for e in episodes}
    return [(m, f) for f in FEEDBACK for m in MODES if (m, f) in present]


def cond_label(cond, multi_feedback: bool) -> str:
    return f"{cond[0]}\n{cond[1]}" if multi_feedback else cond[0]


def heatmap(ax, values, counts, row_labels, col_labels, title, cmap, vmin, vmax, fmt, note):
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=9)
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    mid = (vmin + vmax) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                ax.text(j, i, "no data", ha="center", va="center", fontsize=8, color="0.5")
            else:
                ax.text(j, i, f"{values[i, j]:{fmt}}\nn={counts[i, j]}", ha="center", va="center", fontsize=9,
                        color="white" if values[i, j] > mid else "black")
    ax.set_xlabel(note, fontsize=8, color="0.3")
    for s in ax.spines.values():
        s.set_visible(False)


def overview(groups: dict, agents: list, recorded: set, path: Path) -> None:
    episodes = [e for g in groups.values() for e in g]
    conds = conditions_in(episodes)
    multi = len({f for _, f in conds}) > 1
    budget = max(e["task"]["budget"] for e in episodes)
    shape = (len(agents), len(conds))
    to_target, success = np.full(shape, np.nan), np.full(shape, np.nan)
    counts = np.zeros(shape, dtype=int)
    agg = {}
    for i, agent in enumerate(agents):
        for j, (mode, fb) in enumerate(conds):
            eps = groups.get((agent, mode, fb))
            if eps:
                a = agg[(agent, mode, fb)] = aggregate(eps)
                success[i, j], counts[i, j] = a["success_rate"], a["episodes"]
                if a["mean_mixes_to_target"] is not None:
                    to_target[i, j] = a["mean_mixes_to_target"]
    rows = [f"{a}  (recorded)" if a in recorded else f"{a}  (scripted)" for a in agents]
    cols = [cond_label(c, multi) for c in conds]
    colour = {a: PALETTE[k % len(PALETTE)] for k, a in enumerate(sorted(recorded))}
    marker = {a: MARKERS[k % len(MARKERS)] for k, a in enumerate(sorted(recorded))}

    fig = plt.figure(figsize=(15, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.05, 1])
    modes = [m for m in MODES if any(c[0] == m for c in conds)]
    ks = np.arange(1, budget + 1)
    for n, mode in enumerate(modes):
        span = 6 // len(modes)
        ax = fig.add_subplot(gs[0, n * span:(n + 1) * span])
        fb = "adaptive" if (mode, "adaptive") in conds else conds[0][1]
        base = agg.get(("random_pairs", mode, fb))
        if base:
            ax.plot(ks, base["produced_within"], color="0.55", ls="--", marker="x", lw=1.3,
                    label=f"random pair search (n={base['episodes']})")
        for agent in sorted(recorded):
            a = agg.get((agent, mode, fb))
            if a:
                ax.plot(ks, a["produced_within"], color=colour[agent], marker=marker[agent], lw=2, ms=7,
                        label=f"{agent} (n={a['episodes']})")
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks(ks)
        ax.set_xlabel("number of mixes used", fontsize=9)
        if n == 0:
            ax.set_ylabel("share of episodes that had\nproduced the red complex", fontsize=9)
        ax.set_title(MODE_TITLES[mode], fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right", frameon=False)

    heatmap(fig.add_subplot(gs[1, 0:2]), to_target, counts, rows, cols, "Mixes needed to produce the red complex",
            "Blues", 1, budget, ".2f", "mean over episodes that produced it; fewer is better")
    heatmap(fig.add_subplot(gs[1, 2:4]), success, counts, [""] * len(rows), cols, "Correct answer submitted", "Greens",
            0, 1, ".2f", f"share of episodes; with a budget of {budget}, elimination alone can reach 1.00")

    ax = fig.add_subplot(gs[1, 4:6])
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect calibration")
    shown = sorted(recorded) + [a for a in agents if a not in recorded]
    for agent in shown:
        m = [episode_metrics(e) for (ag, _, _), g in groups.items() if ag == agent for e in g]
        cal = calibration([c for x in m for c in x["confidence_trace"]],
                          [o == "target" for x in m for o in x["outcome_trace"]])
        if not cal["n"]:
            continue
        xs = [b["mean_confidence"] for b in cal["bins"]]
        ys = [b["observed_rate"] for b in cal["bins"]]
        rec = agent in recorded
        ax.plot(xs, ys, lw=1.6 if rec else 1, color=colour.get(agent, "0.6"), alpha=1 if rec else 0.7)
        ax.scatter(xs, ys, s=[30 + 10 * np.sqrt(b["n"]) for b in cal["bins"]], color=colour.get(agent, "0.6"),
                   marker=marker.get(agent, "x"), zorder=3, label=f"{agent}: Brier {cal['brier']:.2f} (n={cal['n']})")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("stated confidence that the mix gives the red complex", fontsize=9)
    ax.set_ylabel("share of those mixes that did", fontsize=9)
    ax.set_title("Calibration of per-mix confidence", fontsize=11, loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", frameon=False, markerscale=0.6)
    ax.grid(alpha=0.25)
    fig.suptitle(f"Stage 3: find the red-complex pair in at most {budget} mixes of two samples (simulated lab)", fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def tokens_figure(recorded: list, path: Path) -> bool:
    """Efficiency and correctness against token use, one large marker per model and condition."""
    rows = [(e["agent"], e["task"]["mode"], episode_metrics(e)) for e in recorded]
    rows = [(a, mode, m) for a, mode, m in rows if m["tokens_total"] is not None]
    if not rows:
        return False
    agents = sorted({a for a, _, _ in rows})
    mode_colour = {"labelled": "#1b9e77", "listed": "#d95f02", "unknown": "#7570b3"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for k, agent in enumerate(agents):
        for mode in MODES:
            sel = [m for a, md, m in rows if a == agent and md == mode]
            if not sel:
                continue
            x = np.array([m["tokens_total"] for m in sel]) / 1000.0
            used = np.array([m["mixes_used"] for m in sel], dtype=float)
            ok = np.array([m["success"] for m in sel], dtype=float)
            mk, col = MARKERS[k % len(MARKERS)], mode_colour[mode]
            ax1.scatter(x, used, color=col, marker=mk, s=22, alpha=0.3)
            ax1.scatter(x.mean(), used.mean(), color=col, marker=mk, s=170, edgecolor="black", linewidth=1,
                        label=f"{agent}, {mode} (n={len(sel)})", zorder=3)
            ax2.scatter(x.mean(), ok.mean(), color=col, marker=mk, s=170, edgecolor="black", linewidth=1,
                        label=f"{agent}, {mode} (n={len(sel)})", zorder=3)
    budget = max(m["budget"] for _, _, m in rows)
    ax1.set_ylim(budget + 0.4, 0.6)
    ax1.set_yticks(range(1, budget + 1))
    ax1.set_ylabel("mixes used before answering (fewer is better, so up is better)", fontsize=9)
    ax1.set_title("Efficiency against token use", fontsize=11, loc="left", fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("share of episodes with the correct answer", fontsize=9)
    ax2.set_title("Correct answers against token use", fontsize=11, loc="left", fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_xlabel("tokens per episode, thousands (input, cached input and output)", fontsize=9)
        ax.grid(alpha=0.25)
    ax1.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("Stage 3: what the tokens buy  (large markers = mean per model and condition; small = single episodes)",
                 fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")
    return True


def timeline(episodes: list, agent: str, path: Path) -> None:
    conds = conditions_in(episodes)
    eps = sorted(episodes, key=lambda e: (conds.index((e["task"]["mode"], e["task"]["feedback"])), e["task"]["seed"],
                                           e["episode_id"]))
    budget = max(e["task"]["budget"] for e in eps)
    n = len(eps)
    fig, ax = plt.subplots(figsize=(4.4 + 1.5 * budget, 1.7 + 0.36 * n), constrained_layout=True)
    prev = None
    for row, e in enumerate(eps):
        y = n - 1 - row
        m = episode_metrics(e)
        cond = (e["task"]["mode"], e["task"]["feedback"])
        if prev is not None and cond != prev:
            ax.axhline(y + 0.5, color="0.4", lw=0.8)
        prev = cond
        for k, (outcome, conf) in enumerate(zip(m["outcome_trace"], m["confidence_trace"])):
            ax.add_patch(plt.Rectangle((k + 0.04, y - 0.42), 0.92, 0.84, color=OUTCOME_COLOURS[outcome]))
            ax.text(k + 0.5, y, f"{conf:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if outcome == "null" else "white")
        for k in range(len(m["outcome_trace"]), budget):
            ax.add_patch(plt.Rectangle((k + 0.04, y - 0.42), 0.92, 0.84, fill=False, ec="0.85", lw=0.8))
        if m["answered"]:
            ok = m["success"]
            ax.text(budget + 0.15, y, f"{'correct' if ok else 'WRONG'}  ({m['answer_confidence']:.2f})", va="center",
                    fontsize=8.5, color="#1e7b34" if ok else "#b00020", fontweight="bold")
        else:
            ax.text(budget + 0.15, y, "no answer", va="center", fontsize=8.5, color="0.4")
    ax.set_yticks(range(n), [f"{e['task']['mode']} / seed {e['task']['seed']}"
                             + ("" if e["task"]["feedback"] == "adaptive" else " / batch") for e in eps][::-1], fontsize=8)
    ax.set_xticks([k + 0.5 for k in range(budget)] + [budget + 0.75],
                  [f"mix {k + 1}" for k in range(budget)] + ["answer (confidence)"], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_xlim(0, budget + 1.7)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [Patch(color=OUTCOME_COLOURS[k], label=OUTCOME_NAMES[k]) for k in OUTCOME_COLOURS]
    handles.append(Patch(fc="white", ec="0.7", label="mix not used"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=8, frameon=False)
    wins = sum(episode_metrics(e)["success"] for e in eps)
    ax.set_title(f"{agent}: every episode, mix by mix  ({wins}/{n} correct)\n"
                 "cell colour = what the mix produced\nnumber = confidence stated beforehand",
                 fontsize=10, loc="left", pad=30)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="*", help="folders with recorded episodes (episode.json files)")
    p.add_argument("--baseline-episodes", type=int, default=100, help="scripted-baseline episodes per condition; 0 = none")
    p.add_argument("--include-pilot", action="store_true", help="also load episodes played under the old rules")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    version = None if a.include_pilot else RULES_VERSION
    everything = [e for root in a.roots for e in load_episodes(root)]
    recorded = [e for root in a.roots for e in load_episodes(root, version)]
    if len(everything) != len(recorded):
        print(f"skipped {len(everything) - len(recorded)} pilot episode(s) played under the old rules "
              f"(use --include-pilot to plot them)")
    if a.roots and not recorded:
        print(f"warning: no usable episode.json found under {a.roots}")
    budgets = {e["task"]["budget"] for e in recorded}
    feedbacks = sorted({e["task"]["feedback"] for e in recorded}) or ["adaptive"]
    baselines = []
    if a.baseline_episodes:
        budget = budgets.pop() if len(budgets) == 1 else None      # match the recorded budget when unambiguous
        baselines = run_grid(POLICIES.values(), range(a.baseline_episodes), feedbacks=feedbacks, budget=budget)
    rec_agents = sorted({e["agent"] for e in recorded})
    base_agents = sorted({e["agent"] for e in baselines})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    overview(group(recorded + baselines), rec_agents + base_agents, set(rec_agents), out / "stage3_overview.png")
    for agent in rec_agents:
        timeline([e for e in recorded if e["agent"] == agent], agent, out / f"stage3_timeline_{agent}.png")
    if recorded:
        if not tokens_figure(recorded, out / "stage3_tokens.png"):
            print("no token usage recorded for these episodes, so stage3_tokens.png was not written")
        write_csv(out / "stage3_recorded_episodes.csv", [episode_metrics(e) for e in recorded])


if __name__ == "__main__":
    main()
