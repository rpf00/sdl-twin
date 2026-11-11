"""Figures for Stage 3: recorded agent episodes next to the scripted baselines.

    python experiments/stage3_plots.py ~/sdl-runs-grid
    python experiments/stage3_plots.py ~/sdl-runs-grid ~/other-runs --baseline-episodes 200

Writes to results/ (or --out):
  stage3_overview.png            success, mixes used, distractor mixes and calibration, all agents
  stage3_timeline_<agent>.png    every episode of each recorded agent, mix by mix
  stage3_recorded_episodes.csv   the recorded episodes as a table

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

CONDITIONS = [("named", "online"), ("named", "offline"), ("blinded", "online"), ("blinded", "offline")]
COND_LABELS = ["named\nonline", "named\noffline", "blinded\nonline", "blinded\noffline"]
OUTCOME_COLOURS = {"target": "#c0392b", "distractor": "#a9743a", "null": "#d9d9d9"}
OUTCOME_NAMES = {"target": "showed the red complex", "distractor": "precipitate only", "null": "no reaction"}


def heatmap(ax, values, counts, row_labels, title, cmap, vmin, vmax, fmt, note):
    shown = np.ma.masked_invalid(values)
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CONDITIONS)), COND_LABELS, fontsize=9)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=9)
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    mid = (vmin + vmax) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                ax.text(j, i, "no data", ha="center", va="center", fontsize=8, color="0.5")
                continue
            dark = (values[i, j] > mid) if cmap != "Greys_r" else (values[i, j] < mid)
            ax.text(j, i, f"{values[i, j]:{fmt}}\nn={counts[i, j]}", ha="center", va="center",
                    fontsize=9, color="white" if dark else "black")
    ax.set_xlabel(note, fontsize=8, color="0.3")
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def overview(groups: dict, agents: list, recorded: set, path: Path) -> None:
    shape = (len(agents), len(CONDITIONS))
    success, mixes, distractor, counts = (np.full(shape, np.nan) for _ in range(4))
    counts = np.zeros(shape, dtype=int)
    for i, agent in enumerate(agents):
        for j, (mode, fb) in enumerate(CONDITIONS):
            eps = groups.get((agent, mode, fb))
            if eps:
                a = aggregate(eps)
                success[i, j], mixes[i, j] = a["success_rate"], a["mean_mixes_used"]
                distractor[i, j], counts[i, j] = a["mean_distractor_mixes"], a["episodes"]
    budget = max(e["task"]["budget"] for eps in groups.values() for e in eps)
    rows = [f"{a}  (recorded)" if a in recorded else f"{a}  (scripted)" for a in agents]

    fig, axes = plt.subplots(2, 2, figsize=(13, 2.6 + 1.5 * len(agents)), constrained_layout=True)
    heatmap(axes[0, 0], success, counts, rows, "Success rate", "Greens", 0, 1, ".2f",
            "share of episodes with the correct pair submitted within the budget")
    heatmap(axes[0, 1], mixes, counts, rows, "Mixes used per episode", "Blues", 0, budget, ".2f",
            f"mean, out of a budget of {budget}; fewer is more efficient")
    heatmap(axes[1, 0], distractor, counts, rows, "Mixes that only precipitated", "Oranges", 0, max(1.0, np.nanmax(distractor)),
            ".2f", "mean per episode of mixes that formed the precipitate and never showed red at any image")

    ax = axes[1, 1]
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect calibration")
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for k, agent in enumerate(agents):
        eps = [e for (ag, _, _), g in groups.items() if ag == agent for e in g]
        m = [episode_metrics(e) for e in eps]
        conf = [c for x in m for c in x["confidence_trace"]]
        hit = [o == "target" for x in m for o in x["outcome_trace"]]
        cal = calibration(conf, hit)
        if not cal["n"]:
            continue
        xs = [b["mean_confidence"] for b in cal["bins"]]
        ys = [b["observed_rate"] for b in cal["bins"]]
        ns = [b["n"] for b in cal["bins"]]
        ax.plot(xs, ys, lw=1.2, marker=markers[k % len(markers)], ms=0, alpha=0.8)
        ax.scatter(xs, ys, s=[30 + 12 * np.sqrt(n) for n in ns], marker=markers[k % len(markers)],
                   label=f"{agent}: Brier {cal['brier']:.2f} (n={cal['n']})", zorder=3,
                   edgecolor="black" if agent in recorded else "none", linewidth=1.2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("stated confidence that the mix gives the red complex (at any image step)", fontsize=9)
    ax.set_ylabel("share of those mixes that did", fontsize=9)
    ax.set_title("Calibration of per-mix confidence", fontsize=11, loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", frameon=False, markerscale=0.5)
    ax.text(0.98, 0.03, "below the line = overconfident\nmarker size grows with number of mixes",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.3")
    ax.grid(alpha=0.25)
    fig.suptitle("Stage 3: find the red-complex pair within the mix budget (simulated lab)", fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def timeline(episodes: list, agent: str, path: Path) -> None:
    eps = sorted(episodes, key=lambda e: (CONDITIONS.index((e["task"]["mode"], e["task"]["feedback"])), e["task"]["seed"],
                                           e["episode_id"]))
    budget = max(e["task"]["budget"] for e in eps)
    n = len(eps)
    fig, ax = plt.subplots(figsize=(8.6, 1.6 + 0.36 * n), constrained_layout=True)
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
    ax.set_yticks(range(n), [f"{e['task']['mode']} / {e['task']['feedback']} / seed {e['task']['seed']}" for e in eps][::-1],
                  fontsize=8)
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
                 "cell colour = what the mix showed at any image step (red beats precipitate)\n"
                 "number = confidence stated beforehand",
                 fontsize=10, loc="left", pad=30)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="*", help="folders with recorded episodes (episode.json files)")
    p.add_argument("--baseline-episodes", type=int, default=50, help="scripted-baseline episodes per condition; 0 = none")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    recorded = [e for root in a.roots for e in load_episodes(Path(root).expanduser())]
    if a.roots and not recorded:
        print(f"warning: no episode.json found under {a.roots}")
    budgets = {e["task"]["budget"] for e in recorded}
    baselines = []
    if a.baseline_episodes:
        budget = budgets.pop() if len(budgets) == 1 else None      # match the recorded budget when unambiguous
        baselines = run_grid(POLICIES.values(), range(a.baseline_episodes), budget=budget)
    rec_agents = sorted({e["agent"] for e in recorded})
    base_agents = sorted({e["agent"] for e in baselines})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    overview(group(recorded + baselines), rec_agents + base_agents, set(rec_agents), out / "stage3_overview.png")
    for agent in rec_agents:
        timeline([e for e in recorded if e["agent"] == agent], agent, out / f"stage3_timeline_{agent}.png")
    if recorded:
        write_csv(out / "stage3_recorded_episodes.csv", [episode_metrics(e) for e in recorded])


if __name__ == "__main__":
    main()
