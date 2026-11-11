"""Figures for Stages 1 and 2: what the digital twin "sees" and how it is scored.

    python experiments/stage12_plots.py
    open results/stage1_overview.png results/stage2_overview.png

stage1_overview.png   the camera frame of every ordered pair, the two features that separate
                      the classes, the confusion matrix and the order-of-addition check
stage2_overview.png   frames of the dilution series, the calibration line with replicates,
                      and the camera response over a wide concentration range

Everything here comes from the simulated backend: the frames are synthetic and the
scatter between replicates is assumed, not measured.

Needs: pip install matplotlib
"""
from __future__ import annotations

import argparse
import itertools
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from concentration_scan import response
from sdl import Lab
from sdl.schema import Add, Image, Recipe, pair_recipe
from sdl.vision.pipeline import DEFAULT_THRESHOLDS

EXPECTED = {frozenset("AB"): "target", frozenset("AC"): "distractor"}
CLASS_COLOURS = {"target": "#c0392b", "distractor": "#a9743a", "null": "#7f8c8d", "other": "#8e44ad"}
CLASSES = ["target", "distractor", "null", "other"]


def show_frame(ax, frame, title, colour="black"):
    ax.imshow(frame, interpolation="nearest")
    ax.set_title(title, fontsize=8, color=colour, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])


def stage1(seed: int, out: Path) -> None:
    store = tempfile.mkdtemp()
    lab = Lab(seed=seed, store=store)
    names = {k: lab.config.reagents[k].name.split(" (")[0] for k in "ABCD"}
    rng = np.random.default_rng(seed)

    # Stage 0 baselines: each stock on its own.
    alone = {}
    for r in "ABCD":
        res = lab.dispatch(Recipe(f"{r} alone", [Add(r, 2.0), Image(5)]))
        alone[r] = (lab.store.frame(res.run_id), res)

    # Stage 1: 12 ordered pairs x 3 replicates over two sessions, as in stage1_confusion.py.
    pairs = list(itertools.permutations("ABCD", 2))
    plan = {0: [], 1: []}
    for k, (a, b) in enumerate(pairs):
        for rep in range(3):
            plan[(k + rep) % 2].append((a, b))
    rows, frames = [], {}
    for session in (0, 1):
        if session:
            lab.new_session()
        for j in rng.permutation(len(plan[session])):
            a, b = plan[session][j]
            res = lab.dispatch(pair_recipe(a, b))
            f = res.features[-1]
            rows.append({"first": a, "second": b, "session": session, "d": res.d_blank, "turbidity": f.turbidity,
                         "hue": f.hue_deg, "expected": EXPECTED.get(frozenset((a, b)), "null"),
                         "predicted": res.predicted_class})
            frames.setdefault((a, b), (lab.store.frame(res.run_id), res))

    fig = plt.figure(figsize=(15, 8.2), constrained_layout=True)
    left, right = fig.subfigures(1, 2, width_ratios=[1.15, 1])
    left.suptitle("What the camera sees: row = added first, column = added second\n"
                  "(diagonal = that stock alone; left square in each frame = white reference, dark band = black stripe)",
                  fontsize=10)
    axes = left.subplots(4, 4)
    for i, a in enumerate("ABCD"):
        for j, b in enumerate("ABCD"):
            if a == b:
                frame, res = alone[a]
                show_frame(axes[i, j], frame, f"{a} alone: {names[a]}\nd = {res.d_blank:.1f}", "0.35")
            else:
                frame, res = frames[(a, b)]
                cls = res.predicted_class
                show_frame(axes[i, j], frame, f"{a} then {b}: {cls}\nd = {res.d_blank:.1f}", CLASS_COLOURS[cls])

    gs = right.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax = right.add_subplot(gs[0, :])
    for cls in ("target", "distractor", "null"):
        for session, marker in ((0, "o"), (1, "s")):
            pts = [r for r in rows if r["expected"] == cls and r["session"] == session]
            ax.scatter([p["d"] for p in pts], [p["turbidity"] for p in pts], color=CLASS_COLOURS[cls], marker=marker,
                       s=42, alpha=0.75, edgecolor="white", linewidth=0.6, label=f"{cls}, day {session + 1}")
    ax.axvline(DEFAULT_THRESHOLDS["d_null"], color="0.4", ls="--", lw=1)
    ax.axhline(DEFAULT_THRESHOLDS["turbidity_min"], color="0.4", ls="--", lw=1)
    ax.text(DEFAULT_THRESHOLDS["d_null"] + 1, 0.97, "left of this line: null", fontsize=8, color="0.3", va="top")
    ax.text(101, DEFAULT_THRESHOLDS["turbidity_min"] + 0.02, "above this line: precipitate", fontsize=8, color="0.3",
            ha="right")
    ax.set_xlim(-3, 104)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("d: colour distance from the water blank (CIELAB units)", fontsize=9)
    ax.set_ylabel("turbidity (loss of stripe contrast)", fontsize=9)
    ax.set_title("How the 36 mixes separate (colour = expected class)", fontsize=10, loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, ncol=3, loc="center", frameon=False)
    ax.grid(alpha=0.2)

    ax = right.add_subplot(gs[1, 0])
    cm = np.array([[sum(r["expected"] == e and r["predicted"] == p for r in rows) for p in CLASSES] for e in CLASSES[:3]])
    ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max())
    for i in range(3):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(4), CLASSES, fontsize=8)
    ax.set_yticks(range(3), CLASSES[:3], fontsize=8)
    ax.set_xlabel("predicted from the frame", fontsize=9)
    ax.set_ylabel("expected", fontsize=9)
    ax.set_title("Confusion matrix", fontsize=10, loc="left", fontweight="bold")

    ax = right.add_subplot(gs[1, 1])
    labels = []
    for k, (a, b) in enumerate(itertools.combinations("ABCD", 2)):
        fwd = [r["d"] for r in rows if (r["first"], r["second"]) == (a, b)]
        rev = [r["d"] for r in rows if (r["first"], r["second"]) == (b, a)]
        ax.scatter(fwd, [k + 0.13] * len(fwd), color="#2c3e50", s=26, label=f"first-letter reagent first" if k == 0 else None)
        ax.scatter(rev, [k - 0.13] * len(rev), color="#e67e22", s=26, marker="D", label="reversed order" if k == 0 else None)
        labels.append(f"{a}+{b}")
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("d", fontsize=9)
    ax.set_title("Order of addition", fontsize=10, loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.grid(alpha=0.2, axis="x")

    acc = np.mean([r["expected"] == r["predicted"] for r in rows])
    fig.suptitle(f"Stage 1 in the digital twin: discrimination and order invariance "
                 f"({len(rows)} simulated mixes, accuracy {acc:.2f}; replicate scatter is assumed, not measured)", fontsize=12)
    fig.savefig(out / "stage1_overview.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'stage1_overview.png'}")


def stage2(seed: int, out: Path) -> None:
    store = tempfile.mkdtemp()
    lab = Lab(seed=seed, store=store)
    series = ["A4", "A3", "A2", "A1"]
    conc = {r: lab.config.reagents[r].species["Fe3"] / 2 for r in series}      # post-mix mol/L
    rows, frames = [], {}
    for session in (0, 1):
        if session:
            lab.new_session()
        for r in series:
            for _ in range(5):
                res = lab.dispatch(pair_recipe(r, "B"))
                rows.append({"reagent": r, "x": conc[r] * 1e6, "session": session, "d": res.d_blank})
                frames.setdefault(r, (lab.store.frame(res.run_id), res))
    res_w = lab.dispatch(Recipe("water", [Add("W", 2.0), Image(5)]))
    res_a = lab.dispatch(pair_recipe("A", "B"))

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[0.8, 1.6])
    top.suptitle("The dilution series as the camera sees it (KSCN fixed, iron increasing left to right)", fontsize=10)
    axes = top.subplots(1, 6)
    show_frame(axes[0], lab.store.frame(res_w.run_id), "water blank\nd = 0 by definition")
    for ax, r in zip(axes[1:5], series):
        frame, res = frames[r]
        show_frame(ax, frame, f"{r}: {conc[r] * 1e6:.0f} µM Fe after mixing\nd = {res.d_blank:.1f}")
    clip = res_a.features[-1].clipped_fraction
    show_frame(axes[5], lab.store.frame(res_a.run_id),
               f"reagent A: 1000 µM Fe (Stage 1)\nd = {res_a.d_blank:.1f}, {clip:.0%} of pixels clipped", "#b00020")

    ax1, ax2 = bottom.subplots(1, 2)
    x = np.array([r["x"] for r in rows])
    y = np.array([r["d"] for r in rows])
    for session, marker, colour in ((0, "o", "#2c3e50"), (1, "s", "#e67e22")):
        sel = [r for r in rows if r["session"] == session]
        ax1.scatter([r["x"] + (session - 0.5) * 3 for r in sel], [r["d"] for r in sel], marker=marker, color=colour,
                    s=30, alpha=0.8, label=f"day {session + 1} (5 replicates per level)")
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(0, x.max() * 1.08, 50)
    ax1.plot(xs, slope * xs + intercept, color="0.4", lw=1, ls="--", label="straight-line fit")
    sd = np.sqrt(np.mean([np.var([r["d"] for r in rows if r["reagent"] == s and r["session"] == k], ddof=1)
                          for s in series for k in (0, 1)]))
    r2 = 1 - ((y - (slope * x + intercept)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    ax1.text(0.97, 0.05, f"slope {slope:.3f} d per µM\nR² = {r2:.3f}\nwithin-day SD of d = {sd:.2f}\n"
             f"smallest detectable difference\n(3 × SD / slope) = {3 * sd / slope:.1f} µM",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.8"))
    ax1.set_xlabel("iron concentration after mixing (µM)", fontsize=9)
    ax1.set_ylabel("d: colour distance from the water blank", fontsize=9)
    ax1.set_title("Calibration line and noise floor", fontsize=10, loc="left", fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left", frameon=False)
    ax1.grid(alpha=0.2)

    grid = np.logspace(-5, np.log10(2e-2), 28)                  # Fe stock, mol/L
    scan = [response(fe, 0.02) for fe in grid]
    d = np.array([s[0].d_blank for s in scan])
    clipped = np.array([s[0].features[-1].clipped_fraction > 0.05 for s in scan])
    post = grid / 2 * 1e6
    ax2.plot(post, d, color="0.6", lw=1)
    ax2.scatter(post[~clipped], d[~clipped], color="#1e7b34", s=22, zorder=3, label="all channels within range")
    ax2.scatter(post[clipped], d[clipped], color="#b00020", s=22, marker="x", zorder=3, label="a colour channel is clipped")
    ax2.axvspan(min(conc.values()) * 1e6, max(conc.values()) * 1e6, color="#1e7b34", alpha=0.12)
    ax2.text(np.sqrt(min(conc.values()) * max(conc.values())) * 1e6, 4, "Stage 2\nseries", ha="center", fontsize=8,
             color="#1e7b34")
    for xv, label in ((1000, "reagent A\n(Stages 1, 3)"), (10000, "first PoC\ndraft")):
        ax2.axvline(xv, color="0.4", ls=":", lw=1)
        ax2.text(xv * 0.93, 30, label, ha="right", fontsize=8, color="0.3")
    ax2.set_xscale("log")
    ax2.set_xlabel("iron concentration after mixing (µM, log scale)", fontsize=9)
    ax2.set_ylabel("d (noise switched off)", fontsize=9)
    ax2.set_title("Why these concentrations: the response flattens once a channel clips", fontsize=10, loc="left",
                  fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left", frameon=False)
    ax2.grid(alpha=0.2)

    fig.suptitle("Stage 2 in the digital twin: noise floor on the iron dilution series "
                 "(simulated; the noise level is assumed, not measured)", fontsize=12)
    fig.savefig(out / "stage2_overview.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'stage2_overview.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stage1(a.seed, out)
    stage2(a.seed, out)


if __name__ == "__main__":
    main()
