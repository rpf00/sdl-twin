"""Stages 1 and 2 under three noise scenarios (optimistic, typical, pessimistic).

The emulator's noise is an assumption. This script shows how much the headline numbers
move across literature-anchored assumptions, and compares the simulated noise floor with
detection limits published for camera-based iron-thiocyanate methods.

    python experiments/noise_scenarios.py
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import yaml

from sdl import Lab
from sdl.backends import SimBackend
from sdl.config import CONFIG_DIR, load_config
from sdl.emulator import EmulatorParams
from sdl.schema import pair_recipe
from sdl.vision.pipeline import delta_e, extract_features

EXPECTED = {frozenset("AB"): "target", frozenset("AC"): "distractor"}
SERIES = ["A1", "A2", "A3", "A4"]


def params_for(cfg, overrides: dict) -> EmulatorParams:
    p = EmulatorParams.from_dict(copy.deepcopy(cfg.emulator))
    for section, values in overrides.items():
        getattr(p, section).update(values)
    return p


def camera_repeatability(cfg, p, seed) -> float:
    """Median colour difference between two frames of the same vial."""
    b = SimBackend(cfg, p, seed=seed)
    b.add("v", "A2", 1.0); b.add("v", "B", 1.0); b.stir("v", 10)
    f = [extract_features(b.image("v", 0)) for _ in range(200)]
    return float(np.median([delta_e(f[i], f[i + 1]) for i in range(0, 200, 2)]))


def stage2(cfg, p, seed) -> dict:
    lab = Lab(backend=SimBackend(cfg, p, seed=seed), config=cfg, n_vials=60)
    # run day by day so that both days share one session each
    data = {rid: {0: [], 1: []} for rid in SERIES}
    for day in (0, 1):
        if day:
            lab.new_session()
        for rid in SERIES:
            for _ in range(5):
                data[rid][day].append(lab.dispatch(pair_recipe(rid, "B")).d_blank)
    x, y, var, shift, cv = [], [], [], [], []
    for rid in SERIES:
        conc = cfg.reagents[rid].species["Fe3"] / 2 * 1e6
        d0, d1 = np.array(data[rid][0]), np.array(data[rid][1])
        within = (d0.var(ddof=1) + d1.var(ddof=1)) / 2
        var.append(within)
        cv.append(100 * np.sqrt(within) / np.concatenate([d0, d1]).mean())
        shift.append(abs(d1.mean() - d0.mean()))
        x += [conc] * 10
        y += list(d0) + list(d1)
    slope = np.polyfit(x, y, 1)[0]
    sd = float(np.sqrt(np.mean(var)))
    return {"sd": sd, "cv": float(np.mean(cv)), "mdd": 3 * sd / slope, "day_shift": float(np.mean(shift))}


def stage1_accuracy(cfg, p, seed) -> float:
    lab = Lab(backend=SimBackend(cfg, p, seed=seed), config=cfg, n_vials=60)
    hits = []
    for rep in range(3):
        if rep == 2:
            lab.new_session()
        for a, b in itertools.permutations("ABCD", 2):
            res = lab.dispatch(pair_recipe(a, b))
            hits.append(res.predicted_class == EXPECTED.get(frozenset((a, b)), "null"))
    return float(np.mean(hits))


def main():
    cfg = load_config()
    with open(CONFIG_DIR / "noise_scenarios.yaml") as fh:
        raw = yaml.safe_load(fh)
    lit = raw.pop("literature_detection_limits_uM")
    seeds = range(5)
    print(f"{'scenario':13s}{'camera repeat dE':>18s}{'SD of d':>9s}{'CV %':>7s}{'day shift':>11s}"
          f"{'detectable dFe (uM)':>21s}{'Stage 1 accuracy':>18s}")
    for name, overrides in raw.items():
        p = params_for(cfg, overrides)
        rep = np.mean([camera_repeatability(cfg, p, s) for s in seeds])
        s2 = [stage2(cfg, p, s) for s in seeds]
        acc = np.mean([stage1_accuracy(cfg, p, s) for s in seeds])
        mean = lambda k: float(np.mean([r[k] for r in s2]))
        print(f"{name:13s}{rep:18.2f}{mean('sd'):9.2f}{mean('cv'):7.1f}{mean('day_shift'):11.2f}"
              f"{mean('mdd'):21.1f}{acc:18.3f}")
    print("\naveraged over 5 seeds; detectable dFe = 3 x within-day SD / slope of d against iron")
    print("\nPublished detection limits for camera-based iron-thiocyanate methods (different setups, for scale):")
    for k, v in lit.items():
        print(f"  {v:4.1f} uM   {k}")


if __name__ == "__main__":
    main()
