"""Data twin, part 2: pull the emulator towards reality.

Given trace records from the real lab, estimate
  - the effective optical path length (the scale of all absorbances), and
  - the replicate noise actually observed,
and report how far emulator predictions are from the measurements.

This is intentionally a small first version (one fitted parameter). Extend it as
real data shows which other priors are wrong.
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict

import numpy as np

from sdl.config import Config
from sdl.emulator.params import EmulatorParams
from sdl.schema.recipe import Recipe


def predict_lab(config: Config, recipe: Recipe, params: EmulatorParams) -> np.ndarray:
    """Noise-free emulator prediction of the final image's (L, a, b)."""
    from sdl.backends.sim import SimBackend
    from sdl.lab import Lab
    lab = Lab(backend=SimBackend(config, params.noise_free()), config=config, n_vials=4)
    return lab.dispatch(recipe).features[-1].lab()


def _groups(records: list) -> dict:
    groups = defaultdict(list)
    for r in records:
        if r.get("kind") == "mix" and r.get("accepted") and r.get("features"):
            f = r["features"][-1]
            groups[json.dumps(r["recipe"]["steps"], sort_keys=True)].append([f["L"], f["a"], f["b"]])
    return {k: np.array(v) for k, v in groups.items()}


def residuals(records: list, config: Config, params: EmulatorParams) -> dict:
    """Per-recipe distance (in Lab units) between observed mean and emulator prediction."""
    out = {}
    for key, obs in _groups(records).items():
        recipe = Recipe.from_dict({"recipe_id": "fit", "steps": json.loads(key)})
        out[key] = float(np.linalg.norm(obs.mean(axis=0) - predict_lab(config, recipe, params)))
    return out


def _rmse(records, config, params) -> float:
    r = np.array(list(residuals(records, config, params).values()))
    return float(np.sqrt((r ** 2).mean())) if len(r) else float("nan")


def fit_path_length(records: list, config: Config, params: EmulatorParams,
                    bounds: tuple = (0.1, 3.0), iterations: int = 30) -> dict:
    """Golden-section search for optics.path_cm minimising the Lab residuals."""
    def loss(path):
        p = copy.deepcopy(params)
        p.optics["path_cm"] = path
        return _rmse(records, config, p)

    before = _rmse(records, config, params)
    gr = (np.sqrt(5) - 1) / 2
    lo, hi = bounds
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = loss(c), loss(d)
    for _ in range(iterations):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = loss(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = loss(d)
    best = 0.5 * (lo + hi)
    return {"path_cm": float(best), "rmse_before": before, "rmse_after": loss(best),
            "n_recipes": len(_groups(records))}


def replicate_noise(records: list) -> dict:
    """Pooled replicate standard deviation of L, a, b across recipes with >= 2 runs."""
    var, dof = np.zeros(3), 0
    for obs in _groups(records).values():
        if len(obs) >= 2:
            var += obs.var(axis=0, ddof=1) * (len(obs) - 1)
            dof += len(obs) - 1
    sd = np.sqrt(var / dof) if dof else np.full(3, np.nan)
    return {"sd_L": float(sd[0]), "sd_a": float(sd[1]), "sd_b": float(sd[2]), "dof": dof}
