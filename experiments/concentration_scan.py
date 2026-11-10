"""Design aid: predicted camera response against Fe stock concentration.

Answers "which concentrations can the camera actually resolve?" before any
chemical is opened. Noise is switched off; this shows the mean response only.
The last line reproduces the first PoC draft (0.02 M Fe + 0.1 M KSCN).

    python experiments/concentration_scan.py
"""
from __future__ import annotations

import copy

from sdl import Lab
from sdl.backends import SimBackend
from sdl.config import Reagent, load_config
from sdl.emulator import EmulatorParams
from sdl.schema import pair_recipe


def response(fe_stock: float, scn_stock: float):
    cfg = copy.deepcopy(load_config())
    cfg.reagents["X"] = Reagent("X", "scan Fe", {"Fe3": fe_stock, "H": 0.01, "Cl": 0.01 + 3 * fe_stock}, ("iron_iii",))
    cfg.reagents["Y"] = Reagent("Y", "scan KSCN", {"SCN": scn_stock, "K": scn_stock}, ("thiocyanate",))
    backend = SimBackend(cfg, EmulatorParams.from_dict(cfg.emulator).noise_free())
    lab = Lab(backend=backend, config=cfg, n_vials=4)
    res = lab.dispatch(pair_recipe("X", "Y"))
    return res, backend.last_truth


def main():
    print(f"{'Fe stock M':>11s}{'KSCN stock M':>14s}{'peak A':>8s}{'d':>8s}{'hue':>6s}{'clipped':>9s}  verdict")
    grid = [(fe, 0.02) for fe in (2e-5, 5e-5, 1e-4, 2e-4, 4e-4, 1e-3, 2e-3, 5e-3, 2e-2)] + [(2e-2, 0.1)]
    prev_d = None
    for fe, scn in grid:
        res, truth = response(fe, scn)
        f = res.features[-1]
        if f.clipped_fraction > 0.05:
            verdict = "a channel is clipped: fine for yes/no, useless for quantification"
        elif res.d_blank < 5:
            verdict = "near the expected noise floor"
        else:
            verdict = "usable for quantification"
        print(f"{fe:11.1e}{scn:14.2f}{truth['peak_absorbance']:8.2f}{res.d_blank:8.1f}{f.hue_deg:6.0f}"
              f"{f.clipped_fraction:9.2f}  {verdict}")
        prev_d = res.d_blank


if __name__ == "__main__":
    main()
