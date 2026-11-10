"""Simulated backend: the outcome emulator behind the Backend interface.

It keeps its own hidden "true" state of every vial (with dispensing errors and
imperfect mixing), separate from the state twin's nominal belief. The gap between
the two is exactly what the real lab will have and what the agent must cope with.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from sdl.config import Config
from sdl.emulator.camera import SessionConditions, render_frame
from sdl.emulator.chemistry import equilibrate
from sdl.emulator.optics import vial_colours
from sdl.emulator.params import EmulatorParams
from sdl.vision.layout import Layout


class _TrueVial:
    def __init__(self):
        self.species_mol: dict = {}
        self.volume_ml = 0.0
        self.order: list = []
        self.extent = 1.0        # how completely the contents are mixed/reacted
        self.t_since_mix = 0.0


class SimBackend:
    name = "sim"

    def __init__(self, config: Config, params: EmulatorParams | None = None, seed: int = 0,
                 layout: Layout = Layout()):
        self.config = config
        self.params = params or EmulatorParams.from_dict(config.emulator)
        self.seed = seed
        self.layout = layout
        self.rng = np.random.default_rng(seed)
        self.vials: dict = {}
        self.last_truth: dict = {}
        self.start_session(0)

    def start_session(self, session: int) -> None:
        self.session = session
        srng = np.random.default_rng([self.seed, 7919, session])
        self.conditions = SessionConditions(self.params.camera, srng)

    def add(self, vial: str, reagent: str, volume_ml: float) -> None:
        v = self.vials.setdefault(vial, _TrueVial())
        cv = self.params.faults.get("volume_cv", 0.0)
        actual = volume_ml * (1 + self.rng.normal(0, cv)) if cv else volume_ml
        actual = max(actual, 0.0)
        for sp, conc in self.config.reagents[reagent].species.items():
            v.species_mol[sp] = v.species_mol.get(sp, 0.0) + conc * actual * 1e-3
        v.volume_ml += actual
        v.order.append(reagent)
        if len(v.order) > 1:     # a new reagent arrived: contents are unmixed until stirred
            chem = self.params.chemistry
            sd = chem.get("unstirred_extent_sd", 0.0)
            v.extent = float(np.clip(chem.get("unstirred_extent", 0.5) + (self.rng.normal(0, sd) if sd else 0), 0.05, 1))
            v.t_since_mix = 0.0

    def stir(self, vial: str, seconds: float) -> None:
        v = self.vials[vial]
        tau = self.params.chemistry.get("mix_tau_s", 2.0)
        v.extent = 1.0 - (1.0 - v.extent) * float(np.exp(-seconds / tau))
        v.t_since_mix += seconds

    def image(self, vial: str, delay_s: float) -> np.ndarray:
        v = self.vials[vial]
        v.t_since_mix += delay_s
        chem, faults = self.params.chemistry, self.params.faults
        sol = equilibrate(v.species_mol, v.volume_ml, chem)
        scale = v.extent * float(np.exp(-chem.get("fade_rate_per_M_s", 0.0) * sol.scn_free * v.t_since_mix))
        bias = faults.get("order_bias", 0.0)
        if bias and len(v.order) > 1 and "iron_iii" in self.config.reagents[v.order[-1]].tags:
            scale *= 1.0 + bias
        colours = vial_colours(sol, self.params.optics, scale)
        self.last_truth = {   # hidden from agents; stored in the trace for analysis only
            "volume_ml": v.volume_ml, "extent": v.extent, "fescn_M": sol.fescn, "fescn2_M": sol.fescn2,
            "feoh3_M": sol.feoh3, "ionic_strength": sol.ionic_strength, "k1": sol.k1,
            "peak_absorbance": colours["peak_absorbance"],
        }
        return render_frame(colours, self.params.camera, self.conditions, self.rng, self.layout)

    def discard(self, vial: str) -> None:
        self.vials.pop(vial, None)

    def describe(self) -> dict:
        blob = json.dumps(self.params.__dict__, sort_keys=True)
        return {"backend": self.name, "seed": self.seed,
                "emulator_params_hash": hashlib.sha256(blob.encode()).hexdigest()[:12]}
