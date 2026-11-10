"""Typed access to configs/emulator_priors.yaml."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class EmulatorParams:
    chemistry: dict = field(default_factory=dict)
    optics: dict = field(default_factory=dict)
    camera: dict = field(default_factory=dict)
    faults: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "EmulatorParams":
        d = copy.deepcopy(d)
        return cls(d.get("chemistry", {}), d.get("optics", {}), d.get("camera", {}), d.get("faults", {}))

    def noise_free(self) -> "EmulatorParams":
        """Copy with every stochastic term switched off (used for calibration fits)."""
        p = copy.deepcopy(self)
        for k in ("session_gain_sd", "session_gradient_sd", "session_gradient_chroma_sd",
                  "flicker_sd", "read_noise_sd"):
            p.camera[k] = 0.0
        p.camera["roi_jitter_px"] = 0
        p.faults["volume_cv"] = 0.0
        p.chemistry["unstirred_extent_sd"] = 0.0
        return p
