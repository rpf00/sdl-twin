"""Scalar reward from the image readout: big colour change, of the target hue, in a clear liquid."""
from __future__ import annotations

import math

DEFAULT = {"d_scale": 100.0, "hue_target_deg": 45.0, "hue_sigma_deg": 30.0}


def reward(d_blank: float, hue_deg: float, turbidity: float, cfg: dict | None = None) -> float:
    c = {**DEFAULT, **(cfg or {})}
    size = min(max(d_blank / c["d_scale"], 0.0), 1.0)
    dh = (hue_deg - c["hue_target_deg"] + 180.0) % 360.0 - 180.0
    hue_match = math.exp(-0.5 * (dh / c["hue_sigma_deg"]) ** 2)
    clarity = 1.0 - (0.0 if turbidity is None or math.isnan(turbidity) else turbidity)
    return float(size * hue_match * clarity)
