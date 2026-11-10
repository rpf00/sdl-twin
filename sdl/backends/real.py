"""Real-hardware backend. TBD until the arm, dispenser and camera specs are known.

See docs/hardware_TBD.md for the list of open questions. Implementing these five
methods (and updating sdl/vision/layout.py with the real camera regions) is all
that is needed to run every experiment script and the agent on the real lab.
"""
from __future__ import annotations

import numpy as np

_MSG = "RealBackend.{} is TBD: waiting for the {} specs (see docs/hardware_TBD.md)"


class RealBackend:
    name = "real"

    def __init__(self, **hardware_config):
        self.hardware_config = hardware_config

    def start_session(self, session: int) -> None:
        raise NotImplementedError(_MSG.format("start_session", "arm and camera"))

    def add(self, vial: str, reagent: str, volume_ml: float) -> None:
        raise NotImplementedError(_MSG.format("add", "arm and dispenser"))

    def stir(self, vial: str, seconds: float) -> None:
        raise NotImplementedError(_MSG.format("stir", "arm and stirrer"))

    def image(self, vial: str, delay_s: float) -> np.ndarray:
        raise NotImplementedError(_MSG.format("image", "camera"))

    def discard(self, vial: str) -> None:
        raise NotImplementedError(_MSG.format("discard", "arm"))

    def describe(self) -> dict:
        return {"backend": self.name, "hardware": "TBD"}
