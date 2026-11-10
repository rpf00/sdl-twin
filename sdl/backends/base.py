"""The contract every lab backend fulfils. The Lab only ever calls these methods,
and only after the state twin has verified the whole recipe."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Backend(Protocol):
    name: str

    def start_session(self, session: int) -> None: ...
    def add(self, vial: str, reagent: str, volume_ml: float) -> None: ...
    def stir(self, vial: str, seconds: float) -> None: ...
    def image(self, vial: str, delay_s: float) -> np.ndarray: ...   # uint8 sRGB, HxWx3
    def discard(self, vial: str) -> None: ...
    def describe(self) -> dict: ...
