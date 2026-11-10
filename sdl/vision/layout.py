"""Where things are in the camera frame.

TBD: these regions are placeholders for the synthetic frame. Once the real camera
is mounted, measure the regions on a real image and put them here (or load them
from a config). The emulator draws its frames from the same layout, so the two
always agree.

Assumed scene: white backdrop, a white reference patch permanently in frame, and
a black stripe on the backdrop behind the vial. The stripe is what makes
turbidity measurable: a clear coloured liquid keeps the stripe's contrast, a
cloudy one washes it out.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    height: int = 80
    width: int = 120
    # (row0, row1, col0, col1), end-exclusive
    white_patch: tuple = (12, 68, 8, 36)
    vial: tuple = (8, 72, 52, 112)          # full vial area
    colour_roi: tuple = (14, 30, 60, 104)   # liquid in front of the white backdrop
    stripe: tuple = (36, 46, 52, 112)       # where the black stripe is drawn
    stripe_roi: tuple = (39, 43, 60, 104)   # liquid in front of the black stripe
