"""Camera layer: vial colours -> a synthetic uint8 sRGB frame with realistic nuisances."""
from __future__ import annotations

import numpy as np

from sdl.vision.color import linear_to_srgb
from sdl.vision.layout import Layout


class SessionConditions:
    """Slow drifts that are fixed within a session ("day") and change between sessions."""

    def __init__(self, cam: dict, rng: np.random.Generator):
        self.gain = 1.0 + rng.normal(0, cam["session_gain_sd"], 3) if cam["session_gain_sd"] else np.ones(3)
        g = rng.normal(0, cam["session_gradient_sd"]) if cam["session_gradient_sd"] else 0.0
        gc = rng.normal(0, cam["session_gradient_chroma_sd"], 3) if cam["session_gradient_chroma_sd"] else np.zeros(3)
        # Illumination at the vial relative to the white patch: not removed by white balance.
        self.vial_illum = (1.0 + g) * (1.0 + gc)


def render_frame(colours: dict, cam: dict, session: SessionConditions,
                 rng: np.random.Generator, layout: Layout = Layout()) -> np.ndarray:
    h, w = layout.height, layout.width
    white = cam["white_level"]
    frame = np.full((h, w, 3), 0.9 * white)            # backdrop
    r0, r1, c0, c1 = layout.white_patch
    frame[r0:r1, c0:c1] = white

    jit = int(cam.get("roi_jitter_px", 0))
    dr, dc = (rng.integers(-jit, jit + 1, 2) if jit else (0, 0))

    t, p, s = colours["transmit"], colours["particle"], colours["scatter"]
    through_white = (1 - s) * t + s * p
    through_black = (1 - s) * t * cam["black_stripe_reflectance"] + s * p
    r0, r1, c0, c1 = layout.vial
    frame[r0 + dr:r1 + dr, c0 + dc:c1 + dc] = white * session.vial_illum * through_white
    r0, r1, c0, c1 = layout.stripe
    frame[r0 + dr:r1 + dr, c0 + dc:c1 + dc] = white * session.vial_illum * through_black

    flicker = 1.0 + (rng.normal(0, cam["flicker_sd"]) if cam["flicker_sd"] else 0.0)
    frame = frame * session.gain * flicker
    if cam["read_noise_sd"]:
        frame = frame + rng.normal(0, cam["read_noise_sd"], frame.shape)
    return np.round(linear_to_srgb(frame) * 255).astype(np.uint8)
