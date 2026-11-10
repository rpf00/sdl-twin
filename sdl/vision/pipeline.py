"""Image pipeline: frame -> white balance -> ROI -> CIELAB features.

This code is backend-agnostic: it runs unchanged on emulator frames and on real
camera frames (uint8 sRGB arrays of shape HxWx3).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from sdl.vision.color import linear_rgb_to_lab, srgb_to_linear
from sdl.vision.layout import Layout

LUMA = np.array([0.2126729, 0.7151522, 0.0721750])


@dataclass
class Features:
    L: float
    a: float
    b: float
    hue_deg: float       # 0 = red, 90 = yellow, 180 = green, 270 = blue
    chroma: float
    contrast: float      # stripe contrast seen through the liquid
    turbidity: float     # 0 = clear, 1 = opaque; relative to the reference contrast
    clipped_fraction: float  # share of ROI pixels at the sensor floor or ceiling

    def lab(self) -> np.ndarray:
        return np.array([self.L, self.a, self.b])

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else round(float(v), 4))
                for k, v in asdict(self).items()}


def _crop(img, box):
    r0, r1, c0, c1 = box
    return img[r0:r1, c0:c1]


def extract_features(image: np.ndarray, layout: Layout = Layout(),
                     reference_contrast: float | None = None) -> Features:
    raw = np.asarray(image)
    lin = srgb_to_linear(raw.astype(float) / 255.0)

    # White balance: scale each channel so the reference patch is (1, 1, 1).
    white = _crop(lin, layout.white_patch).reshape(-1, 3).mean(axis=0)
    lin = lin / np.maximum(white, 1e-6)

    roi = _crop(lin, layout.colour_roi).reshape(-1, 3)
    rgb = roi.mean(axis=0)
    L, a, b = linear_rgb_to_lab(rgb)
    chroma = math.hypot(a, b)
    hue = math.degrees(math.atan2(b, a)) % 360.0

    y_white = float(rgb @ LUMA)
    y_black = float(_crop(lin, layout.stripe_roi).reshape(-1, 3).mean(axis=0) @ LUMA)
    contrast = (y_white - y_black) / max(y_white + y_black, 1e-6)
    if reference_contrast is None or y_white < 0.02:   # too dark to judge turbidity
        turbidity = 0.0 if reference_contrast is None else float("nan")
    else:
        turbidity = float(np.clip(1.0 - contrast / reference_contrast, 0.0, 1.0))

    raw_roi = _crop(raw, layout.colour_roi).reshape(-1, 3)
    clipped = float(((raw_roi <= 2) | (raw_roi >= 253)).any(axis=1).mean())
    return Features(float(L), float(a), float(b), hue, chroma, float(contrast), turbidity, clipped)


def delta_e(f1: Features, f2: Features) -> float:
    """CIE76 colour difference: d = sqrt(dL^2 + da^2 + db^2)."""
    return float(np.linalg.norm(f1.lab() - f2.lab()))


DEFAULT_THRESHOLDS = {   # TBD: replace with values derived from the Stage 2 noise floor
    "d_null": 5.0,           # below this colour change -> null
    "turbidity_min": 0.25,   # above this -> precipitate
    "hue_lo_deg": 15.0,      # target hue window (orange-red)
    "hue_hi_deg": 90.0,
    "chroma_min": 8.0,
}


def classify(post: Features, d_blank: float, thresholds: dict | None = None) -> str:
    """Rule-based class: target | distractor | null | other."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if not math.isnan(post.turbidity) and post.turbidity >= t["turbidity_min"]:
        return "distractor"
    if d_blank < t["d_null"]:
        return "null"
    if t["hue_lo_deg"] <= post.hue_deg <= t["hue_hi_deg"] and post.chroma >= t["chroma_min"]:
        return "target"
    return "other"
