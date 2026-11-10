"""Colour maths: sRGB <-> linear RGB <-> XYZ (D65) <-> CIELAB."""
from __future__ import annotations

import numpy as np

RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
XYZ_TO_RGB = np.linalg.inv(RGB_TO_XYZ)
D65_WHITE = RGB_TO_XYZ @ np.ones(3)   # (0.9505, 1.0, 1.0890)


def srgb_to_linear(v):
    v = np.asarray(v, dtype=float)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.clip(np.asarray(v, dtype=float), 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def linear_rgb_to_lab(rgb) -> np.ndarray:
    xyz = RGB_TO_XYZ @ np.asarray(rgb, dtype=float)
    t = xyz / D65_WHITE
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(np.maximum(t, 0)), t / (3 * d * d) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])
