"""Optics layer: equilibrium concentrations -> the colour a camera would see.

Works on a coarse spectrum (380-730 nm), applies Beer-Lambert, then integrates
against the CIE 1931 observer using the analytic fit of Wyman, Sloan & Shirley
(J. Comput. Graph. Tech. 2, 1 (2013)). A real camera's channel sensitivities are
not the CIE observer; this is an approximation to be corrected by calibration.
"""
from __future__ import annotations

import numpy as np

from sdl.emulator.chemistry import Solution
from sdl.vision.color import XYZ_TO_RGB, D65_WHITE

WL = np.arange(380.0, 731.0, 5.0)


def _lobe(wl, mu, s1, s2):
    s = np.where(wl < mu, s1, s2)
    return np.exp(-0.5 * ((wl - mu) / s) ** 2)


CMF_X = 1.056 * _lobe(WL, 599.8, 37.9, 31.0) + 0.362 * _lobe(WL, 442.0, 16.0, 26.7) - 0.065 * _lobe(WL, 501.1, 20.4, 26.2)
CMF_Y = 0.821 * _lobe(WL, 568.8, 46.9, 40.5) + 0.286 * _lobe(WL, 530.9, 16.3, 31.1)
CMF_Z = 1.217 * _lobe(WL, 437.0, 11.8, 36.0) + 0.681 * _lobe(WL, 459.0, 26.0, 13.8)
_CMF = np.stack([CMF_X, CMF_Y, CMF_Z])
_WHITE_E = _CMF.sum(axis=1)   # XYZ of a flat spectrum (before white balancing)


def spectrum_to_linear_rgb(spec: np.ndarray) -> np.ndarray:
    """Relative spectrum (1.0 = backdrop white) -> linear sRGB with white at (1, 1, 1)."""
    xyz = (_CMF @ spec) / _WHITE_E * D65_WHITE   # white-balanced to D65
    return np.clip(XYZ_TO_RGB @ xyz, 0.0, None)


def absorbance_spectrum(sol: Solution, o: dict, scale: float = 1.0) -> tuple:
    """Return (absorbance of dissolved species, absorbance of precipitate) per cm."""
    gauss = lambda lam, sig: np.exp(-0.5 * ((WL - lam) / sig) ** 2)
    a = scale * (
        sol.fescn * o["fescn_eps_max"] * gauss(o["fescn_lambda_nm"], o["fescn_sigma_nm"])
        + sol.fescn2 * o["fescn2_eps_max"] * gauss(o["fescn2_lambda_nm"], o["fescn2_sigma_nm"])
    )
    a = a + sol.fe_free * o["fe_eps_400"] * np.exp(-(WL - 400.0) / o["fe_decay_nm"])
    a_ppt = sol.feoh3 * o["ppt_eps_400"] * np.exp(-(WL - 400.0) / o["ppt_decay_nm"])
    return a, a_ppt


def vial_colours(sol: Solution, o: dict, signal_scale: float = 1.0) -> dict:
    """Linear RGB of the vial seen against the white backdrop and against the black stripe.

    signal_scale multiplies the complex absorbance (incomplete mixing, fading).
    """
    path = o["path_cm"]
    a, a_ppt = absorbance_spectrum(sol, o, signal_scale)
    transmit = 10.0 ** (-path * (a + a_ppt))
    scatter = 1.0 - np.exp(-o["ppt_scatter_per_M_cm"] * sol.feoh3 * path)
    particle = o["ppt_reflectance"] * 10.0 ** (-path * a_ppt)   # colour of light scattered back
    t_rgb = spectrum_to_linear_rgb(transmit)
    p_rgb = spectrum_to_linear_rgb(particle)
    return {"transmit": t_rgb, "particle": p_rgb, "scatter": float(scatter),
            "peak_absorbance": float(path * (a + a_ppt).max())}
