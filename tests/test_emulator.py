import numpy as np

from sdl.emulator.chemistry import equilibrate, formation_constants
from sdl.emulator.optics import spectrum_to_linear_rgb, WL
from sdl.vision.color import linear_rgb_to_lab

CHEM = {"log_k1_zero_ionic": 2.85, "log_k2_zero_ionic": 1.51, "precipitation_fraction": 1.0}


def test_mass_balance_and_constants():
    mol = {"Fe3": 1e-6, "SCN": 2e-5, "K": 2e-5, "H": 1e-5, "Cl": 1.3e-5}   # in 2 mL
    s = equilibrate(mol, 2.0, CHEM)
    assert np.isclose(s.fe_free + s.fescn + s.fescn2, 5e-4, rtol=1e-6)
    assert np.isclose(s.scn_free + s.fescn + 2 * s.fescn2, 1e-2, rtol=1e-6)
    assert np.isclose(s.fescn / (s.fe_free * s.scn_free), s.k1, rtol=1e-6)
    assert np.isclose(s.fescn2 / (s.fescn * s.scn_free), s.k2, rtol=1e-6)


def test_ionic_strength_lowers_k():
    k1_0, k2_0 = formation_constants(0.0, 2.85, 1.51)
    k1_i, k2_i = formation_constants(0.1, 2.85, 1.51)
    assert np.isclose(k1_0, 10 ** 2.85) and k1_i < k1_0 and k2_i < k2_0


def test_excess_hydroxide_precipitates_all_iron():
    s = equilibrate({"Fe3": 2e-6, "H": 1e-5, "OH": 1e-4}, 2.0, CHEM)
    assert np.isclose(s.feoh3, 1e-3) and s.fe_free < 1e-12 and s.fescn == 0


def test_acid_excess_prevents_precipitation():
    s = equilibrate({"Fe3": 2e-6, "H": 1e-4, "OH": 1e-5}, 2.0, CHEM)
    assert s.feoh3 == 0 and np.isclose(s.fe_free, 1e-3)


def test_flat_spectrum_is_white():
    rgb = spectrum_to_linear_rgb(np.ones_like(WL))
    assert np.allclose(rgb, 1.0, atol=1e-6)
    assert np.allclose(linear_rgb_to_lab(rgb), [100, 0, 0], atol=1e-3)
