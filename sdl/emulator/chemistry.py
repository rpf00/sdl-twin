"""Chemistry layer: what is in the vial once the reagents have met.

Deliberately small. It covers the three behaviours the PoC needs:
  1. Fe3+ + SCN-  <=> FeSCN2+ (+ SCN- <=> Fe(SCN)2+)   target, red
  2. Fe3+ + 3 OH- -> Fe(OH)3(s)                        distractor, brown and turbid
  3. everything else                                    null
Formation constants are corrected for ionic strength with the Davies equation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

CHARGE = {"Fe3": 3, "H": 1, "K": 1, "Na": 1, "Cl": -1, "SCN": -1, "OH": -1, "NO3": -1}
DAVIES_A = 0.509  # water, 25 C


@dataclass
class Solution:
    """Equilibrium concentrations in mol/L."""
    fe_free: float = 0.0
    scn_free: float = 0.0
    fescn: float = 0.0      # FeSCN2+
    fescn2: float = 0.0     # Fe(SCN)2+
    feoh3: float = 0.0      # precipitated Fe, mol per litre of suspension
    ionic_strength: float = 0.0
    k1: float = 0.0
    k2: float = 0.0


def davies_term(ionic_strength: float) -> float:
    i = min(max(ionic_strength, 0.0), 0.5)   # Davies is only reasonable up to ~0.5 M
    s = math.sqrt(i)
    return DAVIES_A * (s / (1 + s) - 0.3 * i)


def formation_constants(ionic_strength: float, log_k1_0: float, log_k2_0: float) -> tuple:
    d = davies_term(ionic_strength)
    # Fe3+ + SCN- -> FeSCN2+ : delta z^2 = 4 - 9 - 1 = -6
    # FeSCN2+ + SCN- -> Fe(SCN)2+ : delta z^2 = 1 - 4 - 1 = -4
    return 10 ** (log_k1_0 - 6 * d), 10 ** (log_k2_0 - 4 * d)


def equilibrate(species_mol: dict, volume_ml: float, chem: dict) -> Solution:
    if volume_ml <= 0:
        return Solution()
    c = {k: v / (volume_ml * 1e-3) for k, v in species_mol.items()}
    fe, scn = c.get("Fe3", 0.0), c.get("SCN", 0.0)
    h, oh = c.get("H", 0.0), c.get("OH", 0.0)

    # Acid-base first; excess hydroxide then precipitates Fe(OH)3.
    net_oh = oh - h
    feoh3 = 0.0
    if net_oh > 0:
        feoh3 = min(fe, net_oh / 3.0) * chem.get("precipitation_fraction", 1.0)
        c["OH"], c["H"] = max(net_oh - 3 * feoh3, 0.0), 0.0
    else:
        c["H"], c["OH"] = -net_oh, 0.0
    fe_avail = fe - feoh3
    c["Fe3"] = fe_avail

    ionic = 0.5 * sum(conc * CHARGE.get(sp, 0) ** 2 for sp, conc in c.items())
    k1, k2 = formation_constants(ionic, chem.get("log_k1_zero_ionic", 2.85), chem.get("log_k2_zero_ionic", 1.51))

    # Solve the mass balances for free thiocyanate s by bisection.
    def fe_free_of(s):
        return fe_avail / (1 + k1 * s + k1 * k2 * s * s)

    lo, hi = 0.0, scn
    for _ in range(80):
        s = 0.5 * (lo + hi)
        bound = fe_free_of(s) * (k1 * s + 2 * k1 * k2 * s * s)
        if s + bound > scn:
            hi = s
        else:
            lo = s
    s = 0.5 * (lo + hi)
    ff = fe_free_of(s)
    return Solution(ff, s, k1 * ff * s, k1 * k2 * ff * s * s, feoh3, ionic, k1, k2)
