"""Entities tracked by the state twin."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Vial:
    id: str
    capacity_ml: float
    status: str = "clean"          # clean | in_use | discarded
    volume_ml: float = 0.0
    additions: list = field(default_factory=list)   # [[reagent_id, volume_ml], ...] in order
    species_mol: dict = field(default_factory=dict)  # nominal moles per species
    stirred: bool = False          # stirred since the last addition
    n_images: int = 0

    def reagents(self) -> list:
        return [r for r, _ in self.additions]


@dataclass
class LabState:
    stocks_ml: dict = field(default_factory=dict)   # reagent_id -> remaining volume
    vials: dict = field(default_factory=dict)       # vial_id -> Vial
    session: int = 0               # "day" counter
    clock_s: float = 0.0           # simulated lab clock
    mixes_used: int = 0
    next_vial: int = 0

    def clean_vials(self) -> list:
        return [v.id for v in self.vials.values() if v.status == "clean"]
