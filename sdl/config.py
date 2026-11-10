"""Loading of the YAML configuration files."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@dataclass(frozen=True)
class Reagent:
    id: str
    name: str
    species: dict  # species -> mol/L in the stock
    tags: tuple = ()


@dataclass(frozen=True)
class Limits:
    vial_capacity_ml: float = 4.0
    min_dispense_ml: float = 0.05
    max_dispense_ml: float = 2.0
    stock_volume_ml: float = 50.0
    max_stir_s: float = 120.0


@dataclass(frozen=True)
class Safety:
    forbidden_tag_pairs: tuple = ()
    banned_tags: tuple = ()


@dataclass
class Config:
    reagents: dict
    limits: Limits
    safety: Safety
    emulator: dict = field(default_factory=dict)

    def hash(self) -> str:
        blob = json.dumps(
            {
                "reagents": {k: [v.species, list(v.tags)] for k, v in self.reagents.items()},
                "limits": self.limits.__dict__,
                "safety": [list(map(list, self.safety.forbidden_tag_pairs)), list(self.safety.banned_tags)],
                "emulator": self.emulator,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _read(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_config(config_dir: str | Path | None = None) -> Config:
    d = Path(config_dir) if config_dir else CONFIG_DIR
    raw = _read(d / "reagents.yaml")
    reagents = {
        rid: Reagent(rid, r.get("name", rid), dict(r.get("species") or {}), tuple(r.get("tags") or ()))
        for rid, r in raw["reagents"].items()
    }
    limits = Limits(**(raw.get("limits") or {}))
    s = _read(d / "safety.yaml")
    safety = Safety(
        tuple(tuple(p) for p in s.get("forbidden_tag_pairs", [])),
        tuple(s.get("banned_tags", [])),
    )
    return Config(reagents, limits, safety, _read(d / "emulator_priors.yaml"))
