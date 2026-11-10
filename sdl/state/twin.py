"""State twin: a persistent, queryable model of what is where in the lab.

It holds the *nominal* state (what we believe happened). The state is never
edited directly: it only changes by applying events, and every event is appended
to a log. That gives persistence, replay and cheap forking for dry runs.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

from sdl.config import Config
from sdl.schema.entities import LabState, Vial
from sdl.schema.events import Event

ADD_DURATION_S = 5.0  # nominal time per dispense. TBD: measure on the real arm


class StateTwin:
    def __init__(self, config: Config, n_vials: int = 48, log_path: str | Path | None = None):
        self.config = config
        self.n_vials_initial = n_vials
        self.state = LabState(
            stocks_ml={rid: config.limits.stock_volume_ml for rid in config.reagents},
        )
        self.events: list[Event] = []
        self.log_path = Path(log_path) if log_path else None
        self._add_vials(n_vials)

    # ---- queries -------------------------------------------------------
    def snapshot(self) -> dict:
        """Plain-dict copy of the full state (JSON serialisable)."""
        return json.loads(json.dumps(asdict(self.state)))

    def inspect(self) -> dict:
        """Compact summary intended for an agent."""
        s = self.state
        return {
            "session": s.session,
            "clock_s": round(s.clock_s, 1),
            "mixes_used": s.mixes_used,
            "clean_vials": len(s.clean_vials()),
            "stocks_ml": {k: round(v, 3) for k, v in s.stocks_ml.items()},
            "reagents": {k: r.name for k, r in self.config.reagents.items()},
            "limits": dict(self.config.limits.__dict__),
        }

    def fork(self) -> "StateTwin":
        """Independent copy for dry runs. Forks never write to the log."""
        twin = copy.copy(self)
        twin.state = copy.deepcopy(self.state)
        twin.events = list(self.events)
        twin.log_path = None
        return twin

    def next_clean_vial(self) -> str | None:
        clean = self.state.clean_vials()
        return clean[0] if clean else None

    # ---- mutation (events only) ---------------------------------------
    def apply(self, event: Event) -> Event:
        event = replace(event, seq=len(self.events))
        self._reduce(event)
        self.events.append(event)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(event.to_dict()) + "\n")
        return event

    def _add_vials(self, n: int) -> None:
        cap = self.config.limits.vial_capacity_ml
        for _ in range(n):
            vid = f"V{self.state.next_vial:04d}"
            self.state.vials[vid] = Vial(vid, cap)
            self.state.next_vial += 1

    def _reduce(self, e: Event) -> None:
        s, p = self.state, e.payload
        if e.type == "session_started":
            s.session = p["session"]
        elif e.type == "vials_restocked":
            self._add_vials(p["n"])
        elif e.type == "stock_refilled":
            s.stocks_ml[p["reagent"]] = p["volume_ml"]
        elif e.type == "vial_assigned":
            s.vials[p["vial"]].status = "in_use"
        elif e.type == "liquid_added":
            v = s.vials[p["vial"]]
            vol = p["volume_ml"]
            s.stocks_ml[p["reagent"]] -= vol
            v.volume_ml += vol
            v.additions.append([p["reagent"], vol])
            for sp, conc in self.config.reagents[p["reagent"]].species.items():
                v.species_mol[sp] = v.species_mol.get(sp, 0.0) + conc * vol * 1e-3
            v.stirred = False
            s.clock_s += ADD_DURATION_S
        elif e.type == "stirred":
            s.vials[p["vial"]].stirred = True
            s.clock_s += p["seconds"]
        elif e.type == "imaged":
            s.vials[p["vial"]].n_images += 1
            s.clock_s += p["delay_s"]
        elif e.type == "vial_discarded":
            s.vials[p["vial"]].status = "discarded"
        elif e.type == "mix_counted":
            s.mixes_used += 1

    # ---- persistence ---------------------------------------------------
    @classmethod
    def replay(cls, config: Config, events: list, n_vials: int = 48) -> "StateTwin":
        twin = cls(config, n_vials=n_vials)
        for e in events:
            twin.apply(e if isinstance(e, Event) else Event.from_dict(e))
        return twin

    @classmethod
    def from_log(cls, config: Config, log_path: str | Path, n_vials: int = 48) -> "StateTwin":
        """Rebuild the state from a log file and keep appending to it."""
        with open(log_path) as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        twin = cls.replay(config, events, n_vials)
        twin.log_path = Path(log_path)
        return twin
