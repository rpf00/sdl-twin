"""The Lab: the single interface that agents, experiment scripts and analysis use.

    lab = Lab()                      # simulated by default
    lab.inspect()                    # what is where
    lab.verify(recipe)               # dry run on a fork of the state twin
    result = lab.dispatch(recipe)    # verify again, execute, read out, record

Swapping `backend=RealBackend(...)` is the only change needed to run on hardware.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from sdl.backends.sim import SimBackend
from sdl.config import Config, load_config
from sdl.data.store import TraceStore
from sdl.schema.events import Event
from sdl.schema.recipe import Add, Image, Recipe, Stir
from sdl.state.twin import StateTwin
from sdl.state.validator import Report, verify
from sdl.vision.layout import Layout
from sdl.vision.pipeline import Features, classify, delta_e, extract_features

BLANK_REAGENT = "W"
BLANK_VOLUME_ML = 2.0


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=2)
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class RunResult:
    run_id: str
    recipe_id: str
    accepted: bool
    issues: list = field(default_factory=list)
    features: list = field(default_factory=list)    # one Features per image step
    d_pre: float | None = None      # colour change from the first to the last image of the recipe
    d_blank: float | None = None    # colour difference of the last image from the session's water blank
    predicted_class: str | None = None
    session: int = 0

    def to_agent_dict(self) -> dict:
        """What an agent is allowed to see (no hidden emulator truth)."""
        return {
            "run_id": self.run_id, "recipe_id": self.recipe_id, "accepted": self.accepted,
            "issues": self.issues,
            "readout": [f.to_dict() for f in self.features],
            "d_pre": None if self.d_pre is None else round(self.d_pre, 3),
            "d_blank": None if self.d_blank is None else round(self.d_blank, 3),
            "predicted_class": self.predicted_class,
        }


class Lab:
    def __init__(self, backend=None, config: Config | None = None, store: TraceStore | str | Path | None = None,
                 n_vials: int = 48, budget_mixes: int | None = None, seed: int = 0,
                 log_path: str | Path | None = None, thresholds: dict | None = None,
                 layout: Layout = Layout()):
        self.config = config or load_config()
        self.backend = backend if backend is not None else SimBackend(self.config, seed=seed, layout=layout)
        self.store = TraceStore(store) if isinstance(store, (str, Path)) else store
        self.twin = StateTwin(self.config, n_vials=n_vials, log_path=log_path)
        self.budget_mixes = budget_mixes
        self.thresholds = thresholds
        self.layout = layout
        self.blank: Features | None = None
        self._run_counter = 0
        self._commit = _git_commit()

    # ---- agent-facing API ------------------------------------------------
    def inspect(self) -> dict:
        out = self.twin.inspect()
        out["budget_mixes"] = self.budget_mixes
        out["backend"] = self.backend.name
        return out

    def verify(self, recipe: Recipe) -> Report:
        return verify(self.twin, recipe, self.budget_mixes)

    def dispatch(self, recipe: Recipe, agent: dict | None = None) -> RunResult:
        """Verify against the live state, execute, read out and record.

        `agent` is free-form metadata stored with the trace, e.g.
        {"reasoning": "...", "confidence": 0.7}.
        """
        if self.blank is None:
            self._measure_blank()
        run_id = self._next_run_id()
        report = self.verify(recipe)
        state_before = self.twin.snapshot()
        issues = [i.__dict__ for i in report.issues]
        if not report.ok:
            result = RunResult(run_id, recipe.recipe_id, False, issues, session=self.twin.state.session)
            self._record(result, recipe, state_before, state_before, [], [], agent, {})
            return result

        vial = report.vial
        events = [self.twin.apply(Event("vial_assigned", {"vial": vial}, run_id)),
                  self.twin.apply(Event("mix_counted", {}, run_id))]
        frames, feats = self._execute(recipe, vial, run_id, events)
        truth = dict(getattr(self.backend, "last_truth", {}) or {})
        self.backend.discard(vial)
        events.append(self.twin.apply(Event("vial_discarded", {"vial": vial}, run_id)))

        result = RunResult(run_id, recipe.recipe_id, True, issues, feats, session=self.twin.state.session)
        if feats:
            result.d_blank = delta_e(self.blank, feats[-1])
            result.d_pre = delta_e(feats[0], feats[-1]) if len(feats) > 1 else None
            result.predicted_class = classify(feats[-1], result.d_blank, self.thresholds)
        self._record(result, recipe, state_before, self.twin.snapshot(), events, frames, agent, truth)
        return result

    # ---- operator API ------------------------------------------------------
    def new_session(self) -> int:
        """Start a new "day": camera/lighting conditions may shift, the blank is re-measured."""
        session = self.twin.state.session + 1
        self.twin.apply(Event("session_started", {"session": session}))
        self.backend.start_session(session)
        self.blank = None
        return session

    def restock_vials(self, n: int = 48) -> None:
        self.twin.apply(Event("vials_restocked", {"n": n}))

    def refill(self, reagent: str) -> None:
        self.twin.apply(Event("stock_refilled",
                              {"reagent": reagent, "volume_ml": self.config.limits.stock_volume_ml}))

    # ---- internals ---------------------------------------------------------
    def _next_run_id(self) -> str:
        self._run_counter += 1
        return f"run_{self._run_counter:05d}"

    def _execute(self, recipe: Recipe, vial: str, run_id: str, events: list) -> tuple:
        frames, feats = [], []
        ref = self.blank.contrast if self.blank else None
        for step in recipe.steps:
            if isinstance(step, Add):
                self.backend.add(vial, step.reagent, step.volume_ml)
                events.append(self.twin.apply(Event(
                    "liquid_added", {"vial": vial, "reagent": step.reagent, "volume_ml": float(step.volume_ml)}, run_id)))
            elif isinstance(step, Stir):
                self.backend.stir(vial, step.seconds)
                events.append(self.twin.apply(Event("stirred", {"vial": vial, "seconds": float(step.seconds)}, run_id)))
            elif isinstance(step, Image):
                frame = self.backend.image(vial, step.delay_s)
                frames.append(frame)
                feats.append(extract_features(frame, self.layout, ref))
                events.append(self.twin.apply(Event("imaged", {"vial": vial, "delay_s": float(step.delay_s)}, run_id)))
        return frames, feats

    def _measure_blank(self) -> None:
        """Image a water-only vial; its colour and stripe contrast are the session reference."""
        vial = self.twin.next_clean_vial()
        if vial is None:
            raise RuntimeError("no clean vial left for the session blank; call restock_vials()")
        run_id = self._next_run_id()
        self.twin.apply(Event("vial_assigned", {"vial": vial}, run_id))
        self.backend.add(vial, BLANK_REAGENT, BLANK_VOLUME_ML)
        self.twin.apply(Event("liquid_added", {"vial": vial, "reagent": BLANK_REAGENT,
                                               "volume_ml": BLANK_VOLUME_ML}, run_id))
        frame = self.backend.image(vial, 0.0)
        self.twin.apply(Event("imaged", {"vial": vial, "delay_s": 0.0}, run_id))
        self.backend.discard(vial)
        self.twin.apply(Event("vial_discarded", {"vial": vial}, run_id))
        first = extract_features(frame, self.layout, None)
        self.blank = extract_features(frame, self.layout, first.contrast)
        if self.store:
            self.store.write({"run_id": run_id, "kind": "blank", "accepted": True,
                              "session": self.twin.state.session, "backend": self.backend.describe(),
                              "features": [self.blank.to_dict()], "config_hash": self.config.hash(),
                              "git_commit": self._commit}, [frame])

    def _record(self, result, recipe, before, after, events, frames, agent, truth) -> None:
        if not self.store:
            return
        self.store.write({
            "run_id": result.run_id, "kind": "mix", "accepted": result.accepted,
            "recipe": recipe.to_dict(), "session": result.session,
            "backend": self.backend.describe(), "config_hash": self.config.hash(), "git_commit": self._commit,
            "issues": result.issues,
            "features": [f.to_dict() for f in result.features],
            "blank": self.blank.to_dict() if self.blank else None,
            "d_pre": result.d_pre, "d_blank": result.d_blank, "predicted_class": result.predicted_class,
            "agent": agent or {},
            "hidden_truth": truth,        # emulator only; never shown to agents
            "events": [e.to_dict() for e in events],
            "state_before": before, "state_after": after,
        }, frames)
