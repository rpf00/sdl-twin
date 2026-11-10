"""Simulation gate: dry-run a recipe on a fork of the state twin.

Nothing reaches a backend (simulated or real) unless `verify(...).ok` is True.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sdl.schema.events import Event
from sdl.schema.recipe import Add, Image, Recipe, Stir
from sdl.state.twin import StateTwin


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    step: int | None = None
    severity: str = "error"   # error | warning


@dataclass
class Report:
    ok: bool
    issues: list = field(default_factory=list)
    events: list = field(default_factory=list)       # events the recipe would produce
    predicted_state: dict = field(default_factory=dict)
    vial: str | None = None

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.severity == "warning"]

    def codes(self) -> list:
        return [i.code for i in self.errors]

    def to_dict(self) -> dict:
        return {"ok": self.ok, "vial": self.vial,
                "issues": [i.__dict__ for i in self.issues]}


def _forbidden(tags_present: set, safety) -> tuple | None:
    for a, b in safety.forbidden_tag_pairs:
        if a in tags_present and b in tags_present:
            return (a, b)
    return None


def verify(twin: StateTwin, recipe: Recipe, budget_mixes: int | None = None) -> Report:
    cfg, lim = twin.config, twin.config.limits
    sim = twin.fork()
    issues: list[Issue] = []
    events: list[Event] = []

    def err(code, msg, step=None):
        issues.append(Issue(code, msg, step))

    def warn(code, msg, step=None):
        issues.append(Issue(code, msg, step, "warning"))

    def emit(etype, **payload):
        events.append(sim.apply(Event(etype, payload)))

    if not recipe.steps:
        err("EMPTY_RECIPE", "recipe has no steps")
    if budget_mixes is not None and sim.state.mixes_used >= budget_mixes:
        err("BUDGET_EXCEEDED", f"mix budget of {budget_mixes} already used")
    vial_id = sim.next_clean_vial()
    if vial_id is None:
        err("NO_CLEAN_VIAL", "no clean vial left; restock vials")
        return Report(False, issues)
    emit("vial_assigned", vial=vial_id)
    emit("mix_counted")
    vial = sim.state.vials[vial_id]

    for i, step in enumerate(recipe.steps):
        if isinstance(step, Add):
            r = cfg.reagents.get(step.reagent)
            if r is None:
                err("UNKNOWN_REAGENT", f"'{step.reagent}' is not on the shelf", i)
                continue
            ok = True
            banned = set(r.tags) & set(cfg.safety.banned_tags)
            if banned:
                err("BANNED_REAGENT", f"'{step.reagent}' carries banned tag(s) {sorted(banned)}", i)
                ok = False
            if not (lim.min_dispense_ml <= step.volume_ml <= lim.max_dispense_ml):
                err("VOLUME_OUT_OF_RANGE",
                    f"{step.volume_ml} mL outside [{lim.min_dispense_ml}, {lim.max_dispense_ml}] mL", i)
                ok = False
            elif sim.state.stocks_ml[step.reagent] < step.volume_ml - 1e-9:
                err("STOCK_DEPLETED",
                    f"only {sim.state.stocks_ml[step.reagent]:.2f} mL of '{step.reagent}' left", i)
                ok = False
            elif vial.volume_ml + step.volume_ml > vial.capacity_ml + 1e-9:
                err("VIAL_OVERFILL",
                    f"{vial.volume_ml + step.volume_ml:.2f} mL exceeds vial capacity {vial.capacity_ml} mL", i)
                ok = False
            tags = set(r.tags)
            for rid in vial.reagents():
                tags |= set(cfg.reagents[rid].tags)
            pair = _forbidden(tags, cfg.safety)
            if pair:
                err("FORBIDDEN_COMBINATION", f"mixing {pair[0]} with {pair[1]} is not allowed", i)
                ok = False
            if ok:
                emit("liquid_added", vial=vial_id, reagent=step.reagent, volume_ml=float(step.volume_ml))
        elif isinstance(step, Stir):
            if vial.volume_ml <= 0:
                err("STIR_EMPTY", "cannot stir an empty vial", i)
            elif not (0 < step.seconds <= lim.max_stir_s):
                err("STIR_OUT_OF_RANGE", f"stir time must be in (0, {lim.max_stir_s}] s", i)
            else:
                emit("stirred", vial=vial_id, seconds=float(step.seconds))
        elif isinstance(step, Image):
            if vial.volume_ml <= 0:
                err("IMAGE_EMPTY", "nothing to image: the vial is empty", i)
            elif step.delay_s < 0:
                err("DELAY_NEGATIVE", "image delay cannot be negative", i)
            else:
                if len(vial.additions) >= 2 and not vial.stirred:
                    warn("UNSTIRRED_IMAGE", "imaging a multi-reagent vial that was not stirred", i)
                emit("imaged", vial=vial_id, delay_s=float(step.delay_s))
        else:
            err("UNKNOWN_STEP", f"unsupported step {step!r}", i)

    if vial.n_images == 0 and not any(i.code == "IMAGE_EMPTY" for i in issues):
        warn("NO_IMAGE", "recipe never images the vial, so it produces no readout")
    emit("vial_discarded", vial=vial_id)

    ok = not any(i.severity == "error" for i in issues)
    return Report(ok, issues, events if ok else [], sim.snapshot() if ok else {}, vial_id)
