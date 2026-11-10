"""Append-only events. The lab state is the fold of these over the initial state."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

EVENT_TYPES = {
    "session_started", "vials_restocked", "stock_refilled",
    "vial_assigned", "liquid_added", "stirred", "imaged", "vial_discarded", "mix_counted",
}


@dataclass(frozen=True)
class Event:
    type: str
    payload: dict = field(default_factory=dict)
    run_id: str | None = None
    seq: int = -1

    def __post_init__(self):
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.type}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(d["type"], d.get("payload", {}), d.get("run_id"), d.get("seq", -1))
