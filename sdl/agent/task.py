"""Stage 3 task definition and label blinding."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import yaml

from sdl.config import CONFIG_DIR, Config

MODES = ("named", "blinded")
FEEDBACK = ("online", "offline")


@dataclass
class Task:
    mode: str = "blinded"       # named: agent sees chemical names. blinded: shuffled labels S1..Sn
    feedback: str = "online"    # online: readout after every mix. offline: all mixes first, readouts after
    budget: int = 3
    seed: int = 0
    show_reward: bool = False   # include the scalar reward in the readout (RL-style) or not (eval-style)
    samples: tuple = ("A", "B", "C", "D")
    target_tags: tuple = ("iron_iii", "thiocyanate")
    distractor_tags: tuple = ("iron_iii", "base")
    spoiler_tag: str = "base"
    reward: dict | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.feedback not in FEEDBACK:
            raise ValueError(f"feedback must be one of {FEEDBACK}")

    @classmethod
    def from_config(cls, path=None, **overrides) -> "Task":
        with open(path or CONFIG_DIR / "task.yaml") as fh:
            raw = yaml.safe_load(fh)
        base = dict(
            samples=tuple(raw["samples"]), budget=raw["budget_mixes"],
            target_tags=tuple(raw["target_tags"]), distractor_tags=tuple(raw["distractor_tags"]),
            spoiler_tag=raw["spoiler_tag"], reward=raw.get("reward"),
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)

    def to_dict(self) -> dict:
        return asdict(self)

    def labels(self) -> dict:
        """label shown to the agent -> true reagent id."""
        if self.mode == "named":
            return {s: s for s in self.samples}
        order = np.random.default_rng([self.seed, 4242]).permutation(len(self.samples))
        return {f"S{i + 1}": self.samples[j] for i, j in enumerate(order)}

    def describe(self, label: str, true_id: str, config: Config) -> str:
        if self.mode == "blinded":
            return "unlabelled aqueous sample"
        return config.reagents[true_id].name.split(" (")[0]   # drop role hints such as "(null control)"

    def outcome(self, true_ids, config: Config) -> str:
        """Ground-truth class of a mix from the tags of what went into the vial."""
        tags = set()
        for rid in true_ids:
            tags |= set(config.reagents[rid].tags)
        if set(self.distractor_tags) <= tags:
            return "distractor"
        if set(self.target_tags) <= tags and self.spoiler_tag not in tags:
            return "target"
        return "null"

    def target_pair(self, config: Config) -> frozenset:
        out = set()
        for tag in self.target_tags:
            out |= {s for s in self.samples if tag in config.reagents[s].tags}
        return frozenset(out)

    def goal(self) -> str:
        fb = ("You see the readout of each mix before choosing the next one."
              if self.feedback == "online" else
              "Readouts are withheld until you call reveal_results; after that no more mixes are allowed. "
              "Plan all your mixes up front.")
        return (
            f"Among the samples on the shelf, exactly one pair reacts to form a red complex. "
            f"Find that pair using at most {self.budget} mixes, then call submit_answer. "
            f"Every accepted recipe costs one mix, whatever it contains. {fb} "
            "With each mix, state your confidence (0 to 1) that it will produce the red complex, "
            "list the samples you intend to combine (planned_samples) and give one sentence of reasoning. "
            "Readouts are colour features of the vial: CIELAB L, a, b; hue angle in degrees "
            "(0 red, 90 yellow, 180 green, 270 blue); chroma; turbidity (0 clear, 1 opaque); "
            "d_blank, the colour distance from plain water; and d_pre, the change since the recipe's first image."
        )
