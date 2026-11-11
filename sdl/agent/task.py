"""Stage 3 task definition: what the agent is told, and the rules of the game."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import yaml

from sdl.config import CONFIG_DIR, Config

# What the agent is told about the bottles:
#   labelled  which chemical is in which bottle
#   listed    which chemicals are on the shelf, but not which bottle is which
#   unknown   nothing about the contents
MODES = ("labelled", "listed", "unknown")
# When the agent sees readouts:
#   adaptive  after every mix
#   batch     only after all mixes are done
FEEDBACK = ("adaptive", "batch")
RULES_VERSION = 2            # 2 = at most two samples per mix (1 = any number; the pilot runs)
LEGACY = {"named": "labelled", "blinded": "unknown", "online": "adaptive", "offline": "batch"}


@dataclass
class Task:
    mode: str = "unknown"
    feedback: str = "adaptive"
    budget: int = 5
    seed: int = 0
    show_reward: bool = False   # include the scalar reward in the readout (RL-style) or not (eval-style)
    max_samples_per_mix: int = 2
    samples: tuple = ("A", "B", "C", "D")
    target_tags: tuple = ("iron_iii", "thiocyanate")
    distractor_tags: tuple = ("iron_iii", "base")
    spoiler_tag: str = "base"
    reward: dict | None = None
    rules_version: int = RULES_VERSION

    def __post_init__(self):
        self.mode = LEGACY.get(self.mode, self.mode)
        self.feedback = LEGACY.get(self.feedback, self.feedback)
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
            max_samples_per_mix=raw.get("max_samples_per_mix", 2),
            target_tags=tuple(raw["target_tags"]), distractor_tags=tuple(raw["distractor_tags"]),
            spoiler_tag=raw["spoiler_tag"], reward=raw.get("reward"),
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)

    def to_dict(self) -> dict:
        return asdict(self)

    def labels(self) -> dict:
        """label shown to the agent -> true reagent id."""
        if self.mode == "labelled":
            return {s: s for s in self.samples}
        order = np.random.default_rng([self.seed, 4242]).permutation(len(self.samples))
        return {f"S{i + 1}": self.samples[j] for i, j in enumerate(order)}

    @staticmethod
    def chemical_name(true_id: str, config: Config) -> str:
        return config.reagents[true_id].name.split(" (")[0]   # drop role hints such as "(null control)"

    def describe(self, label: str, true_id: str, config: Config) -> str:
        if self.mode == "labelled":
            return self.chemical_name(true_id, config)
        return "unlabelled aqueous sample"

    def shelf_contents(self, config: Config) -> list | None:
        """Listed condition only: the chemicals on the shelf, in an order unrelated to the labels."""
        if self.mode != "listed":
            return None
        return sorted(self.chemical_name(s, config) for s in self.samples)

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
        told = {
            "labelled": "Each sample is labelled with what it contains.",
            "listed": "You are told which chemicals are on the shelf, but not which sample is which.",
            "unknown": "You are told nothing about what the samples contain.",
        }[self.mode]
        fb = ("You see the readout of each mix before choosing the next one."
              if self.feedback == "adaptive" else
              "Readouts are withheld until you call reveal_results; after that no more mixes are allowed. "
              "Plan all your mixes up front.")
        return (
            f"Among the samples on the shelf, exactly one pair reacts to form a red complex. {told} "
            f"Find that pair using at most {self.budget} mixes, then call submit_answer. "
            f"A mix may contain at most {self.max_samples_per_mix} different samples; imaging one sample on its own "
            f"is allowed. Every accepted recipe costs one mix. {fb} "
            "As soon as one of your mixes has produced the red complex, stop mixing and call submit_answer: "
            "using fewer mixes is better. "
            "With each mix, state your confidence (0 to 1) that it will produce the red complex, "
            "list the samples you intend to combine (planned_samples) and give one sentence of reasoning. "
            "Readouts are colour features of the vial: CIELAB L, a, b; hue angle in degrees "
            "(0 red, 90 yellow, 180 green, 270 blue); chroma; turbidity (0 clear, 1 opaque); "
            "d_blank, the colour distance from plain water; and d_pre, the change since the recipe's first image."
        )
