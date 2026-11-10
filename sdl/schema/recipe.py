"""Recipe dictionary: the only way anything (agent or human) asks the lab to act.

YAML form (matches the PoC document):

    recipe_id: R0042
    steps:
      - add: {reagent: A, volume_ml: 1.0}
      - add: {reagent: B, volume_ml: 1.0}
      - stir: {seconds: 10}
      - image: {delay_s: 5}
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml


@dataclass(frozen=True)
class Add:
    reagent: str
    volume_ml: float
    op: str = "add"


@dataclass(frozen=True)
class Stir:
    seconds: float
    op: str = "stir"


@dataclass(frozen=True)
class Image:
    delay_s: float = 5.0
    op: str = "image"


_OPS = {"add": Add, "stir": Stir, "image": Image}


class RecipeFormatError(ValueError):
    pass


@dataclass
class Recipe:
    recipe_id: str
    steps: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        if not isinstance(d, dict) or "steps" not in d:
            raise RecipeFormatError("recipe needs 'recipe_id' and 'steps'")
        steps = []
        for i, raw in enumerate(d["steps"]):
            if not isinstance(raw, dict) or len(raw) != 1:
                raise RecipeFormatError(f"step {i}: expected a single-key mapping like {{add: {{...}}}}")
            (op, args), = raw.items()
            if op not in _OPS:
                raise RecipeFormatError(f"step {i}: unknown operation '{op}' (allowed: {sorted(_OPS)})")
            args = dict(args or {})
            args.pop("op", None)
            try:
                steps.append(_OPS[op](**args))
            except TypeError as exc:
                raise RecipeFormatError(f"step {i}: {exc}") from None
        return cls(str(d.get("recipe_id", "unnamed")), steps)

    @classmethod
    def from_yaml(cls, text: str) -> "Recipe":
        return cls.from_dict(yaml.safe_load(text))

    def to_dict(self) -> dict:
        out = []
        for s in self.steps:
            args = asdict(s)
            op = args.pop("op")
            out.append({op: args})
        return {"recipe_id": self.recipe_id, "steps": out}


def pair_recipe(first: str, second: str, volume_ml: float = 1.0, stir_s: float = 10.0,
                delay_s: float = 5.0, pre_image: bool = True, recipe_id: str | None = None) -> Recipe:
    """Standard two-reagent mix: add first, (image), add second, stir, image."""
    steps = [Add(first, volume_ml)]
    if pre_image:
        steps.append(Image(delay_s))
    steps += [Add(second, volume_ml), Stir(stir_s), Image(delay_s)]
    return Recipe(recipe_id or f"{first}+{second}", steps)
