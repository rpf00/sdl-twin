import pytest

from sdl.schema import Add, Recipe, pair_recipe
from sdl.schema.recipe import RecipeFormatError

YAML = """
recipe_id: R0042
steps:
  - add: {reagent: A, volume_ml: 1.0}
  - add: {reagent: B, volume_ml: 1.0}
  - stir: {seconds: 10}
  - image: {delay_s: 5}
"""


def test_yaml_roundtrip():
    r = Recipe.from_yaml(YAML)
    assert r.recipe_id == "R0042" and len(r.steps) == 4
    assert r.steps[0] == Add("A", 1.0)
    assert Recipe.from_dict(r.to_dict()).steps == r.steps


def test_unknown_operation_rejected():
    with pytest.raises(RecipeFormatError):
        Recipe.from_dict({"recipe_id": "x", "steps": [{"heat": {"celsius": 90}}]})


def test_bad_arguments_rejected():
    with pytest.raises(RecipeFormatError):
        Recipe.from_dict({"recipe_id": "x", "steps": [{"add": {"reagent": "A"}}]})


def test_pair_recipe_shape():
    ops = [s.op for s in pair_recipe("A", "B").steps]
    assert ops == ["add", "image", "add", "stir", "image"]
