import dataclasses
import json

from sdl.schema import Add, Image, Recipe, Stir, pair_recipe
from sdl.schema.events import Event
from sdl.state import StateTwin, verify


def codes(twin, steps, **kw):
    return verify(twin, Recipe("t", steps), **kw).codes()


def test_valid_recipe_passes_and_does_not_mutate(config):
    twin = StateTwin(config, n_vials=2)
    before = twin.snapshot()
    report = verify(twin, pair_recipe("A", "B"))
    assert report.ok and report.vial == "V0000"
    assert twin.snapshot() == before and twin.events == []
    assert report.predicted_state["stocks_ml"]["A"] == config.limits.stock_volume_ml - 1.0


def test_unknown_reagent(config):
    assert codes(StateTwin(config), [Add("Z", 1.0), Image()]) == ["UNKNOWN_REAGENT", "IMAGE_EMPTY"]


def test_volume_out_of_range(config):
    assert "VOLUME_OUT_OF_RANGE" in codes(StateTwin(config), [Add("A", 0.001), Image()])
    assert "VOLUME_OUT_OF_RANGE" in codes(StateTwin(config), [Add("A", 3.0), Image()])


def test_vial_overfill(config):
    steps = [Add("A", 2.0), Add("B", 2.0), Add("D", 1.0), Image()]
    assert codes(StateTwin(config), steps) == ["VIAL_OVERFILL"]


def test_stock_depleted(config):
    twin = StateTwin(config)
    twin.apply(Event("stock_refilled", {"reagent": "A", "volume_ml": 0.5}))
    assert codes(twin, [Add("A", 1.0), Image()]) == ["STOCK_DEPLETED", "IMAGE_EMPTY"]


def test_forbidden_combination_and_banned_reagent(config_with_bleach):
    cfg = config_with_bleach
    # banned outright
    assert "BANNED_REAGENT" in codes(StateTwin(cfg), [Add("BLEACH", 1.0), Image()])
    # allowed on its own once un-banned, but never together with thiocyanate, in either order
    cfg.safety = dataclasses.replace(cfg.safety, banned_tags=())
    assert codes(StateTwin(cfg), [Add("OX", 1.0), Image()]) == []
    assert codes(StateTwin(cfg), [Add("B", 1.0), Add("OX", 1.0), Image()]) == ["FORBIDDEN_COMBINATION"]
    assert codes(StateTwin(cfg), [Add("OX", 1.0), Add("B", 1.0), Image()]) == ["FORBIDDEN_COMBINATION"]


def test_stir_and_image_need_liquid(config):
    assert codes(StateTwin(config), [Stir(10)]) == ["STIR_EMPTY"]
    assert codes(StateTwin(config), [Image()]) == ["IMAGE_EMPTY"]


def test_no_clean_vial_and_budget(config):
    assert codes(StateTwin(config, n_vials=0), [Add("A", 1.0), Image()]) == ["NO_CLEAN_VIAL"]
    twin = StateTwin(config)
    twin.apply(Event("mix_counted"))
    assert "BUDGET_EXCEEDED" in codes(twin, [Add("A", 1.0), Image()], budget_mixes=1)


def test_unstirred_image_is_a_warning_not_an_error(config):
    report = verify(StateTwin(config), Recipe("t", [Add("A", 1.0), Add("B", 1.0), Image()]))
    assert report.ok and [w.code for w in report.warnings] == ["UNSTIRRED_IMAGE"]


def test_replay_reproduces_state(config, tmp_path):
    log = tmp_path / "events.jsonl"
    twin = StateTwin(config, n_vials=3, log_path=log)
    for e in verify(twin, pair_recipe("A", "C")).events:
        twin.apply(Event(e.type, e.payload))
    assert twin.state.vials["V0000"].status == "discarded"
    assert abs(twin.state.vials["V0000"].species_mol["Fe3"] - 2e-6) < 1e-12
    replayed = StateTwin.replay(config, [json.loads(l) for l in open(log)], n_vials=3)
    assert replayed.snapshot() == twin.snapshot()
    assert StateTwin.from_log(config, log, n_vials=3).snapshot() == twin.snapshot()
