import copy
import itertools
import json

import numpy as np

from sdl import Lab
from sdl.backends import RealBackend, SimBackend
from sdl.data.calibrate import fit_path_length, replicate_noise
from sdl.emulator import EmulatorParams
from sdl.schema import Add, Image, Recipe, pair_recipe

EXPECTED = {frozenset("AB"): "target", frozenset("AC"): "distractor"}


def test_all_twelve_ordered_pairs_classified():
    lab = Lab(seed=3)
    for a, b in itertools.permutations("ABCD", 2):
        res = lab.dispatch(pair_recipe(a, b))
        assert res.predicted_class == EXPECTED.get(frozenset((a, b)), "null"), (a, b)


def test_signal_increases_with_iron_and_series_is_unclipped():
    lab = Lab(seed=0)
    d = []
    for rid in ["A4", "A3", "A2", "A1"]:
        res = lab.dispatch(pair_recipe(rid, "B"))
        assert res.features[-1].clipped_fraction == 0
        d.append(res.d_blank)
    assert d == sorted(d) and d[0] > 5


def test_same_seed_same_result():
    a = Lab(seed=11).dispatch(pair_recipe("A", "B")).d_blank
    b = Lab(seed=11).dispatch(pair_recipe("A", "B")).d_blank
    c = Lab(seed=12).dispatch(pair_recipe("A", "B")).d_blank
    assert a == b and a != c


def test_rejected_recipe_changes_nothing_but_is_recorded(tmp_path):
    lab = Lab(seed=0, store=tmp_path / "runs")
    lab.dispatch(pair_recipe("A", "B"))
    before = lab.twin.snapshot()
    res = lab.dispatch(Recipe("bad", [Add("A", 3.5), Add("B", 3.5), Image()]))
    assert not res.accepted and res.features == []
    assert lab.twin.snapshot() == before
    assert len(lab.store.records(accepted_only=False)) == 3      # blank, good mix, rejected mix
    assert len(lab.store.records()) == 2


def test_budget_is_enforced():
    lab = Lab(seed=0, budget_mixes=2)
    assert lab.dispatch(pair_recipe("A", "B")).accepted
    assert lab.dispatch(pair_recipe("A", "C")).accepted
    res = lab.dispatch(pair_recipe("A", "D"))
    assert not res.accepted and res.issues[0]["code"] == "BUDGET_EXCEEDED"


def test_trace_record_is_complete_and_agent_view_hides_truth(tmp_path):
    lab = Lab(seed=0, store=tmp_path / "runs")
    res = lab.dispatch(pair_recipe("A", "B"), agent={"confidence": 0.8, "reasoning": "Fe + SCN"})
    rec = json.load(open(tmp_path / "runs" / res.run_id / "record.json"))
    for key in ("recipe", "features", "events", "state_before", "state_after", "backend", "config_hash", "agent"):
        assert key in rec
    assert rec["backend"]["backend"] == "sim" and rec["agent"]["confidence"] == 0.8
    assert rec["hidden_truth"]["fescn_M"] > 0
    assert "hidden_truth" not in json.dumps(res.to_agent_dict())
    assert lab.store.frame(res.run_id).shape == (80, 120, 3)


def test_new_session_remeasures_blank():
    lab = Lab(seed=0)
    lab.dispatch(pair_recipe("B", "D"))
    first = lab.blank
    lab.new_session()
    lab.dispatch(pair_recipe("B", "D"))
    assert lab.twin.state.session == 1 and lab.blank is not first


def test_calibration_recovers_path_length(tmp_path, config):
    truth = EmulatorParams.from_dict(config.emulator)
    truth.optics["path_cm"] = 0.7                      # the "real" lab differs from the prior
    lab = Lab(backend=SimBackend(config, truth, seed=5), config=config, store=tmp_path / "runs")
    for rid in ["A1", "A2", "A3", "A4"]:
        for _ in range(3):
            lab.dispatch(pair_recipe(rid, "B"))
    records = lab.store.records()
    prior = EmulatorParams.from_dict(copy.deepcopy(config.emulator))
    fit = fit_path_length(records, config, prior)
    assert abs(fit["path_cm"] - 0.7) < 0.03
    assert fit["rmse_after"] < fit["rmse_before"] / 5
    assert replicate_noise(records)["dof"] == 8


def test_real_backend_is_an_explicit_stub():
    try:
        RealBackend().image("V0000", 5)
    except NotImplementedError as exc:
        assert "TBD" in str(exc)
    else:
        raise AssertionError("RealBackend should not pretend to work")
