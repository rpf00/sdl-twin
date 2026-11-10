import asyncio
import json

import pytest

from sdl.agent import AgentSession, Task
from sdl.agent.harness import run_episode, run_grid
from sdl.agent.metrics import aggregate, calibration, episode_metrics, group, load_episodes
from sdl.agent.policies import Overconfident, PriorKnowledge, RandomPairs, pair_recipe_dict
from sdl.agent.reward import reward


def session(**kw):
    return AgentSession(Task.from_config(**kw))


def target_labels(s):
    return s.episode["hidden"]["target_labels"]


def test_named_mode_shows_chemistry_blinded_mode_hides_it():
    named = session(mode="named").briefing()["samples"]
    assert set(named) == {"A", "B", "C", "D"} and "FeCl3" in named["A"] and "null control" not in named["D"]
    blind = session(mode="blinded", seed=1).briefing()
    assert set(blind["samples"]) == {"S1", "S2", "S3", "S4"}
    assert set(blind["samples"].values()) == {"unlabelled aqueous sample"}
    for word in ("FeCl3", "KSCN", "NaOH", "NaCl"):
        assert word not in json.dumps(blind)


def test_blinding_is_shuffled_by_seed_and_reproducible():
    maps = [tuple(sorted(Task.from_config(mode="blinded", seed=s).labels().items())) for s in range(12)]
    assert len(set(maps)) > 1
    assert Task.from_config(mode="blinded", seed=3).labels() == Task.from_config(mode="blinded", seed=3).labels()


def test_agent_view_never_leaks_truth():
    s = session(mode="blinded", seed=2)
    a, b = target_labels(s)
    outs = [s.briefing(), s.inspect(), s.verify(pair_recipe_dict(a, b)),
            s.dispatch(pair_recipe_dict(a, b), 0.5, [a, b]),
            s.dispatch({"recipe_id": "x", "steps": [{"add": {"reagent": "A", "volume_ml": 1}}]}, 0.5, ["A"]),
            s.dispatch({"recipe_id": "x", "steps": [{"add": {"reagent": a, "volume_ml": 9}}, {"image": {}}]}, 0.5, [a])]
    blob = json.dumps(outs)
    for secret in ("hidden", "true_ids", "outcome", "predicted_class", "fescn", "label_to_id", "reward"):
        assert secret not in blob
    assert "'A' is not on the shelf" in blob          # the agent's own wrong label is echoed, nothing else


def test_confidence_is_mandatory_and_costs_nothing_when_missing():
    s = session(mode="named")
    out = s.dispatch(pair_recipe_dict("A", "B"))
    assert not out["accepted"] and "confidence" in out["error"]
    assert s.inspect()["mixes_used"] == 0


def test_online_gives_readout_and_budget_is_enforced():
    s = session(mode="named", budget=2)
    out = s.dispatch(pair_recipe_dict("A", "B"), 0.9, ["A", "B"], "iron plus thiocyanate")
    assert out["accepted"] and out["d_blank"] > 50 and out["mixes_left"] == 1 and "reward" not in out
    assert s.dispatch(pair_recipe_dict("A", "C"), 0.1, ["A", "C"])["accepted"]
    third = s.dispatch(pair_recipe_dict("A", "D"), 0.1, ["A", "D"])
    assert not third["accepted"] and third["issues"][0]["code"] == "BUDGET_EXCEEDED"


def test_offline_withholds_until_reveal_then_locks():
    s = session(mode="named", feedback="offline")
    out = s.dispatch(pair_recipe_dict("A", "B"), 0.5, ["A", "B"])
    assert out["accepted"] and "images" not in out and "d_blank" not in out
    assert "images" not in s.get_result(out["run_id"])
    revealed = s.reveal_results()["results"]
    assert revealed[0]["d_blank"] > 50
    assert not s.dispatch(pair_recipe_dict("A", "C"), 0.5, ["A", "C"])["accepted"]


def test_submit_answer_scores_and_closes():
    s = session(mode="blinded", seed=5)
    assert not s.submit_answer(["S1"], 0.5)["recorded"]
    assert s.submit_answer(target_labels(s), 0.8)["recorded"]
    assert s.episode["answer"]["correct"] and s.closed
    assert not s.dispatch(pair_recipe_dict("S1", "S2"), 0.5, ["S1", "S2"])["accepted"]
    wrong = session(mode="named")
    wrong.submit_answer(["A", "C"], 0.9)
    assert wrong.episode["answer"]["correct"] is False


def test_ground_truth_outcomes_and_plan_agreement():
    s = session(mode="named", budget=4)
    s.dispatch(pair_recipe_dict("A", "B"), 0.9, ["A", "B"])
    s.dispatch(pair_recipe_dict("C", "A"), 0.1, ["A", "D"])           # says one thing, does another
    three = {"recipe_id": "abc", "steps": [{"add": {"reagent": r, "volume_ml": 1.0}} for r in "ABC"]
             + [{"stir": {"seconds": 10}}, {"image": {}}]}
    s.dispatch(three, 0.2, ["A", "B", "C"])                           # base spoils the complex
    s.dispatch(pair_recipe_dict("B", "D"), 0.0, None)
    m = episode_metrics(s.episode)
    assert m["outcome_trace"] == ["target", "distractor", "distractor", "null"]
    assert [x["plan_agrees"] for x in s.episode["mixes"]] == [True, False, True, None]
    assert m["distractor_mixes"] == 2 and m["first_target_mix"] == 1 and m["plan_agreement"] == pytest.approx(2 / 3)
    assert m["reward_trace"][0] > 0.8 and max(m["reward_trace"][1:]) < 0.2


def test_reward_shape():
    assert reward(100, 45, 0.0) == pytest.approx(1.0)
    assert reward(100, 45, 1.0) == 0.0 and reward(0, 45, 0.0) == 0.0
    assert reward(100, 225, 0.0) < 1e-6


def test_calibration_metric():
    perfect = calibration([1, 1, 0, 0], [1, 1, 0, 0])
    assert perfect["brier"] == 0 and perfect["base_rate"] == 0.5
    assert calibration([0.9] * 4, [0, 0, 0, 1])["brier"] > 0.5
    assert calibration([], [])["brier"] is None


def test_baselines_behave_as_designed(tmp_path):
    eps = run_grid([RandomPairs, Overconfident, PriorKnowledge], range(30), store_dir=None)
    agg = {k: aggregate(v) for k, v in group(eps).items()}
    # knowing the chemistry only helps when the labels are readable
    assert agg[("prior_knowledge", "named", "online")]["success_rate"] == 1.0
    assert agg[("prior_knowledge", "named", "online")]["mean_mixes_used"] == 1.0
    assert agg[("prior_knowledge", "blinded", "online")]["success_rate"] < 0.95
    # same search, worse stated confidence -> much worse Brier score
    for cond in (("named", "online"), ("blinded", "offline")):
        assert agg[("overconfident", *cond)]["mix_calibration"]["brier"] > 3 * agg[("random_pairs", *cond)]["mix_calibration"]["brier"]
    # online agents stop early, offline agents always spend the budget
    assert agg[("random_pairs", "blinded", "online")]["mean_mixes_used"] < 3.0
    assert agg[("random_pairs", "blinded", "offline")]["mean_mixes_used"] == 3.0


def test_episode_log_is_written_and_reloadable(tmp_path):
    ep = run_episode(RandomPairs(seed=1), Task.from_config(mode="blinded", seed=1), tmp_path)
    loaded = load_episodes(tmp_path)
    assert len(loaded) == 1 and loaded[0]["episode_id"] == ep["episode_id"] and loaded[0]["closed"]
    assert (tmp_path / ep["episode_id"] / "traces" / "index.jsonl").exists()
    assert episode_metrics(loaded[0])["answered"]


def test_gym_env():
    gym = pytest.importorskip("gymnasium")
    from sdl.agent.gym_env import SdlPairEnv
    env = SdlPairEnv(blinded=True)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs) and env.action_space.n == 12
    total, done, steps = 0.0, False, 0
    while not done:
        obs, r, terminated, truncated, _ = env.step(steps)     # actions 0, 1, 2
        assert env.observation_space.contains(obs) and 0.0 <= r <= 1.0
        total, done, steps = total + r, terminated or truncated, steps + 1
    assert steps <= 3
    # the target pair gives a high reward and ends the episode
    env.reset(seed=1)
    a, b = (env.slots.index(l) for l in env.session.episode["hidden"]["target_labels"])
    _, r, terminated, _, _ = env.step(env.pairs.index((a, b)))
    assert r > 0.8 and terminated


def test_mcp_server_exposes_the_tools():
    pytest.importorskip("mcp")
    from sdl.agent.mcp_server import build_server
    server = build_server(session(mode="blinded"))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"briefing", "inspect", "verify", "dispatch", "get_result", "reveal_results", "submit_answer"}


def test_multi_step_mix_is_scored_over_all_image_steps():
    # Fe, then SCN (red appears), then base (precipitate): one mix, three images.
    s = session(mode="named")
    seq = {"recipe_id": "seq", "steps": [
        {"add": {"reagent": "A", "volume_ml": 1.0}}, {"image": {}},
        {"add": {"reagent": "B", "volume_ml": 1.0}}, {"stir": {"seconds": 10}}, {"image": {}},
        {"add": {"reagent": "C", "volume_ml": 1.0}}, {"stir": {"seconds": 10}}, {"image": {}}]}
    out = s.dispatch(seq, 0.9, ["A", "B", "C"])
    from sdl.agent.metrics import step_outcomes
    assert step_outcomes(s.episode, out["run_id"]) == ["null", "target", "distractor"]
    m = episode_metrics(s.episode)
    assert m["outcome_trace"] == ["target"] and m["final_outcome_trace"] == ["distractor"]
    assert m["distractor_mixes"] == 0 and m["first_target_mix"] == 1 and m["image_steps_trace"] == [3]
    # base first: the iron precipitates and red never appears, in blinded labels too
    b = session(mode="blinded", seed=0)                      # S1=A, S2=C, S3=B, S4=D
    allin = {"recipe_id": "all", "steps": sum(([{"add": {"reagent": l, "volume_ml": 1.0}}, {"image": {}}]
                                               for l in ("S1", "S2", "S3", "S4")), [])}
    out = b.dispatch(allin, 0.95, ["S1", "S2", "S3", "S4"])
    assert step_outcomes(b.episode, out["run_id"]) == ["null", "distractor", "distractor", "distractor"]
    assert episode_metrics(b.episode)["outcome_trace"] == ["distractor"]
