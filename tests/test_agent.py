import asyncio
import json

import pytest

from sdl.agent import AgentSession, Task
from sdl.agent.harness import run_episode, run_grid
from sdl.agent.metrics import aggregate, calibration, episode_metrics, group, load_episodes, step_outcomes
from sdl.agent.policies import Overconfident, PriorKnowledge, RandomPairs, pair_recipe_dict
from sdl.agent.reward import reward

CHEMICALS = ("FeCl3", "KSCN", "NaOH", "NaCl")


def session(**kw):
    return AgentSession(Task.from_config(**kw))


def target_labels(s):
    return s.episode["hidden"]["target_labels"]


def test_three_information_conditions():
    labelled = session(mode="labelled").briefing()
    assert set(labelled["samples"]) == {"A", "B", "C", "D"} and "FeCl3" in labelled["samples"]["A"]
    assert "null control" not in labelled["samples"]["D"] and "shelf_contents" not in labelled

    listed = session(mode="listed", seed=1).briefing()
    assert set(listed["samples"]) == {"S1", "S2", "S3", "S4"}
    assert set(listed["samples"].values()) == {"unlabelled aqueous sample"}
    assert len(listed["shelf_contents"]) == 4 and all(any(c in x for x in listed["shelf_contents"]) for c in CHEMICALS)

    unknown = session(mode="unknown", seed=1).briefing()
    assert set(unknown["samples"]) == {"S1", "S2", "S3", "S4"} and "shelf_contents" not in unknown
    assert not any(c in json.dumps(unknown) for c in CHEMICALS)


def test_listed_contents_do_not_reveal_which_bottle_is_which():
    # the list is in the same order whatever the shuffle, so it carries no positional information
    lists = {tuple(session(mode="listed", seed=s).briefing()["shelf_contents"]) for s in range(8)}
    maps = {tuple(sorted(Task.from_config(mode="listed", seed=s).labels().items())) for s in range(8)}
    assert len(lists) == 1 and len(maps) > 1
    # listed and unknown use the same shuffle for the same seed, so they are directly comparable
    assert Task.from_config(mode="listed", seed=3).labels() == Task.from_config(mode="unknown", seed=3).labels()


def test_old_condition_names_still_load():
    t = Task.from_config(mode="blinded", feedback="online")
    assert (t.mode, t.feedback) == ("unknown", "adaptive")
    assert Task.from_config(mode="named", feedback="offline").feedback == "batch"


def test_agent_view_never_leaks_truth():
    s = session(mode="unknown", seed=2)
    a, b = target_labels(s)
    outs = [s.briefing(), s.inspect(), s.verify(pair_recipe_dict(a, b)),
            s.dispatch(pair_recipe_dict(a, b), 0.5, [a, b]),
            s.dispatch({"recipe_id": "x", "steps": [{"add": {"reagent": "A", "volume_ml": 1}}]}, 0.5, ["A"]),
            s.dispatch({"recipe_id": "x", "steps": [{"add": {"reagent": a, "volume_ml": 9}}, {"image": {}}]}, 0.5, [a])]
    blob = json.dumps(outs)
    for secret in ("hidden", "true_ids", "outcome", "predicted_class", "fescn", "label_to_id", "reward"):
        assert secret not in blob
    assert "'A' is not on the shelf" in blob          # the agent's own wrong label is echoed, nothing else


def test_at_most_two_samples_per_mix_and_single_sample_allowed():
    s = session(mode="unknown", seed=0)
    three = {"recipe_id": "x", "steps": [{"add": {"reagent": l, "volume_ml": 1.0}} for l in ("S1", "S2", "S3")]
             + [{"image": {}}]}
    assert s.verify(three)["issues"][0]["code"] == "TOO_MANY_SAMPLES"
    out = s.dispatch(three, 0.9, ["S1", "S2", "S3"])
    assert not out["accepted"] and out["issues"][0]["code"] == "TOO_MANY_SAMPLES"
    assert s.inspect()["mixes_used"] == 0                                   # a rejected recipe costs nothing
    single = {"recipe_id": "one", "steps": [{"add": {"reagent": "S1", "volume_ml": 2.0}}, {"image": {}}]}
    assert s.dispatch(single, 0.0, ["S1"])["accepted"] and s.inspect()["mixes_used"] == 1
    # the same two samples may be added in several portions
    twice = {"recipe_id": "t", "steps": [{"add": {"reagent": "S1", "volume_ml": 1.0}},
                                         {"add": {"reagent": "S2", "volume_ml": 0.5}}, {"stir": {"seconds": 10}}, {"image": {}},
                                         {"add": {"reagent": "S2", "volume_ml": 0.5}}, {"stir": {"seconds": 10}}, {"image": {}}]}
    assert s.dispatch(twice, 0.3, ["S1", "S2"])["accepted"]


def test_goal_states_rules_and_stopping():
    goal = session(mode="listed").briefing()["goal"]
    for phrase in ("at most 5 mixes", "at most 2 different samples", "one sample on its own", "stop mixing"):
        assert phrase in goal, phrase


def test_confidence_is_mandatory_and_costs_nothing_when_missing():
    s = session(mode="labelled")
    out = s.dispatch(pair_recipe_dict("A", "B"))
    assert not out["accepted"] and "confidence" in out["error"]
    assert s.inspect()["mixes_used"] == 0


def test_adaptive_gives_readout_and_budget_is_enforced():
    s = session(mode="labelled", budget=2)
    out = s.dispatch(pair_recipe_dict("A", "B"), 0.9, ["A", "B"], "iron plus thiocyanate")
    assert out["accepted"] and out["d_blank"] > 50 and out["mixes_left"] == 1 and "reward" not in out
    assert s.dispatch(pair_recipe_dict("A", "C"), 0.1, ["A", "C"])["accepted"]
    third = s.dispatch(pair_recipe_dict("A", "D"), 0.1, ["A", "D"])
    assert not third["accepted"] and third["issues"][0]["code"] == "BUDGET_EXCEEDED"


def test_batch_withholds_until_reveal_then_locks():
    s = session(mode="labelled", feedback="batch")
    out = s.dispatch(pair_recipe_dict("A", "B"), 0.5, ["A", "B"])
    assert out["accepted"] and "images" not in out and "d_blank" not in out
    assert "images" not in s.get_result(out["run_id"])
    assert s.reveal_results()["results"][0]["d_blank"] > 50
    assert not s.dispatch(pair_recipe_dict("A", "C"), 0.5, ["A", "C"])["accepted"]


def test_submit_answer_scores_and_closes():
    s = session(mode="unknown", seed=5)
    assert not s.submit_answer(["S1"], 0.5)["recorded"]
    assert s.submit_answer(target_labels(s), 0.8)["recorded"]
    assert s.episode["answer"]["correct"] and s.closed
    assert not s.dispatch(pair_recipe_dict("S1", "S2"), 0.5, ["S1", "S2"])["accepted"]
    wrong = session(mode="labelled")
    wrong.submit_answer(["A", "C"], 0.9)
    assert wrong.episode["answer"]["correct"] is False


def test_ground_truth_outcomes_plan_agreement_and_mixes_to_target():
    s = session(mode="labelled")
    s.dispatch(pair_recipe_dict("C", "A"), 0.1, ["A", "D"])           # says one thing, does another
    s.dispatch(pair_recipe_dict("B", "D"), 0.0, None)
    s.dispatch(pair_recipe_dict("A", "B"), 0.9, ["A", "B"])
    m = episode_metrics(s.episode)
    assert m["outcome_trace"] == ["distractor", "null", "target"] and m["first_target_mix"] == 3
    assert [x["plan_agrees"] for x in s.episode["mixes"]] == [False, None, True]
    assert m["distractor_mixes"] == 1 and m["plan_agreement"] == pytest.approx(0.5) and m["rules_version"] == 2
    assert m["reward_trace"][2] > 0.8 and max(m["reward_trace"][:2]) < 0.2
    assert step_outcomes(s.episode, s.episode["mixes"][2]["run_id"]) == ["null", "target"]


def test_reward_shape():
    assert reward(100, 45, 0.0) == pytest.approx(1.0)
    assert reward(100, 45, 1.0) == 0.0 and reward(0, 45, 0.0) == 0.0
    assert reward(100, 225, 0.0) < 1e-6


def test_calibration_metric():
    perfect = calibration([1, 1, 0, 0], [1, 1, 0, 0])
    assert perfect["brier"] == 0 and perfect["base_rate"] == 0.5
    assert calibration([0.9] * 4, [0, 0, 0, 1])["brier"] > 0.5
    assert calibration([], [])["brier"] is None


def test_baselines_behave_as_designed():
    eps = run_grid([RandomPairs, Overconfident, PriorKnowledge], range(60))
    agg = {k: aggregate(v) for k, v in group(eps).items()}
    assert set(k[1] for k in agg) == {"labelled", "listed", "unknown"} and set(k[2] for k in agg) == {"adaptive"}
    # knowing the chemistry only helps when the bottles are labelled
    assert agg[("prior_knowledge", "labelled", "adaptive")]["produced_within"][0] == 1.0
    assert agg[("prior_knowledge", "listed", "adaptive")]["produced_within"][0] < 0.5
    # random pair search finds the pair at a steady rate of about one in six per mix
    pw = agg[("random_pairs", "unknown", "adaptive")]["produced_within"]
    assert len(pw) == 5 and pw == sorted(pw) and 0.05 < pw[0] < 0.35 and 0.65 < pw[4] < 0.97
    # with five of six pairs testable, elimination always gives the right answer
    assert agg[("random_pairs", "unknown", "adaptive")]["success_rate"] == 1.0
    # same search, worse stated confidence -> much worse Brier score
    assert (agg[("overconfident", "unknown", "adaptive")]["mix_calibration"]["brier"]
            > 2.5 * agg[("random_pairs", "unknown", "adaptive")]["mix_calibration"]["brier"])


def test_episode_log_reload_and_pilot_filter(tmp_path):
    ep = run_episode(RandomPairs(seed=1), Task.from_config(mode="unknown", seed=1), tmp_path)
    # a pilot episode written before the rule change: old names, no rules_version
    legacy = json.loads(json.dumps(ep))
    legacy["episode_id"] = "pilot"
    legacy["task"].update(mode="blinded", feedback="online")
    legacy["task"].pop("rules_version")
    (tmp_path / "pilot").mkdir()
    (tmp_path / "pilot" / "episode.json").write_text(json.dumps(legacy))
    everything = load_episodes(tmp_path)
    assert len(everything) == 2 and {e["task"]["mode"] for e in everything} == {"unknown"}
    current = load_episodes(tmp_path, rules_version=2)
    assert [e["episode_id"] for e in current] == [ep["episode_id"]] and current[0]["closed"]
    assert (tmp_path / ep["episode_id"] / "traces" / "index.jsonl").exists()


def test_gym_env():
    pytest.importorskip("gymnasium")
    from sdl.agent.gym_env import SdlPairEnv
    env = SdlPairEnv(blinded=True)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs) and env.action_space.n == 12
    done, steps = False, 0
    while not done:
        obs, r, terminated, truncated, _ = env.step(steps)
        assert env.observation_space.contains(obs) and 0.0 <= r <= 1.0
        done, steps = terminated or truncated, steps + 1
    assert steps <= 5
    env.reset(seed=1)
    a, b = (env.slots.index(l) for l in env.session.episode["hidden"]["target_labels"])
    _, r, terminated, _, _ = env.step(env.pairs.index((a, b)))
    assert r > 0.8 and terminated


def test_mcp_server_exposes_the_tools():
    pytest.importorskip("mcp")
    from sdl.agent.mcp_server import build_server
    server = build_server(session(mode="listed"))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"briefing", "inspect", "verify", "dispatch", "get_result", "reveal_results", "submit_answer"}


def test_token_usage_is_parsed_and_attached(tmp_path):
    from sdl.agent.metrics import usage_from_claude_json
    result = {"type": "result", "num_turns": 7, "duration_ms": 21000, "total_cost_usd": 0.12,
              "usage": {"input_tokens": 100, "output_tokens": 900, "cache_creation_input_tokens": 2000,
                        "cache_read_input_tokens": 30000}, "modelUsage": {"some-model": {}}}
    u = usage_from_claude_json(json.dumps(result))
    assert u["total_tokens"] == 33000 and u["output_tokens"] == 900 and u["cost_usd"] == 0.12
    assert u["num_turns"] == 7 and u["duration_s"] == 21.0 and u["models"] == ["some-model"]
    assert usage_from_claude_json(json.dumps([{"type": "system"}, result]))["total_tokens"] == 33000
    assert usage_from_claude_json("not json") is None and usage_from_claude_json("{}") is None

    # usage.json next to an episode, and a batch-runner transcript, are both picked up on load
    a = run_episode(RandomPairs(seed=1), Task.from_config(mode="unknown", seed=1), tmp_path)
    b = run_episode(RandomPairs(seed=2), Task.from_config(mode="listed", seed=2), tmp_path)
    c = run_episode(RandomPairs(seed=3), Task.from_config(mode="labelled", seed=3), tmp_path)
    (tmp_path / a["episode_id"] / "usage.json").write_text(json.dumps(u))
    (tmp_path / "_transcripts").mkdir()
    (tmp_path / "_transcripts" / f"{b['episode_id']}.json").write_text(json.dumps(result))
    eps = {e["episode_id"]: e for e in load_episodes(tmp_path)}
    assert episode_metrics(eps[a["episode_id"]])["tokens_total"] == 33000
    assert episode_metrics(eps[b["episode_id"]])["cost_usd"] == 0.12
    assert episode_metrics(eps[c["episode_id"]])["tokens_total"] is None
    agg = aggregate(list(eps.values()))
    assert agg["mean_tokens"] == 33000 and agg["episodes_with_usage"] == 2
