"""Stage 3: the dual-interface agent task, run with the scripted baseline agents.

Runs every baseline in the three information conditions (labelled / listed / unknown)
and prints the evaluation table. LLM episodes recorded through the MCP server are
scored with experiments/stage3_report.py and can be compared against these rows.

    python experiments/stage3_agent.py --episodes 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import write_csv
from sdl.agent.harness import run_grid
from sdl.agent.metrics import aggregate, episode_metrics, group
from sdl.agent.policies import POLICIES


def print_table(groups: dict) -> list:
    budget = max(e["task"]["budget"] for eps in groups.values() for e in eps)
    within = "".join(f"{'<=' + str(k):>6s}" for k in range(1, budget + 1))
    print(f"{'agent':17s}{'told':10s}{'feedback':10s}{'n':>4s}{'correct':>9s}{'to red':>8s}   red complex produced within"
          f"{within}{'Brier':>8s}{'plan ok':>9s}{'tokens/ep':>11s}{'$/ep':>8s}")
    rows = []
    fmt = lambda v, f: "-" if v is None else format(v, f)
    for (agent, mode, fb), eps in sorted(groups.items()):
        a = aggregate(eps)
        cal = a["mix_calibration"]
        pw = a["produced_within"] + [float("nan")] * (budget - len(a["produced_within"]))
        print(f"{agent:17s}{mode:10s}{fb:10s}{a['episodes']:4d}{a['success_rate']:9.2f}"
              f"{fmt(a['mean_mixes_to_target'], '8.2f'):>8s}{'':31s}" + "".join(f"{x:6.2f}" for x in pw)
              + f"{fmt(cal['brier'], '8.3f'):>8s}{fmt(a['plan_agreement'], '9.2f'):>9s}"
              + f"{fmt(a['mean_tokens'], '11,.0f'):>11s}{fmt(a['mean_cost_usd'], '8.3f'):>8s}")
        rows.append({"agent": agent, "told": mode, "feedback": fb, "episodes": a["episodes"],
                     "correct_answer_rate": a["success_rate"], "mean_mixes_to_red": a["mean_mixes_to_target"],
                     **{f"red_within_{k + 1}": x for k, x in enumerate(a["produced_within"])},
                     "never_produced_red": a["never_produced"], "mean_mixes_used": a["mean_mixes_used"],
                     "mean_tokens_per_episode": a["mean_tokens"], "mean_cost_usd_per_episode": a["mean_cost_usd"],
                     "mix_brier": cal["brier"], "plan_agreement": a["plan_agreement"],
                     "answer_brier": a["answer_calibration"]["brier"]})
    print("\ncorrect = share of episodes with the right pair submitted (elimination alone can reach 1.00)")
    print("to red  = mean number of mixes before the red complex was produced, over episodes that produced it")
    print("within  = share of episodes that had produced the red complex after 1, 2, ... mixes")
    print("Brier   = mean squared error of stated per-mix confidence (0 is perfect)")
    print("tokens  = mean tokens per episode (input, cached input and output); '-' = not recorded")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=100, help="episodes (seeds) per agent and condition")
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--store", default=None, help="keep full episode logs and frames in this folder")
    p.add_argument("--batch", action="store_true", help="also run the batch-feedback variant of each condition")
    a = p.parse_args()
    feedbacks = ("adaptive", "batch") if a.batch else ("adaptive",)
    episodes = run_grid(POLICIES.values(), range(a.episodes), feedbacks=feedbacks, store_dir=a.store, budget=a.budget)
    rows = print_table(group(episodes))
    write_csv(Path(a.out) / "stage3_summary.csv", rows)
    per_episode = [{k: v for k, v in episode_metrics(e).items()} for e in episodes]
    write_csv(Path(a.out) / "stage3_episodes.csv", per_episode)


if __name__ == "__main__":
    main()
