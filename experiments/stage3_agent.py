"""Stage 3: the dual-interface agent task, run with the scripted baseline agents.

Runs every baseline in all four conditions (named / blinded x online / offline)
and prints the evaluation table. LLM episodes recorded through the MCP server are
scored with experiments/stage3_report.py and can be compared against these rows.

    python experiments/stage3_agent.py --episodes 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import write_csv
from sdl.agent.harness import run_grid
from sdl.agent.metrics import aggregate, episode_metrics, group
from sdl.agent.policies import POLICIES


def print_table(groups: dict) -> list:
    print(f"{'agent':17s}{'labels':9s}{'feedback':10s}{'n':>4s}{'success':>9s}{'mixes':>7s}"
          f"{'distractor':>12s}{'Brier':>8s}{'base rate':>11s}{'plan ok':>9s}")
    rows = []
    for (agent, mode, fb), eps in sorted(groups.items()):
        a = aggregate(eps)
        cal = a["mix_calibration"]
        fmt = lambda v, f: "-" if v is None else format(v, f)
        print(f"{agent:17s}{mode:9s}{fb:10s}{a['episodes']:4d}{a['success_rate']:9.2f}{a['mean_mixes_used']:7.2f}"
              f"{a['mean_distractor_mixes']:12.2f}{fmt(cal['brier'], '8.3f'):>8s}{fmt(cal['base_rate'], '11.2f'):>11s}"
              f"{fmt(a['plan_agreement'], '9.2f'):>9s}")
        rows.append({"agent": agent, "labels": mode, "feedback": fb, "episodes": a["episodes"],
                     "success_rate": a["success_rate"], "mean_mixes_used": a["mean_mixes_used"],
                     "mean_distractor_mixes": a["mean_distractor_mixes"], "mix_brier": cal["brier"],
                     "mix_base_rate": cal["base_rate"], "plan_agreement": a["plan_agreement"],
                     "answer_brier": a["answer_calibration"]["brier"]})
    print("\nsuccess    = correct pair submitted within the budget")
    print("distractor = mean mixes per episode spent on the precipitate-forming pair")
    print("Brier      = mean squared error of stated per-mix confidence (0 is perfect; lower is better)")
    print("base rate  = share of mixes that actually produced the red complex")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=50, help="episodes (seeds) per agent and condition")
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--store", default=None, help="keep full episode logs and frames in this folder")
    a = p.parse_args()
    episodes = run_grid(POLICIES.values(), range(a.episodes), store_dir=a.store, budget=a.budget)
    rows = print_table(group(episodes))
    write_csv(Path(a.out) / "stage3_summary.csv", rows)
    per_episode = [{k: v for k, v in episode_metrics(e).items()} for e in episodes]
    write_csv(Path(a.out) / "stage3_episodes.csv", per_episode)


if __name__ == "__main__":
    main()
