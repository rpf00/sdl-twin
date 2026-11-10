"""Score recorded Stage 3 episodes, whoever played them (Claude Code over MCP, scripted, RL).

    python experiments/stage3_report.py runs/claude
"""
from __future__ import annotations

import sys

from sdl.agent.metrics import aggregate, episode_metrics, group, load_episodes
from stage3_agent import print_table


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    episodes = load_episodes(root)
    if not episodes:
        sys.exit(f"no episode.json found under {root}")
    open_eps = [e["episode_id"] for e in episodes if not e["closed"]]
    print(f"{len(episodes)} episode(s) under {root}" + (f"; {len(open_eps)} without a submitted answer" if open_eps else ""))
    print()
    print_table(group(episodes))
    print("\nPer episode")
    for e in episodes:
        m = episode_metrics(e)
        print(f"  {m['episode_id']}: success={m['success']} mixes={m['mixes_used']} "
              f"outcomes={m['outcome_trace']} confidence={[round(c, 2) for c in m['confidence_trace']]} "
              f"answer_conf={m['answer_confidence']}")
    for key, eps in sorted(group(episodes).items()):
        bins = aggregate(eps)["mix_calibration"]["bins"]
        if bins:
            print(f"\nCalibration of per-mix confidence, {' / '.join(key)}")
            for b in bins:
                print(f"  stated {b['range']}: n={b['n']:3d}  mean stated {b['mean_confidence']:.2f}  observed {b['observed_rate']:.2f}")


if __name__ == "__main__":
    main()
