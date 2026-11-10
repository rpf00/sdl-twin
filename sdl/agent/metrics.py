"""Scoring of Stage 3 episodes. Works on the episode.json dicts written by AgentSession,
whoever the agent was (scripted policy, Claude Code over MCP, an RL policy)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def episode_metrics(ep: dict) -> dict:
    mixes = [m for m in ep["mixes"] if m.get("accepted")]
    rejected = [m for m in ep["mixes"] if not m.get("accepted")]
    outcomes = [m["outcome"] for m in mixes]
    answer = ep.get("answer")
    agrees = [m["plan_agrees"] for m in mixes if m.get("plan_agrees") is not None]
    first_target = outcomes.index("target") + 1 if "target" in outcomes else None
    return {
        "episode_id": ep["episode_id"], "agent": ep["agent"],
        "mode": ep["task"]["mode"], "feedback": ep["task"]["feedback"],
        "budget": ep["task"]["budget"], "seed": ep["task"]["seed"],
        "answered": answer is not None,
        "success": bool(answer and answer["correct"]),       # correct pair submitted within budget
        "answer_confidence": answer["confidence"] if answer else None,
        "mixes_used": len(mixes),
        "rejected_recipes": len(rejected),
        "distractor_mixes": outcomes.count("distractor"),
        "made_target": first_target is not None,              # actually produced the red complex
        "first_target_mix": first_target,
        "plan_agreement": float(np.mean(agrees)) if agrees else None,
        "reward_trace": [round(m["reward"], 4) for m in mixes],
        "confidence_trace": [m["confidence"] for m in mixes],
        "outcome_trace": outcomes,
    }


def calibration(confidences, outcomes, n_bins: int = 5) -> dict:
    """Brier score and a reliability table for stated confidence vs what happened."""
    c, y = np.asarray(confidences, float), np.asarray(outcomes, float)
    if len(c) == 0:
        return {"n": 0, "brier": None, "base_rate": None, "bins": []}
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            bins.append({"range": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(sel.sum()),
                         "mean_confidence": float(c[sel].mean()), "observed_rate": float(y[sel].mean())})
    return {"n": int(len(c)), "brier": float(np.mean((c - y) ** 2)), "base_rate": float(y.mean()), "bins": bins}


def aggregate(episodes: list) -> dict:
    """Summary over a group of episodes (one agent in one condition)."""
    m = [episode_metrics(e) for e in episodes]
    conf = [c for e in m for c in e["confidence_trace"]]
    hit = [o == "target" for e in m for o in e["outcome_trace"]]
    ans = [(e["answer_confidence"], e["success"]) for e in m if e["answered"]]
    agree = [e["plan_agreement"] for e in m if e["plan_agreement"] is not None]
    return {
        "episodes": len(m),
        "success_rate": float(np.mean([e["success"] for e in m])),
        "mean_mixes_used": float(np.mean([e["mixes_used"] for e in m])),
        "mean_distractor_mixes": float(np.mean([e["distractor_mixes"] for e in m])),
        "mean_rejected_recipes": float(np.mean([e["rejected_recipes"] for e in m])),
        "plan_agreement": float(np.mean(agree)) if agree else None,
        "mix_calibration": calibration(conf, hit),
        "answer_calibration": calibration([a for a, _ in ans], [s for _, s in ans]),
    }


def load_episodes(root: str | Path) -> list:
    return [json.load(open(p)) for p in sorted(Path(root).rglob("episode.json"))]


def group(episodes: list) -> dict:
    out = {}
    for e in episodes:
        out.setdefault((e["agent"], e["task"]["mode"], e["task"]["feedback"]), []).append(e)
    return out
