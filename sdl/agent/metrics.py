"""Scoring of Stage 3 episodes. Works on the episode.json dicts written by AgentSession,
whoever the agent was (scripted policy, Claude Code over MCP, an RL policy)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_CONFIG = None


def _tags(reagent_id: str) -> set:
    global _CONFIG
    if _CONFIG is None:
        from sdl.config import load_config
        _CONFIG = load_config()
    return set(_CONFIG.reagents[reagent_id].tags)


def step_outcomes(ep: dict, run_id: str) -> list:
    """Ground-truth class of the vial at every image step of one recipe, rebuilt from the episode log.

    A recipe can add several samples and image in between, so one mix can show the red
    complex at an early image and a precipitate at the last one.
    """
    task, label_to_id = ep["task"], ep["hidden"]["label_to_id"]
    call = next((c for c in ep["calls"] if c["tool"] == "dispatch" and c["result"].get("run_id") == run_id), None)
    if call is None:
        return []
    tags, out = set(), []
    for step in call["args"]["recipe"]["steps"]:
        (op, args), = step.items()
        if op == "add":
            tags |= _tags(label_to_id[args["reagent"]])
        elif op == "image":
            if set(task["distractor_tags"]) <= tags:
                out.append("distractor")
            elif set(task["target_tags"]) <= tags and task["spoiler_tag"] not in tags:
                out.append("target")
            else:
                out.append("null")
    return out


def mix_outcome(ep: dict, mix: dict) -> str:
    """What a mix showed at any image step: red complex beats precipitate beats nothing."""
    steps = step_outcomes(ep, mix["run_id"]) or [mix["outcome"]]
    for kind in ("target", "distractor"):
        if kind in steps:
            return kind
    return "null"


def episode_metrics(ep: dict) -> dict:
    mixes = [m for m in ep["mixes"] if m.get("accepted")]
    rejected = [m for m in ep["mixes"] if not m.get("accepted")]
    outcomes = [mix_outcome(ep, m) for m in mixes]
    answer = ep.get("answer")
    agrees = [m["plan_agrees"] for m in mixes if m.get("plan_agrees") is not None]
    first_target = outcomes.index("target") + 1 if "target" in outcomes else None
    return {
        "episode_id": ep["episode_id"], "agent": ep["agent"],
        "mode": ep["task"]["mode"], "feedback": ep["task"]["feedback"],
        "rules_version": ep["task"].get("rules_version", 1),
        "budget": ep["task"]["budget"], "seed": ep["task"]["seed"],
        "answered": answer is not None,
        "success": bool(answer and answer["correct"]),       # correct pair submitted within budget
        "answer_confidence": answer["confidence"] if answer else None,
        "mixes_used": len(mixes),
        "rejected_recipes": len(rejected),
        "distractor_mixes": outcomes.count("distractor"),       # precipitated and never showed red
        "made_target": first_target is not None,              # actually produced the red complex
        "first_target_mix": first_target,
        "plan_agreement": float(np.mean(agrees)) if agrees else None,
        "reward_trace": [round(m["reward"], 4) for m in mixes],
        "confidence_trace": [m["confidence"] for m in mixes],
        "outcome_trace": outcomes,                            # per mix, over all image steps
        "final_outcome_trace": [m["outcome"] for m in mixes],  # per mix, final vial contents only
        "image_steps_trace": [len(step_outcomes(ep, m["run_id"])) for m in mixes],
        "tokens_total": (ep.get("usage") or {}).get("total_tokens"),
        "tokens_output": (ep.get("usage") or {}).get("output_tokens"),
        "cost_usd": (ep.get("usage") or {}).get("cost_usd"),
        "num_turns": (ep.get("usage") or {}).get("num_turns"),
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
    budget = max(e["budget"] for e in m)
    first = [e["first_target_mix"] for e in m]
    tokens = [e["tokens_total"] for e in m if e["tokens_total"] is not None]
    cost = [e["cost_usd"] for e in m if e["cost_usd"] is not None]
    return {
        "episodes": len(m),
        "mean_tokens": float(np.mean(tokens)) if tokens else None,     # per episode, all token types
        "mean_cost_usd": float(np.mean(cost)) if cost else None,
        "episodes_with_usage": len(tokens),
        # share of episodes in which the red complex had been produced within k mixes, k = 1..budget
        "produced_within": [float(np.mean([f is not None and f <= k for f in first])) for k in range(1, budget + 1)],
        "mean_mixes_to_target": float(np.mean([f for f in first if f is not None])) if any(first) else None,
        "never_produced": float(np.mean([f is None for f in first])),
        "success_rate": float(np.mean([e["success"] for e in m])),
        "mean_mixes_used": float(np.mean([e["mixes_used"] for e in m])),
        "mean_distractor_mixes": float(np.mean([e["distractor_mixes"] for e in m])),
        "mean_rejected_recipes": float(np.mean([e["rejected_recipes"] for e in m])),
        "plan_agreement": float(np.mean(agree)) if agree else None,
        "mix_calibration": calibration(conf, hit),
        "answer_calibration": calibration([a for a, _ in ans], [s for _, s in ans]),
    }


_LEGACY = {"named": "labelled", "blinded": "unknown", "online": "adaptive", "offline": "batch"}
_TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def usage_from_claude_json(text: str) -> dict | None:
    """Token usage from the output of `claude -p --output-format json`.

    Returns None when the text is not that kind of output. total_tokens counts everything the
    model processed: fresh input, cached input (written and read) and output.
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if isinstance(data, list):                       # some versions print a list of messages
        data = next((m for m in reversed(data) if isinstance(m, dict) and m.get("type") == "result"), None)
    if not isinstance(data, dict) or not isinstance(data.get("usage"), dict):
        return None
    u = data["usage"]
    out = {k: int(u.get(k) or 0) for k in _TOKEN_KEYS}
    out["total_tokens"] = sum(out[k] for k in _TOKEN_KEYS)
    out["cost_usd"] = data.get("total_cost_usd")
    out["num_turns"] = data.get("num_turns")
    out["duration_s"] = None if data.get("duration_ms") is None else data["duration_ms"] / 1000.0
    out["models"] = sorted((data.get("modelUsage") or {}).keys())
    return out


def _find_usage(episode_path: Path, root: Path, episode_id: str) -> dict | None:
    """usage.json next to the episode, else the batch runner's saved transcript for it."""
    f = episode_path.parent / "usage.json"
    if f.exists():
        return json.loads(f.read_text())
    t = root / "_transcripts" / f"{episode_id}.json"
    return usage_from_claude_json(t.read_text()) if t.exists() else None


def load_episodes(root: str | Path, rules_version: int | None = None) -> list:
    """Load episode logs. Old condition names are mapped to the current ones. Pass
    rules_version to keep only episodes played under that rule set (pilot runs are version 1)."""
    out = []
    root = Path(root).expanduser()
    for p in sorted(root.rglob("episode.json")):
        ep = json.load(open(p))
        if "usage" not in ep:
            ep["usage"] = _find_usage(p, root, ep["episode_id"])
        t = ep["task"]
        t["mode"], t["feedback"] = _LEGACY.get(t["mode"], t["mode"]), _LEGACY.get(t["feedback"], t["feedback"])
        t.setdefault("rules_version", 1)
        if rules_version is None or t["rules_version"] == rules_version:
            out.append(ep)
    return out


def group(episodes: list) -> dict:
    out = {}
    for e in episodes:
        out.setdefault((e["agent"], e["task"]["mode"], e["task"]["feedback"]), []).append(e)
    return out
