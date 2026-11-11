"""AgentSession: the tool surface of the Stage 3 task.

Every agent, scripted or LLM, local or over MCP, goes through these methods, so
every episode produces the same log and is scored the same way. All methods take
and return plain JSON-serialisable data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from sdl.agent.reward import reward as reward_fn
from sdl.agent.task import Task
from sdl.config import load_config
from sdl.data.store import TraceStore
from sdl.lab import Lab
from sdl.schema.recipe import Add, Recipe, RecipeFormatError

RECIPE_EXAMPLE = {"recipe_id": "my-mix-1", "steps": [
    {"add": {"reagent": "<label>", "volume_ml": 1.0}}, {"image": {"delay_s": 5}},
    {"add": {"reagent": "<label>", "volume_ml": 1.0}}, {"stir": {"seconds": 10}}, {"image": {"delay_s": 5}}]}


class AgentSession:
    def __init__(self, task: Task | None = None, store_dir: str | Path | None = None,
                 lab: Lab | None = None, agent_name: str = "unknown", episode_id: str | None = None):
        self.task = task or Task.from_config()
        self.config = lab.config if lab else load_config()
        self.store_dir = Path(store_dir) if store_dir else None
        store = TraceStore(self.store_dir / "traces") if self.store_dir else None
        self.lab = lab or Lab(config=self.config, seed=self.task.seed, budget_mixes=self.task.budget,
                              store=store, n_vials=self.task.budget + 2)
        self.label_to_id = self.task.labels()
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.results: dict = {}       # run_id -> full agent-visible readout
        self.revealed = self.task.feedback == "adaptive"
        self.closed = False
        self.episode = {
            "episode_id": episode_id or f"{agent_name}-{self.task.mode}-{self.task.feedback}-seed{self.task.seed}",
            "agent": agent_name, "task": self.task.to_dict(),
            "backend": self.lab.backend.describe(),
            "hidden": {"label_to_id": self.label_to_id,
                       "target_labels": sorted(self.id_to_label[i] for i in self.task.target_pair(self.config))},
            "calls": [], "mixes": [], "answer": None, "closed": False,
        }
        self._save()

    # ---- read-only tools ----------------------------------------------------
    def briefing(self) -> dict:
        return self._log("briefing", {}, {
            "goal": self.task.goal(),
            "samples": {lab: self.task.describe(lab, rid, self.config) for lab, rid in self.label_to_id.items()},
            **({"shelf_contents": self.task.shelf_contents(self.config)} if self.task.mode == "listed" else {}),
            "budget_mixes": self.task.budget, "max_samples_per_mix": self.task.max_samples_per_mix,
            "feedback": self.task.feedback,
            "tools": ["inspect", "verify", "dispatch", "get_result", "reveal_results", "submit_answer"],
            "recipe_format": RECIPE_EXAMPLE,
            "recipe_notes": "Steps: add, stir, image. A vial holds up to "
                            f"{self.config.limits.vial_capacity_ml} mL. A recipe may use at most "
                            f"{self.task.max_samples_per_mix} different samples (one sample alone is fine). "
                            "Recipes are dry-run first; rejected recipes cost nothing.",
        })

    def inspect(self) -> dict:
        s = self.lab.twin.state
        return self._log("inspect", {}, {
            "samples": {lab: {"description": self.task.describe(lab, rid, self.config),
                              "stock_ml": round(s.stocks_ml[rid], 2)} for lab, rid in self.label_to_id.items()},
            "mixes_used": s.mixes_used, "mixes_left": self.task.budget - s.mixes_used,
            "clean_vials": len(s.clean_vials()), "limits": dict(self.config.limits.__dict__),
            "feedback": self.task.feedback, "results_revealed": self.revealed, "episode_closed": self.closed,
        })

    def verify(self, recipe: dict) -> dict:
        translated, issues = self._translate(recipe)
        if translated is None:
            return self._log("verify", {"recipe": recipe}, {"ok": False, "issues": issues})
        report = self.lab.verify(translated)
        return self._log("verify", {"recipe": recipe},
                         {"ok": report.ok, "issues": [self._relabel(i.__dict__) for i in report.issues]})

    def get_result(self, run_id: str) -> dict:
        if run_id not in self.results:
            out = {"error": f"unknown run_id '{run_id}'"}
        elif not self.revealed:
            out = {"run_id": run_id, "readout": "withheld until reveal_results"}
        else:
            out = self.results[run_id]
        return self._log("get_result", {"run_id": run_id}, out)

    # ---- acting tools ---------------------------------------------------------
    def dispatch(self, recipe: dict, confidence: float | None = None,
                 planned_samples: list | None = None, reasoning: str = "") -> dict:
        args = {"recipe": recipe, "confidence": confidence, "planned_samples": planned_samples,
                "reasoning": reasoning}
        if self.closed:
            return self._log("dispatch", args, {"accepted": False, "error": "episode is closed"})
        if self.task.feedback == "batch" and self.revealed:
            return self._log("dispatch", args, {"accepted": False, "error": "results were revealed; no more mixes"})
        if confidence is None or not (0.0 <= float(confidence) <= 1.0):
            return self._log("dispatch", args, {
                "accepted": False,
                "error": "state your confidence (0 to 1) that this mix produces the red complex before mixing"})
        translated, issues = self._translate(recipe)
        if translated is None:
            self.episode["mixes"].append({"accepted": False, "issues": issues, "confidence": float(confidence)})
            return self._log("dispatch", args, {"accepted": False, "issues": issues})

        labels = [s.reagent for s in Recipe.from_dict(recipe).steps if isinstance(s, Add)]
        true_ids = [self.label_to_id[lab] for lab in labels]
        res = self.lab.dispatch(translated, agent={"episode": self.episode["episode_id"], "labels": labels,
                                                   "confidence": float(confidence),
                                                   "planned_samples": planned_samples, "reasoning": reasoning})
        issues = [self._relabel(i) for i in res.issues]
        mix = {"run_id": res.run_id, "accepted": res.accepted, "labels": labels, "true_ids": true_ids,
               "confidence": float(confidence), "planned_samples": planned_samples, "reasoning": reasoning,
               "plan_agrees": None if planned_samples is None else set(planned_samples) == set(labels),
               "issues": issues}
        if not res.accepted:
            self.episode["mixes"].append(mix)
            return self._log("dispatch", args, {"accepted": False, "issues": issues})

        last = res.features[-1]
        r = reward_fn(res.d_blank, last.hue_deg, last.turbidity, self.task.reward)
        mix.update({"outcome": self.task.outcome(true_ids, self.config), "reward": r,
                    "d_blank": res.d_blank, "hue_deg": last.hue_deg, "turbidity": last.turbidity})
        self.episode["mixes"].append(mix)
        readout = {"run_id": res.run_id, "accepted": True, "samples": labels, "warnings": issues,
                   "images": [f.to_dict() for f in res.features],
                   "d_pre": None if res.d_pre is None else round(res.d_pre, 3),
                   "d_blank": round(res.d_blank, 3),
                   "mixes_left": self.task.budget - self.lab.twin.state.mixes_used}
        if self.task.show_reward:
            readout["reward"] = round(r, 4)
        self.results[res.run_id] = readout
        if self.revealed:
            return self._log("dispatch", args, readout)
        return self._log("dispatch", args, {"run_id": res.run_id, "accepted": True, "warnings": issues,
                                            "readout": "withheld until reveal_results",
                                            "mixes_left": readout["mixes_left"]})

    def reveal_results(self) -> dict:
        if self.task.feedback == "adaptive":
            return self._log("reveal_results", {}, {"note": "feedback is adaptive; results are already visible",
                                                    "results": list(self.results.values())})
        self.revealed = True
        return self._log("reveal_results", {}, {"results": list(self.results.values()),
                                                "note": "no further mixes are allowed"})

    def submit_answer(self, samples: list, confidence: float | None = None) -> dict:
        args = {"samples": samples, "confidence": confidence}
        if self.closed:
            return self._log("submit_answer", args, {"recorded": False, "error": "an answer was already submitted"})
        unknown = [s for s in samples if s not in self.label_to_id]
        if unknown or len(set(samples)) != 2:
            return self._log("submit_answer", args, {
                "recorded": False, "error": f"give exactly two distinct sample labels from {sorted(self.label_to_id)}"})
        if confidence is None or not (0.0 <= float(confidence) <= 1.0):
            return self._log("submit_answer", args, {"recorded": False, "error": "give a confidence between 0 and 1"})
        true = frozenset(self.label_to_id[s] for s in samples)
        self.episode["answer"] = {"labels": sorted(samples), "true_ids": sorted(true),
                                  "confidence": float(confidence),
                                  "correct": true == self.task.target_pair(self.config)}
        self.closed = self.episode["closed"] = True
        return self._log("submit_answer", args, {"recorded": True, "note": "episode closed"})

    # ---- internals ----------------------------------------------------------------
    def _translate(self, recipe: dict):
        """Agent labels -> true reagent ids. Returns (Recipe | None, issues)."""
        try:
            parsed = Recipe.from_dict(recipe)
        except (RecipeFormatError, TypeError, AttributeError) as exc:
            return None, [{"code": "BAD_FORMAT", "message": str(exc), "step": None, "severity": "error"}]
        issues, steps = [], []
        for i, s in enumerate(parsed.steps):
            if isinstance(s, Add):
                if s.reagent not in self.label_to_id:
                    issues.append({"code": "UNKNOWN_REAGENT", "step": i, "severity": "error",
                                   "message": f"'{s.reagent}' is not on the shelf; available: {sorted(self.label_to_id)}"})
                    continue
                s = Add(self.label_to_id[s.reagent], s.volume_ml)
            steps.append(s)
        distinct = {s.reagent for s in steps if isinstance(s, Add)}
        if len(distinct) > self.task.max_samples_per_mix:
            issues.append({"code": "TOO_MANY_SAMPLES", "step": None, "severity": "error",
                           "message": f"a mix may contain at most {self.task.max_samples_per_mix} different samples; "
                                      f"this recipe uses {len(distinct)}"})
        if issues:
            return None, issues
        return Recipe(parsed.recipe_id, steps), []

    def _relabel(self, issue: dict) -> dict:
        """Never leak true reagent ids through error messages."""
        msg = issue.get("message", "")
        msg = re.sub(r"'([^']+)'", lambda m: f"'{self.id_to_label.get(m.group(1), m.group(1))}'", msg)
        return {**issue, "message": msg}

    def _log(self, tool: str, args: dict, result: dict) -> dict:
        self.episode["calls"].append({"i": len(self.episode["calls"]), "tool": tool, "args": args, "result": result})
        self._save()
        return result

    def _save(self) -> None:
        if self.store_dir:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            with open(self.store_dir / "episode.json", "w") as fh:
                json.dump(self.episode, fh, indent=2)
