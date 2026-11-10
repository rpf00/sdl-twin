"""Scripted baseline agents. They use exactly the tools an LLM agent gets, so they
(1) test the harness end to end and (2) give reference points for LLM results."""
from __future__ import annotations

import itertools

import numpy as np

from sdl.agent.session import AgentSession


def pair_recipe_dict(first: str, second: str, rid: str = "mix") -> dict:
    return {"recipe_id": rid, "steps": [
        {"add": {"reagent": first, "volume_ml": 1.0}}, {"image": {"delay_s": 5}},
        {"add": {"reagent": second, "volume_ml": 1.0}}, {"stir": {"seconds": 10}}, {"image": {"delay_s": 5}}]}


def redness(readout: dict) -> float:
    """A policy's own reading of a readout: 0 = clearly not the red complex, 1 = clearly is."""
    img = readout["images"][-1]
    if (img["turbidity"] or 0) >= 0.25 or readout["d_blank"] < 5 or img["chroma"] < 8:
        return 0.0
    return 1.0 if 15 <= img["hue_deg"] <= 90 else 0.0


class RandomPairs:
    """Tries untested pairs in random order. Confidence = 1 / (pairs not yet ruled out)."""
    name = "random_pairs"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng([seed, 99])

    def confidence(self, n_remaining: int) -> float:
        return 1.0 / n_remaining

    def max_mixes(self, session: AgentSession) -> int:
        return session.task.budget

    def order(self, labels: list, session: AgentSession) -> list:
        pairs = list(itertools.combinations(labels, 2))
        return [pairs[i] for i in self.rng.permutation(len(pairs))]

    def run(self, session: AgentSession) -> None:
        brief = session.briefing()
        labels = sorted(brief["samples"])
        todo = self.order(labels, session)
        n_pairs, tested, found = len(todo), [], None
        online = session.task.feedback == "online"
        for k, (a, b) in enumerate(todo[:self.max_mixes(session)]):
            out = session.dispatch(pair_recipe_dict(a, b, f"{a}+{b}"), self.confidence(n_pairs - k if online else n_pairs), [a, b],
                                   f"try untested pair {a}+{b}")
            tested.append((a, b, out))
            if online and out.get("accepted") and redness(out) > 0.5:
                found = (a, b)
                break
        if not online:
            results = {r["run_id"]: r for r in session.reveal_results()["results"]}
            for a, b, out in tested:
                if out.get("accepted") and redness(results[out["run_id"]]) > 0.5:
                    found = (a, b)
        if found:
            session.submit_answer(list(found), 0.95)
        else:
            left = [p for p in todo if p not in [(a, b) for a, b, _ in tested]]
            guess = left[0] if left else todo[0]
            session.submit_answer(list(guess), 1.0 / max(len(left), 1) if left else 0.05)


class Overconfident(RandomPairs):
    """Same search as RandomPairs but always claims 90%. Exists to show the calibration metric bites."""
    name = "overconfident"

    def confidence(self, n_remaining: int) -> float:
        return 0.9


class PriorKnowledge(RandomPairs):
    """Stands in for "the agent already knows the chemistry": if the sample descriptions name
    iron and thiocyanate, mix those first with high confidence. Blinded, it has nothing to go on
    and behaves like RandomPairs."""
    name = "prior_knowledge"

    def order(self, labels: list, session: AgentSession) -> list:
        todo = super().order(labels, session)
        desc = session.briefing()["samples"]
        fe = [l for l in labels if "Fe" in desc[l]]
        scn = [l for l in labels if "SCN" in desc[l]]
        self.known = (fe[0], scn[0]) if fe and scn else None
        if self.known:
            todo.sort(key=lambda p: set(p) != set(self.known))
        return todo

    def max_mixes(self, session: AgentSession) -> int:
        return 1 if self.known else session.task.budget     # one confirming mix is enough

    def confidence(self, n_remaining: int) -> float:
        if getattr(self, "known", None) and not getattr(self, "_used", False):
            self._used = True
            return 0.95
        return 1.0 / n_remaining


POLICIES = {p.name: p for p in (RandomPairs, Overconfident, PriorKnowledge)}
