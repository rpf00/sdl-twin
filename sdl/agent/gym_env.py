"""Gymnasium environment: the RL face of the same Stage 3 task.

    action       one of the 12 ordered pairs of sample slots (first added, second added)
    observation  [d_blank/100, cos(hue), sin(hue), chroma/100, turbidity, mixes_left/budget] of the last mix
    reward       sdl.agent.reward (size of colour change x hue match x clarity), in [0, 1]
    terminated   when a mix scores reward >= success_reward
    truncated    when the mix budget is spent

blinded=True reshuffles which slot holds which reagent on every reset, so a policy
cannot memorise "slot 0 + slot 1" and has to use the readouts. blinded=False keeps
the mapping fixed, the RL analogue of the named condition.

Needs: pip install -e ".[rl]"
"""
from __future__ import annotations

import itertools
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sdl.agent.policies import pair_recipe_dict
from sdl.agent.session import AgentSession
from sdl.agent.task import Task


class SdlPairEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, blinded: bool = True, budget: int | None = None, success_reward: float = 0.5,
                 store_dir=None):
        self.blinded, self.budget, self.success_reward, self.store_dir = blinded, budget, success_reward, store_dir
        self.n = len(Task.from_config().samples)
        self.pairs = list(itertools.permutations(range(self.n), 2))
        self.action_space = spaces.Discrete(len(self.pairs))
        self.observation_space = spaces.Box(-1.0, 2.0, shape=(6,), dtype=np.float32)
        self.session: AgentSession | None = None
        self._episode = 0

    def _obs(self, readout: dict | None) -> np.ndarray:
        left = self.session.task.budget - self.session.lab.twin.state.mixes_used
        frac = left / self.session.task.budget
        if readout is None:
            return np.array([0, 0, 0, 0, 0, frac], dtype=np.float32)
        img = readout["images"][-1]
        h = math.radians(img["hue_deg"])
        return np.array([readout["d_blank"] / 100, math.cos(h), math.sin(h), img["chroma"] / 100,
                         img["turbidity"] or 0.0, frac], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        ep_seed = int(self.np_random.integers(0, 2**31 - 1))
        task = Task.from_config(mode="blinded" if self.blinded else "named", feedback="online",
                                seed=ep_seed, budget=self.budget, show_reward=True)
        self._episode += 1
        sdir = None if self.store_dir is None else f"{self.store_dir}/rl-episode-{self._episode:05d}"
        self.session = AgentSession(task, store_dir=sdir, agent_name="rl")
        self.slots = sorted(self.session.label_to_id)
        return self._obs(None), {"slots": self.slots}

    def step(self, action: int):
        i, j = self.pairs[int(action)]
        a, b = self.slots[i], self.slots[j]
        out = self.session.dispatch(pair_recipe_dict(a, b, f"{a}+{b}"), confidence=0.5, planned_samples=[a, b])
        if not out.get("accepted"):       # only possible if the budget is already spent
            return self._obs(None), 0.0, False, True, {"issues": out.get("issues")}
        r = float(out["reward"])
        terminated = r >= self.success_reward
        truncated = (not terminated) and out["mixes_left"] <= 0
        return self._obs(out), r, terminated, truncated, {"samples": [a, b], "run_id": out["run_id"]}
