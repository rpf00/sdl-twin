"""Run Stage 3 episodes with a scripted policy."""
from __future__ import annotations

from pathlib import Path

from sdl.agent.session import AgentSession
from sdl.agent.task import Task


def run_episode(policy, task: Task, store_dir: str | Path | None = None) -> dict:
    sdir = None
    if store_dir:
        sdir = Path(store_dir) / f"{policy.name}-{task.mode}-{task.feedback}-seed{task.seed}"
    session = AgentSession(task, store_dir=sdir, agent_name=policy.name)
    policy.run(session)
    return session.episode


def run_grid(policy_classes, seeds, modes=("labelled", "listed", "unknown"), feedbacks=("adaptive",),
             store_dir: str | Path | None = None, **task_overrides) -> list:
    episodes = []
    for cls in policy_classes:
        for mode in modes:
            for fb in feedbacks:
                for seed in seeds:
                    task = Task.from_config(mode=mode, feedback=fb, seed=seed, **task_overrides)
                    episodes.append(run_episode(cls(seed=seed), task, store_dir))
    return episodes
