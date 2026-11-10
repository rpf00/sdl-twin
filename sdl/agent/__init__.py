"""Agent layer: the Stage 3 dual-interface task.

  task.py        task definition, named vs blinded labels
  session.py     AgentSession: the tool surface an agent uses (inspect / verify / dispatch / ...)
  reward.py      scalar reward from the image readout
  metrics.py     per-episode and aggregate evaluation
  policies.py    scripted baseline agents
  harness.py     run episodes with a policy
  mcp_server.py  the same tools over MCP, for Claude Code      (needs: pip install -e ".[agent]")
  gym_env.py     Gymnasium environment for RL                   (needs: pip install -e ".[rl]")
"""
from sdl.agent.task import Task  # noqa: F401
from sdl.agent.session import AgentSession  # noqa: F401
