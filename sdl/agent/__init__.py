"""Agent interface (step 3 of the build plan, not implemented yet).

Planned contents:
  - mcp_server.py : exposes Lab.inspect / verify / dispatch / get_result as MCP tools for Claude Code
  - gym_env.py    : Gymnasium wrapper (action = recipe, reward = d and hue match) for the RL interface
  - harness.py    : Stage 3 evaluation (budget, stated confidence, calibration, mixes spent on the distractor)

The Lab already returns `RunResult.to_agent_dict()`, which is the payload these
tools will hand to an agent, and `Lab.dispatch(recipe, agent={...})` already
stores reasoning and confidence in the trace.
"""
