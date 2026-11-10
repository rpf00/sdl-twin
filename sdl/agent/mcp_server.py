"""MCP server: the Stage 3 tools for Claude Code (or any MCP client).

One server process = one episode. Start it with the condition you want to test:

    python -m sdl.agent.mcp_server --mode blinded --feedback online --seed 0 --store runs/claude

The episode log is written to <store>/<episode_id>/episode.json after every call
and scored later with experiments/stage3_report.py.

Needs: pip install -e ".[agent]"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sdl.agent.session import AgentSession
from sdl.agent.task import FEEDBACK, MODES, Task

try:                                    # mcp >= 2
    from mcp.server.mcpserver import MCPServer
except ImportError:                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer


def build_server(session: AgentSession) -> "MCPServer":
    server = MCPServer("sdl-lab", instructions=session.task.goal() + " Call briefing first.")

    @server.tool()
    def briefing() -> dict:
        """Read this first: the goal, the samples on the shelf, the mix budget and the recipe format."""
        return session.briefing()

    @server.tool()
    def inspect() -> dict:
        """Current lab state: remaining stock of each sample, mixes used and left, lab limits."""
        return session.inspect()

    @server.tool()
    def verify(recipe: dict) -> dict:
        """Dry-run a recipe without executing it. Free: costs no mix. Returns ok plus any issues."""
        return session.verify(recipe)

    @server.tool()
    def dispatch(recipe: dict, confidence: float, planned_samples: list[str], reasoning: str = "") -> dict:
        """Execute a recipe in the lab. Costs one mix if accepted.

        confidence: your probability (0 to 1) that this mix produces the red complex.
        planned_samples: the sample labels you intend to combine in this recipe.
        reasoning: one sentence on why this mix.
        Returns the image readout (or a note that it is withheld, in offline feedback mode).
        """
        return session.dispatch(recipe, confidence, planned_samples, reasoning)

    @server.tool()
    def get_result(run_id: str) -> dict:
        """Fetch the readout of an earlier mix again."""
        return session.get_result(run_id)

    @server.tool()
    def reveal_results() -> dict:
        """Offline feedback mode only: reveal all readouts. No further mixes are allowed afterwards."""
        return session.reveal_results()

    @server.tool()
    def submit_answer(samples: list[str], confidence: float) -> dict:
        """Final answer: the two sample labels that form the red complex, and your confidence (0 to 1).
        Closes the episode."""
        return session.submit_answer(samples, confidence)

    return server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, default="blinded")
    p.add_argument("--feedback", choices=FEEDBACK, default="online")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=None, help="mix budget k (default: configs/task.yaml)")
    p.add_argument("--agent-name", default="claude-code")
    p.add_argument("--store", default="runs/claude", help="folder that receives one sub-folder per episode")
    a = p.parse_args()
    task = Task.from_config(mode=a.mode, feedback=a.feedback, seed=a.seed, budget=a.budget)
    root = Path(a.store).expanduser().resolve()
    base = f"{a.agent_name}-{a.mode}-{a.feedback}-seed{a.seed}"
    episode_id, n = base, 1
    while (root / episode_id).exists():      # never overwrite an earlier episode
        n += 1
        episode_id = f"{base}-r{n}"
    session = AgentSession(task, store_dir=root / episode_id,
                           agent_name=a.agent_name, episode_id=episode_id)
    build_server(session).run()


if __name__ == "__main__":
    main()
