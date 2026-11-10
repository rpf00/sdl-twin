# Running Stage 3 with Claude Code

Claude Code plays the task through the MCP server in `sdl/agent/mcp_server.py`.
One server process is one episode in one condition.

## 1. Install

```bash
pip install -e ".[all]"
```

## 2. Keep the answer key out of reach

This repository contains the answers: the reagent configs, and each episode's
hidden label map in `episode.json`. An agent that can read files could look them
up. So:

- start Claude Code from an **empty folder**, not from this repo;
- point `--store` at a folder **outside** that working folder;
- limit the session to the `sdl-lab` tools (see the Claude Code permission
  settings in its documentation).

## 3. Register the server

Use absolute paths: Claude Code starts the server itself, outside your virtualenv.

```bash
mkdir -p ~/sdl-arena ~/sdl-runs && cd ~/sdl-arena

claude mcp add --transport stdio sdl-lab -- \
  /ABSOLUTE/PATH/TO/sdl-twin/.venv/bin/python -m sdl.agent.mcp_server \
  --mode blinded --feedback online --seed 0 --store ~/sdl-runs
```

Options: `--mode named|blinded`, `--feedback online|offline`, `--seed N`,
`--budget K`, `--agent-name NAME` (use it to tag the model, e.g. `claude-fable-5-2`).

To switch condition or seed, remove and re-add the server:

```bash
claude mcp remove sdl-lab
```

MCP command syntax changes between Claude Code versions. If the command above is
rejected, check https://docs.claude.com/en/docs/claude-code/mcp.

## 4. Play one episode

Start `claude` in `~/sdl-arena` and give it a neutral prompt, the same every time:

> Use the sdl-lab tools to complete the task. Call `briefing` first and follow
> the goal it gives you. Finish by calling `submit_answer`.

Each time Claude Code starts, the server begins a fresh episode in a new folder
(`...-seed0`, `...-seed0-r2`, ...). Sessions that end without `submit_answer`
are kept and flagged by the report.

## 5. Score

```bash
python experiments/stage3_report.py ~/sdl-runs
python experiments/stage3_agent.py            # baselines to compare against
```

## What to run

For the named-vs-blinded comparison, run the same seeds in all four conditions.
With a 3-mix budget, several episodes per condition are needed before success
rates mean anything: blind pair-by-pair search succeeds about two times in three
by chance. Calibration needs more still, since each episode contributes at most
three confidence statements.
