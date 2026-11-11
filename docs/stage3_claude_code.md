# Running Stage 3 with Claude Code

Claude Code plays the task through the MCP server in `sdl/agent/mcp_server.py`.
One server process is one episode in one condition.

## 1. Install

```bash
pip install -e ".[all]" matplotlib
```

## 2. Keep the answer key out of reach

This repository contains the answers: the reagent configs, and each episode's
hidden label map in `episode.json`. An agent that can read files, or search the
web for this public repository, could look them up. So:

- Claude Code runs in an **empty folder**, not in this repo;
- episode logs and server configs live **outside** that folder;
- the session is limited to the `sdl-lab` tools: built-in tools, including web
  access, are switched off.

`experiments/run_claude_grid.py` sets all of this up.

## 3. Run the six tests

```bash
cd ~/sdl-twin && source .venv/bin/activate
python experiments/run_claude_grid.py --models opus,haiku --seeds 0-4 --dry-run   # show the commands
python experiments/run_claude_grid.py --models opus,haiku --seeds 0-4             # 2 models x 3 conditions x 5 seeds
```

- `--models` are passed to `claude --model` and tag the episodes. Use the aliases
  your Claude Code accepts, or full model names.
- `--modes labelled,listed,unknown` selects the information conditions.
- Episodes go to `~/sdl-runs-v2`. Finished episodes are skipped on a rerun.
- If the first episode fails, the script stops and points to the error output.
  The likeliest cause is a `claude` flag that changed; check `claude --help` and
  https://docs.claude.com/en/docs/claude-code/mcp.

## 4. Score and plot

```bash
python experiments/stage3_report.py ~/sdl-runs-v2
python experiments/stage3_plots.py ~/sdl-runs-v2
open results/stage3_overview.png
```

## Playing one episode by hand

Useful for watching what the agent does.

```bash
mkdir -p ~/sdl-arena ~/sdl-configs
cat > ~/sdl-configs/listed-seed0.json <<'JSON'
{"mcpServers": {"sdl-lab": {"type": "stdio",
  "command": "/ABSOLUTE/PATH/TO/sdl-twin/.venv/bin/python",
  "args": ["-m", "sdl.agent.mcp_server", "--mode", "listed", "--seed", "0",
           "--store", "/ABSOLUTE/PATH/TO/sdl-runs-manual", "--agent-name", "manual"]}}}
JSON
cd ~/sdl-arena
claude --mcp-config ~/sdl-configs/listed-seed0.json --strict-mcp-config --tools ""
```

Then give it the same prompt the batch runner uses:

> Use the sdl-lab tools to complete the task. Call briefing first and follow the
> goal it gives you. Finish by calling submit_answer.

## How many episodes

Five seeds per test show large effects only. The headline number is how many
mixes an agent needs before the red complex appears; random pair search finds it
at about one chance in six per mix, so differences of a mix or so need more
seeds to be convincing.
