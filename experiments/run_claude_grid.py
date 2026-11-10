"""Run Stage 3 episodes with Claude Code, unattended, over a grid of conditions and seeds.

For every (labels, feedback, seed) it writes a one-episode MCP config, starts
Claude Code in print mode inside an empty "arena" folder with only the lab tools,
and lets it play. Episodes that already have a submitted answer are skipped, so
the script can be stopped and restarted.

    python experiments/run_claude_grid.py --agent-name opus-5.5 --seeds 0-4
    python experiments/run_claude_grid.py --agent-name opus-5.5 --seeds 0-4 --dry-run   # show commands only
    python experiments/stage3_report.py ~/sdl-runs-grid

Keep --arena, --store and --configs outside this repository and outside each
other: the repo and the episode logs contain the answer key.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROMPT = ("Use the sdl-lab tools to complete the task. Call briefing first and follow the goal it "
          "gives you. Finish by calling submit_answer.")


def parse_seeds(text: str) -> list:
    out = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def finished(store: Path, base: str) -> bool:
    """True if an episode of this condition and seed already has a submitted answer."""
    for d in store.glob(f"{base}*"):
        f = d / "episode.json"
        if d.name in (base,) or d.name.startswith(base + "-r"):
            if f.exists() and json.loads(f.read_text()).get("closed"):
                return True
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-name", default="claude-code", help="tag for the model being tested, e.g. opus-5.5")
    p.add_argument("--model", default=None, help="passed to `claude --model`; default is your Claude Code default")
    p.add_argument("--seeds", default="0-4", help="e.g. 0-4 or 0,2,5")
    p.add_argument("--modes", default="named,blinded")
    p.add_argument("--feedbacks", default="online,offline")
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--arena", default="~/sdl-arena", help="empty working folder for Claude Code")
    p.add_argument("--store", default="~/sdl-runs-grid", help="where episode logs go")
    p.add_argument("--configs", default="~/sdl-configs", help="where the per-episode MCP configs go")
    p.add_argument("--timeout", type=int, default=900, help="seconds per episode")
    p.add_argument("--claude-bin", default="claude")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    arena, store, configs = (Path(x).expanduser().resolve() for x in (a.arena, a.store, a.configs))
    for d in (arena, store, configs, store / "_transcripts"):
        d.mkdir(parents=True, exist_ok=True)
    leftovers = [f.name for f in arena.iterdir() if not f.name.startswith(".")]
    if leftovers:
        sys.exit(f"arena {arena} is not empty ({leftovers[:3]}...). Use an empty folder.")

    grid = [(m, f, s) for m in a.modes.split(",") for f in a.feedbacks.split(",") for s in parse_seeds(a.seeds)]
    print(f"{len(grid)} episodes planned, agent tag '{a.agent_name}', logs in {store}\n")
    done = failed = skipped = 0
    for mode, fb, seed in grid:
        base = f"{a.agent_name}-{mode}-{fb}-seed{seed}"
        if finished(store, base):
            print(f"skip  {base} (already answered)")
            skipped += 1
            continue
        args = ["-m", "sdl.agent.mcp_server", "--mode", mode, "--feedback", fb, "--seed", str(seed),
                "--store", str(store), "--agent-name", a.agent_name]
        if a.budget is not None:
            args += ["--budget", str(a.budget)]
        cfg = configs / f"{base}.json"
        cfg.write_text(json.dumps({"mcpServers": {"sdl-lab": {
            "type": "stdio", "command": sys.executable, "args": args}}}, indent=2))
        cmd = [a.claude_bin, "-p", PROMPT, "--mcp-config", str(cfg), "--strict-mcp-config",
               "--tools", "", "--allowedTools", "mcp__sdl-lab", "--output-format", "json"]
        if a.model:
            cmd += ["--model", a.model]
        if a.dry_run:
            print("would run (in", arena, "):\n  " + " ".join(repr(c) if " " in c or not c else c for c in cmd))
            continue
        print(f"run   {base} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            out = subprocess.run(cmd, cwd=arena, capture_output=True, text=True, timeout=a.timeout)
            (store / "_transcripts" / f"{base}.json").write_text(out.stdout or "")
            if out.returncode != 0:
                (store / "_transcripts" / f"{base}.stderr.txt").write_text(out.stderr or "")
            status = "ok" if out.returncode == 0 else f"claude exited with code {out.returncode}"
        except subprocess.TimeoutExpired:
            status = f"timed out after {a.timeout}s"
        answered = finished(store, base)
        print(f"{status}; {'answer submitted' if answered else 'NO ANSWER SUBMITTED'} ({time.time() - t0:.0f}s)")
        done += answered
        failed += not answered
        if not answered and done == 0 and failed == 1:
            err = store / "_transcripts" / f"{base}.stderr.txt"
            sys.exit(f"\nThe first episode did not complete, so the run was stopped. "
                     f"Check {err if err.exists() else store / '_transcripts'} and the flags in this script "
                     f"against `claude --help`.")
    if not a.dry_run:
        print(f"\n{done} completed, {failed} without an answer, {skipped} skipped.")
        print(f"Score with: python experiments/stage3_report.py {store}")


if __name__ == "__main__":
    main()
