# Self-Driving Lab - Digital Twin (sdl-twin)

Digital twins for the Fe/SCN self-driving-lab proof of concept. Everything here
runs without hardware, and the same code runs on the real lab once
`sdl/backends/real.py` is filled in.

| Twin | What it does | Where |
|---|---|---|
| State twin | Tracks what is where, dry-runs every recipe, rejects unsafe or impossible ones | `sdl/state/` |
| Outcome emulator | Predicts what the camera would see: chemistry, then optics, then camera | `sdl/emulator/`, `sdl/backends/sim.py` |
| Data twin | Stores every run in one format; refits the emulator from real runs | `sdl/data/` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
python experiments/concentration_scan.py
python experiments/stage1_confusion.py
python experiments/stage2_noise_floor.py
```

```python
from sdl import Lab
from sdl.schema import Recipe

lab = Lab(seed=0, store="runs")          # simulated backend by default
recipe = Recipe.from_yaml("""
recipe_id: R0042
steps:
  - add: {reagent: A, volume_ml: 1.0}
  - image: {delay_s: 5}
  - add: {reagent: B, volume_ml: 1.0}
  - stir: {seconds: 10}
  - image: {delay_s: 5}
""")
print(lab.verify(recipe).to_dict())      # dry run, nothing happens
result = lab.dispatch(recipe)            # verify, execute, read out, record
print(result.to_agent_dict())
```

## Stage 3: the agent task

The task: one pair among four samples forms a red complex; find it in at most
k mixes (default 3), stating a confidence with every mix. It runs in four
conditions:

| | What changes |
|---|---|
| `named` vs `blinded` labels | The agent sees chemical names, or shuffled labels S1-S4 |
| `online` vs `offline` feedback | Readout after every mix, or only after all mixes are done |

```bash
pip install -e ".[all]"                       # adds the MCP and Gymnasium extras
python experiments/stage3_agent.py            # scripted baselines, all four conditions
python experiments/stage3_report.py runs/     # score any recorded episodes
```

- **Tools** (`sdl/agent/session.py`): `briefing`, `inspect`, `verify`, `dispatch`,
  `get_result`, `reveal_results`, `submit_answer`. Scripted agents, the MCP server
  and the Gymnasium environment all go through them, so every episode is logged
  and scored the same way.
- **Claude Code** plays through the MCP server: see `docs/stage3_claude_code.md`.
- **RL** uses `sdl.agent.gym_env.SdlPairEnv` (reward = size of colour change x hue
  match x clarity).
- **Metrics** (`sdl/agent/metrics.py`): success within budget, mixes spent on the
  distractor, calibration of stated confidence (Brier score and reliability
  table), and whether the samples the agent said it would mix match the recipe
  it ran.
- **Baselines** (`sdl/agent/policies.py`): `random_pairs`, `overconfident` (same
  search, always claims 90%) and `prior_knowledge` (reads the names, so it only
  helps in the named condition).

## How it fits together

```
recipe -> Lab.verify  (fork of the state twin, safety and logistics rules)
       -> Lab.dispatch -> Backend.add / stir / image   (SimBackend or RealBackend)
                       -> vision pipeline (white balance, CIELAB, d, hue, chroma, turbidity)
                       -> TraceStore (record.json + frames, one folder per run)
```

- The state twin holds the **nominal** state. `SimBackend` keeps a separate hidden
  "true" state with dispensing error and imperfect mixing. Agents only ever see
  the readout, never the truth (`RunResult.to_agent_dict()`).
- The state only changes through events (`sdl/schema/events.py`), so it can be
  replayed from the log and forked for dry runs.
- Two colour distances are reported. `d_pre` compares the last image with the
  first image of the same recipe, as in the PoC draft. `d_blank` compares it with
  the session's water blank. Use `d_blank` for order-invariance: `d_pre` differs
  between A->B and B->A simply because the first reagent differs.

## What is real and what is a guess

- Formation constants: literature values (de Berg, Maeder & Clifford, Inorg.
  Chim. Acta 466, 249 (2017)), corrected for ionic strength.
- Molar absorptivities, precipitate optics, camera noise, dispensing error:
  **priors**, each marked with a confidence level in
  `configs/emulator_priors.yaml`. Noise numbers produced by the emulator are
  therefore placeholders. The real noise floor comes from Stage 0-2 on hardware.
- Hardware: **TBD**, see `docs/hardware_TBD.md`.

## Concentrations

`configs/reagents.yaml` uses lower concentrations than the first PoC draft and
explains why. In short: the draft's 0.02 M Fe would drive the blue channel to
zero, so the camera could say "red" but could not quantify anything. The Stage 2
series (0.05-0.4 mM Fe stock) is the range the emulator predicts to be resolvable
with a 1 cm path. Check with `experiments/concentration_scan.py`, then confirm on
the bench.

## Roadmap

1. Schemas and state twin: done
2. Emulator and vision pipeline, Stages 1-2 in simulation: done
3. Agent tools (MCP), Gymnasium env, Stage 3 harness: done
   - next: run Claude Code episodes in all four conditions and compare with the baselines
4. Real backend and calibration against bench data: waiting for hardware
