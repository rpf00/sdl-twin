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
pip install -e ".[all]" matplotlib
python -m pytest

python experiments/concentration_scan.py      # which concentrations the camera can resolve
python experiments/stage1_confusion.py        # Stage 1: discrimination and order invariance
python experiments/stage2_noise_floor.py      # Stage 2: noise floor on the dilution series
python experiments/noise_scenarios.py         # Stages 1-2 under optimistic / typical / pessimistic noise
python experiments/stage12_plots.py           # figures for Stages 1-2 (written to results/)
python experiments/stage3_agent.py            # Stage 3: scripted baseline agents
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

## The experiment

| # | Sample | Role |
|---|---|---|
| A | FeCl3 2 mM in 0.01 M HCl | Iron |
| B | KSCN 0.02 M | Thiocyanate |
| C | NaOH 0.1 M | Base |
| D | NaCl 0.1 M | Null control |
| A1-A4 | FeCl3 0.4, 0.2, 0.1, 0.05 mM in 0.01 M HCl | Stage 2 dilution series |
| W | DI water | Blank, imaged at the start of each session |

A + B gives the orange-red complex (target), A + C a turbid precipitate
(distractor), everything else is a null. `configs/reagents.yaml` explains why the
concentrations are lower than in the first PoC draft.

## Stages 1 and 2

- **Stage 1** asks whether the camera can tell target, distractor and null apart,
  and whether the order of addition matters: 12 ordered pairs x 3 replicates over
  two simulated days.
- **Stage 2** asks how small a concentration difference the camera can detect:
  the dilution series, 5 replicates per level on two days. The detectable
  difference is 3 x (scatter of d) / (slope of d against concentration).
- Two colour distances are reported. `d_pre` compares the last image with the
  first image of the same recipe. `d_blank` compares it with the session's water
  blank and is the one to use for order-invariance.

## Stage 3: the agent task

One pair among four samples forms the red complex. The agent must find it in at
most 5 mixes, stating a confidence with every mix, and stop as soon as a mix has
produced the red complex.

**Rules**

- A mix may contain at most two different samples. One sample on its own is
  allowed and costs a mix.
- Every accepted recipe costs one mix. Rejected recipes cost nothing.
- The agent sees only the colour readout of each image, never the true contents,
  the classifier's verdict or the reward.

**The six tests: what the agent is told x which model plays**

| What the agent is told | Expert (stronger model) | Novice (weaker model) |
|---|---|---|
| `labelled`: which chemical is in which bottle | Test 1 | Test 2 |
| `listed`: which chemicals are on the shelf, but not which bottle is which | Test 3 | Test 4 |
| `unknown`: nothing about the contents | Test 5 | Test 6 |

Labelled vs listed shows what labels add, listed vs unknown what knowing the
contents adds, expert vs novice what model capability adds.

**What is measured**

- **Mixes to red:** how many mixes before the red complex was produced, and the
  share of episodes that had produced it within 1, 2, ... 5 mixes. This is the
  headline number. With five of the six pairs testable, the correct answer can
  always be reached by elimination, so plain success saturates.
- **Calibration:** stated per-mix confidence against what happened (Brier score
  and reliability curve).
- **Plan agreement:** whether the samples the agent said it would mix match the
  recipe it ran.
- **Token use:** tokens and cost per episode, recorded by the Claude Code batch
  runner, and plotted against mixes used and correct answers
  (`results/stage3_tokens.png`).

**How to run it**

```bash
python experiments/stage3_agent.py                              # scripted baselines
python experiments/run_claude_grid.py --models opus,haiku --seeds 0-4   # the six tests with Claude Code
python experiments/stage3_report.py ~/sdl-runs-v2               # score recorded episodes
python experiments/stage3_plots.py ~/sdl-runs-v2                # figures
```

- **Tools** (`sdl/agent/session.py`): `briefing`, `inspect`, `verify`, `dispatch`,
  `get_result`, `reveal_results`, `submit_answer`. Scripted agents, the MCP server
  and the Gymnasium environment all go through them.
- **Claude Code** plays through the MCP server: see `docs/stage3_claude_code.md`.
- **RL** uses `sdl.agent.gym_env.SdlPairEnv`.
- **Baselines** (`sdl/agent/policies.py`): `random_pairs`, `overconfident` (same
  search, always claims 90 %) and `prior_knowledge` (reads the labels, so it only
  helps in the labelled condition).
- A second axis, `--feedback adaptive|batch` (readout after every mix, or only
  after all mixes), exists but is not part of the six tests.

Episodes recorded before the two-samples rule (the first 20 Opus 5.5 runs) are
pilot data. They load, but reports and plots leave them out by default.

## What is real and what is a guess

- Formation constants: literature values (de Berg, Maeder & Clifford, Inorg.
  Chim. Acta 466, 249 (2017)), corrected for ionic strength.
- Camera repeatability and dispensing precision: anchored to published figures
  (see `configs/noise_scenarios.yaml`) and reported as three scenarios.
- Precipitate optics, mixing time, fading rate, day-to-day lighting: **guesses**,
  each marked with a confidence level in `configs/emulator_priors.yaml`.
- The real noise floor comes from running Stages 1-2 on hardware.
- Hardware: **TBD**, see `docs/hardware_TBD.md`.

## How it fits together

```
recipe -> Lab.verify  (fork of the state twin, safety and logistics rules)
       -> Lab.dispatch -> Backend.add / stir / image   (SimBackend or RealBackend)
                       -> vision pipeline (white balance, CIELAB, d, hue, chroma, turbidity)
                       -> TraceStore (record.json + frames, one folder per run)
```

- The state twin holds the **nominal** state. `SimBackend` keeps a separate hidden
  "true" state with dispensing error and imperfect mixing.
- The state only changes through events (`sdl/schema/events.py`), so it can be
  replayed from the log and forked for dry runs.
- Turbidity is read from a black stripe on the backdrop behind the vial: a clear
  liquid keeps the stripe's contrast, a cloudy one washes it out.

## Roadmap

1. Schemas and state twin: done
2. Emulator and vision pipeline, Stages 1-2 in simulation: done
3. Agent tools (MCP), Gymnasium env, Stage 3 harness: done
   - next: run the six tests and compare with the baselines
4. Real backend and calibration against bench data: waiting for hardware
