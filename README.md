# drone-training

A UAV flight simulator and research sandbox for FPV and large multirotor
aircraft, with a machine learning pipeline for target detection and autonomous
engagement **inside the simulation**.

Built from scratch on Panda3D and the Bullet physics engine. A human pilot and
a trained agent drive the aircraft through the same interface, so an autonomous
policy is simply another implementation of `ControlSource`.

> **Scope.** This is a training simulator and game. Every "target" is a virtual
> object inside the simulation, in the same sense as in any flight simulator or
> video game. The project does not interface with physical hardware and is not
> intended for real-world guidance. It exists to study flight control, computer
> vision and reinforcement learning in a sandbox.

---

## What it does

**Flight.** Acro/Rate mode with the handling of a 5-inch FPV quad, plus a heavy
multirotor with Angle, altitude-hold and position-hold modes. Keyboard or
PlayStation gamepad (DualShock 4 / DualSense) input, with configurable deadzone,
expo and axis inversion.

**Physics.** Motor inertia and spin-up lag, battery sag under load, ground
effect, anisotropic quadratic drag, airmode, feed-forward PID terms and
deterministic turbulent wind. Each effect is individually switchable through
`configs/vehicles/*.yaml`, so the model can be reduced to a simpler one for
reproducible experiments.

**Scenarios.** Three mission types — `gate_race` (fly a circuit through oriented
gates), `patrol` (spot targets appearing on a schedule) and `strike_range`
(approach and engage stationary targets) — each with several difficulty levels
and seeded, reproducible layouts.

**Computer vision.** Synthetic dataset generation directly from the simulator
(frame plus projected bounding boxes), YOLO training and inference, and a
detection overlay on the FPV feed.

**Reinforcement learning.** A Gymnasium-compliant environment and a PPO agent
trained to approach and engage a target autonomously, driving the aircraft
through the same control path a human pilot uses.

**Tooling.** A GUI launcher for flight, autonomous runs, input calibration and
checkpoint management; a headless evaluation harness; and 61 test modules
covering physics, controllers, scenarios and both ML pipelines.

### One deliberate limitation, stated up front

The autonomous agent currently receives the target's relative position **from
the simulation directly**, not from the detector. YOLO runs in parallel and
displays its bounding boxes as a picture-in-picture overlay, but detection does
not yet drive flight. Closing that loop — feeding noisy, intermittent detections
into the policy instead of ground truth — is the next substantial piece of work.
The compromise is recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## Autonomous mode: measured results

The policy was trained on a single task — approach one nearly stationary target
— then evaluated unchanged on all three scenarios, 20 seeded episodes each,
headless. Raw data: [`docs/phase7_autonomous_eval.json`](docs/phase7_autonomous_eval.json).

| Scenario | Result | Mean score | Mean time |
|---|---|---|---|
| `strike_range` | 16/20 episodes engaged **all 4** targets | 3.70 / 4 | 51.8 s |
| `patrol` | Spotted 2 of 3 targets in most episodes, never all 3 | 1.80 / 3 | 60.0 s (full limit) |
| `gate_race` | **0/20** completed 3 laps; 4/20 completed exactly 1 lap | 0.25 laps | 90.0 s (full limit) |

The `gate_race` result is the interesting one, and it is not an integration bug.
Unit tests confirm the gate centre is passed to the agent correctly. The policy
simply never encountered the concept of *flying through* an oriented opening
during training — only *approaching a point in space* — so it clears gates only
when its approach heading happens to align. This is a known limit of the current
observation strategy, and it defines the next step: training on trajectory-shaped
objectives rather than point targets.

Full analysis, including a methodological note on why the first evaluation run
produced byte-identical results across all seeds, is in
[`docs/phase7_autonomous_eval.md`](docs/phase7_autonomous_eval.md).

---

## Architecture

Everything runs on a fixed-step simulation loop (1/240 s for physics, lower
rates for render and control):

```
 input ──▶ ControlSource ──▶ FlightController ──▶ Vehicle ──▶ Physics
(keyboard/                  (Acro / Angle /      (motors,     (Bullet via
 gamepad/                    AltHold / PosHold)   mass, thrust) Panda3D)
 RL agent)                                                          │
                                                                    ▼
 HUD / Render ◀── Renderer ◀── FPV Camera ◀──────────────── World state
                                    │
                                    ▼
                            ML: Detection / RL
```

The governing rule is that a human and an agent are interchangeable: both emit
the same `ControlCommand` vector. The data contracts (`ControlCommand`,
`VehicleState`, `CameraFrame`, `Detection`) live in `core/contracts.py` and are
the only coupling between packages.

| Package | Responsibility |
|---|---|
| `core/` | Simulation loop, clock, configuration, data contracts |
| `physics/` | Bullet world, aerodynamics, wind |
| `vehicles/` | Aircraft models and flight-mode controllers |
| `input/` | Keyboard, gamepad and RL-agent control sources |
| `render/` | Panda3D scene, PBR materials, FPV camera, HUD, effects |
| `world/`, `scenarios/` | Scene composition, missions, scoring |
| `ml/data`, `ml/detection`, `ml/rl` | Dataset generation, YOLO, PPO environment |
| `gui/` | Launcher application |

Design notes: [`ARCHITECTURE.md`](ARCHITECTURE.md).
Phase-by-phase development history with acceptance criteria: [`ROADMAP.md`](ROADMAP.md).
Decision log: [`docs/DECISIONS.md`](docs/DECISIONS.md).
Terminology: [`docs/GLOSSARY.md`](docs/GLOSSARY.md).

---

## Getting started

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -e ".[dev]"
```

Manual flight:

```bash
python -m dronesim.app fly --vehicle fpv_5inch --input keyboard
python -m dronesim.app fly --vehicle fpv_5inch --input gamepad
python -m dronesim.app fly --scenario gate_race --level circuit8
```

Autonomous mode, using a trained checkpoint:

```bash
python -m dronesim.ml.rl.train --config ml/ppo_strike    # train
python -m dronesim.app autonomous --scenario strike_range
```

GUI launcher, instead of the terminal:

```bash
pip install -e ".[gui]"
python -m dronesim.gui.launcher      # or run_gui.bat on Windows
```

Tests and linting:

```bash
pytest -q
ruff check
```

### 3D assets

Generated models in `assets/models/*.bam` are committed. Regeneration is only
needed when editing `assets/blender/build_*.py`; those scripts run inside
Blender, and `gltf2bam` converts the exported `.glb`. If the assets are missing,
rendering falls back to procedural primitives and nothing breaks.

---

## Technology

Python 3.11 throughout, so there is no bridge between the engine and the ML
code. Panda3D provides both rendering and, through `panda3d.bullet`, physics;
`panda3d-simplepbr` handles PBR lighting. Reinforcement learning uses Gymnasium
with Stable-Baselines3, detection uses Ultralytics YOLO, gamepad input goes
through pygame/SDL, configuration through OmegaConf, and tests through pytest.

---

## The preprint's experiments

**Feasibility Ceilings Before Learning: Sizing and Timing Bounds for
Drone-Mesh Interception** — [arXiv:2610.20823](https://arxiv.org/abs/2610.20823)
([PDF](https://arxiv.org/pdf/2610.20823)), cs.RO / cs.LG / cs.MA, CC BY 4.0.

The paper's analysis is reproduced end to end by the scripts in
[`experiments/`](experiments/). A single entry point regenerates every number
and every figure in it:

```bash
python experiments/run_all.py
```

It runs the eight experiment scripts in order, then the figure generator, then
prints every reproduced value against the one printed in the manuscript. Exit
status is non-zero if any assertable check fails, so it doubles as a regression
test on the analysis. Expect roughly four minutes on a laptop CPU.

| script | produces |
|---|---|
| `exp1_prediction_uncertainty.py` | Table I, Figure 1 |
| `exp2_planar_coverage.py` | Table II, Figure 2 |
| `exp3_volumetric_coverage.py` | Figure 3 |
| `exp4_sensor_cadence.py` | Table III |
| `exp5_placement_strategies.py` | Table IV, Figure 4 |
| `exp6_error_model_robustness.py` | Table V, Figure 5 |
| `exp7_commit_window.py` | Table VI, Figure 6 |
| `exp8_policy_comparison.py` | Tables VII and VIII |
| `make_figures.py` | all six figures, from the recorded results |

Three properties are worth stating explicitly.

**The scripts import the prototype, they do not restate it.** The propagated
prediction uncertainty comes from
`dronesim.ml.swarm.intercept_predictor._propagate_crossing_uncertainty`, the
mesh geometry from `analytic_policy.hex_lattice_slots`, the closed-form sizing
and timing laws from `swarm.coverage_laws`, and experiment 8 drives the
policies through `scripts/eval_swarm_intercept.py`'s own `run_episode`. A
validation run therefore tests the implementation that actually flies, not a
parallel copy of the algebra.

**Every Monte Carlo seed is fixed and stated**, in `experiments/_common.py`.
Each run writes a JSON record to `experiments/results/` carrying the table it
produced, the parameters behind it, the published values it was checked
against, and the environment it ran in.

**The archived policy checkpoints are in version control**, under
[`experiments/checkpoints/`](experiments/checkpoints/) — the three trainer
seeds behind Table VII, plus the original and the determinism-retrain copies
of seed 0. Their training configurations are
`configs/ml/ppo_swarm_intercept*.yaml`. The working `runs/` directory stays
gitignored; these are the copies the paper's numbers came from.

A handful of reported values are deliberately *not* asserted, because the
manuscript does not record the grid or construction they depend on — the
fitted exponent of Section V-B, the random-placement row of Table IV, and the
narrow and wide corridors of Table VI. Those are printed as "reported, not
asserted" and carried in the JSON records with both values side by side.

The commit the paper cites is `4924613`; later commits may refine the code,
but that one is what produced the published numbers and figures.

```bibtex
@misc{babarytskyi2026feasibility,
  title         = {Feasibility Ceilings Before Learning: Sizing and Timing
                   Bounds for Drone-Mesh Interception},
  author        = {Babarytskyi, Dmytro},
  year          = {2026},
  eprint        = {2610.20823},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

---

## Status

Phases 0 through 7 are complete: scaffolding, physics, manual flight, large-UAV
flight modes, missions and scoring, computer vision, reinforcement learning, and
autonomous-mode integration. Fixed-wing aircraft remain unimplemented and are
marked optional.

Training artifacts (`runs/`, `datasets/`, model weights) are excluded from
version control and must be regenerated or restored separately.
