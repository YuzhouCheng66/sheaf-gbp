# Sheaf-GBP

Distributed manipulation through a narrow passage with local feedback and
communication-efficient Gaussian belief propagation.

<p align="center">
  <img src="assets/demo.gif" width="920" alt="PianoMovers-Force exact-collision demonstration">
</p>

## Task

Eight to twenty agents are rigidly attached to force handles on a non-convex
payload. Each agent receives a local observation of the payload, nearby wall,
mission goal, its handle geometry, and causal contact history. Agents may send
messages only along the radius graph. Their joint planar forces must translate
and rotate the payload through a `0.720 m` slit without penetration, then bring
it to rest at the goal.

The current public checkpoint solves the short-horizon **PianoMovers-Force**
skill. It deliberately excludes navigation, map construction, free-moving
robots, grasp acquisition, and contact switching.

## Current policy

```text
per-agent local observation (27)
  -> local recurrent planner features (96)
  -> next-SE(2) plan (8)
  -> local inverse-dynamics wrench proposal
  -> correction stalk (16)
  -> analytic restriction [I3  0] : R16 -> R3
  -> four rounds of diagonal Gaussian BP on radius edges
  -> posterior net wrench (3)
  -> analytic distributed grasp decoder
  -> local planar force (2)
```

The eight planner coordinates are
`[dx, dy, dtheta, vx, vy, omega, z1, z2]`. Planning is recomputed from current
geometry and feedback; deployment has no letter ID, stored route, teacher query,
or carried path index. A one-step exact collision check selects a safe local
target near the wall.

GBP sends a 3-D diagonal Gaussian as mean plus precision: **6 float32 values per
directed edge per round**, for four rounds. The restriction is analytic rather
than a learned bottleneck: the first three coordinates of the 16-D local stalk
are the normalized wrench proposal, and `[I3  0]` exposes exactly those public
coordinates to the edge sheaf.

For a fixed attachment, agents perform one tree all-reduce of four sufficient
statistics `[N, sum(bx), sum(by), sum(||b||^2)]`. Afterwards each agent builds
the same `3 x 3` grasp Gram matrix locally and independently maps the posterior
wrench to its own 2-D force. No grasp-map data is transmitted per control step.

## Verified results

The route-free p5 checkpoint has SHA-256
`3f4f1529378b08ef8de644564a1acc43f9de56b7d389e55e563fae7538c576a6`.
Checkpoints and bulk rollouts are intentionally excluded from Git.

| Evaluation | Result | Exact penetrations |
|---|---:|---:|
| canonical T, 8–20 agents, four fixed micro-perturbations each | **52/52 (100%)** | **0** |
| previous p4, A–Z nominal agent-count grid | **336/338 (99.41%)** | **0** |
| previous p4, non-T letters only | **325/325 (100%)** | **0** |

The A–Z rows are reported separately because the complete p5 A–Z rerun is not
yet finished. They are not presented as p5 evidence.

For the earlier two-wall matched benchmark, the archived 12-agent Sheaf-GBP
arm achieved `96.83%` success versus `27.15%` for raw-full while sending 25%
fewer float payload bytes. That result concerns the earlier task and architecture,
not the p5 single-slit audit above.

## Install

Python 3.10+ and PyTorch 2.1+ are required.

```bash
git clone https://github.com/YuzhouCheng66/Sheaf-GBP.git
cd Sheaf-GBP
python -m pip install -e ".[dev]"
pytest -q
```

## Run the interactive demo

Supply the p5 checkpoint explicitly:

```bash
piano-movers-demo \
  --checkpoint /path/to/sheaf_gbp_route_free_all_shapes_p5.pt \
  --device cuda \
  --host 127.0.0.1 \
  --port 7861 \
  --open
```

The page streams policy computation while the animation is running. Every
displayed packet is a real directed GBP message on a physical radius edge; the
browser does not synthesize decorative communication. Drag the two wall tips,
change the T proportions or ant count, press Random for another admitted scene,
then press Play.

### Multi-wall waypoint composition

The validated **p6 development demo** must load the complete recipe, not just
the weights (the handoff, continuous-collision and GBP settings are required):

```bash
PYTHONPATH=src python -m piano_movers.multi_waypoint_demo \
  --recipe configs/p6_force_crossing_hold_control.json --device cpu --port 7862
```

The recipe verifies the checkpoint checksum and requires every control option.
`/api/model` exposes the served identity/configuration; the tab title shows the
checkpoint prefix. The three-wall A–Z × N8–20 reference is 338/338, all
continuously certified. See the development reports for scope and limitations.
The frozen-p5 command and results below are **historical**, not this p6 deployment.

The same frozen p5 manipulation skill can be invoked sequentially along a
hand-supplied SE(2) route:

```bash
piano-movers-waypoints \
  --checkpoint /path/to/sheaf_gbp_route_free_all_shapes_p5.pt \
  --device cpu \
  --port 7862 \
  --open
```

The default course has three offset `0.720 m` slits and four waypoints. Each
leg exposes only its locally relevant slit to the policy, while exact collision
is evaluated against every wall in the world. The bundled 12-agent course was
verified zero-shot at **1/1 success**, **0 contact events**, and **0 exact
penetrations**. This demonstrates waypoint-conditioned skill composition; it
does not claim that the present checkpoint builds a map or discovers the route.

Intermediate waypoints constrain position only; only the final target requires
a stationary SE(2) pose. Non-T glyphs use a harder large-letter OOD profile:
each is enlarged until its best heading on the online planner's 72-heading grid
has a `0.648 m` span inside the `0.660 m` physical opening. The same frozen
checkpoint completes **9/26 shapes (34.62%)** with **0 exact penetrations**:
`H N R S T V W X Y`. This lower number replaces the earlier small-glyph audit;
it exposes the next planner-training problem rather than hiding it. The complete
nominal manifest is
[`assets/multi_wall_a_z_12agent_audit.json`](assets/multi_wall_a_z_12agent_audit.json).

To create p5 from its admitted p4 parent without modifying any tensor:

```bash
PYTHONPATH=src python scripts/bridge_single_slit_route_free_geometry.py \
  --parent-checkpoint /path/to/p4.pt \
  --output /path/to/p5.pt
```

The bridge verifies identical state-dict keys, bitwise-identical parameter
tensors, and an unchanged wire protocol before writing the child checkpoint.

## Reproduce the public checks

```bash
PYTHONPATH=src pytest -q \
  tests/test_single_slit_cspace.py \
  tests/test_single_slit_local_planner.py \
  tests/test_shape_agnostic_distributed_wrench.py \
  tests/test_single_slit_interactive_demo.py

PYTHONPATH=src python scripts/audit_single_slit_asymmetric_openings.py \
  --checkpoint /path/to/p5.pt \
  --n-agents 8 12 20 \
  --device cuda \
  --output asymmetric_openings.json
```

## Repository map

```text
src/piano_movers/single_slit_interactive_demo.py  streamed monochrome demo
src/piano_movers/multi_waypoint_demo.py           multi-wall waypoint composition
src/piano_movers/single_slit_local_planner.py     local recurrent/geometry planner
src/piano_movers/single_slit_cspace.py            exact current-geometry C-space
src/piano_movers/models.py                        stalk, restriction, GBP, decoder
src/piano_movers/gbp.py                           diagonal Gaussian BP
src/piano_movers/env.py                           dynamics and exact collision
scripts/bridge_single_slit_route_free_geometry.py tensor-preserving p4 -> p5 bridge
scripts/audit_single_slit_asymmetric_openings.py  asymmetric/open-wall audit
tests/                                            CPU architecture and semantics tests
```

## Scope

- Agents are fixed force handles, not walking ants.
- Collision uses the exact compound-rectangle body and wall slabs used by the
  evaluator.
- The once-per-attachment statistic exchange and per-step GBP messages are
  counted separately.
- The demo is explanatory; quantitative claims come from fixed evaluation
  panels.
