# p6: reusable local manipulation

Status: active. Parent: p5, SHA-256
`3f4f1529378b08ef8de644564a1acc43f9de56b7d389e55e563fae7538c576a6`.

**Mainline clarification:** retain the original architecture. Follow
[the current experiment contract](p6-mainline-experiment-contract.md). The
separate recurrent planner and goal-residual runs below are independent
controls, not a decision to replace p5; composition failure does not prove
route memorization or failure of the original architecture.

The learned skill consumes local physical observations, attached-body geometry,
a relative target with an orientation mask, and causal feedback memory. It must
work when called repeatedly from different poses and contact states. Mapping,
global route discovery, and attachment are separate capabilities.

## Ordered work

1. Reproduce frozen-parent failures. Vary one factor at a time: glyph size,
   target orientation, body clearance at handoff, temporal state, and observed
   obstacles. Identify the physical wall rejecting each attempted motion and
   measure whether the learned planner or analytic geometry branch controls it.
2. Correct target semantics and obstacle/history handoff. Test geometry and
   force-level behavior, including counterexamples with a centre beyond a wall
   but the body's trailing part still inside its slit.
3. Build a feasible-state curriculum and exact-checked executable local
   teachers. Query teachers on the student's actual visited states, including
   retreat, reorientation, contact, nonzero velocity, and target changes.
4. Train local planner/recovery components on the Linux RTX 5090. Keep the
   physical GBP and grasp interface stable initially. Check that trained outputs
   actually affect deployed actions; loss reduction in an overridden branch is
   insufficient evidence.
5. Regress old single-slit tests and assess held-out multi-wall layouts and
   entry states. Release a candidate demo only after independent validation.

## Data and evaluation contract

- A–Z exact compound geometries; 8–20 agents; nominal and enlarged shapes;
  broad-to-tight feasible clearances; easy passages remain in the training mix.
- Split whole layout families and episodes before generating teacher labels.
  Preserve S/Z and odd-count holdouts where not previously consumed. Disclosed
  inspected examples are development data, never a fresh unbiased test.
- Group jointly transformed scenes together. Test translation and rotation
  consistency without exposing absolute-world coordinates or waypoint indices
  to the learned skill. A wall-relative chart is a local geometric coordinate,
  not an absolute route coordinate.
- Record teacher failures and unresolved feasibility; do not hide them through
  success-only data or evaluation selection. Verify force limits and complete
  executed transitions, not only collision-free pose waypoints.
- Target >=90% success per agent count and on held-out shape/layout/entry-state
  combinations. Report denominators, uncertainty, contacts, stall duration,
  control steps, path length, actual message traffic, and exact penetration.
- Parent single-slit and canonical T behavior must not regress. The historical
  336/338 grid belongs to p4; it is not a completed p5 grid.

## Initial evidence, 2026-09-16

- Reproduced B/12: first waypoint at step 208; first rejected transition at
  step 211. Wall clearances of that attempt: [-0.0238513, 0.7892244, 1.1805027] m.
  The rejecting obstacle is the **first** wall, omitted after waypoint handoff.
  The 1100-step episode has 889 rejected attempts. This corrects the earlier
  interpretation that entering stage two implied hitting wall two.
- Reproduced L/12: no waypoint completed and no rejected contact in 1100 steps.
  Its failure requires separate diagnosis.
- Reproduced T/12: 844 steps, four goals completed, zero rejected contact.
- p5's saved world scales are (3.9, 1.6), equal to this demo's scales. A changed
  normalization between those two configurations is therefore ruled out.
- The public p5 process remains on its original worktree. Experiments run here.
- A body-clearance handoff guard alone changes B/12 to 656 steps, all four goals
  completed, zero rejected contact, with identical checkpoint weights. T and X
  remain successful (858 and 770 steps respectively). This is an interface
  correction, not a learned improvement.
- L's stalled centre reaches the goal x but remains 0.300 m away in y. Its rear
  is still 0.000316 m before the first wall's right face. The geometric corridor
  continues to require exit before recentering, while its forward target is
  already reached. A merely position-only orientation mask does not resolve it.
  This is a reproducible local planning deadlock to include in teacher queries.
- During the stalled L state, 9/12 local plans are overridden by the analytic
  geometry branch. Training an upstream planner alone may therefore have little
  effect; measure the deployed branch before selecting trainable parameters.
- The first v1 diagnostic logger did not record grasp setup traffic correctly;
  its `grasp_setup_counter` is invalid and must not be used as communication
  evidence. v2 records actual setup traces and respects implicit GBP precision.

These are development diagnostics, not new generalization claims.

## First controlled panel

Five letters (A, B, L, T, X), 12 agents, six controls, 30 completed rollouts:

| Control | A | B | L | T | X |
| --- | --- | --- | --- | --- | --- |
| shipped p5 composition | fail | fail | fail | pass | pass |
| body-clear handoff | fail | pass | fail | pass | pass |
| position-only control | fail | fail | fail | pass | pass |
| both interface changes | fail | pass | fail | pass | pass |
| both + preserve private memory | fail | pass | fail | pass | pass |
| original glyph sizes, old interface | fail | fail | pass | pass | pass |

All controls kept the same p5 weights. Independent resolved-pose collision
checks found no penetration. Passing the handoff guard is insufficient for A:
it can rotate back into the previous wall after the guard has fired. Observation
of nearby obstacles must remain available across goal changes. The private
memory switch alone made no difference on this panel; reset is not established
as the primary cause. Dataset targets must include exit-and-recenter maneuvers
and retain the previous obstacle when it is still within sensing range.

Run the auditable v2 logger from this worktree:

```bash
PYTHONPATH=src python scripts/audit_local_skill_composition.py \
  --checkpoint /path/to/sheaf_gbp_route_free_all_shapes_p5.pt \
  --output output/p6_controls_new_run \
  --shapes A B L T X --agents 12
```

Each NPZ includes the student's actual states, local observations, private
memory, forces, learned and executed plans, contact attempts, and clearance.
These records are inputs for future teacher queries, **not labelled DAgger
data yet**. The JSON manifest records parent and source hashes, all requested
cases, episode outcomes, physical-message payload counts, and artifact hashes.

## Local-learning pipeline and first closed-loop results

The complete corrected-interface p5 panel is 15/26 letters at N=12 on the
enlarged three-wall course (`output/p6_composed_AZ12_20260916`). This is still a
development course, not the historical nominal single-slit grid.

An isolated candidate planner now receives 78 body-frame features per node:
handle offset (2), own velocity (2), relative active goal (2), masked heading
and orientation mask (3), radius-limited lidar (32), attached geometry (32),
and causal mechanical feedback (5). It uses 78→128→96, GRU(96), 96→64→3
normalized desired twist, converted to the physical eight-coordinate plan.
It has 84,771 parameters. No absolute route, waypoint index/list, letter ID,
or teacher is available at deployment. All nearby obstacles remain in sensing;
private memory persists at goal switches. Physical GBP and grasp decode are
the parent implementation. The final-plan interface prevents the old analytic
planner from silently replacing the newly trained output.

Training-only teacher search proposes paths in (x,theta) slices with continuous
feasible y intervals, permits retreat, and intersects all supplied slit walls.
A separate recursive separating-axis motion-bound checker validates continuous
segments. It uses double precision, not interval arithmetic or a general exact
algebraic proof. Search exhaustion is resolution/budget-relative; generic
non-slit obstacle search is not implemented. Force execution respects 15%
per-handle force reserve and separately certifies every executed transition.

Artifacts and results, all kept out of the public demo:

| Artifact | Meaning |
| --- | --- |
| `output/p6_local_dataset_seed20260916` | 214/214 admitted teacher continuations; 22 p5 failure-state queries + 192 procedural cases; 22,226 frames, 307,604 agent-frames |
| `output/p6_local_train_v1` | 2,000 gradient updates on RTX 5090; not a success claim |
| `output/p6_local_eval_v1_multiwall` | Learned candidate: 0/5 complete courses |
| `output/p6_local_train_long_v1` | 20,000 additional updates; 0/5 complete courses in corresponding evaluation |
| `output/p6_local_dagger_v1b` | 70 actual p6 student-state queries; 66 certified executable continuations, four unadmitted cases retained |
| `output/p6_local_train_dagger_v2` | Warm-start plus actual-student DAgger, 20,000 additional updates |
| `output/p6_local_eval_dagger_v2_multiwall` | Still 0/5 complete courses; L completes the first local target |
| `output/p6_local_eval_dagger_v2_seen` | 1/12 seen-start smoke cases; L's old stalled state recovers; **not generalization evidence** |
| `output/p6_local_train_collective_v3` | Longer recurrent windows and collective imitation surrogate; separate development candidate |

The DAgger shards retain preceding actual student observations as unlabelled
recurrent burn-in; their first teacher label is verified to match the exact
student observation. They are not merely new trajectories from nominal starts.
The initial `p6_local_dagger_v1` folder is an interrupted JSON serialization
attempt and is **not a training dataset**. v1b is the complete restart.

Two further audited limitations are important:

1. The first teacher dataset used a 0.04 m parking gate. 90/96 oriented episodes
   stopped more than 0.03 m from the target, outside the evaluator's stricter
   tolerance. Its 214/214 means success under that documented teacher gate,
   **not 214/214 strict task success**. New teacher generation uses a 0.015 m
   search target region, a 0.020 m execution gate, tighter angle/speed gates,
   and six stable steps. The old dataset remains unchanged for reproducibility.
2. Four GBP sweeps do not converge on every attachment tree. In an E/20 teacher
   replay the tree diameter is five: max next-state coordinate error drops from
   0.0248481 (four sweeps) to 0.0000218749 (five). This mixes coordinate units
   and is a diagnostic discrepancy, not a distance metric. p6's explicit
   `--diameter-rounds` option uses max(parent rounds, tree diameter); actual
   communication must include the extra sweeps. Public p5 remains unchanged.

The stricter dataset run `output/p6_local_dataset_strict_seed20260917` requests
24 training letters × seven even counts × four scenarios = 672 cases. It adds
reverse travel, nearby terminal goals, arbitrary terminal heading, and correctly
scales the T's physical rectangles (the old glyph-scale field did not resize T).
S/Z and odd counts remain held out. Failed teachers stay in the manifest.

The longer-window collective candidate v3 also remains 0/5 full courses; it
does not justify a release. A second actual-student query set,
`output/p6_local_dagger_v2_strict`, contains 30 distinct visited-state queries
(26 admitted continuations), with repeated identical stalled states explicitly
deduplicated. It uses the tightened terminal criteria.

An additional opt-in hypothesis, `goal_residual`, adds a generic body-frame
proportional goal field and asks the same recurrent network to predict bounded
detours. A smooth residual envelope vanishes at the complete required goal;
orientation error can still permit translational retreat. This is neither a
stored route nor a collision-avoidance guarantee. The first v4 run starts from
scratch on a recorded partial snapshot of the stricter dataset plus the second
DAgger set. Its checkpoint explicitly records the changed output semantics;
the loader rejects silent semantic changes when resuming training. It is an
unvalidated candidate, not the new public model.

Feature tests cover translation invariance, masked target heading, a joint
90-degree rotation of local geometry and goals, private recurrence, agent-order
equivariance, and the goal-residual zero equilibrium. They do not establish
closed-loop maze success. The certificate tests include a path with safe
endpoints but an intervening wall, which sampled endpoint checking misses.

These experiments have **not** passed the p6 release gate. In particular,
successful teachers, lower fit loss, and one recovered L case do not establish
robust skill composition. Next gates are strict teacher completion, fresh
model-induced recovery data, unchanged-parent regression, and held-out closed
loops, before replacing the p5 demo.

## Archived alternative-planner controls and stronger audit

These runs are controls under the mainline contract, not replacements for p5.
The strict dataset completed with 668/672 admitted teachers. Adding a
radius-clipped boundary-segment sensor produces 162 features; the 256-hidden
GRU variant has 518,915 parameters. The local segment ordering is canonical
in the body frame and tested for obstacle order, 90-degree joint rotation,
and remote-obstacle irrelevance. It is idealized geometric proximity sensing,
not an occlusion-aware camera model.

Matched 78/96, 78/256 and 162/256 controls all failed the five complete
three-wall development courses. The 162/256 control passed 8/20 seen-start
smoke episodes. Adding independently stopped-state teacher queries, 19 actual
student-state continuations and an expanded residual range gave 10/20 seen
starts but still 0/5 complete courses. Resetting recurrence did not resolve
those courses. These results do not diagnose the original architecture.

An accidental four-scenario data correlation omitted zero-heading,
left-to-right, position-only initial cases. An additional crossed dataset
requests 24 letters × seven even counts × eight combinations = 1,344 teachers;
1,340 are admitted. Four failures remain in the manifests. Its four canonical
162-feature shards are `output/p6_factorial_{A_F,G_L,M_R,T_Y}_boundary_v2`.
The already-running v7 control finished 30,000 updates on 2,695 trajectory or
independent-query groups, 213,813 frames, in 249.35 optimization seconds. Its
training fit loss is not a closed-loop success result. No automatic continuation
of that alternative architecture is scheduled.

The original p4 file was recovered read-only from the Windows experiment
archive; SHA-256 `d946da7219691774f068f56bace14852de470cc6e776e218aa8b237c14837c58`
matches p5's parent metadata. Under the **current audited code and benchmark-T
profile**, both frozen p4 and p5 reach the terminal gate on 338/338 nominal
single-slit cases. This does not reproduce the historical p4 336/338 under its
original source/runtime. Independently certified continuous trajectories are
335/338 for p4 and 337/338 for p5, respectively. Safe sampled poses alone are
not a continuous zero-penetration certificate.

p5's T/8 segment 60 has safe endpoints but a recursive-check witness of
−0.0572 mm. A subsequent dense 1,001-point probe finds a −0.1126 mm witness at
interpolation fraction 0.062; this is not a proven global minimum. Coordinates
and provenance are in
`output/p6_frozen_p5_single_slit_grid/T_8_continuous_witness.json`.
The alternative p6 evaluator now has an explicitly declared continuous-contact
rejection option, with a tunnelling regression test. The frozen-mainline
controls retain original sampled physics and certify trajectories afterward,
so interface comparisons do not silently change physics. Public p5 is unchanged.
