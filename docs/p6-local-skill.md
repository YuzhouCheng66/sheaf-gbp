# p6: reusable local manipulation

Status: active. Parent: p5, SHA-256
`3f4f1529378b08ef8de644564a1acc43f9de56b7d389e55e563fae7538c576a6`.

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
