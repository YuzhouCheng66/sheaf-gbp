# Fixed-frame physics and interface audit

This audit keeps the original neural architecture, learned weights, Sheaf-GBP
and grasp decoder. It does **not** add or promote a new planner. It uses the
separately versioned `p6_force_position_heading_control.json` candidate, whose
position-goal heading control was already isolated in the preceding matched
experiment. That geometric control must not be described as learned recovery.
Public p5 is unchanged.

Checkpoint SHA256:
`a493af24bdf6de88d478b5c28a1b6f7f69a82aca008dfec054abb1e62e8f9f4f`.

## What changes coordinates

A supplied, fixed frame maps chart points to world points by
`p_world = R(alpha) p_chart + t`. Positions, walls, arena, waypoints and body
orientation move together. Forces and linear velocities rotate without a
translation; angular velocity and torque are unchanged. The policy still
operates in its existing chart and receives no future waypoint or new input.

The world collision checker explicitly represents every wall half and the
arena as oriented rectangles. It projects world vertices onto the body and
obstacle axes; it does **not** inverse-transform geometry into the old AABB
checker. Nonconvex letters remain unions of rectangles. Every pair must be
separated. The same 0.03 m wall reserve is materialized in the geometry.

Continuous sweeps are tested recursively using the endpoint separating gap
and the displacement bound `||delta_xy|| + body_radius * |delta_theta|`.
Uncertified intervals are rejected. Tests include a collision between two
safe endpoints and an obstacle inside an L's convex hull but outside its
actual material. Arithmetic is float64, not interval arithmetic: these are
conservative numerical certificates, not exact-real proofs.

## Replay audit — not new success-rate evidence

Source: the previously authenticated three-wall A–Z count grid. Selection is
A–Z at N=8,12,20, independent of outcome: 78 episodes, 31,058 recorded steps.
Every source NPZ and checkpoint hash is checked before use. Six transforms
are declared before evaluation: identity, translation (0.63,-0.40) m, and
that translation with rotations +37°, +90°, 180° or -90°.

All **468/468** frame checks pass, including continuous world-SAT. Maximum
absolute differences:

| Check | Maximum difference |
| --- | ---: |
| Independent versus legacy pose clearance | 4.45e-16 m |
| Transformed versus source clearance | 1.34e-15 m |
| Existing dynamics under rigid transform | 8.89e-16, state-coordinate units |
| Float64 recomputation versus recorded float32 attempt | 2.07e-7, state-coordinate units |

The replay rotates archived forces and reevaluates one-step dynamics. It is
not a fresh rollout and does not test a policy's response to new states.

Artifact: `output/p6_rigid_world_replay/report.json`, SHA256
`d7e5cfe363ab2a3a3a1f8723063c7371e628279980c6ed6d1f09d81de0331e0c`.

## Closed loop — fixed-chart interface control

The second audit actually invokes the model at every step. Physical world
state persists in float64, forces are rotated before integration, and the
independent world-SAT guard accepts or rejects each physical move. The next
float32 policy observation is built from the inverse-chart state. No archived
action is replayed. Goal checks and terminal orientation are additionally
verified in the physical world.

Identity also uses this float64 physics backend: it is the precision-matched
control, not the historical float32 trajectory. This distinction prevents
precision/backend changes from being attributed to rotation.

On the locked development draw 81201 (`combined`, `entry_reference`),
A/B/L/T/X/Z at N=12 pass all **36/36** cases across the six frames. All have
zero collision attempts, zero independently measured penetration and a
continuous full-world path certificate. For every shape, all transforms
have the same step count and GBP payload bytes as their identity control:

| Shape | Steps in each frame |
| --- | ---: |
| A | 354 |
| B | 285 |
| L | 239 |
| T | 789 |
| X | 388 |
| Z | 222 |

Total: 13,662 steps, 65,359,872 GBP float-payload bytes and 139,392 grasp-setup
payload bytes; wire headers excluded. Chart path length is 260.895 m across
all 36 runs (rigid transforms preserve this length). Runtime was 93.16 s on
Linux CPU; no GPU training occurred.

Artifact: `output/p6_fixed_chart_closed_loop_81201/manifest.json`, SHA256
`e3b73b884fab4ff3c34eaf0d46754eadb293a77b8f867cd162627fb5f5cf0547`.
The catalog, source hashes, local policy traces and physical world traces are
retained on Linux. Only the small JSON reports were copied to Windows.

## Scope and next gates

These are rigid copies of one parallel-wall layout, not 36 independent maze
draws. They validate a **known fixed-frame adapter**, not learned rotation
equivariance. They do not show that the private MLP's original absolute-y
input has become invariant, or test memory transport between changing
frames. No new architecture, checkpoint promotion or public demo replacement
is justified solely by these results.

The next capability gates remain different wall normals within one episode,
causal contact/history at target/frame transitions, relative-goal coverage,
and independent held-out layouts for the final recipe. Repeated zero-contact
success should not substitute for a contact-recovery evaluation. Diagnose
those states with the original model first and collect genuine on-policy
DAgger where learning has action authority; keep analytic alternatives
separately named and matched.

## Reproduce

From the project root with `PYTHONPATH=src`, NumPy and PyTorch installed:

```sh
python scripts/audit_mainline_rigid_replay.py \
  --manifest output/p6_force_heading_counts_all/manifest.json \
  --output output/new_rigid_audit/report.json

python scripts/audit_mainline_framed_closed_loop.py \
  --output output/new_framed_closed_loop --shapes A B L T X Z

python -m pytest -q
```

The only default-engine edit is an overridable physics-call seam. Tests
verify both default branches are bitwise identical to their original
transition functions. Framed physics is used only by the audit subclass.
