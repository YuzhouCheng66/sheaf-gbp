# Crossing/alignment interface control

## Outcome and scope

With **unchanged source weights, physical Sheaf-GBP, precision, message budget
and grasp decoder**, a default-off geometric command-interface control repairs
the three development A failures and one additional validation A failure.
No learned residual head or replacement planner is used. Public p5 is untouched.

| Matched panel | Original | Interface control | Candidate certified / penetration |
| --- | ---: | ---: | --- |
| Early impulse, seed 81201, A/B/L/T/X/Z × N8/12/20 | 15/18 | 18/18 | 18/18; zero |
| No impulse plus near impulse, same shapes/counts | 36/36 | 36/36 | 36/36; zero |
| Early impulse, locked-validation seed 82202, same shapes × N9/13/19 | 17/18 | 18/18 | 18/18; zero |

These are correlated cells from two layout draws, not independent Bernoulli
trials, not all A–Z, not all agent counts, and not a complete generalization
claim. The matched source and candidate use the same per-case physical setup,
goal criteria, disturbance rule, checkpoint and horizon. Four repaired cases
also have exactly identical recorded impulse events. A few other paired
successful trajectories change before the state-triggered impulse, so do not
claim all disturbance states are identical across all pairs.

Development A/8 and A/12 each experience one contact and subsequently complete
the entire three-wall course, in 318 and 354 steps. A/20 completes in 350 steps
with no contact after the impulse; validation A/19 completes in 491 steps with
no contact. The latter two demonstrate avoidance, **not** post-contact recovery.
All four source episodes time out at 1,100 steps.

## Why this is an interface repair to test before changing the network

The prior read-only audit found 17 states where every node agreed on a local
twist that was itself unsafe. The maximum actuation tracking error on the
51 failed-continuation frames was 4.14e-5 (componentwise m/s or rad/s), with
no force saturation. The geometry branch classified the body as aligned and
allowed forward movement, but simultaneously adjusted y and yaw. A static
body-fit predicate and an x-held alignment check do not certify this combined
advancing/rotating command.

The new option is `hold_feasible_crossing` in the composition recipe, mapped
to `local_support_hold_feasible_crossing` in the environment configuration.
Both default to false. The control only applies to a visible, already-aligned
support-function crossing, excluding terminal parking and explicit C-space
threading. It uses the existing local slit observation and attached geometry.

For each body vertex `(vx, vy)`, its vertical offset is
`vx sin(theta) + vy cos(theta)`. Exact sinusoidal arc extrema, combined with
conservative bounds on the proposed linear y displacement, give a sufficient
whole-body containment check over one control step. If the current pose fits
but the combined command cannot be certified by that bound, preserve its
longitudinal velocity and set desired lateral velocity and angular velocity to
zero. At constant y and yaw, longitudinal translation preserves whole-body
slit containment. The existing physical continuous collision guard remains
authoritative after GBP/force allocation; this local check is not a global
obstacle or maze planner, and floating-point evaluation is not interval arithmetic.

This sufficient bound can reject a safe combined command; it must not be
described as an exact unsafe-command classifier. Unaffected rows preserve the
original expressions. No teacher, route index, future waypoint list, new
learned mode, recurrent-state reset, or runtime confidence gate is introduced.
Existing model/interface limitations outside this repair remain unchanged.

## Regression and fidelity checks

- Disabled control: all **15 saved arrays** in each of the three A traces are
  bitwise identical to the pre-change failed source traces.
- Early panel: 14/18 complete traces are bitwise unchanged; three failures are
  repaired; the fourth changed case is a successful L/20, one step shorter.
- No/near impulse panel: 33/36 complete traces are bitwise unchanged, no success
  regression. The three changed cases are L; every T trace in this panel is
  bitwise unchanged.
- Locked-validation odd-count panel: no success regression; 9/18 complete
  traces are bitwise unchanged. The changed successful T cases still succeed;
  this is not a claim of universal canonical bitwise parity with the option on.
- Tests: **140 passed**, including containment, dense angular-motion checks,
  and previous physical/GBP and teacher-label tests.
- The source checkpoint remains
  `a493af24bdf6de88d478b5c28a1b6f7f69a82aca008dfec054abb1e62e8f9f4f`.

Full A–Z × 14 entry/velocity/scale profiles at N12 (the previous 364-case
development catalog) is running in
`output/p6_crossing_hold_all_states_81201/manifest.json`.
Its partial results must not be reported as the full 364-case result. Broader
agent-count, held-out layout and sequence regression is still required before
any public promotion. The overall p6 goal is not complete.

## Artifacts

Recipe: `configs/p6_force_crossing_hold_control.json`.
Comparison runner: `scripts/compare_crossing_interface.py` authenticates
checkpoint, scene, options and NPZ hashes, then compares every saved array.
Historical zero-angular-impulse metadata is normalized through its declared
default; other disturbance differences are rejected. Paired steps, contacts,
path lengths and actual GBP float payload bytes are in
`output/p6_crossing_interface_comparison/{early,near_and_nominal,validation_odd,disabled}.json`.
Wire headers remain excluded and cannot be inferred from these payload counts.

Manifest SHA256:

- Early panel: `c2b488a1f16fb222f9b6ad69570a39868f7e9527f3df6642ccae3214b45c1d16`
- No/near panel: `d307080ea19785b84ef8a54c539cf1afc5daeee8c454678cc97d09c214237a56`
- Odd validation candidate: `6fbbedbe95895268f88a2444ee39dbe63e6311f27709858c92cb883747d24d09`
- Odd validation source: `f563d5ce52a21454f87cde382109834267d2eac1f8e70c02112245c4bd882d94`

Only small reports are mirrored to Windows. Bulk trajectories stay on Linux.
This cycle used CPU audits and no GPU training; no other user's GPU job was touched.
