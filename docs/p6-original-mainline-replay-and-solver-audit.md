# Original-mainline replay and GBP budget audit

Development evidence, 2026-09-16. **Not a new public release.** The original
local planner → physical Sheaf-GBP → analytic grasp decoder remains the
mainline. The public p5 checkpoint/demo has not been changed. None of these
results establishes route memorization or requires a replacement planner.

## Successful-state coverage matters

The matched training experiment keeps the original morphology MLP and motion
head: 52 inputs, 213,000 trainable parameters. All other checkpoint tensors
remain frozen. It uses 92 admitted, authentic actor-induced teacher queries,
plus 491 independently certified successful development episodes from the
same v2 replay actor (2,755,340 agent-frames). Successful replay now includes
large entry headings, velocities and scales, not only nominal starts.

The new optional four-goal-column input is a **separate control**, with 1,024
additional parameters. Both fits use the same data, optimizer, seed, 20,000
updates, batch size 1,024, learning rate 1e-4 and replay weight 3. GPU fitting
takes 29.47 s for the original input and 33.59 s with the additional inputs;
data loading and closed-loop evaluation are not included in these timings.
Successful anchors exclude S/Z and all odd agent counts; reserved evaluation
episodes are never admitted to training. Teacher supervision is on private
plans, so closed-loop evaluation remains essential wherever an analytic plan
overrides the private prediction.

| Matched panel | Original inputs | +4 goal inputs |
| --- | ---: | ---: |
| A/B/C/I/X/Z × 14 state profiles, N=12 | 82/84 | 82/84 |
| Default enlarged three-wall A–Z, N=12 | 26/26 | 26/26 |

Both 84-case arms fail X at `entry_reference` and `shape_large`. There is no
extra-input benefit on this panel; the original input remains primary.

The original-input fit was then evaluated more widely:

| Development panel | Consistent-interface v2 reference | Original broad-replay fit |
| --- | ---: | ---: |
| A–Z × 14 state profiles, N=12 | 357/364 | **362/364** |
| Default enlarged three-wall A–Z × N=8–20 | 332/338¹ | **328/338** |

¹ The 332-case reference uses the rejoin/alignment-screen recipe. Its default
course traces were verified bitwise equal with and without the alignment
screen; the goal-region control subsequently repairs a different perturbed
panel. These are not the old nominal single-slit 338 cases.

All 364 and 338 trajectories of the new fit pass independent continuous
collision certification, including the unsuccessful episodes. The remaining
state-panel failures are X/reference and X/large. The count-grid failures are
D/9,10,15,16,17,18,19,20 and J/9,10. **The gain in entry-state coverage comes
with count-grid regressions; this is not a promoted checkpoint.** The +4-input
fit has not received the full 364-case evaluation: do not attribute 362/364
to it. The new original fit's old nominal single-slit regression and independent
joint-OOD evaluation remain pending.

Earlier narrow-replay alternatives are also not promoted: command-only
training gave 355/364 and the joint original-MLP/+4-column fit gave 353/364,
both under the *alignment-screen-only* recipe. The correct reference for
those two comparisons is 354/364, not the later consistent-interface 357/364.

Artifacts:

- `configs/p6_mainline_original_broad_replay_v1.json`:
  checkpoint SHA256 `98c60d1d612f6d8ad707394918c70518c2c62103f757a54411ce7879234ab547`.
- `configs/p6_mainline_goal_broad_replay_v1.json`:
  checkpoint SHA256 `19da5b9feae840b9dbe627e0b59199c93e21d23fb621f0f203b901799106973d`.
- `output/p6_mainline_original_broad_state_81201_all/manifest.json`.
- `output/p6_mainline_original_broad_all_counts/manifest.json`.
- `output/p6_mainline_goal_full_v1_state_81201_all/manifest.json`.

## Physical feedback at goal handoff

An independent unchanged-weight control retains the previous force, real hit
indicator and rejected-motion feedback while explicitly resetting the
goal/frame-conditioned private latent. It gives **357/364**, the same success
count as the matching reset-feedback reference. The audit checks state
continuity and feedback equality at **1,074 actual target switches**. This is
evidence for one-step mechanical continuity, not preservation or coordinate-
correct transport of an arbitrary long recurrent memory.

Artifacts: `configs/p6_mainline_physical_history_control.json` and
`output/p6_mainline_physical_history_state_81201_all/history_audit.json`.

## Two distinct physical solver issues

### Reweighting partial statistics is not reweighting complete estimates

For handle i, local measured velocity is c_i = A_i v, with
`A_i = [[1, 0, -r_y], [0, 1, r_x]]`. The original inverse-dynamics proposal uses
the partial twist statistic `s_i = N M^-1 A_i^T c_i`, where
`M = sum_i A_i^T A_i` depends only on fixed attachment geometry. Its **uniform
mean**, not each individual statistic, equals the body twist v.

Giving these already-inverted partial statistics heterogeneous weights breaks
that identity. The actual policy uses precision boost 4 for geometry-active
nodes and 1 for other nodes. On 711 mixed-confidence moving-state probes,
an identical coasting command that should request zero total wrench acquires
maximum normalized weighted bias `[0.1631, 0.2165, 0.2854]`. This is an algebraic
and counterfactual diagnosis, not proof that it causes every failed episode.

The corrected finite-solver audit verifies factual force parity first, then
retains the actual confidence masks while replaying the real GBP and grasp
decoder. Representative coasting torque changes include I: 10.27 Nm →
-0.00019 Nm with equal weights; for D, equal weights alone leave -2.12 Nm at
four rounds, while six rounds reduce that residual to 0.00011 Nm. These are
diagnostic commanded-coasting cases, not normal-policy performance scores.

`output/p6_mainline_weighted_velocity_audit/finite_solver_corrected.json` is
authoritative. **The earlier `finite_solver.json` probe is invalid for the
precision comparison**: its plan-override API resets the geometry mask, so
both supposed confidence arms actually used uniform confidence. It is retained
as an audit trail and must not be cited as a valid comparison.

`local_wrench_factors.py` is a mathematical **independent control only**. It
forms a rank-two local information factor directly from each handle's velocity
and desired motion, preserving its full 3×3 information matrix. Diagnostic
central aggregation recovers the common coasting wrench to 7.23e-8. This is
not a deployed centralized policy and is not wired into the current diagonal
GBP. Arbitrary full node covariance cannot be made exact merely by passing it
through the diagonal solver. A full 3D Gaussian packet would need nine floats,
rather than the present six, absent a separately justified encoding.

### Four rounds do not cover every morphology's communication tree

The legacy four-round assumption holds for the canonical topology but not all
enlarged letters/counts: the actual selected radius-edge tree can have diameter
6 or 7. The opt-in budget is `max(4, actual_tree_diameter)`, computed once after
attachment. It changes no weights, private planner or decoder. Messages still
traverse only real radius edges and every additional round is charged.

Current solver factors have strong selected-tree edges and very weak non-tree
edges; the code **still transmits on the non-tree edges**. Therefore this budget
is not a theorem of exact loopy convergence. No nonexistent sparsification is
credited to its communication account.

## Unchanged-weight factorial: physical consistency is not enough

Predeclared development panel: A/B/C/I/L/O/T/X at N=12 and D at every N=8–20,
all at seed 81201 with the same `entry_reference` course. These are **21 paired
episodes per arm**, not 21 independent layout draws. All weights are v2 SHA256
`44be6ea0cbd21b036b2fbcc5c18e0fb5035d79f77d92d987fca39306b742f795`.

| Control | Success | Steps | Collision attempts | GBP payload bytes |
| --- | ---: | ---: | ---: | ---: |
| Original confidence, 4 rounds | 21/21 | 10,773 | 221 | 57,557,376 |
| Equal confidence, 4 rounds | 19/21 | 10,259 | 1,261 | 54,814,272 |
| Original confidence, cover diameter | **21/21** | **8,174** | **0** | 58,446,432 |
| Equal confidence, cover diameter | 19/21 | 8,694 | 1,029 | 61,083,792 |

All 84 trajectories pass independent continuous certification. Equal confidence
loses I and X, with or without the longer budget. It must not replace the policy
on the strength of a clean coasting diagnostic. Its lower total payload includes
failed, shorter-travel episodes and is not an efficiency win.

Covering diameter with original confidence reduces steps by 24.13%, preserves
all successes and eliminates these 221 rejected collision attempts, **but raises
GBP payload by 1.54%**. All arms also pay the same 107,104 bytes of attachment
grasp setup. Transport headers/topology discovery are not included; these are
float-payload counts, not total on-air bytes and not a raw-baseline Pareto claim.

The summarizer authenticates catalogs/trace hashes, verifies every recorded
per-tick payload against the actual rounds and directed endpoints, and checks
all stored arrays of nine overlapping reference episodes bitwise against the
pre-control implementation. Thus disabled controls preserve those complete
trajectories, not merely aggregate success.

Reproduce the report:

```bash
PYTHONPATH=src python scripts/summarize_mainline_solver_controls.py \
  --old-reference output/p6_mainline_consistent_v2_state_81201_all/manifest.json \
  --output output/solver_factorial_fresh/report.json
```

Inputs: `configs/p6_mainline_solver_{reference,equal,diameter,equal_diameter}.json`;
report: `output/p6_mainline_solver_factorial_81201/report.json`.

The enlarged default all-count and full 364-state diameter-budget audits are
running next. No public replacement is authorized by this small panel alone.
