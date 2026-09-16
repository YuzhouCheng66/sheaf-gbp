# Mainline composition gates, coordinates and action-level training

2026-09-16. Original architecture remains the mainline; the public p5 demo is
unchanged. No result below proves route memorization or the need for a new
planner. Independent controls are not promoted merely because one property
improves.

## Completed mainline gates

| Panel | v2 weights + consistent interface + covered GBP diameter | Broad-replay v1 weights + same interface/solver |
| --- | ---: | ---: |
| Enlarged three-wall A–Z × N=8–20 | **338/338** | **338/338** |
| A–Z × 14 entry/motion/scale profiles, N=12 | 359/364 | **362/364** |
| Old nominal single-slit A–Z × N=8–20 | **338/338** | **338/338** |
| Validation 82202: S/Z × N=8–20 × five profiles | **130/130** | **130/130** |

All trajectories in these panels pass independent continuous collision
certification. Both full count grids and both nominal/validation panels have
zero collision attempts. The broader 364-state panel has one attempt for the
v2-weight control and 1,032 for broad-replay v1: the latter's large-X failure
accounts for 1,029 attempts. Higher aggregate success is not an across-the-board
improvement in every metric.

For broad-replay v1, the two state-panel failures are X/entry-reference and
X/large. Their **entire NPZ arrays are bitwise identical** under four rounds and
the diameter-covered recipe, because those two actual trees have diameter 4.
Thus the already collected ten actor-induced teacher queries remain relevant;
they were not regenerated or relabelled as new data.

The default composed grid costs:

| Weights | Control steps | GBP float payload bytes |
| --- | ---: | ---: |
| v2 | 138,904 | 1,066,383,024 |
| broad-replay v1 | 137,832 | 1,058,437,632 |

These counts exclude transport headers and add the separately recorded grasp
setup payload. They compare two versions of the same Sheaf system, **not** a
raw-full baseline or a new communication Pareto claim.

The strongest validated development candidate remains
`configs/p6_mainline_original_broad_diameter_v1.json`, checkpoint SHA256
`98c60d1d612f6d8ad707394918c70518c2c62103f757a54411ce7879234ab547`.
Artifacts are `output/p6_mainline_original_broad_diameter_{all_counts,state_81201_all,nominal_all_counts,validation_82202_joint_counts}/manifest.json`.

## A further private-plan DAgger fit was not a net improvement

Appending ten certified continuations from the two real X failures gives 102
queries. The original-width fit with 497 current-actor successful replay
episodes, 20,000 steps and learning rate 5e-5 passes 26/26 at default A–Z/N=12
but only **11/14** X state profiles, versus 12/14 for its source. It repairs
large X while losing forward- and backward-velocity entries; entry-reference
still fails. It is **not promoted**. Recipe:
`configs/p6_mainline_original_broad_diameter_v2.json`.

## Initial-state coordinate sensitivity: a specific input issue

The audit uses actual initial states, zero initial memory and fixed model
weights. It shifts the load, slit geometry and active goal together by
±0.75 m horizontally or ±0.4 m vertically. The normalization and outer arena
constants are retained. Consequently it is a **local-scene coordinate probe**,
not a complete physically translated-arena rollout or an arbitrary SE(2)
equivariance claim. No incorrect transport of old recurrent hidden state is
used to manufacture a difference.

There are 196 probes per axis (seven shapes × 14 profiles × two shifts).
For both tested mainline checkpoints, horizontal shifts change no output
force. Vertical shifts change only column 3 of the private 52-D input above
1e-5: normalized absolute payload-centre y.

| Weights | Median maximum force difference per vertical probe | Maximum |
| --- | ---: | ---: |
| v2 | 0.3534 N | 2.1769 N |
| broad-replay v1 | 0.3758 N | 2.2604 N |

A counterfactual retains the original private plan **before** the existing
geometry branch. All 196 geometry authority masks remain identical, and
maximum force differences fall to 1.19e-6 N and 9.54e-7 N, respectively. This
isolates the private-plan pathway on these probes. It does not establish that
the model memorizes a route, and it does not attribute every failure to this
single input.

Scripts/artifacts:

- `scripts/audit_mainline_coordinate_sensitivity.py`.
- `output/p6_mainline_diameter_coordinate_sensitivity/causal_report.json`.
- `output/p6_mainline_original_broad_coordinate_sensitivity/causal_report.json`.

## Same-width relative-y input control: invariant but currently worse

An optional, default-off input-semantic control replaces that one column by
`(centre_y - current_goal_y) / world_y`. It retains **52 inputs, the original
MLP width/depth, 213,000 trainable parameters, restrictions, GBP and grasp
decoder**. No extra obstacle, route, letter or future-waypoint input is added.
All checkpoint tensors are bitwise unchanged at the initial conversion; this
is not an exact behavioral bridge for nonzero goal-y. At goal-y=0 the input
extractor retains the exact old operation path. Optional +4 goal-context
columns, when separately enabled, are adjusted to avoid subtracting the goal
twice.

On off-policy initial-state probes of that untrained interface conversion,
maximum vertical force difference is **1.91e-6 N**, with no difference above
1e-3 N. This verifies that input property, not task capability or long-memory
transport. Artifact:
`output/p6_mainline_relative_y_coordinate_sensitivity/untrained_bridge.json`.

Matched fits then use the same source tensors, 102 teacher queries/89,856
teacher agent-frames, 502 successful actor episodes/2,594,296 replay
agent-frames, seed 7101, 20,000 updates, batch 1,024, learning rate 5e-5 and
replay weight 3. Both use the latest diameter-covered successful replay.

| Input column 3 | X × 14 state profiles | Default A–Z/N=12 |
| --- | ---: | ---: |
| Original absolute centre-y | **11/14** | **26/26** |
| Relative centre-minus-goal-y | **6/14** | **25/26** |

The relative-y control is **not adopted**. Its source-plan replay loss stays
much higher, which is consistent with difficulty preserving the source's
absolute-coordinate-sensitive function after changing the representation.
That is a hypothesis about the objective/representation interaction, not a
proof of irreducible label conflict. The observed result specifically rejects
promotion of this fit, not the general usefulness of relative coordinates.

Recipes: `configs/p6_mainline_absolute_y_control_v1.json` and
`configs/p6_mainline_goal_relative_y_v1.json`. Checkpoints:

- Absolute control: `a7bb6d95d76f95955a4d93107af65bbd813fab05f8049b90bb78189ad87aef28`.
- Relative control: `adf1114f3adfc3de1ceea5fc45ea41be0fdd14b683b895e33d88d3e58db199af`.

## Verified differentiable replay of actual actuation

The existing private-plan objective supervises coordinates 0:6, but the current
inverse dynamics uses **only coordinates 3:6** (desired body twist). Moreover,
geometry-active nodes' private predictions are overwritten before GBP. A good
private-plan imitation loss therefore does not by itself establish a good
executed force. This is a reason to test an action-level objective, not to
replace the planner architecture.

`src/piano_movers/actuation_training.py` now exposes an **offline training
replay of the original actuation path**:

1. Retain the original geometry proposal and authority mask at that observed
   state; pass private predictions only where they actually have authority.
2. Apply the original local inverse dynamics, confidence values, [I3,0]
   restriction, actual-radius topology and original GBP module/round count.
3. Apply the original distributed grasp-statistics decoder and force clipping.

No new solver or deployed global observation is introduced. Cached geometry
decisions are safe to use for this offline calculation because the original
geometry routine's first six output coordinates and active mask do not depend
on the supplied private plan. The cached last two coordinates do not influence
the physical actuation path.

An audit on **448 authentic states** from eight shapes × 14 profiles verifies:

- **Zero force error** against stored original-policy forces on every state.
- Exact agreement in actual GBP payload count at every sampled state.
- 321 states have a nonzero gradient from force back to the private plan.
- Coordinates 0:3 and 6:8 have zero action gradient; geometry-overridden nodes
  also have zero private-plan gradient, as required by the actual program.

A unit test additionally checks the active action gradient against finite
differences. This is readiness for force-/wrench-level supervised training,
**not** a newly trained checkpoint or a new success-rate claim. Next compare an
action-supervised original-network fit with the private-plan objective under
matched replay and held-out evaluation. Keep all public behavior unchanged
until closed-loop results justify a promotion.

Audit: `output/p6_mainline_actuation_training_audit/report.json`.
Code validation: **107 repository tests pass** on CPU; changed-script lint passes.

## Reserved evaluation remains independent

The frozen broad-replay-v1/diameter recipe is being evaluated on predeclared
reserved seeds 83201–83204: combined layouts/goals, S/Z, all N=8–20 and all 14
non-nominal entry/motion/scale profiles (**1,456 requested episodes**). The
catalog denominator is frozen before execution; failures remain visible.
S/Z and odd counts are held out from p6 fine-tuning, not claimed unseen by
ancestral p5. These four layout draws, with many correlated cases per layout,
must not be reported as 1,456 independent layout samples. No reserved-test
state or outcome is admitted to training or used to select these already
specified input/objective controls.

Pending broader goal requirements include full completed reserved reports
with cluster-aware uncertainty, full translation/rotation composition and
causal-memory transport tests, genuinely changed waypoint-sequence/layout
lengths, and a validated p6 demo. High nominal counts alone do not complete
the goal or establish maze mapping/global planning.
