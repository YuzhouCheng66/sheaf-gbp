# Action losses, position-only handoff and sequence composition

2026-09-17. Original architecture remains the mainline. No public p5 demo was
changed. Windows holds the source/checkpoint/report mirror; computation uses
the existing uk-linux environment. Bulk NPZ traces and the tensor cache remain
on Linux, as requested during the import.

## Matched training completed; neither loss alone was promoted

Both fits start from broad-replay v1 (`98c60d1d...`) and train the original
52-input morphology MLP only: 213,000 parameters. The original geometry
authority, physical Sheaf-GBP, restrictions, message budget and grasp decoder
remain unchanged. The offline differentiable action path is authenticated
against deployed forces before training.

The dataset contains 102 admitted model-induced teacher queries (7,488 whole
multi-agent frames / 89,856 agent-frames) and 502 successful development replay
episodes (204,114 whole frames / 2,594,296 agent-frames). S/Z and odd counts are
excluded from p6 training. Each fit uses 3,000 updates, 64 complete-agent
frames from each source per update, learning rate 2e-5, replay weight 3 and
seed 7102. Frames are sampled with the same bucket/frame RNG in both fits.
GPU update time is 32.63 s for the force objective and 9.62 s for the plan
objective: update/data matching is not equal wall-clock compute.

| Original-network variant | X, 14 states | Default A–Z/N=12 | Full A–Z × 14 states |
| --- | ---: | ---: | ---: |
| Source broad-replay v1 + diameter | 12/14 | 26/26 | 362/364 |
| Executed-force loss | 14/14 | 26/26 | 361/364 |
| Matched grouped private-plan loss | 14/14 | 26/26 | 361/364 |

Both new fits repair X but fail L at −180°, −135° and −90° entries. This does
not establish superiority of the force loss, and neither replaces the best
fully regressed candidate.

Checkpoints:

- Force: `a493af24bdf6de88d478b5c28a1b6f7f69a82aca008dfec054abb1e62e8f9f4f`.
- Plan: `42db33e5e159ae8cc70e95e067c579da653571c0c0e8837bb66ac2f065e62a4d`.

Configs: `p6_mainline_force_loss_v1.json`, `p6_mainline_grouped_plan_loss_v1.json`.
Merged result files on Linux: `output/p6_mainline_{force_loss,grouped_plan_loss}_v1_state_81201_all.json`.

An exhaustive offline deployed-force audit uses all 7,488 teacher frames and
all 204,114 replay frames. **5,497 teacher frames have zero influence from the
trained morphology head** because the geometry branch controls every node;
their residual is effectively unchanged by either fit. Among the remaining
1,991 frames, teacher force MAE is 1.3517 N for the source, 1.2729 N for force
loss and 1.3598 N for plan loss. Across source replay, force MAE drift is
0.01363 N and 0.00333 N respectively. Thus force supervision improves this
off-policy force metric but perturbs the source more, without a superior
closed-loop success rate. This is not evidence that reducing that metric
alone solves the task. Artifact: `output/p6_mainline_action_objective_audit/report_v2.json`.

## The L regression exposes a remaining interface conflict

All three force-fit L failures are motionless at stage 2, not colliding or
oscillating. Final centre x is 2.644886 m, while the current goal is x=2.489213 m.
The unchanged goal-disk radius is 0.14 m. At the acquired heading, satisfying
the unchanged 0.03 m full-body exit reserve requires x≥2.639886 m; the disk
requires x≤2.629213 m. **That fixed-heading intersection is empty by 10.67 mm.**
The existing terminal geometry branch also holds the current yaw because the
target is position-only. It therefore parks outside the disk indefinitely.
All nodes are geometry-controlled at that state: changing morphology-head
weights has no instantaneous influence on the executed forces there.

Six distinct actual-student queries were collected from the three failed
episodes, retaining authentic observations and histories. All six have
certified executable teacher continuations satisfying the original handoff
contract. The three stalled-state queries have zero morphology-head authority.
They are retained as data, but no claim is made that merely adding their loss
can override an unchanged analytic branch.

Artifacts: `output/p6_mainline_force_loss_L_dagger_81201/{manifest,goal_region_diagnosis}.json`.
The authenticated 108-query union with previous development rounds is
`output/p6_mainline_round3_L_recovery/manifest.json`; no additional training on
this union has yet been claimed or run.
The diagnostic's attempted fixed-heading counterfactual explicitly fails the
goal-disk and reserve checks; a collision-free segment alone is not a valid
handoff witness.

## Independent position-goal heading control

Default-off `resolve_position_goal_heading` changes no learned tensors, force
solver or success tolerance. Only when the current heading has an empty
goal-disk/full-body-halfspace intersection after the body has cleared the wall:

1. Search relative heading offsets in 5° steps.
2. Analytically bound every rigid vertex over the entire proposed rotation
   arc, using extrema of `a cos(theta) + b sin(theta)`; sampling endpoints alone
   is not treated as a sweep certificate.
3. Accept only an in-place arc staying beyond the same reserved wall plane and
   inside the y bounds, whose final heading admits a translation into the
   unchanged goal disk. No valid candidate leaves the legacy action unchanged.
4. Keep position fixed while rotating. Ordinary terminal translation resumes
   once the goal region becomes feasible.

This is a separately identified **geometric interface control**, not a new
learned planner and not evidence that the neural network learned this recovery.
It uses the current local wall/chart, attached geometry and active goal; no
later waypoint is supplied. The full simulator/continuous collision guard
still checks other walls: the local halfspace certificate is not an all-maze
collision certificate. Unrelated implementations retain default-off behavior.

Same-checkpoint L/X test: **28/28**, zero contact attempts, all continuously
certified. All arrays in **25 of 28 trajectories are bitwise identical** to
the control-off traces; exactly the three failed L cases change and succeed.
Full regression completes **364/364**, with zero collision attempts, zero
resolved penetration and all 364 continuous certificates. **361 complete
traces remain bitwise unchanged**, and exactly those same three L failures
are repaired; no other trajectory changes. The full panel costs 100,513
control steps, 2,443.78 m of path, 489,920,592 GBP float-payload bytes and
1,409,408 grasp-setup payload bytes. Headers are excluded. The enlarged
three-wall **A–Z × every N=8–20 grid also completes 338/338**, all certified,
zero collision attempts, in 134,525 control steps. The force-fit checkpoint's
old nominal single-slit regression completes **338/338**, all certified,
zero collision attempts, in 31,825 steps. The new heading control is inactive
for those final oriented goals. This is still not a public release or an
arbitrary-maze generalization claim.

Recipe: `configs/p6_force_position_heading_control.json`.
Artifacts: `output/p6_force_heading_control_LX/{manifest,paired_trace_report}.json`.
Full-panel artifacts: `output/p6_force_heading_control_all/{manifest,paired_trace_report}.json`.
Count grid: `output/p6_force_heading_counts_all/manifest.json`.
Nominal regression: `output/p6_force_v1_nominal_all_counts/manifest.json`.

## Variable sequence length without additional training

New outcome-independent draws use 1, 2, 3 or 5 walls and one position target
after each wall plus a final SE(2) target. All lengths use the same arena
half-size (7 m, 1.6 m), preserving normalization across this panel, and share
identical local prefixes. This arena differs from the earlier three-wall
panel, so its three-wall member is a necessary matched control. The horizon
is predeclared as `ceil(1100 * wall_count / 3)`; it is not increased after
observing a failure. Shape and N never enter course generation.

The original broad-replay-v1/diameter candidate, **without either new loss or
heading control**, passes **20/20** for A/B/L/T/X at N=12 across all four wall
counts on development seed 81201, with zero contact attempts and continuous
certification. Five-wall T takes 1,194 steps and completes six targets. This
is evidence of repeated invocation across new sequence lengths, not arbitrary
wall orientations, maze navigation or global planning. The S/Z × counts
8/11/16/19/20 × three entry profiles × four lengths validation completes
**120/120**, zero collision attempts and all continuous certificates. It costs
25,790 control steps and 207,581,808 GBP float-payload bytes, plus 813,312
setup bytes. This is one validation layout seed with correlated length/profile
cases, not 120 independent maze draws or evidence for all integer counts in
the new sequence family.

The force-fit/heading-control candidate separately completes the same locked
120-case sequence panel **120/120**, again zero collision attempts and all
continuous certificates, in 25,687 steps. These are paired validation results,
not a fresh 120-layout test set. Artifact:
`output/p6_force_heading_sequence_validation_82201/manifest.json`.

Artifact: `output/p6_mainline_sequence_length_81201/manifest.json`.
Validation: `output/p6_mainline_sequence_validation_82201/manifest.json`.
Generator: `sequence_length_course`; runner: `audit_mainline_layout_validation.py
--sequence-wall-counts 1 2 3 5 --families combined`.

## Relative-y force-loss control: do not adopt merely for invariance

An additional matched fit retains the same 52-dimensional MLP and changes
only column 3 to centre-minus-current-goal-y. It uses the same cached source
forces, original source tensors, data, 3,000 updates and other hyperparameters
as the absolute-y force fit; no heading-search control is enabled in this
comparison. Training took 33.68 s on the Linux RTX 5090. Checkpoint SHA256:
`337eeb6d200934325cd50431a08546ced8e74929fb55b638d6f3a6114736535a`.

Its local-scene coordinate probe has zero horizontal force difference and
maximum vertical difference 1.91e-6 N across 196 probes per axis, but the outer
arena was not translated: this is not full physical SE(2) equivariance. It
passes default A–Z/N=12 **26/26**, but only **23/28** in L/X state profiles
versus the absolute-y force fit's 25/28. The same three L failures remain and
X/+90° and X/+135° newly fail. Thus it is **not promoted**, despite satisfying
the intended input-sensitivity property. A more invariant representation is
not sufficient evidence of stronger closed-loop capability.

Artifacts: `output/p6_relative_y_force_v1_{LX,AZ12}/manifest.json` and
`output/p6_relative_y_force_v1_coordinate/report.json`.
Recipe: `configs/p6_relative_y_force_loss_v1.json`.

## Remaining gates

The force-fit/heading-control candidate now passes the declared development
and nominal/count/sequence-validation gates. It remains a distinctly versioned
candidate, not an unannounced replacement of public p5. Remaining work includes
actual rotated/translated scene composition and memory/frame handling,
relative-goal command coverage, reserved generalization for the chosen final
recipe and a validated interactive p6 demo. Earlier four-layout reserved
results do not automatically transfer to a new checkpoint or interface. The
full objective remains active. **113 repository tests pass** on Linux CPU.
