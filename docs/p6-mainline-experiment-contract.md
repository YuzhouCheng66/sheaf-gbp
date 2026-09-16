# Original-architecture mainline

Latest measured results: [original-architecture v2 report](p6-mainline-v2-results.md).
The trained original modules with the screened interface now pass 332/338 on
the enlarged three-wall A–Z/8–20 development panel, and 338/338 on the old
nominal single-slit regression. These do not complete the fresh generalization,
joint count/entry-state or public-release gates.

This contract supersedes any implication in earlier p6 experiment notes that
the separate `LocalSkillPlanner` must replace p5. The user's direction is to
keep the original architecture as the mainline and diagnose interfaces before
changing models.

## Evidence boundary

Frozen B/12 changes from failure to success with a body-clear waypoint handoff
and no weight changes. Three-wall failures therefore do not establish an
architectural limitation, and do not establish route memorization. The source
code's geometric calculations and the learned components must be described
separately; neither the nominal success rate nor a failed composition alone
identifies what the network learned.

## Mainline, in order

1. Freeze p5 weights, encoder, private planner/recovery modules, physical
   Sheaf-GBP, and analytic grasp decoder. Test independently: position-only
   intermediate targets, trailing-body handoff, nearby obstacle visibility,
   private-memory continuity, and previous-step mechanical feedback continuity.
2. Keep the original 27-D observation contract for these controls. An opt-in
   sensor fills its previously dummy second wall slot with each handle's
   nearest non-active wall, masked by the **existing horizontal visibility**
   rule. This does not constitute a new all-obstacle or occlusion-aware sensor:
   two simultaneously visible secondary walls exceed its capacity. Record
   that limitation instead of describing the interface as unlimited.
3. Collect authentic mainline rollouts with the corrected, documented
   interface. Preserve actual observations, private states, contacts, active
   goal, checkpoint/source hashes, and scene provenance. Query the training
   teacher from those physical states; retain failed queries. New entrants and
   contact states must not be replaced by unrelated nominal trajectories.
4. Train the original planner/recovery parameters where they actually affect
   executed plans and forces. Verify that an unchanged geometric branch does
   not overwrite the trained output. Do not silently bypass that branch or
   insert a new planner to make the training appear effective. Any necessary
   branch/interface change is a separately declared controlled experiment.
5. Regress old single-slit/canonical cases and evaluate unseen local entries,
   goal switches, shape/count/layout combinations, then multi-wall composition.
   Keep continuous collision checks and real communication accounting separate
   from terminal success.

## Independent controls, not replacements

The `local_skill_policy.LocalSkillPlanner` experiments v1–v7 (78/162 inputs,
96/256 hidden dimensions, optional goal residual, stopped-state labels,
collective imitation, and factorial data) are **alternative-architecture
controls**. Their datasets may be reusable, but their labels/features must be
converted and audited for the original interface before mainline training.
Their low success rates are not evidence against the original architecture.
No such checkpoint is authorized for public-demo replacement without matched
closed-loop validation and parent regression.

The goal-command interface experiment is distinct from those new planners:
counterfactual audits found that the original neural MLP lacks true goal y and
the orientation-required command, even though its analytic controller receives
them. Its independent zero bridge adds four first-layer input columns (1,024
weights), retaining the original MLP's width/depth, original tensors, GBP and
decoder. Actual-state and full-rollout parity are checked before training.
Even this minimal input change is not promoted without matched closed-loop
regression; repairing X alone does not excuse A/Z regressions. The required
active-goal command is not a route, future target list or teacher input.

The v7 run already in progress when this direction was clarified may finish
as a recorded control. Do not automatically continue that series at the expense
of the original-architecture mainline. The public p5 worktree remains untouched.

## Frozen-mainline evidence after the clarification

All entries below use the same p5 checkpoint; none changes model weights.
They are inspected development cases, not fresh held-out generalization.

| Interface | A | B | L | T | X |
| --- | --- | --- | --- | --- | --- |
| Earlier position-only + body-clear handoff | fail | 656 | fail | 858 | 770 |
| Also expose nearest secondary wall | 593 | 573 | fail | 859 | 653 |
| Only retain previous-step physical feedback | fail | 656 | fail | 858 | 770 |
| Secondary wall + private/physical history continuity | 593 | 573 | fail | 859 | 653 |

Numbers are control steps to complete all four targets. All successful entries
in the three new rows have zero rejected contacts. All 15 new trajectories,
including failures, pass the independent recursive continuous-motion check.
See `output/p6_mainline_frozen_interface_v3/manifest.json`.

The complete A–Z/N=12 secondary-wall development grid is **14/26**, not an
overall improvement over the earlier 15/26 interface. A is repaired, while D
and S regress. The previous-wall-only variant also finishes 14/26. Therefore
neither new sensor option is promoted to a default or a release. The first
five-letter improvement is useful causal evidence, not a selection shortcut.
Artifacts: `output/p6_mainline_nearby_remaining_AZ12` plus the five-letter
panel above; `output/p6_mainline_previous_wall_AZ12`.

The original-interface DAgger collector is
`scripts/collect_mainline_skill_dagger.py`. It authenticates actual 27-D
observations and private states, re-observes certified teacher continuations
through the same interface, and does not introduce path-progress supervision.
The first L query from the nominal source has a certified 93-step executable
continuation satisfying the body-clear handoff. Its actual stalled state at
tick 366 has an 18-step certified continuation to the position target, but
fails the supervisor's additional body-clear condition. This second query is
retained with `admitted_for_training=false`, not silently trained or counted
as successful recovery. At that stalled state, three nodes retain direct
learned-plan authority and nine use the geometric branch; training must be
checked at the executed-action level. Artifact:
`output/p6_mainline_dagger_coherent_v1`.

Removing the body-clear gate while retaining secondary-wall visibility is
also tested, not assumed to be a fix. On A/B/L/T/X it yields 3/5: A/T/X pass,
L remains stalled, and B again hits its trailing wall after an early handoff.
All trajectories pass the independent continuous checker. Artifact:
`output/p6_mainline_position_handoff_v1`. Thus the next mainline teacher work
must target the existing declared handoff region (including body clearance),
not quietly discard that condition to admit L's labels. A terminal-region
teacher can search for another feasible pose within the supervisor's position
tolerance; it must not supply a memorized deployment route.
