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
development catalog) has completed **364/364**, with zero contact and zero
penetration; all continuous certificates pass. The manifest is
`output/p6_crossing_hold_all_states_81201/manifest.json`.
The full agent-count and reused held-out panels have also completed, as
recorded below. The final input-locality/goal-expression audit, updated
sequence-composition validation and validated development demo remain pending;
the overall p6 goal is not complete and there is no public promotion.

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

## Follow-up regression work

The fixed-chart closed-loop regression has completed for A/B/L/T/X/Z, N12,
and six rigid copies of the seed-81201 parallel-wall scene:
**36/36 successful**, all independent world-frame continuous certificates pass,
zero contact and zero penetration. All 30 nonidentity copies match their
corresponding identity-world case in success, control steps and actual GBP
payload bytes. Physics runs in float64 in the transformed world; the policy
is rerun each step through the existing supplied chart adapter. This is not
force replay, learned rotation equivariance, changing local frames, or a
mixed-wall-normal maze test.

Report: `output/p6_crossing_hold_framed_81201/manifest.json`, SHA256
`52ccbbe05f9663e97f7d2af08a495304de68a5bdbe9517080a4c4aff8ed4e7e8`.

Larger regressions:

- `output/p6_crossing_hold_all_states_81201/manifest.json`: the complete
  previous A–Z × 14 state-profile × N12 catalog, **364/364 complete**.
- `output/p6_crossing_hold_counts_all/manifest.json`: A–Z × every integer
  N8–20 on the previous default three-wall course, **338/338 complete**.

The paired comparator `scripts/compare_mainline_trace_controls.py` now rejects
duplicate, missing, or extra cases and can require that exactly one named
composition option changes (`--only-option hold_feasible_crossing`). It checks
course and checkpoint identity, authenticates all saved trace hashes, and
reports success, contacts, steps, path length, attachment setup and GBP payload,
plus bitwise trace parity. Low-motion fraction requires both speed <0.01 m/s
and |angular speed| <0.01 rad/s, so in-place rotation is not mislabeled as
stationary. This metric alone does not determine whether the model is stuck.
Test coverage has increased to **143 passing tests**. No public promotion has
been made while the complete regressions remain pending.

### Complete 364-case paired result

The comparator authenticated the whole case set and verified that only
`hold_feasible_crossing` changes. Source and candidate both pass 364/364;
334 complete traces are bitwise identical. Every T case in this panel is
bitwise unchanged. Total control steps change from 100,513 to 100,483;
GBP float payload changes from 489,920,592 to 489,790,224 bytes. These small
aggregate differences are not evidence of a universal efficiency improvement:
one A/-135-degree case takes 229 instead of 224 steps, and 16 cases have a
slightly longer path (maximum increase about 0.0258 m). No success or collision
regression occurs. Both panels have zero collision attempts.

Manifest SHA256:
`9554c6433b4d590fbc11203da95416ac6b5052816ab9f2e8aa658b2b75a077b3`.
Detailed paired metrics: `output/p6_crossing_interface_comparison/all_states.json`.

### Remaining evaluation, no new teacher labels

Four single-thread CPU shards have completed the predeclared S/Z ×
N8–20 × 14 state profiles × seeds 83201–83204 panel (1,456 cells). Their
manifests are `output/p6_crossing_hold_reserved_<seed>/manifest.json`.
These generator seeds were examined with earlier candidates: this is a reused
regression/OOD panel, **not a newly sealed blind test**. S/Z and odd counts remain
excluded from this DAgger training branch; that is not a claim that ancestral
p5 never encountered those glyphs. No labels are generated from these runs.
The existing layout-cluster report retains the four-draw uncertainty caveat;
individual shape/count/profile cells must not be counted as independent layouts.

The old single-slit runner now has an explicit `--hold-feasible-crossing`
option. Without it, rerunning that benchmark would only test the default-off
source, not the new interface. The current audit uses the same source weights,
384-step horizon, benchmark T profile, exact-contact guard, 32 planning samples,
and diameter-covering GBP budget as the previous 338/338 source grid, changing
only this option. It has completed **338/338** at
`output/p6_crossing_hold_single_slit_all/manifest.json`. All 13 canonical T
counts have completed successfully; their entire state trajectories and GBP
payload counts are bitwise/exactly unchanged. The shard merger rejects mixed
values of this new option.

## Complete count, legacy and reused-OOD panels

| Panel | Strict success | Continuous certificates | Contacts / penetration |
| --- | ---: | ---: | --- |
| Default three-wall A–Z × N8–20 | 338/338 | 338/338 | zero / zero |
| Old single-slit A–Z × N8–20 | 338/338 | 338/338 | zero / zero |
| S/Z × N8–20 × 14 profiles × 4 reused layouts | 1,456/1,456 | 1,456/1,456 | zero / zero |
| Odd-N subset of the preceding panel | 672/672 | 672/672 | zero / zero |

Every count in 8–20 scores 26/26 on each nominal grid and 112/112 on the reused
S/Z panel. These are empirical panel results, not proof of a >=90% population
success probability on arbitrary scenes. The source weights still have not
changed during this crossing-interface experiment.

The complete three-wall comparison has 262/338 bitwise-identical traces and no
success regressions; all 13 T traces are identical. Total steps change from
134,525 to 134,438 and GBP payload from 1,030,303,056 to 1,029,599,328 bytes.
Some individual F cases take up to five additional steps, so this is not a
per-case efficiency dominance claim. The legacy single-slit comparison has
337/338 identical state trajectories. Only J/17 changes, with unchanged 86
steps, zero contacts and unchanged 1,213,632 GBP payload bytes. State equality
does not independently establish equality of every individual ant force.

The S/Z aggregate uses exactly four original layout draws. Its descriptive
layout-cluster bootstrap interval is [1,1], a degenerate result from four
all-success clusters, **not certainty**. The independent-bounded-layout
Hoeffding interval is approximately [0.321,1]. Neither interval corrects for
candidate selection or earlier inspection of the same panel. The report
explicitly marks panel reuse and makes no blind-test claim. It cannot certify
>=90% true success or unseen maze layouts. S/Z and odd counts are held out of
the current DAgger branch, not necessarily ancestral p5 exposure.

All requested cases, including any invalid-endpoint or uncertified cases,
must remain in the report denominator. Reporting now handles a missing trace
for an invalid endpoint without crashing or dropping that failure; unit tests
cover this and equal layout-cluster denominators. **145 tests pass.**

Key artifacts (bulk traces remain on Linux):

- Three-wall manifest: `output/p6_crossing_hold_counts_all/manifest.json`,
  SHA256 `d596248ca7240a07c083f183c37cabbbac27ba647a36f323ca2529528e0c6f05`.
- Legacy manifest: `output/p6_crossing_hold_single_slit_all/manifest.json`,
  SHA256 `688bc2c468b40ac0f80c7f1fa0da32c84da0dbb891c707dac01068489ec73e80`.
- Merged reused panel: `output/p6_crossing_hold_reserved_all/manifest.json`,
  SHA256 `696be567d463eb763e924b746b887cc8af3e656817c93e14465f88a59fa8160e`.
- Cluster/per-agent report: `output/p6_crossing_hold_reserved_all/report.json`,
  SHA256 `f7d9c111fae3851e7212d4895b64f75d6ca063cd980b6e05553d03596e7d6052`.
- Detailed comparisons: `output/p6_crossing_interface_comparison/counts_all.json`
  and `single_slit.json`.
