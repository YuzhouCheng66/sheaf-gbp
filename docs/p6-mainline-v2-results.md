# Original-architecture p6 v2: development results

**Not a release or an OOD success claim.** The public p5 demo is unchanged.
Use the complete candidate recipe in
[`configs/p6_mainline_candidate_v2.json`](../configs/p6_mainline_candidate_v2.json),
not just its checkpoint: target handoff and collision settings are part of the
validated system.

## Extended count and first held-out-course audit

The original v2 recipe has now completed the **enlarged three-wall A–Z ×
N=8–20 grid: 331/338 (97.93%)**, with all 338 trajectories independently
continuously certified. This is the harder composed course, unlike the
338/338 nominal single-slit regression below.

| Agent counts | Successes per count |
| --- | ---: |
| 8, 11, 12, 13, 14, 15, 20 | 26/26 |
| 9, 16, 17, 18, 19 | 25/26 |
| 10 | 24/26 |

The seven failures are D/9,10,16,17,18,19 and T/10. Every count exceeds 90%
as a point estimate on this grid; this is not a lower-confidence-bound or
arbitrary-layout guarantee. D's failures include slow transit and late final
arrival, not seven identical wall deadlocks. T/10 is a genuine repeated-contact
deadlock (922 rejected moves, zero motion over its last 100 steps).

Predeclared course-family validation at seed 82201, A–Z/N=12 gives **103/104**:
layout-only 26/26, entry-pose-only 25/26, goal-only 26/26, combined 26/26.
All 104 trajectories pass continuous certification. No teacher-admission or
outcome-based filtering is applied, and the case catalog is hashed before any
rollout. These 104 cases share only **one draw per family**, not 104 independently
sampled layouts. The actual entry heading is only 0.01784 rad, despite the
generator's wider declared range. Thus this is useful first held-out-course
evidence, not broad orientation, contact-entry, or velocity generalization.
Source velocity is zero; reserved test seeds have not been evaluated.

The validation failure is C at the second wall. A frozen-state replay finds
nearly identical local plans and a ~1.4 micrometre interior collision while
aligning; the previous wall is not the obstructing wall. The continuous guard
correctly rejects it. This validation case is not added to training data.

Artifacts:
`output/p6_mainline_v2_composed_all_counts/manifest.json` and
`output/p6_mainline_v2_validation_82201_all/manifest.json`.

### Isolated rejoin-screen control, unchanged v2 weights

The T/10 failure identifies an independent geometric-interface defect: the
original fast C-space branch uses a reference route with 10 mm planning
reserve as if it also certified rejoining that route from the actual tracked
pose. At the deadlock, each agent's predicted motion matches the realized
force-driven proposal, but its midpoint penetrates by ~0.259 mm even though
its endpoint has positive clearance. This is not evidence of failed GBP
consensus, nor does fitting an overwritten neural plan repair that action.

An **opt-in, default-off** control screens these previously unchecked actual-
pose lookaheads with the existing local candidate checker, keeping their
original gain and 16-point lookahead. Neural weights, restrictions, GBP and
grasp decoding are unchanged. Same-source replay gives T/10:
failure/1,100 steps/922 contacts → success/932 steps/9 contacts, with independent
continuous certification in both cases. The control also passes T/8–20
(13/13) and the enlarged A–Z/N=12 panel (26/26). Its complete all-count paired
comparison is separate; it is not silently enabled in the v2 recipe or public
demo. Implementation: `72f7b97`; control name `complete_exit_screened`.

### Additional authentic-state DAgger, original modules only

The 18 new D queries come exclusively from the actual v2 development failures
at **even** N=10,16,18. All have admitted executable teacher continuations.
Odd-count failures and held-out-course C are not used as training examples.
Four certified T/10 queries are archived separately: the deadlock has zero
neural-plan authority, and those labels are not casually mixed into an
ineffective neural-only objective.

Combining the original 51 queries with the 18 D queries produces 69 admitted
queries/57,582 agent-frames. Actor/root-parent and source trace hashes remain
distinct. A fresh original-module v3 fit uses the same 213,000 trainable
parameters, 20,000 updates and replay objective; GPU optimization takes
29.60 seconds. It preserves 26/26 at N=12 and changes D/N=8–20 from 7/13 to
10/13, retaining failures D/10,18,19. This is not sufficient for promotion:
the full matched all-count audit is required, including any new regressions.
The v2 recipe remains the locked reference, not an automatically replaced
checkpoint.

## What changed

The original p5 encoder/private recovery machinery, physical Sheaf-GBP, and
analytic grasp decoder remain in place. Only the existing morphology planner
MLP and motion-mean head were trained: **213,000 parameters**, original
52-dimensional input (12 physical/geometric features, 8 causal recovery
coordinates, 32 geometry descriptors). No letter label, trajectory index,
future waypoint, or teacher query was added at deployment. The independently
developed 78/162-input recurrent planners are not used here.

The training set consists of **51 queries of states actually visited in failed
p5 development episodes**, covering A/C/E/G/I/J/L/O/Q/Y with 12 agents. All 51
have bounded-force, recursively certified executable continuations meeting the
declared handoff conditions: 3,604 control frames, 43,248 agent-frames. Another
94,260 frozen-parent agent-frames provide function-preserving replay. S/Z and
odd counts are excluded from this fine-tuning data; that is not a claim that
the parent had never encountered them.

Training used 20,000 AdamW updates on the local RTX 5090, batch 1,024,
learning rate 1e-4. Optimization took 29.74 seconds after CPU feature preparation.
The objective is private motion-plan imitation plus parent-output replay, not
an unrolled-GBP reward objective. Frozen tensors are checked bitwise at every
saved checkpoint. The first two-query pilot did not repair L despite a lower
fit loss; its canonical T trajectories remained bitwise identical for N=8–20.

Teacher admission now matches the original supervisor: intermediate targets
allow its 0.14 m position region **and** require the complete body to clear
the wall. Search uses a tighter 0.125 m region plus a body-support halfspace;
force execution independently checks the real handoff. L's stalled-state
teacher changes from an inadmissible endpoint to a valid 19-step recovery.
These constraints are training targets, not deployment route inputs.

## Separate training and interface effects

All rows use the same enlarged A–Z three-wall development course, 12 agents,
1,100-step budget and unchanged p5 neural architecture.

| Weights | Original position-only/body-clear interface | Corrected corridor completion |
| --- | ---: | ---: |
| Frozen p5 | 15/26 | 16/26 |
| Original-module DAgger v2 | 23/26 | 26/26 |

With the original interface, the trained model preserves all 15 previous
successes and repairs A/C/E/G/J/Q/Y/Z. I/L/O remain unsuccessful.

The completion correction addresses a concrete interface conflict. A rotated
body can reach the target **centre x** while its trailing part has not crossed
the wall, although the supervisor still requires that crossing. For a feasible
rightward position-only corridor, the existing geometric controller now holds
the centre beyond the geometry-derived exit plane while recentering. It does
not add a new planner, change final oriented parking, or modify explicit
C-space entry targets. Its opt-in control fixes frozen-p5 L/12 without training
(547 steps, no rejected contact). Combined with v2, it also repairs I and O.

Adding a second visible-wall record is **not** part of the selected recipe.
It repaired A in an early frozen panel but regressed other letters: frozen p5
14/26, or 17/26 with completion; the v2 combination gives 19/26. These controls
are retained rather than promoted from their five-letter successes. Likewise,
removing the body-clear handoff reintroduced B's trailing-wall collision.

## Continuous collision and matched accounting

The original four-sample motion checks missed a frame-interior penetration on
nominal canonical T/8. Simply rejecting continuously uncertified transitions
keeps that trajectory safe but stalls it. Increasing the **original local
C-space candidate screen** from 4 to 32 samples lets T/8 finish in 183 steps;
the environment still uses the separate recursive continuous checker. More
samples alone are not presented as a continuous-motion proof.

All 13 canonical T counts (8–20) now finish with that continuous guard and pass
the post-rollout continuous check. The **uniform nominal A–Z × N=8–20 panel is
also complete: 338/338 terminal successes and 338/338 continuous certificates**,
all using 32-sample planning and the continuous environment guard. This is
nominal single-slit regression, not enlarged three-wall count generalization.
The entire three-wall panel is also rerun
with this same stronger physics and candidate screening for both models:

| Metric, all 26 attempted episodes | Frozen p5 + same corrected interface | DAgger v2 + same corrected interface |
| --- | ---: | ---: |
| Completed courses | 16/26 | 26/26 |
| Independently checked continuous trajectories | 26/26 | 26/26 |
| Total control steps, including failures | 20,375 | 11,749 |
| Rejected contact attempts | 2,490 | 55 |
| GBP floating-point payload, bytes | 86,684,352 | 50,284,608 |
| Attachment/grasp setup payload, bytes | 100,672 | 100,672 |

Combined recorded payload decreases **41.94%**. These are paired development
descriptions, not a population confidence claim or a comparison against a
non-sheaf baseline. Headers and control-plane traffic are not yet counted.
The recursive checker uses float64 SAT plus a conservative motion bound, not
formal interval arithmetic. A subdivision-limit rejection is distinguished
from a measured negative-clearance witness; its V1 feedback field is only a
nonpositive rejection marker.

## Remaining gates

- Multiple fresh layouts, substantial initial headings/velocities, contacts,
  goal changes and geometry scales, crossed with all N=8–20. The default
  enlarged multi-wall count grid above is now measured, not a remaining run.
- Explicit parent-lineage accounting for what is held out. S/Z exclusion from
  this fine-tune does not establish unseen-shape status for the original p5.
- Goal-switch history coherence on fresh episodes. The selected development
  recipe preserves the original reset-at-handoff convention; its successful
  panel does not establish that discarding last-step physical feedback or
  private history is appropriate for every future skill composition.
- Goal-interface audit: in the observed stalled L state, changing target y or
  target heading/mask leaves the original 27-D observation and private plan
  identical. The analytic branch separately reads goal configuration. This
  identifies an information-interface limitation, not proof of memorization
  or proof that the complete hybrid policy cannot solve such tasks.
- Full wire accounting, confidence intervals on genuinely held-out episodes,
  reproducible checkpoint/config bundling, and a validated demo handoff.

No v3 training has been started just to optimize these now-passing development
cases. Nine additional certified queries from v2's actual I/L/O states are
archived for possible later DAgger; fresh generalization evidence comes next.

## Principal artifacts

- `output/p6_mainline_markov_dagger_v2`: checkpoint and training record.
- `output/p6_mainline_dagger_composed_region_AZ_v3`: original-policy DAgger data.
- `output/p6_mainline_markov_v2_composed_AZ12`: 23/26, original interface.
- `output/p6_trained_certified32_AZ12`: 26/26, final development recipe.
- `output/p6_parent_certified32_AZ12`: matched frozen-parent control, 16/26.
- `output/p6_mainline_markov_v2_canonical_guard32`: canonical T, N=8–20.
- `output/p6_mainline_markov_v2_nominal_guard32_AZ_all_counts`: uniform 338/338
  nominal terminal successes and continuous certificates.
- `output/p6_mainline_goal_observability/L_stalled.json`: goal counterfactual.
- `output/p6_mainline_dagger_v2_actual_queries`: nine later student queries.

Source implementation is committed as `3de6def`; artifact manifests include
checkpoint, source and trace hashes. Checkpoint SHA-256:
`44be6ea0cbd21b036b2fbcc5c18e0fb5035d79f77d92d987fca39306b742f795`.
