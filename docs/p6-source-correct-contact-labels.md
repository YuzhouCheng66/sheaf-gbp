# Source-correct contact supervision: development diagnosis

The original model remains the mainline. These experiments do not justify
replacing it, and do not change the public demo. In particular, an interface
repair that solves a waypoint case without changing weights is evidence
against inferring architectural failure or route memorization from that case.

## What was tested

All results below concern the deliberately perturbed A/8, A/12 and A/20
development cases from the contact-recovery audit, not the nominal A–Z
benchmark. The original policy checkpoint is
`a493af24bdf6de88d478b5c28a1b6f7f69a82aca008dfec054abb1e62e8f9f4f`.
The physical Sheaf-GBP and grasp decoder are unchanged. The contact heads
are independent, unpromoted controls, not the original architecture.

On the 579 active-contact-memory training frames, the tested linear control
had nonzero force-loss gradients in every frame; none pointed away from the
teacher-twist correction, and no node was saturated. This rules out that
particular gradient/saturation explanation **on these training frames**,
not on every possible rollout state. In-sample ridge fits favored the existing
256-dimensional encoding difference over the tested simpler and bilinear
features. Neither result proves that the representation generalizes.

Closed-form teacher-twist fitting of the same 768-parameter linear head also
failed all three perturbed A cases. More capacity is therefore not the next
default assumption.

## Can the original model resume after a short teacher recovery?

For five original failed-state queries, tested teacher prefixes of 4–16 steps
were sufficient for the unchanged original policy to finish the **current local
goal**, without further contacts. The complete teacher sequences were about
50 steps. These were the earliest *tested* successful prefixes, not a proof
of continuous or globally minimal recovery duration.

We then evaluated every active-contact-memory frame in the two-round DAgger
union: 579 frames across 11 teacher sequences, from 23 total queries. Each
continuation starts from the actual teacher state, observation and causal
private memory; the policy gets no future teacher states or waypoint list.
It has 512 steps to reach the current goal with the existing body-clearance
criterion. A failed continuation means timeout under this test, not geometric
infeasibility. Successful paths also pass the continuous collision validator.

- 528 frames: the original policy finishes the local goal.
- 51 frames: it does not finish within the horizon.
- All 30 overlapping sampled-prefix traces reproduce bit-for-bit.

These are correlated states along training sequences, not 579 independent
test episodes and not a deployment success rate. No whole-course success
is implied. There were no success-to-failure reversals along these particular
sampled sequences; this is not an assumed monotonic basin property.

## Training-only relabelling and matched result

The original NPZ arrays, including teacher forces and states, are retained
unchanged as trajectory witnesses. Two separate arrays hold the alternative
one-step training forces and a preservation mask. Where the certified original
continuation succeeds, use its first action and zero residual target; elsewhere
retain the teacher target. The alternative labels are **not** presented as an
executable trajectory matching the retained teacher states. There is no runtime
teacher lookup, confidence gate, or teacher-to-student switching rule.

The same linear control was trained for 3,000 steps from linear v2 with the same
seed, optimizer, batch size, learning rate and replay weighting as matched
linear v3. Only the teacher labels changed. Training took 24.56 seconds on the
uk-linux RTX 5090; original policy parameters were checked unchanged.

| Independent control | A/8 | A/12 | A/20 | Contact attempts |
| --- | --- | --- | --- | --- |
| Matched linear v3, full teacher labels | fail | fail | fail | 991 / 1 / 961 |
| Source-correct labels | fail | fail | fail | 1 / 1 / 961 |

All three new paths pass continuous collision validation with zero penetration.
A/8 and A/12 retreat but settle before the first local goal; A/20 remains in
contact. Therefore the relabelling idea is **not yet a successful recovery
method**, and neither control is promoted. Different labels mean their raw
training losses should not be compared as if they shared the same targets.

This establishes a useful data boundary, not a solution: only 51/579 contact
frames need teacher intervention under the source counterfactual. Next work
should check minority recovery-frame fitting and actual newly induced states,
with source-success preservation and closed-loop tests, rather than replacing
the original planner or assuming deeper networks solve the problem.

## Reproducibility

Scripts:

- `scripts/diagnose_contact_learning.py`
- `scripts/fit_contact_twist_ridge.py`
- `scripts/audit_teacher_recovery_handoff.py`
- `scripts/build_source_correct_contact_dataset.py`
- `scripts/train_contact_residual_control.py`

Key artifact SHA256 values:

- `output/p6_source_correct_contact_audit/manifest.json`:
  `d4e2ae8983a06164e5c032d44510456aade064c9f040c24dafb725f20c9b03d2`
- `output/p6_source_correct_contact_data/manifest.json`:
  `eb11277c091b75e2df3d1b9f33dc4be32d76255cb9f4607e107b222fe29cefe1`
- `output/p6_contact_source_correct_v1/control_step3000.pt`:
  `b2fb65b2096f708f12154c77988ec04151f4de69ea116775e25bcdc7fd352596`
- `output/p6_contact_source_correct_v1_early_A/manifest.json`:
  `d345da8bd029b310ddd353effdb37a84213cf98aa26d41d18f03e58295d70140`

Small reports and the final sidecar are mirrored to Windows; bulk traces and
caches stay on Linux. The full test suite passes: **134 tests**. No public
checkpoint, baseline, GPU job belonging to another user, or demo was replaced.
