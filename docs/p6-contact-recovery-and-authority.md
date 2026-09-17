# Contact recovery: an action-authority failure, not a failed GBP solver

The original neural architecture remains the mainline. None of the contact
residual controls below is promoted; public p5 and the validated demo are
unchanged. Base weights:
`a493af24bdf6de88d478b5c28a1b6f7f69a82aca008dfec054abb1e62e8f9f4f`.

## Locked environment stressors

All cases use development seed 81201, the combined three-wall course and
`entry_reference`; A/B/L/T/X/Z at N=8,12,20. No case is outcome-filtered.
The policy, individual-force limits, communication graph and continuously
guarded contact dynamics are unchanged. An explicitly separate environment
stressor applies one velocity impulse, after the current policy action has
been computed. The direction is the negative finite-difference gradient of
exact-SAT clearance. Its magnitude, trigger and tie rule are declared before
evaluation, not chosen by success. This is not a new nominal benchmark.

Two distinct panels:

| Panel | First clearance at/below | Linear velocity increment | Angular increment |
| --- | ---: | ---: | ---: |
| Near-contact | 0.030 m | 0.65 m/s | 0 |
| Earlier displacement + rotation | 0.120 m | 0.65 m/s | 0.80 rad/s |

The earlier impulse actually moves and rotates A before a later collision.
It is not just a fabricated collision bit or a reset at an unchanged pose.
Every agent receives the ordinary physical aftereffect, not the environment's
clearance gradient, impulse direction or future action. The all-body contact
bit and per-handle rejected motion follow the existing benchmark sensor
contract; this experiment does not make that contract more spatially local.

## Original model outcomes

Near-contact panel (54 runs):

| Interface | Success | Actual contact episodes | Successful contact recoveries |
| --- | ---: | ---: | ---: |
| No impulse, original resets | 18/18 | 0 | 0 |
| Impulse, original resets | 18/18 | 17 | 17/17 |
| Impulse, retain physical feedback at handoff | 18/18 | 17 | 17/17 |

B/N=12 is perturbed but does not collide; it is **not** counted as a contact
recovery. The other cases move at least 2 cm again within 1–3 steps after
the rejected move. Retaining physical feedback versus resetting it produces
bitwise-identical complete force/state arrays in all 18 matched cases. This
does not prove that arbitrary learned hidden state can be transported across
changing coordinate frames.

The earlier-displacement panel is harder (36 runs). Both the original reset
and the independent preserve-all-memory control succeed **15/18**. Each has
five contact episodes: two L cases recover, while A at N=8,12,20 fails with
991, 945 and 961 attempted contacts. All 36 trajectories remain continuously
certified and penetration-free. Keeping all memory does not repair A.

These are correlated development cases from one layout draw. They are not
independent maze trials, and neither 17/17 nor 15/18 establishes a population
success bound. Perturbation recovery is reported separately from nominal
performance. No existing 338-case result is relabelled or discarded.

## Why ordinary additional planner training cannot repair these states

At the first post-contact A states and final deadlocks, **all nodes use the
analytic geometry branch**. Its overwrite leaves zero executed-force
derivative with respect to the original private learned plan. This authority
property is directly covered by the differentiable actuation test. The
failure is therefore not evidence that the original MLP is remembering a
route or that GBP cannot coordinate the required forces.

The teacher was queried on authenticated student observations/hidden states,
including actual post-collision states. **11/11** first-round queries have
certified executable continuations satisfying the same unchanged goal disk
and full-body handoff. Queries include initial/pre-contact states as controls;
do not call all 11 distinct deadlocks. No S/Z or odd-count teacher is admitted.

## Independent contact-authority learning controls — not promoted

Keep the original encoder and 8-D causal recovery GRU frozen. For the
52-D morphology input `(geometry12, recovery8, shape32)`, form

`d = E(geometry, recovery, shape) - E(geometry, 0, shape)`.

A trained bias-free readout produces a bounded correction to the existing
executed body twist **after** the geometry overwrite, before the unchanged
physical GBP and grasp decoder. The original geometry mask and precision
are retained. This differs from the older final-plan override, which clears
that mask. Tests check full-forward versus differentiable-replay force and
message parity, live gradients at geometry-overridden nodes, and exact
zero-residual parity.

When recovery memory is zero, `d` and the correction are identically zero
for any learned weights. There is no confidence threshold or explicit mode
selector. The control does add local computation, learned parameters and
post-geometry authority; it must not be described as unchanged architecture.

Results on the three harder A cases:

| Independent control | Trainable weights | Success | Contact attempts N=8 / 12 / 20 |
| --- | ---: | ---: | --- |
| Original model | 0 new | 0/3 | 991 / 945 / 961 |
| Linear v1, 2,000 updates | 768 | 0/3 | 1 / 1 / 945 |
| Linear v2, real second-round DAgger, 3,000 updates | 768 | 0/3 | 1 / 1 / 1 |
| Linear v3, matched continuation, 3,000 updates | 768 | 0/3 | 991 / 1 / 961 |
| Nonlinear readout, matched continuation, 3,000 updates | 17,344 | 0/3 | 990 / 945 / 961 |

v1/v2 demonstrate changed recovery behavior but **not increased success**:
the policy can retreat and then settle at an incorrect equilibrium. A larger
readout is not an established remedy. Both matched continuations start at v2,
use the same data, seed, batch size, optimizer rate and update count. The
nonlinear branch is exactly zero at initialization, preserving the starting
function; its extra parameters and compute are explicit, not parameter-matched.

Second-round teacher queries are from v1's actual resulting trajectories,
not a repetition of a fixed teacher route: **12/12** are admitted. The union
contains 23 authenticated queries from two actor identities (same base
checkpoint, different sidecar), 1,571 teacher frames, including 579 with
active contact memory. Initial/pre-contact prefixes overlap; these are not
23 independent layouts. Successful source-contact replay retains 6,133 frames,
including 1,016 active-memory frames. Only active-memory frames are fitted;
the excluded zero-memory frames have structurally zero correction, and all
counts remain visible. Exact no-contact preservation does not imply preservation
of previously successful contact behavior; that still requires rollout tests.

All base parameters are asserted bitwise unchanged at each saved sidecar.
Training used only uk-linux RTX 5090. Update times: v1 16.37 s, v2 24.23 s,
nonlinear 24.99 s, matched linear 24.31 s. Preparation and CPU evaluation are
separate. No GPU occupation process or other user's job was modified.

## Check whether the execution layer can realize the labels

On all 579 teacher contact-memory frames, insert the recorded teacher next
twist after geometry and use the **same** GBP/decoder. This training-only
oracle is not a deployed model or a closed-loop success claim.

| Prediction | Force MAE | Mean net-force error | Mean torque error |
| --- | ---: | ---: | ---: |
| Original model | 0.78683 N | 14.5069 N | 1.6290 Nm |
| Teacher-twist oracle | 0.001887 N | 0.003505 N | 0.01152 Nm |
| Linear v2 | 0.58712 N | 8.6939 N | 2.4485 Nm |
| Matched linear v3 | 0.44843 N | 8.6100 N | 0.5160 Nm |
| Nonlinear readout | 0.49035 N | 7.1743 N | 1.9060 Nm |

The oracle maximum individual-force error is 0.5363 N: do not claim every
teacher force is exactly represented. The small average/net errors support
investigating recovery prediction/optimization before replacing GBP. Lower
offline force error, fewer collisions and more capacity did not imply task
completion. Next diagnostics should test whether the recovery representation
distinguishes retreat from re-entry and whether the clipped-force objective
supplies the needed gradients, rather than assuming either is already the cause.

## Artifacts and audit scope

All paths are relative to the experimental repository. JSON reports and final
small sidecars are copied to Windows; bulk NPZ traces and caches stay on Linux.

- `output/p6_contact_recovery_81201/manifest.json`: near-contact full panel;
  SHA256 `35711484973829a5bc37ff0065cc7d4fd2f13727c3a9fa5e093cf19819de18dc`.
- `output/p6_early_contact_recovery_81201/manifest.json`: earlier-impulse panel;
  SHA256 `63cf7d7ea5fa1de61bbc55b17a59c0f8e48ac43911d00cacd5603f54167a82c6`.
- `output/p6_early_contact_dagger_81201/manifest.json`: first 11 teacher queries;
  SHA256 `eec7853844095128925d49ce2fd3cb8c1cf7096261335297531ed6052fa83031`.
- `output/p6_contact_residual_round2_data/manifest.json`: 23-query union;
  SHA256 `d60f1dacda1b9a979c7e2d69d6dbb44965f967ebaef225b494abdbfa35cf0857`.
- `output/p6_contact_residual_{v1,v2,linear_matched_v3}/` and
  `output/p6_contact_residual_nonlinear_v1/`: training manifests and sidecars.
- Corresponding `*_early_A/manifest.json`: all three evaluated outcomes.
- `output/p6_contact_teacher_actuation/report.json`: teacher execution audit;
  SHA256 `215d86a5249c504168fb59520c8b527ffd2ebff9a6c669d1bf61062f65984948`.

The near-contact process was interrupted after 25 recorded cases. The runner
authenticated the completed prefix, source/checkpoint/catalog hashes and each
trace before continuing; the manifest records the resume. No completed case
was rerun or silently lost. Total pre-resume runtime was unavailable and is
left null, not reconstructed from guesses. There is no claim that this work
completes the broader p6 generalization goal.
