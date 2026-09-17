# Matched input-interface training from the current original mainline

Four independent controls start from checkpoint
`a493af24bdf6de88d478b5c28a1b6f7f69a82aca008dfec054abb1e62e8f9f4f`
with the validated crossing/heading interfaces. The original planner depth,
physical Sheaf-GBP and analytic grasp decoder remain unchanged. No attractor,
replacement planner, public deployment or promotion is introduced.

## Fixed experiment

Every fit uses the same cache, source tensors, seed 7102, 3,000 updates,
64 whole-agent states per teacher/replay sample, learning rate 2e-5, source
replay weight 3, AdamW and the original differentiable force objective.
Only the original morphology MLP and motion head, plus the declared extra
goal columns when present, are trained. Frozen tensors are checked at every
saved checkpoint. The declared final checkpoint is evaluated; intermediate
checkpoints are not selected by test results.

The cache contains **102 historical model-induced DAgger queries / 7,488
teacher frames** and **504 current-actor successful development episodes /
163,282 replay frames**. S/Z and all odd agent counts are excluded from fitting.
These are existing DAgger labels, not fresh failures of the four new actors.
Force replay is checked against the source with maximum error 4.18e-7 N.

Source data: `output/p6_mainline_original_broad_round2_data/manifest.json`.
Replay: `output/p6_crossing_hold_{counts_all,all_states_81201}/manifest.json`.
Cache: `output/p6_current_goal_input_action_cache/cache.pt` (Linux only).

The +4 command term is an additional first-layer affine projection of relative
goal-y, masked relative-heading sine/cosine, and orientation-required flag.
It adds 1,024 parameters. The projection starts at zero, preserves the original
tensors and random-generator state, and uses the existing exact-forward bridge.
Replacing column 3 by centre-minus-goal-y is a separate semantic change and
does NOT claim zero-initialization behavior parity.

## Completed closed-loop results

| Input control | Trainable parameters | GPU training seconds | L/X, 14 profiles, N12 | Default A–Z, N12 | Steps, L/X | Steps, A–Z |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original absolute-y | 213,000 | 31.12 | 28/28 | 26/26 | 8,075 | 10,235 |
| Relative-y | 213,000 | 30.32 | 28/28 | 26/26 | 9,198 | 10,411 |
| Absolute-y + goal fields | 214,024 | 32.68 | 28/28 | 26/26 | 8,064 | 10,265 |
| Relative-y + goal fields | 214,024 | 32.80 | 28/28 | 26/26 | 9,252 | 10,410 |

All 216 rollouts pass continuous certification, with zero contact attempts and
zero penetration. These are four paired controls over 54 tasks, not 216
independent test tasks. Relative-y is slower on this aggregate; equal success
does not establish superiority. The default enlarged three-wall T succeeds
in all four controls, but its complete trace is not bitwise identical to the
source. Source T takes 858 steps; fitted controls take 857/858/857/858.
This does not replace the required canonical single-slit regression.

## The new goal input is causally used, not only present

At the final source tick of each control's L/entry-reference/N12 rollout,
reconstruct the exact observation, recorded causal hidden state and active
configuration. Verify source replay, then change only the current target.
Maximum private **effective twist coordinates 3:6** changes are:

| Control | Target-y +0.3 m | Target heading to 1 rad | Disable heading constraint |
| --- | ---: | ---: | ---: |
| Original absolute-y | 0 | 0 | 0 |
| Relative-y | 0.13184 | 0 | 0 |
| Absolute-y + goal fields | 0.00504 | 0.00597 | 0.01121 |
| Relative-y + goal fields | 0.12527 | 0.00500 | 0.00799 |

These are normalized plan-coordinate differences, not Newtons or task success
rates. They establish command sensitivity at the probed state, not correctness
for arbitrary goals. States differ slightly between fitted actors, so this is
not a matched-state comparison of response magnitude between models. The
downstream analytic branch remains active and must not be credited to learning.

## Data limitation and next gate

The teacher cache has only four distinct goal-command tuples; successful replay
has eight. Teacher goal-y spans about [-0.15, 0.17] m. **5,497/7,488 teacher
frames have all nodes under analytic authority**, so their force error cannot
be corrected by modifying the private planner alone. Raw minibatch loss mixes
this fixed error with the learnable part; more updates are not automatically
the right response.

The +goal controls now warrant the complete 364-state A–Z regression, not
promotion. Then use an independently specified wider relative-goal/entry/
contact distribution to find authentic new actor failures and acquire certified
continuations where needed. Preserve the source and matched controls, retain
unproved-feasibility failures in evaluation denominators, and do not silently
replace the successful public model. Strict radius-local obstacle sensing,
canonical-branch inputs and changing-chart memory remain separate unresolved
contract checks; this fit does not complete the full p6 objective.

## Artifacts and checks

Under `output/p6_current_input_<variant>_v1/` (`absolute`, `relative`,
`absolute_goal`, `relative_goal`) are `training.json`, `checkpoint_step3000.pt`,
the authenticated unpromoted `recipe.json`, and `goal_twist_probe.json`.
Rollout manifests and hash-authenticated traces are in the corresponding
`..._LX/` and `..._AZ12/` directories. Large artifacts remain on Linux.

Checkpoint SHA256 values, in table order:

- `97de2a079e653e67732a2fb0ebd70cddd24bed8bd6220a61a60cf65e70998423`
- `880ac1f7cf053168f25048d3112fde97d7442053863266bc3536c6accc4e4d46`
- `85e22bc2f901a249bae67105bd7c70d38989fe50fd8eae2d37db9574f009afd6`
- `e84af5f6303d272a4e9e09e0f39afebaf093328557b93d2abadb8e508bd7f6e3`

Trainer: `scripts/train_mainline_action_dagger.py`, options
`--input-column3 {absolute_y,goal_relative_y}` and optional `--goal-context`.
Recipe materializer verifies source recipe, actor and final checkpoint hashes.
**149 repository tests pass.** GPU training used only the Linux RTX 5090,
verified without other compute processes before launch; CPU handled rollouts.
