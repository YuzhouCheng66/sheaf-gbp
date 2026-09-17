# Original mainline: goal contract and repeated invocation

The original architecture remains the mainline. Three-wall failures do not
establish route memorization or architectural inadequacy. The B waypoint
repair and the later same-weight crossing controls are evidence for testing
interfaces before replacing a network. No new planner, attractor, checkpoint,
or public demo is promoted by this audit.

Candidate: `configs/p6_force_crossing_hold_control.json`, checkpoint SHA256
`a493af24bdf6de88d478b5c28a1b6f7f69a82aca008dfec054abb1e62e8f9f4f`.
This is a hybrid learned/analytic local policy with physical Sheaf-GBP and
analytic grasp allocation, not an entirely learned global planner.

## Repeated invocation: completed same-weight regression

The existing sequence panel is replayed with only `hold_feasible_crossing`
changed. S/Z, N=8/11/16/19/20, three entry/motion profiles, seed 82201, and
1/2/3/5 walls give 120 cases. Each length succeeds **30/30**; total **120/120**,
zero contact attempts, zero penetration, all continuous certificates passed.
Intermediate targets are position-only; the final target includes orientation.

The comparator authenticates checkpoint, complete case set, physical courses,
options and trace hashes. **112/120 complete trace arrays are bitwise unchanged**;
there are no success regressions. Total steps change from 25,687 to 25,695:
this is not universal efficiency dominance. This is a reused single-layout
panel of parallel walls, not 120 independent layouts, arbitrary wall normals,
or a maze-navigation demonstration.

- Source: `output/p6_force_heading_sequence_validation_82201/manifest.json`.
- Candidate: `output/p6_crossing_hold_sequence_validation_82201/manifest.json`.
- Candidate manifest SHA256:
  `8cd3b9092f182d268b173fa58588e6dfbd17959be13e836694390aad5088c481`.
- Paired arrays and metrics: `output/p6_crossing_interface_comparison/sequence.json`.

## What actually receives the goal?

`audit_mainline_goal_observability.py` now reconstructs all current composition
options, accepts an explicit source tick, authenticates the source trace, and
verifies reference observation, private plan, executed plan and force before
performing command counterfactuals. It also records the actual 52-D private
features, rather than inferring their semantics from the final force.

Six authentic states are used: first and last control ticks of A/L/T, N=12,
`combined_entry_reference_s81201`. State, geometry and recorded causal memory
are held fixed. Target-y is changed by +/-0.3 m, target heading is changed to
1 rad, or the orientation-required flag is toggled. All six reference replays
pass. In all six probes, the **52-D private features and private 8-D plan remain
bitwise identical** under these command changes.

The whole controller does not ignore the goal. At the final-state probes,
maximum force differences across these counterfactual commands are 4.0030 N
(A), 3.9635 N (L), and 4.0080 N (T). These responses originate downstream of
the private plan, in the existing goal-dependent analytic branch. This is
one-step command sensitivity, not a closed-loop success test for the modified
goals, and does not prove their feasibility.

Artifacts: `output/p6_crossing_hold_goal_contract/{A_0,A_-1,L_0,L_final,T_0,T_-1}.json`.

## Input and memory limitations that remain explicit

| Component | Current implementation | Claim boundary |
| --- | --- | --- |
| Private morphology planner | Canonical physical features 12 + recovery state 8 + geometry signature 32; original 52-D MLP | Includes centre coordinates; no explicit goal-y/heading-mask in this branch. Do not claim a fully relative-goal learned policy. |
| Target handling | Caller supplies current goal and active slit chart; analytic branch receives goal x/y/theta and orientation-required | No future waypoint list is passed to the private MLP, but the caller selects the active portal. Autonomous portal selection is untested. |
| Feedback | Per-handle previous force and rejected motion; shared payload collision bit | The collision bit is not a localized contact sensor. Mechanical feedback is not an extra GBP packet. |
| Handoff | Physical pose/velocity continue; current recipe resets private hidden state and one-step feedback at a new target | Do not claim memory is transported continuously across chart/goal changes. Earlier carry controls remain separate. |
| Wall observation | Legacy horizontal wall-distance visibility plus slit descriptors | This is not a generic Euclidean-radius sensor for arbitrary finite obstacles. |
| Communication | Genuine handle-distance radius edges; original GBP/grasp | Current tree construction requires connectivity and errors on disconnected graphs. No disconnected-graph guarantee. |

Sources are `models.py` (`local_se2_recovery_canonical_features`,
`local_se2_markov_shape_inputs_from_planner_inputs`), `env.py` (`radius_edges`
and wall observations), `single_slit_contact_observation.py`, and
`multi_waypoint_demo.py` (active-goal selection and handoff).

## Next experiment, without replacing the mainline

1. Keep this checkpoint and validated recipe frozen as the reference.
2. Treat explicit relative-goal/mask input as a separately versioned interface
   control using the same local planner, Sheaf-GBP and grasp decoder. Earlier
   goal-context/relative-y fits had regressions; do not silently adopt them.
3. Use diverse feasible current-goal/entry/contact states and actual actor
   failures for DAgger. Supervise executed force where the learned branch has
   authority; retain successful-source replay and report teacher admission
   failures. Goal-responsive analytic overrides must not be misreported as
   learned-planner improvement.
4. Require matched old-task and new-command closed-loop validation before
   promotion. Generic obstacle sensing and changing-chart memory are separate
   controls, not bundled unexplained architecture changes.

This audit and the sequence replay use Linux CPU only. The repository suite
passes **145 tests**. The full goal is still active: successful existing grids
do not discharge the stricter local-input, generic-goal and demo requirements.
