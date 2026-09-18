# Local demo deployment — 2026-09-18

The user explicitly requested replacing the old demo with the best validated
current model. Port **7862** now serves the p6 original-mainline candidate from
this worktree, not frozen p5 from `sheaf-gbp-public-clean`. The old p5 source
and checkpoint remain intact for rollback. No GitHub release is implied.

- Checkpoint: `output/p6_mainline_force_loss_v1/checkpoint_step3000.pt`.
- SHA256: `a493af24bdf6de88d478b5c28a1b6f7f69a82aca008dfec054abb1e62e8f9f4f`.
- Recipe: `configs/p6_force_crossing_hold_control.json`.
- Recipe SHA256: `024629a9391916494397be4124c039ab6bbf124ede1abb3bb864168e2c77fcab`.

This version is selected for its completed A–Z × N8–20, old-task, state,
contact and sequence regressions. The more recently fitted goal-input controls
remain experimental: passing their 364-state panel does not replace the still
required full-count/canonical/contact validations, nor establish better efficiency.

## Deployment cannot silently drop the tested interfaces

`python -m piano_movers.multi_waypoint_demo --recipe <recipe> --device cpu
--host 127.0.0.1 --port 7862` authenticates checkpoint bytes and requires every
composition option to be explicitly present. It does not fall back to legacy
defaults. `/api/model` reports the served checkpoint/configuration and recipe
hash; the browser title and dock tooltip expose the loaded checkpoint identity.
The original `--checkpoint` invocation remains available as explicit legacy mode.

## Actual HTTP verification, not only an offline score

Before cutover, a temporary server on 7863 streamed all **26 letters, N12**.
`scripts/verify_demo_http.py` checked its metadata, every scene/header and final
outcome against `output/p6_crossing_hold_counts_all/manifest.json`. Both `state`
and `next_state` at **every control step** were bitwise identical to the
hash-authenticated, continuously certified reference NPZ trajectories.
All 26 succeeded, with zero contact attempts and zero penetration.

After replacing the old 7862 process, A/B/L were repeated through the actual
live HTTP endpoint: **3/3**, again bitwise-identical states and exact header/
outcome agreement. The Windows endpoint `/api/model` was independently fetched
through its existing loopback SSH tunnel and returned the expected hash and all
options. The in-app browser displayed checkpoint prefix `a493af24`.

Reports (Linux and small local JSON copies):

- `output/p6_demo_deployment_20260918/http_staging_AZ12.json`.
- `output/p6_demo_deployment_20260918/http_live_ABL12.json`.

The HTTP verification covers all shapes at N12, not a fresh 338-case web audit.
The larger N8–20 result belongs to the referenced engine regression. No
arbitrary maze or goal-distribution claim follows from this deployment.

**152 repository tests pass**, including recipe hash, exact-option loading and
missing-option rejection. Serving/inference uses CPU; GPU experiments are not
started or changed by the demo migration.
