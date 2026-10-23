# Root ledger — Sheaf-GBP 64-D map fusion (Claude takeover, 2026-10-21)

## Facts verified (02:46 UTC)
- SSH to uk-linux works from the desktop sandbox with a session-specific ed25519 key (user appended it). GPU-238bdbd5… free (120 MiB), no training processes; `/` free 33.2 GB (do not write there), `/mnt/datasets` free 1.69 TB.
- Anchors SHA-verified: v310 E12 ckpt 31da2f39…, v299 control 28cbc27c…, v310 source manifest a0154cb7…, v306 bank manifest 3e450d06…. Local 70-file sealed copy verified 70/70.
- New diagnostic root: /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/ (scripts/, runs/). Nothing under old roots touched.

## Diagnostic 1 — privileged 64-D hard-8 code fit through the FROZEN v310 E12 decoder (VALIDATION, rooms 12/20, N 8/16, idx 0–3; 300 Adam steps from the actual code; BCE objective; log-variance features fixed at actual GBP values)
| size | stage | actual wall F1 | oracle-code wall F1 (hard8) | oracle (continuous) | note |
|---|---|---|---|---|---|
| 128 | K0 | 0.851 | 0.934 | 0.934 | encoder gap 8 pp; decoder ceiling gap 7 pp |
| 128 | R8 | 0.583 | 0.783 | 0.783 | fusion/encoder gap 20 pp; ceiling gap 22 pp |
| 256 | K0 | 0.533 | 0.854 | 0.854 | encoder gap 32 pp (!) |
| 256 | R8 | 0.393 | 0.439 | 0.439 | ceiling gap 56 pp: decoder family is THE limiter |
Per-cohort R8 ceilings at 128: rooms12/N8 0.885, rooms12/N16 0.831, rooms20/N8 0.785, rooms20/N16 0.630. At 256 R8: 0.32–0.54.
- 8-bit quantisation costs nothing (continuous == hard8 everywhere) → H3 (bit precision) is NOT the binding constraint at 128/256.
- Dice-objective fits on the same frozen decoder: 128/20/16 R8 0.68/0.57 (vs BCE fit 0.62/0.51); 256 R8 0.36/0.31 (no change) → the frozen decoder family, not merely the fitting objective, caps multi-room unions.
- Model hash identical before/after (read-only confirmed).

## Diagnostic 2 — decoder-family ceiling (auto-decoder, free hard-8 codes, TRAIN idx 0–63 × rooms 6/12/20 × N 8/16, 128): families cnn_bce (v299-shaped), cnn_dt, mlp_bce, mlp_dt. Running.

## Diagnostic 2 results — decoder-family ceilings (VAL oracle code fits; TRAIN idx 0–63; 4000 steps)
| family | H | K0 wall F1 | R8 wall F1 | R8 rooms20/N16 | note |
|---|---|---|---|---|---|
| cnn_bce (v299-shaped) | 128 | 0.840 | 0.805 | 0.765 | fresh decoder, BCE |
| cnn_dt | 128 | 0.873 | **0.865** | **0.856** | DT head: +6 pp union ceiling, +9 pp hardest cohort |
| mlp_bce | 128 | 0.908 | 0.767 | 0.719 | MLP lift overfits TRAIN (0.98) |
| mlp_dt | 128 | 0.873 | 0.808 | 0.781 | |
| cnn_bce | 256 | 0.707 | 0.746 | 0.703 | vs frozen v310 oracle 0.44 → v310's 256 decoder was simply under-trained |
| cnn_dt | 256 | **0.864** | **0.776** | 0.699 | K0 ceiling at 256 ≈ 128 level |
Reference: frozen v310 oracle ceilings 128: K0 .934 / R8 .783; 256: K0 .854 / R8 .439. Actual v310: 128 .851/.583; 256 .533/.393.

## Diagnostic 2b — local K0 refinement on frozen v310 (30 steps, own observation only, rooms12/N8/idx0, 128)
K0 0.841→0.929 but R1 0.717→0.607, R8 0.756→0.681: individually-optimal dense codes fuse worse → dense averaging ≠ union (H2 evidence).

## Math tests for v342 (sealed v295 primitives): ALL PASS (complementary owned sources fuse to the union within one hard8 step through the profile-4 wire; dense contrast mixes by 0.913).
## Preflight v342 owned on 5090: 592 TRAIN scenes, ~0.7–1.0 s/update under shared GPU, VAL(48) ≈ 40 s.

## v342 matched pair, H128 (VALIDATION 48 scenes, equal-weight 6 strata; sealed metrics; raw predictions)
Batch 1 (8 ep, 4736 upd): owned K0 .798 / R8 .566 (acc .889, fpr .081, own@R8 .535); dense K0 .747 / R8 .523 (acc .840, fpr .099). Fresh panel (idx 8–15) within 1 pp. Local refinement (50 steps): K0 owned .856, dense .863; R8 unchanged (±0.4 pp) — no longer harmful (source-anchored readout).
Batch 2 (36 ep, lease 7200 s) at epoch 28 (16.6k upd): **owned K0 .904 / R8 .838 (acc .942, fpr .046, own@R8 .828)**; dense K0 .843 / R8 .696 (acc .896, fpr .065, own@R8 .708). Anchor v310 E12: K0 .831 / R8 .626 (acc .894, fpr .065, own@R8 .599).
Owned ceiling (free codes fitted on own observation, exact averaging): K0 .871 / R8 .499 — the end-to-end owned encoder learned averageable block codes far beyond the non-co-adapted ceiling.
