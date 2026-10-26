# Analyst A — decoder and localisation (round 2, 2026-10-26, ~20:00 UTC)

**Short answer.** The one-voxel error is not caused by (a) the block readout or (b) the 16x16 start grid; both were ruled out by measurement.
The one grid structure it follows is the **4-voxel code column**, and that effect sits almost entirely in **furniture**. There, the code does
carry a weak, monotone in-column position signal, but the decoder does not use it. So (d) is the main cause, with a real (c) share.
Two larger parts of the gap lie outside (a)–(d): the known head vetoes 71 % of the false-negative (FN) surface voxels, and most furniture
error does not depend on phase at all. Perfect in-column placement would add only about **+1.7 pp** K0 exact (estimate).

## 0. What was run (CPU only, `nice 19`; outputs in `claude-diag-20261021/analysis_20261026/A/`; scripts saved next to this memo)

- Checkpoints: `runs/v429-stack/c2-s342101|s342202/final.pt`.
- Data: a **VAL subset** of 9 houses (indices 0–2 of each stratum). On this subset K0 exact is .888/.890 and tol-1 .947/.947 (full VAL: .901/.902).
- Probes: `probe_A_errors` (error budget), `probe_A_shift` (geometry-controlled phase test), `probe_A_readout` (heads retrained on frozen
  features), `probe_A_veto` / `probe_A_rule` (known-head veto), `probe_A_sweep` (synthetic in-column sweep).

## 1. Evidence

**1.1 Error budget (K0, s1 / s2).**
- Counts: FN 60.8k / 60.3k and FP 67.5k / 66.6k, on 583k surface voxels. Tol-1 forgives 55 % of FN and 67–68 % of FP.
- **71 % of FN are predicted *unknown*** (known-head veto); on 92 % of those the occupancy head is right.
- FN by class: furniture .56, wall .22, slab .22.
- Error type, over xy boundary pairs: thickening into free space 5–12 %, holes 12–18 %, coherent shift 1–4 % (into free space) and 1–2 % (into solid).
  There is **no systematic one-voxel shift**; the errors are dilation into free space plus vetoed holes.

**1.2 Grid-phase test with geometry held fixed** (`probe_A_shift`). The whole world is translated by s voxels in x and y, and the *same* physical boundary
pairs are re-scored. Only their grid phase changes. Values are Δ pair-error rate in pp (xy pairs, K0, s1 / s2).

| move of the same pairs | tests | all | walls | furniture |
|---|---|---|---|---|
| start-cell edge ↔ code edge inside a start cell (s=4) | (b) | +0.2,+0.7 / −0.2,−0.2 | −0.2,+0.6 / 0.0,+0.1 | +0.8,+0.9 / −0.5,−1.1 |
| shift-noise control (s=4, non-edge pairs) | — | −0.04,+0.07 / +0.22,+0.09 | | |
| mid-column → code-column edge (s=2, parity kept) | (c)/(d) | **−7.1 / −6.9** | −2.5 / −2.0 | **−14.7 / −15.1** |
| code edge → mid-column (s=2) | (c)/(d) | **+8.0 / +8.6** | +2.6 / +4.4 | **+19.1 / +17.1** |
| mid-column → odd plane inside a compute cell (s=1) | (a) | +1.8 / +0.9 | +0.2 / −0.1 | +4.5 / +2.6 |

Without the control, start-cell edges looked worse than code edges (.278 vs .198). That difference came from geometry, not the grid.

**1.3 Readout heads on frozen h** (64x64x16, 72 ch; s1; 8 TRAIN scenes, 240 steps). Every head is residual on the original 1x1 readout and zero-initialised.

| head (VAL subset, K0) | params | exact | tol-1 |
|---|---|---|---|
| original block readout | — | .8884 | .9468 |
| 1x1 fine-tuned (control) | 1,168 | .8909 | .9478 |
| + 3x3x3 conv on the compute grid | 31,120 | .8803 | .9417 |
| + voxel-resolution conv (d2s → 3³ conv at 128x128x32) | 8,178 | .8888 | .9461 |

**1.4 Synthetic in-column sweep** (`probe_A_sweep`). One agent; a boundary plane at 4 positions inside one code column.

| | xa % 4 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| boundary wall | column μ, hard-8 levels (s1) | −107 | −75 | −41 | −5 |
| | exact, s1 / s2 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |
| furniture box face | column μ (s1) | +19 | +25 | +39 | +49 |
| | exact, s1 / s2 | **1.00/1.00** | .08/.00 | .29/.00 | .17/.42 |
| | face offset, voxels | 0/0 | −.9/−1.0 | −.7/−1.0 | −.8/−.5 |

The readout places walls exactly at odd positions, so it can split a compute cell. Both objects get a monotone code, but furniture gets only
6–15 levels per voxel against 34–42 for walls. Off a column edge, the decoder draws the furniture face one voxel into free space.

**1.5 Known-head veto.**
- Relaxing the veto (`known_logit ≥ −1`): K0 exact +1.65 / +1.67 pp, but R8 FPR rises +1.2 pp, which breaks the guard.
- A parameter-free **surface-adjacency rule** keeps a voxel occupied when the occupancy head says so and it is 6-adjacent to a voxel predicted free.
  The basis: an observed surface touches observed free space by definition.
  K0 exact +0.78 / +0.83 pp, R8 exact +0.42 / +0.39 pp. FPR rises only +0.10 / +0.11 pp (K0) and +0.19 pp (R8).

## 2. Diagnosis

| cause | binds? | confidence | evidence |
|---|---|---|---|
| **(a)** block readout | no | ≈ 85 % | walls placed exactly at every parity; parity effect ≤ 2 pp (≈ 0 for walls); frozen-h heads, voxel-resolution conv included, gain nothing |
| **(b)** start grid / fold 2 | no | ≈ 90 % | swapping start edges and code edges changes nothing (\|Δ\| ≤ 1.1 pp, the shift-noise level) |
| **(d)** code | yes, main | ≈ 60 % | the error tracks code-column edges (7–9 pp; furniture 15–19 pp); the code spends little scalar range on furniture position, mixed with shape; one Gaussian per 4×4×32 column cannot hold both; consistent with L1 lever / L2 not a lever |
| **(c)** decoder depth | partly | ≈ 30 % | the weak furniture signal goes unused; I cannot exclude that a deeper code-resolution decoder would read it |

**The ceiling on the gain.** If every boundary had the code-edge error rate, K0 exact would rise by only ≈ +1.7 pp (estimate). Furniture pairs
exactly on code edges are still 47–54 % wrong. Reaching K0 .95 needs more furniture content in the code, not a sharper decoder.

## 3. Recommended batch: one decoder block at code resolution (tests (c) against (d))

- **Change (in `train_3d_v431.py`).**
  - Add `--code-depth {0,1}`; default 0 is bit-for-bit.
  - In `Decoder3D.__init__`, *after* `self.output` (l.220), create `self.extra = Sequential(Conv3d(ch+4, ch, 3, p=1), SiLU(), Conv3d(ch, ch, 3, p=1))`, with ch = 96 at the 32x32x16 stage. Creating it last means every old tensor is drawn identically.
  - Zero-initialise the last conv. In `forward` (loop at l.229), after the 32x32x16 block: `h = h + self.extra(cat(h, public3))`.
  - Then the initial function equals the control's. A test asserts that all old `initial.pt` tensors are equal.
- **Cost.** +508,224 parameters (decoder 1.85 M → 2.36 M) and +8.3 GMAC per decode (+25 %). Estimated (round-1 time model): **≈ 0.235 s/update**,
  so 16k updates ≈ 3,760 s and fit one 7,200 s lease; **≈ 9.5 GB** GPU memory.
- **Control.** L1, `runs/v429-ladder/l1-s*` (32x32x1 @ 8+8, d3 W2, 16k), paired by seed; the same control as 3S-Z.
- **Pre-registered prediction (from my diagnosis): not better.**
  - ΔR8 surface in [−0.5, +1.0] pp and ΔR8 var in [−0.5, +1.5] pp on both seeds; guards pass.
  - Mechanism, measured on CPU: the off-edge furniture sweep stays exact < .30 (control ≈ .16); the `probe_A_shift` code-edge penalty stays within ±2 pp of the control.
- **What would falsify (d).** All three on both seeds:
  - the lever rule passes (ΔR8 var ≥ +3 **and** ΔR8 surface ≥ +1 pp);
  - the off-edge sweep exact reaches ≥ .50;
  - the code-edge penalty drops by ≥ 3 pp.

  Then (c) binds, and decoder depth becomes the next axis.
- **Honest ranking.** For expected F1, this batch ranks below code-side batches. Do not spend a lease on (a) (voxel-resolution readout) or (b) (no-fold start).
- **Free side finding (no lease).** Pre-register the surface-adjacency rule as a flag in `predict()` (l.377), default = the current veto.
  Score it on the 3S-X checkpoints using VAL indices 3–7, which I did not touch, plus the fresh panel. Guard: FPR ≤ control + 0.3 pp.

## 4. Dependence on 3S-Z

- **3S-Z is a lever.** Coordinates (d) are confirmed as the lever, so stack 64x64x1 into two cycles first.
  Before any decoder batch, re-run `probe_A_shift` (s = 1, 2, 4, 8) and `probe_A_sweep` on the 3S-Z checkpoints (CPU, ~10 min). This is needed for two reasons:
  - at 64x64 the code column equals the compute cell;
  - the fold-4 start, which puts 4×4 columns into one start cell, was not covered by my s = 4 test.

  The depth block would then go at 64x64x16, with the 3S-Z arm as control.
- **3S-Z is not better.** Either the decoder cannot absorb 4,096 coordinates (two convs at code resolution, fold-4 start), or DOF per coordinate is not the limit.
  The sweep on 3S-Z checkpoints decides between these:
  - monotone furniture codes still ignored by the decoder → (c) gates the whole ladder, and this batch becomes top priority;
  - flat codes → the encoder's code shape is the limit.
