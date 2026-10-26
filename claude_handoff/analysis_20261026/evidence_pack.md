# Evidence pack — round 2 analysis (2026-10-26 19:40 UTC)

Purpose: after batches 3S-W … 3S-Y, find the ONE change most likely to break through toward the user's final targets on
furnished synthetic storeys (R1 grid 128 x 128 x 32): **R8 surface F1 >= .90, K0 surface F1 >= .95, variable-structure F1 >= .75**.
Primary goal F1; secondary goal compression (deployment budget chosen by the user at the end; the diagnostic ladder may exceed 512 B).
Principles the user demands: first principles + Occam's razor; pre-registered single-variable batches; two seeds (342101, 342202) must both pass.

## 1. The task and the model (read the code: `scripts/claude_v342/train_3d_v431.py`, the current trainer)

- World: HouseExpo plan extruded to a 3.2 m storey, slabs, lintels, furniture library (footprint share .30). One chart of L metres
  (L = 8–20 m per house) is rasterised to 128 x 128 x 32 voxels, so the voxel is L/128 = 6–16 cm laterally and 10 cm vertically.
  8 agents explore; each has a private partial map (unknown / free / occupied).
- Encoder: `PatchTrunk3D` (block-local: a coordinate sees only its own block's voxels) + `OwnedHead` -> per agent a code of D Gaussian
  means (tanh, 8-bit) and D learned diagonal precisions on public blocks; coverage mask.
- Fusion (Sheaf-GBP, mandatory): max-precision lattice join over 8 rounds on a radio Kruskal forest; 8+8-bit wire
  (`train_v378.kbit_wire_quantize`, precision floored on a 255-level log codebook over [1e-8, 128]); latent prior 1e-4 / (n D)
  (v431 floors it at 1e-8).
- Decoder (`Decoder3D`, lines ~199–233): reads ONLY the fused posterior (mean, log-precision/8) + public coordinate planes.
  Structure, current best recipe (32x32x1 code, W = 2):
  * code grid 32 x 32 x 1 is space-to-depth folded by 2 onto a **16 x 16 start grid** (8 input channels); `pre` = 2 convs (c0 = 128);
    trilinear upsample to 16 x 16 x 8; `lift` (1 conv); then 2 blocks: trilinear x2 + 2 convs each, to 32 x 32 x 16 and **64 x 64 x 16
    (the COMPUTE grid, SUB = 2)**; then a 1 x 1 conv to 2 x 8 channels and a parameter-free **depth_to_space(2)** to 128 x 128 x 32
    (the v401 "block readout"). No convolution ever runs at voxel resolution. ~7 conv layers in total.
  * for the 64x64x1 code (3S-Z) the fold is 4 (32 input channels) onto the same 16 x 16 start; everything else identical.
- Loss: known BCE + surface-weighted (w = 4) occupancy BCE on stream targets per hop (K0 = own map, R8 = after 8 rounds).
- Metrics (24 VAL houses, 8 per room stratum 1/2/3): surface F1 (exact voxel), tol-1 surface F1, variable-structure F1 (slabs removed),
  furniture-region F1, unknown->known FPR; fresh panel of another 24 houses. No-furniture oracle: K0 .915 / R8 .900 / var .81.

## 2. Programme so far (paired by seed vs control; full write-ups in `claude_handoff/OVERNIGHT_REPORT_20261021.md` D8-2 … D8-30)

| batch | single variable | VAL R8 surface (s1/s2) | var | verdict |
|---|---|---|---|---|
| 3S-A | base 8x8x4, D 256, W1, 16k | .636/.631 | .38/.40 | baseline (interior walls recalled ~.1) |
| 3S-W | decoder width x2 | .665/.655 | +8/+6 pp | lever |
| 3S-C | layout 16x16x1 | .653/.707 | +11/+16 pp | lever |
| D2 | W2 + 16x16x1 | .769/.779 | .643/.654 | stack |
| 3S-P | 2.5-D plan+column decoder | worse than D2 | | rejected |
| 3S-R | known-BCE boundary weighting | ambiguous | | not adopted |
| 3S-Q | coverage-derived precision | -2.1 pp | | worse |
| 3S-T | two warm-started cycles 2 x 25.5k | .821/.818 | .717/.712 | lever |
| 3S-D L1 | 32x32x1 @ 8+8 (D 1,024, 2 KB), 16k | .814/.816 | .709/.713 | lever (+4.5/+3.7) |
| 3S-D L2 | 16x16x1 @ 16+16 (1 KB) | | | not better (more bits useless) |
| 3S-D L3 | 16x16x2 @ 4+4 (512 B) | | | ambiguous (+1–2.6, furniture +5) |
| 3S-S | full 15,812-house cache, 16k | ±0.8 pp | | not better |
| **3S-X** | **L1 + two cycles (current best, `runs/v429-stack/c2-s*`)** | **.877/.877** | **.804/.805** | lever; K0 .901/.902, furniture .57 |
| 3S-Y | 3S-X + full 15,812 houses | .8805/.8806 (+0.4/+0.3) | +0.6/+0.5 | not better |
| 3S-Z | 64x64x1 @ 8+8 (D 4,096, 8 KB) vs L1 at 16k | **running, result ~21:45 UTC** | | — |

Rejected/settled: data quantity (twice), more bits per coordinate, coverage precision, plan+column decoder, generalisation
(fresh panel within 0.3 pp of VAL; TRAIN–VAL gap 2–4 pp earlier). Earlier analyst memos (round 1): `claude_handoff/analysis_20261025/`.

## 3. Where the residual is (3S-Y arm / 3S-X control, nearly identical; D8-30)

1. **One-voxel misplacement dominates the K0 gap**: K0 exact surface F1 .904, tol-1 .955 (target .95 is exact). Same signature as
   batch 3D-B (D6-4): moving to the block-readout decoder at 128^3 dropped exact F1 3–4 pp while tol-1 stayed above the 64^3 exact
   numbers.
2. **Furniture**: K0 furniture recall .72 (structure .93, slab .95); 72 % of K0 false positives lie in furniture regions. Height
   profile of K0 furniture recall: 0–0.3 m .58, 0.3–0.9 m .70, 0.9–1.5 m .77, 1.5–2.1 m .78, 2.1–3.2 m .83. Structure recall is
   also lowest near the floor (0–0.3 m .86 vs .94 above 1.5 m).
3. **House size**: per-house F1 falls with chart size L (each code coordinate covers L/32 = 0.25–0.63 m): r(L, K0) ≈ −0.58,
   r(L, R8) ≈ −0.61; partial r controlling room count and observed fraction −0.37 … −0.46; negative inside the 1-room and
   3-room strata. Houses ≤ 13 m already reach R8 .902 / K0 .92; houses ≥ 17 m R8 .857 / K0 .880.
4. **Fusion**: R8 − K0 fusion gap −2.4 pp (was −3 pp at D2). Known-head veto: ~7 % of target-known occupied voxels output unknown
   (16 % of furniture voxels).
5. Interior walls (between two observed free spaces): K0 recall .85, boundary walls .93.

Per-house diagnostics: `runs/v430-stackdata/diag-{a,control}-s*/diag.json` (`per_scene` has chart_m, rooms, K0/K8 surface, var,
furniture, mask fraction, q_mean); slice dumps `slices-*-K0.npz`; interior/veto JSONs alongside.

## 4. Constraints (hard)

- Sheaf-GBP is mandatory; the decoder reads only the fused posterior + public planes (no private side channel). Messages stay a fixed
  code of D Gaussians on public blocks.
- One variable per batch; control = an existing run; two seeds; a lever needs ΔR8 var >= +3 pp and ΔR8 surface >= +1 pp on both seeds
  (cycle batches ΔR8 surface >= +2 pp); guards FPR, slab recall, K0 var.
- GPU (RTX 5090, uk-linux) is SOLO and busy with 3S-Z until ~21:45 UTC. Analysts may run **CPU-only** probes on uk-linux
  (`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19`), writing ONLY under
  `/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/analysis_20261026/`. Never touch running processes, chains, locks,
  `runs/`, or `scripts/`. Never use the local GPU. One run lease = 7,200 s; ~0.20 s/update at W2 32x32x1 → ~25k updates per lease.
- Python on uk-linux: `/home/yuzhou/anaconda3/envs/jax-gpu/bin/python`; sealed source `--source /home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source`;
  env `HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json`.
  The diag scripts (`scripts/diag_storey_v429.py`, `diag_interior_v429.py`, `diag_veto_v429.py`) show how to load a checkpoint and run VAL.
- TEST split is sealed: never generate or evaluate it.
