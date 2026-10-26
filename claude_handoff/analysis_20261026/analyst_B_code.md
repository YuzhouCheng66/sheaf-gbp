# Analyst B, round 2: code representation and budget

Written 2026-10-26, 19:48–20:10 UTC. CPU-only probes on uk-linux, VAL panel of 24 houses only, TEST untouched. Scripts: `probe_B_*.py` next to this memo. Outputs: `/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/analysis_20261026/B/`. Probed model: 3S-X `runs/v429-stack/c2-s342101` (32x32x1 @ 8+8). Correlations over 24 houses are indicative only (both seeds share the houses).

## 1. Measured evidence

**1.1 Information in the K0 target** (`probe_B_info`, 192 own maps). The causal 6-neighbour context model is pooled over all maps, so its estimate is optimistic. lzma gives an upper bound.

| | 16x16 | 32x32 | 64x64 |
|---|---|---|---|
| evidenced coordinates per agent | 61 | 212 | 772 |
| context bits per evidenced coordinate: all / ≤13 m / ≥17 m | 153 / 149 / 150 | **45** / 42 / 47 | **12.6** / 11.7 / 13.3 |
| lzma surface bits per evidenced coordinate | n/a | 35 | 9.8 |

- **One own map:** 9.5 kbit (context model) or 17.2 kbit (lzma; known mask 11.4, surface bitmap 7.0, of which 4.9 without furniture).
- **Furniture:** 16 % of surface voxels, but 30 % of surface bits.
- **Height:** the band below 1.5 m holds 65 % of the bits (0–0.3 m alone: 24 %), and 81 % of furniture surfaces sit below 1.5 m.

**1.2 What the code carries** (`probe_B_code` modes stats, bits and wire).
- **Usage per agent:** 212 of 1,024 coordinates carry evidence.
  - The mean uses 246 levels (H 7.25 bit).
  - The precision uses 117 of 255 codes (H 6.38 bit). Its codebook spans [1e-8, 128], but the head emits values in [1e-4, 4].
- **What it describes:** μ tracks the free fraction (Spearman .90). Quadratic R² from (μ, log q) is .87 for free fraction, .46 for furniture-occupied and .23 for occupancy in the low band.
- **Post-hoc bit sweep:** source and wire are re-quantised without retraining (12 houses). "Matched" means the precision codebook is restricted to [1e-4, 4] with nearest rounding.

| bits (mean + precision) | B/msg | K0 exact | K0 tol-1 | R8 exact | R8 var | R8 furn |
|---|---|---|---|---|---|---|
| 8+8 (as trained) | 2048 | .892 | .949 | .871 | .793 | .588 |
| 8+4 matched | 1536 | .890 | .948 | .870 | .790 | .584 |
| **4+4 matched** | **1024** | **.886** | .946 | **.867** | .785 | .574 |
| 4+3 / 3+4 matched | 896 | .861 / .844 | | .846 / .840 | | |
| 4+4, original full-range floor codebook | 1024 | .670 | .933 | .703 | .668 | .417 |

- **Bits actually used:** about 4 + 4 bit per coordinate. Anything below that costs 2.5–5 pp; half of the 16-bit wire is waste.
- **The decoder places surfaces partly from log q.** Under the full-range floor codebook, exact F1 drops 22 pp while tol-1 drops only 1.6 pp: the floor bias moves surfaces by one voxel.
- **Wire sparsity:** the fraction of a message's coordinates that carry evidence is 21 % at K0 and 33 % on average at R8 (p95 55 %, max 76 %).

**1.3 Does the decoder place folded coordinates?** (`--mode routing`) We measured the footprint of a ±0.25 change to one fused mean.

| model | own column share | each fold sibling | each other neighbour | centroid projection (1 = own column, 0 = fold parent) |
|---|---|---|---|---|
| 3S-X, fold 2, 51k updates | 11.9 % | 4.6 % | 3.9 % | .98 |
| L1, fold 2, 16k | 10.3 % | 4.1 % | 3.6 % | .85 |
| z1 (64x64x1), fold 4, **u4000 checkpoint** (mechanism only) | 1.5 % | 1.15 % | 1.12 % | .74 |

The fold-2 decoder reads its coordinates, and placement sharpens with training. The fold-4 decoder is only partly placing them at u4000.

**1.4 Who gained from the 16→32 rung** (`probe_B_sizegain`, `probe_B_gainvsinfo`). Values are seed 342101 / seed 342202.

| | 16k: L1 − D2 | 2 cycles: 3S-X − 3S-T |
|---|---|---|
| ΔR8, houses ≤13 m (n 10) | +5.1 / +5.0 | +5.7 / +6.2 |
| ΔR8, houses ≥17 m (n 7) | +3.3 / +2.3 | +5.8 / +5.6 |
| r(ΔK0, L) | −.44 / −.52 | −.29 / −.34 |
| r(ΔK0, bits per coordinate), plain [partial on L] | +.35 / +.39 [+.51 / +.61] | +.41 / +.52 [+.51 / +.65] |
| r(ΔK0, furniture share) | +.40 / +.50 | +.68 / +.68 |
| R8–L slope, arm vs control (pp per m) | −.47 / −.61 vs −.14 / −.25 | −.63 / −.60 vs −.64 / −.55 |

- **Size did not drive the gain.** Big houses did not gain more, and the size slope did not flatten.
- **Metric pitch is not the variable.** At equal metres per coordinate (0.45–0.66 m), the 32-grid on big houses scores K0 .857 against .826 for the 16-grid on small houses.
- **Information density is the variable.** The gain went to information-dense, furniture-rich houses; information per coordinate is flat in L. Per-house K0 of 3S-X tracks it (r −.77): lowest quartile (33 bit) K0 **.936**, R8 .90; highest (56 bit) .864–.868.
- **The rest of the size effect is readout.** The K0 exact-vs-tol-1 gap is 4.7 pp for houses ≤13 m and 5.9 pp for houses ≥15 m (r +.51), because features are thinner in voxel units in big houses.

**Answer to question 2.** The 2 KB code is **information-limited, not decoder-limited**: the target needs about 45 bit per evidenced coordinate (lzma about 80) and the decoder uses about 8, a deficit of 5× or more, concentrated in furniture and the low band. The fold-2 decoder does read its coordinates, and the wire is twice as wide as what is used.

## 2. 3S-Z prediction (fixed 19:48 UTC by `date -u`, before any 3S-Z VAL number was read)

Read beforehand: only z1's loss and rate lines, the u100 preflight line, and the u4000 checkpoint (routing).

**z1 vs L1 at 16k, paired:**
- ΔR8 surface **+2.0 pp** per seed (80 % interval +0.5 to +3.5)
- ΔR8 var **+3.0** (+1 to +5)
- ΔK0 **+2.5** (+1 to +4)
- ΔR8 furniture-region **+6** (+2 to +10)
- K0 exact-vs-tol-1 gap shrinks by at most 1 pp

**Verdict probabilities:** lever .45, ambiguous .40, not better .12, worse .03.

**House size:**
- The gain will **not** grow with L. Δ(≥17 m) − Δ(≤13 m) on R8 falls in [−3, +1] pp on both seeds, and r(ΔK0, L) ≤ +.1.
- The arm's F1–L slope stays at least 0.8× the control's.
- Instead, r(ΔK0, bits per coordinate) ≥ +.3 and r(ΔK0, furniture share) ≥ +.3 on both seeds.
- **Falsifier of this reading:** big houses gain at least 1.5 pp more on both seeds, with r(ΔK0, bits) ≤ 0. That would mean metric pitch matters after all.

**Reasoning:**
- **Why a gain:** demand per coordinate falls from 45 to 12.6 bit. That is the same 3.4–3.6× step as the last rung, and it lands near the supply of about 8 bit.
- **Why smaller than the last rung's +4.1:**
  - Part of the K0 gap is not code-limited: low-density houses plateau at .91 at 16k, and the readout gap is 5–6 pp.
  - Fold-4 placement is still forming.

## 3. Recommended next batch

**If 3S-Z is a lever, or ambiguous with Δfurniture-region ≥ +4 pp and r(ΔK0, bits) ≥ +.3 on both seeds → 3S-Z2: 64x64x1 @ 8+8 on the two-cycle recipe.**
- **Control:** 3S-X (`runs/v429-stack/c2-s*`). Only the layout changes.
- **Size:** 8,192 B per message.
- **Speed and memory:** z1 measured **0.206 s/update** (updates 100→3,650 in 732 s) and about 8.5 GB in use (nvidia-smi).
- **Budget (estimate):** about 5,250 s per cycle (fits the 6,300 s training cap); 4 leases, about 6 h.
- **Prediction (estimate):**
  - R8 +2.5 (→ about .90), K0 +3 (→ about .93), var +3.5, furniture-region +7.
  - This exceeds the 16k gain: the 32 rung went from +4.1 to +5.75 with two cycles, and fold-4 placement needs training.
- **Lever rule:** ΔR8 var ≥ +3 and surface ≥ +2 on both seeds.
- **Falsified by:** ΔR8 < +1 on either seed, or gains uncorrelated with bits per coordinate.

**If 3S-Z is not better:** first run `probe_B_code --mode routing` and `--mode stats` on z1 `final.pt` (2 minutes of CPU).
- **(a) Projection < .8, or column R² below the 32-grid values:** fold 4 or the 1x1x16 column encoder is failing. Run **3S-V: 32x32x2 @ 8+8** (D 2,048, 4,096 B) at 16k vs L1.
  - **Why this layout:** it keeps fold 2 (measured to work) and the lateral encoder patch. It splits each column at 1.6 m, below which lie 65 % of the bits and 81 % of furniture surfaces.
  - **Code change:** add `'32x32x2': (32, 32, 2)` to `LAYOUTS` and `DIAGNOSTIC_LAYOUTS`.
  - **Cost (estimate):** the trunk sees the same voxels, so about 0.20–0.21 s/update and about 8 GB; about 3,300 s per seed, 2 leases.
  - **Prediction (estimate):** ΔR8 +1.5 (+0.5 to +2.5), Δvar +2.5, furniture-region +5, low-band furniture recall +5. P(lever) about .35.
  - **Falsified by:** Δvar < +1 and no furniture gain on both seeds.
- **(b) Placement and encoder are fine:** coordinate count is no longer binding. B has **no code batch** to offer; the residual is readout (5–6 pp) and fusion (−2.4 pp).

## 4. Deployment: F1 per byte

**One byte per coordinate.** Use a 4-bit mean plus a 4-bit precision on a codebook matched to the head's range ([1e-4, 4], nearest rounding, an exact zero for "no evidence"), and spend every remaining byte on coordinates.
- **Measured:** 32x32x1 at **1 KB** gives R8 .867 vs .871 at 2 KB, without retraining.
- **Grid by budget:**
  - 1 KB → 32x32x1.
  - 4 KB → 64x64x1, if 3S-Z is a lever. Price it on CPU with `--mode bits`.
  - 512 B → 16x16x2 @ 4+4 (L3). Alternatively, 32x32x1 @ 4+4 with lossless sparse coding: estimated about 440 B mean and 850 B worst case, not a fixed size.
- **Avoid:** fewer than 4+4 bits, and any precision codebook wider than the head's range.
