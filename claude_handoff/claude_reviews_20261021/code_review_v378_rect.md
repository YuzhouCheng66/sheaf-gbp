VERDICT: LAUNCH OK WITH CHANGES

# Pre-launch code review: v378 rectangular house-fitted chart (batch O)

Reviewer: independent Claude subagent, 2026-10-23, CPU only (no GPU). Nothing under `diag/`, `handoff_src/` or `houseexpo_src/` was modified.
Scratch scripts and outputs: `/tmp/claude-0/-home-claude/4bad606d-6bc2-5614-a287-aa8ba2b79cab/scratchpad/review378/`.

## Files reviewed (sha256)

| file | sha256 |
|---|---|
| `/home/claude/sheafgbp/diag/mapping_houseexpo_data_v378.py` (**new module under review**) | `cacf69a65310f2197cbbcc11bd6cf2ca0ea6994918e59612b57d7dc524a0ce7a` |
| `/home/claude/sheafgbp/diag/test_v378_rect.py` | `987029685328badf9de6fda923e1d6bcec39e343d5f4bc9e011918e75cd2e3ca` |
| `/home/claude/sheafgbp/diag/train_v378.py` (**trainer the chain launches**) | `db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff` |
| `/home/claude/sheafgbp/diag/gen_hx_cache_v378.py` | `7fa0a45d552471b0acc61bb8e0ba3494d33c845a2a690e69896ebd9742b76a48` |
| `/home/claude/sheafgbp/diag/chain_day4_O.sh` (launch script) | `2ad26407b566831d022d647aa5620f419271936e17b0292e218adc82d06c4965` |
| `/home/claude/sheafgbp/diag/analyze_v378.py` (pre-registered analysis) | `90947d8b3ddb29a2419cc95628c8d27afe8a25666400f7eacf33d7d47bffe674` |
| `/home/claude/sheafgbp/diag/mapping_houseexpo_data_v371.py` (square control, reviewed base) | `6abdc221f01c323b968409b1281d0f1c6d2b03526901419088cd33aea915e009` |
| `/home/claude/sheafgbp/diag/mapping_houseexpo_data_v365.py` (imported base) | `e1e6b5aab5105190b0b4e2f7cf40c42ac71504867ac84ecf789634787aa68b91` |
| `/home/claude/sheafgbp/diag/train_v376.py` (diff base) | `ea1c26df5814ad52dd333874e914a906dd097bb8359c6100c0e342316d3604c8` |
| `/home/claude/sheafgbp/diag/gen_hx_cache_v375.py` (diff base) | `b3a0792d40ffe7bc9bf255ce51a48a886887ce81bf33353ab5c36a527115eee5` |
| `/home/claude/sheafgbp/diag/analyze_v375.py` (imported `eq_weight`; batch-N analysis read by chain O) | `4a004073c622fc23b30430fe518180521871e3a0c71b8a4066bee3c556ef166e` |
| sealed `scripts/prepare_houseexpo_pilot_v121.py` (`make_rays`, `raycast_observe`, `radius_adjacency`) | `ac9d15670d08c5a7e7c21958dc7230b62fa34ea7157a287e48bcb9500b57c169` |
| sealed `scripts/mapping_multiroom_data_v293.py` (`observe_trajectory` reference) | `6e4bc0766e22b3b720332fde59bfcf84c2284d27f1067357835c31df2cfd5445` |
| sealed `scripts/mapping_native_bank_v306.py` | `bfe42a06b08b33df00ecf97fa05d74f6fe88d030ac446d075c13cfa6ce18096d` |
| sealed `scripts/mapping_recognition_metrics_v277.py` | `17f0755ed0fd9d3641b60beb38eaa67e714245aee39c998d717fc644935efdf3` |
| sealed `scripts/mapping_multiroom_forest_v295.py` | `d3b71b660eae29af49c662441570ec849b025c6a45f1636b46faa3877e6b458e` |
| sealed `scripts/train_mapping_implicit64_v192.py` (`door_measures`) / `scripts/mapping_topology_v211.py` | `66970f70ada8aaf9b6e9689de7f2ee14a64b6ae659eb8b1260c1eba11728efa7` / `2973c35446cabaf7f14da9f24f676aacccb5756b1512b407519a52b0dab7d973` |
| HouseExpo index used locally | `7b4730cc52689c9044228a6f9bf8e1b4785cb5f6f40f54a422e603c6000101c1` |

## Summary

**The module is physically correct. I found no x/y, row/col, border or tie bug.** I checked it against independent references rather than only re-reading it:
- The anisotropic ray traversal matches an independent exact segment traversal on all 8,192 beams of 32 chart shapes (0 mismatches, 436 exact corner ties handled identically).
- The same metric poses observed on the square and on the rectangular chart give the same metric observation (known-area IoU 0.978, label agreement 0.993, observed area ratio 0.995). Two injected axis bugs are clearly detected by the same comparison.
- The anisotropic EDT equals a brute-force metric distance exactly.
- The opening mask gives the right metric answer per axis on synthetic doorways.
- `square=True` reproduces v371 bit-exactly on 156 scenes.
- The trainer runs end to end on the new world (CPU smoke), and the full k2400 TRAIN cache generates without failure (§ Checks, item 7).

Two changes are required before the result can be read as a one-change comparison. Neither is a bug in the sensor or the geometry:

- **M1. Pair the poses across the two arms.** At present v378 re-draws every pose (the world_id prefix changes, and so does the candidate raster). Both training seeds are evaluated on the same single VAL pose draw per arm. The pose-draw term in "rect − square" is therefore shared by the two seeds, and the "both seeds" rule does not protect against it. With a fixed instrument I measured its size at about **0.76 pp SD on VAL R8 and 1.3 pp SD on K0**, against decision thresholds of 2 pp and 1 pp. Translating the v371 poses into the rect chart is about 6 lines and was tested here. It makes poses, radio graph and forest identical in both arms, makes the 27 % of panel houses whose chart does not change bit-identical (a built-in placebo), and roughly halves the residual noise.
- **M2. Guard the control's identity in `analyze_v378.py`.** The d2400 control is "batch N d2400", but batch N picks its chart by a rule. It falls back to the 20 m v365 chart if either v372 run did not complete 72 epochs or missed +2 pp. Nothing in chain O or `analyze_v378.py` checks that the control is a `houseexpo_fit` run. If it is not, "rect vs 20 m chart" would be read as "rect vs square".

The anisotropy effects on the metrics are **semantic, not bugs**, and have to be disclosed (§ Anisotropy):
- per-cell F1 on a raster that is finer only along the short axis;
- the DT loss counted in cells;
- topology and border measured in raster units;
- no scale input to the decoder;
- 15 % of houses (27 % of the VAL panel) where the chart does not change at all.

---

## Required changes

**M1. Pair the poses: translate the v371 (square-chart) metric poses into the rect chart.**
- *What happens now:* `v378.world_spec` renames the world_id from `v365:` to `v378:`, so the pose RNG seed changes. `metric_trajectory` also samples from the rect 512 raster, so the candidate list `np.argwhere(...)` changes as well. Even with the v371 seed, a re-sampled draw would therefore not reproduce the v371 poses. Every scene in both splits gets new poses, a new radio graph and a new forest.
- *Why it matters for this experiment:*
  - VAL fixtures depend only on the world_id, not on the training seed. Both seeds of each arm are scored on **one** VAL pose draw, and the control's two seeds on another, so the draw-to-draw difference enters both seed comparisons identically.
  - *Measurement:* I used `pose_noise.py`, with the v293-trained owned/max/posterior_only checkpoint `long-s342202-final.pt` as a fixed instrument (zero-shot; only the *variation* matters). I scored the trainer's VAL panel (48 scenes, equal weight over 6 strata) under the official v371 poses and under 4 alternative draws (same chart and houses, world_id suffixed).

    | equal-weight VAL wall F1 | 5 draws | SD over draws | SD of a 2-draw difference | max pairwise difference |
    |---|---|---|---|---|
    | R8 | .5452 .5390 .5523 .5482 .5512 | 0.54 pp | **0.76 pp** | 1.34 pp |
    | K0 | .5455 .5393 .5559 .5365 .5560 | 0.91 pp | **1.29 pp** | 1.96 pp |

  - Against the pre-registered rule (adopt ≥ 2 pp on both seeds, no gain < 1 pp on both), a shared ~0.8 pp term has about an 18 % chance of moving the gain by more than 1 pp, in the same direction for both seeds.
  - The treatment itself is moderate: on the VAL panel, the blocks containing floor go from 46.1 to 53.0 of 64. For comparison, v365 → v371 went from about 24 to about 45 blocks for +10 pp. So the effect to be detected is of the same order as the thresholds.
- *Patch (tested by monkeypatch in `paired_patch_test.py`; the reviewed file is untouched):*
  ```python
  import mapping_houseexpo_data_v371 as v371
  def metric_trajectory(spec, agents=8):
      """Paired design: the v371 square-chart poses of the same split/house/N, translated into the rect chart."""
      sq = v371.world_spec(spec['split'], spec['index'], spec['stratum'])
      L = sq['chart_m']; lx, ly = spec['chart_xy_m']
      return v371.metric_trajectory(sq, agents) + np.array([lx / 2 - L / 2, ly / 2 - L / 2])
  ```
  The house is centred in both charts, so this is the exact metric identity of the same physical pose.
- *Verified with the patch on the 96 VAL/fresh panel scenes:*
  - Every fixture builds (`observe_trajectory` accepts every pose).
  - Adjacency and the selected forest are identical to v371 in 96/96 scenes.
  - `square=True` still equals v371 in 96/96 (the shift is exactly 0.0).
  - The 26 scenes whose chart does not change (Lx = Ly = L) give **bit-identical private maps**.
  - **Over the full k2400 TRAIN plan plus the panel (7,296 scenes), every translated pose is inside the rect chart and on a free native cell: 0 infeasible** (`paired_feasibility.py`). This is also expected analytically: the continuous clearance is ≥ 0.34 m, while a rect cell's half-diagonal is ≤ 0.09 m.
- *Effect on the noise (same instrument, rect chart with paired poses, 5 draws):*
  - "rect − square" per draw is −4.69 / −3.93 / −3.80 / −4.50 / −3.54 pp on R8. The negative level is expected: the instrument was trained on isotropic 20 m charts, so this is not a prediction of the treatment effect.
  - Paired-difference SD is **0.48 pp (R8) and 0.55 pp (K0)**. The unpaired rect − square SD with this instrument is 0.99 / 1.23 pp, measured both as sqrt(var_R + var_S) and as the empirical off-diagonal SD. Pairing therefore halves the noise; the draw-level correlation between arms is 0.83 / 0.80.
  - With in-distribution models for each arm the chart × pose interaction is probably smaller than for this out-of-distribution instrument, but that is not measured here.
- *Also required with M1:*
  - Add to `test_v378_rect.py`: (a) rect adjacency == v371 adjacency, and (b) unchanged-chart scenes have bit-identical private maps.
  - Regenerate `houseexpo_rect-k2400` with the new module. The trainer's `data_module_sha256` assert rejects a stale cache.
  - Write the new module sha into `REVIEW_OK`.
- *Cheap chain cross-check that the pairing reached the cache (recommended):* `adjacency` and `edges` of every row of `houseexpo_rect-k2400` must equal those of `houseexpo_fit-k2400` row by row. The rows are the same houses because `scene_plan` is v365's for both. On the H600 path, compare the first 8 VAL fixtures' adjacency the same way.
- *Fallback if relaunch timing matters:*
  - Launch unchanged, but then (i) state in the analysis that the primary carries an unmeasured shared pose-draw term of about 1 pp, and (ii) pool val + fresh (96 scenes) as a pre-registered secondary.
  - This does not make the comparison one-change; M1 does.

**M2. Assert the control's world, recipe and initial weights in `analyze_v378.py`, or in chain O before launch.**
- *The risk:*
  - `chain_day4_N.sh` sets `WORLD=houseexpo_fit` only if **both** v372 runs have `epochs_completed == 72` and R8 ≥ v367 + 2 pp; otherwise batch N ran on `houseexpo` (the v365 20 m chart).
  - Chain O then takes `runs/v375-datascale/d2400-s<seed>` as the "square" control without checking.
  - A deadline-truncated v372 run is enough to cause this, and the v375 review M1 showed that this deadline is not slack.
- *Required:* in `load()` / `main()`:
  - assert the control's `run.json['args']['world'] == 'houseexpo_fit'` and the arm's `== 'houseexpo_rect'`;
  - assert `initial_weights_sha256` is equal for arm and control (as `analyze_v375.py` already does);
  - assert the recipe-defining args are equal: `epochs, lr, fusion, readout, precision, wire, mu_bits, q_bits, dt_trunc, consistency, edge_select, size, arm, seed`, and `per_stratum` from the cache manifest on the d2400 path;
  - on failure, set the verdict to INCONCLUSIVE.
- *Current state:* I checked the CLI lines by hand and they match (chain O d2400 = chain N line apart from `--world` / cache path; chain O h600 = chain M v372 line). The assert is the guard against the batch-N fallback and against later edits.

## Minor

- **m1. Recommended secondary analysis: a common-grid evaluation.** It needs M1 and is analysis only.
  - With paired poses, the square and rect targets of a scene are the same physical observations: known-area IoU 0.97, label agreement 0.99.
  - Map each arm's VAL prediction to the other arm's raster (nearest cell, through metric coordinates) and score it with `node_metrics` against that arm's target. Do this in both directions and report it next to the native-raster primary. A consistent sign in both directions removes the pixel-scale confound from the conclusion.
  - Sub-sampling the finer rect prediction onto the square raster slightly penalises rect, and the reverse direction penalises square.
  - It can be run post hoc from `final.pt` with a small evaluation script; the trainer does not need to change.
  - Also report the gain on chart-changed houses only (73 % of the panel, 56 % in rooms 1). The unchanged houses are, with M1, a placebo that measures only the effect of the training distribution.
- **m2. `public_frame.cell_size_m` and `extent_m` are now lists in (x, y) order, while the raster is [row = y, col = x].**
  - No consumer exists today: grep over `train_v378.py`, `gen_hx_cache_v378.py` and the sealed bank, metrics and forest modules finds only `communication_radius_m` being read.
  - Any future consumer that indexes `cell_size_m` by raster axis would get the axes swapped. State the order in the docstring, or add `cell_size_rc_m`.
- **m3. `edge_selection` (train L116-124) hardcodes the 20 m world's 2.5 m block centres.**
  - It is wrong on any fitted chart, v371 included.
  - It is inactive at `--edge-select 0`, which is what chain O uses.
  - Add `assert EDGE_SELECT_M <= 0 or args.world == 'v293'`.
- **m4. Cache bookkeeping.**
  - `gen_hx_cache_v378.py` still writes `script='gen_hx_cache_v375.py'` in the manifest.
  - Nothing in the repo writes the `houseexpo_rect k2400 exit 0` line that chain O waits for in `$C/done_v378.txt`. Make sure the generation command appends it only on exit 0.
  - `--out` uses `mkdir(exist_ok=False)`, so a pre-M1 cache directory must be moved away first. A stale cache would otherwise make both trainers die at the sha assert, after the batch-J wait.
- **m5. Clearance wording.**
  - "0.45 m clearance" is the distance from a candidate *cell centre* to the nearest occupied 512-cell centre, before the ±0.1 m deltas.
  - The actual continuous clearance from poses to the contour is ≥ 0.340 m for square and ≥ 0.349 m for rect (5th percentile 0.381 / 0.377 m), over 246 scenes.
  - It is inherited from v365 and identical in both arms, so it is not a confound; it is wording only.
- **m6. Inherited inefficiency.** `metric_trajectory` calls `rasterize(spec, 512)`, which also computes the Python-loop opening mask on the 512 raster and then discards it. The redundant `size != 512` clause in `rasterize` is harmless. This is unchanged from v371 and costs about 0.2 s per scene in total.
- **m7. Corner ties are much more frequent on anisotropic cells.**
  - 436 exact ties in 8,192 beams. The aspect ratio Ly/Lx is rational, so for example 45° beams hit cell corners exactly whenever (2k+1)·Ly = (2m+1)·Lx.
  - They are handled conservatively, with both side cells before the diagonal, exactly as v121.
  - The tie tolerance `1e-10·min(cx, cy)` is safe: my exact traversal, using direct (not accumulated) boundary parameters and a 1e-9·max tolerance, agrees beam for beam.
  - Effect: on rect charts slightly more beams stop at corner side cells. This is the intended conservative semantics, and I mention it only for completeness.

## Anisotropy: semantic changes to disclose vs bugs

None of the following is a bug. Each is part of what "changing the chart" means and should be stated with the result:

| effect | where | size (VAL panel, 48 houses) |
|---|---|---|
| The raster is finer **only along the short axis**. The long-axis cell equals the square-chart cell in every house; the aspect rule only ever raises the short side. | `rasterize`, all per-cell metrics | short-axis cell median 0.094 m vs 0.105 m (range 0.062-0.141); cell anisotropy median 1.17, max 1.75, > 1.25 in 33 % of houses; +17 % cells per m² |
| Per-cell wall F1 is not a per-metre-of-wall measure. With conservative rasterisation a thin wall is about 1-2 cells across, so its weight scales with length in cells along its own axis. In rect, walls along the short axis are counted more finely than walls along the long axis. Cells are equal-area within one house; across houses they were already unequal in v371. | `node_metrics` | same metric poses: +12 % wall cells in rect (chart-changed scenes) |
| The truncated DT target and loss are in **cell** units: T = 4 cells, near-wall weight band DT ≤ 1.5 cells, prediction wall := DT < 0.5. On rect cells the metric band is anisotropic (1.5·cx vs 1.5·cy). The decoded wall definition is still exactly "wall cell". | `truncated_dt`, `stage_loss`, `predict` | band 0.09-0.23 m depending on axis and house |
| Topology (4-neighbour pair connectivity), the 6-px interior border, and opening recall are all measured on the anisotropic raster. A narrow gap can be open on one raster and closed on the other. | `mapping_topology_v211`, `INTERIOR_BORDER`, `door_measures` | secondary metrics only |
| The decoder receives no (Lx, Ly). Per-axis scale is now two unknowns rather than one (v371 already hid L). Metric priors such as door width or room aspect become per-axis ambiguous in the 64-coordinate code. | model inputs | the price of the rect design; arguably part of the treatment |
| Blocks are Lx/8 × Ly/8 m (anisotropic). This is the treatment. | 8 × 8 block grid | blocks with floor 46.1 → 53.0 of 64 (median 47 → 54); bbox share 0.63 → 0.76 |
| Treatment dilution: houses with both axes ≤ 7 m (min-8 rule on both axes) get an unchanged chart. | `chart_rule` | 15.0 % of TRAIN / 16.3 % of VAL in the index; **27 % of the 48-house VAL panel** (rooms 1: 44 %, rooms 2/3: 19 %) |
| The aspect rule binds rarely. | `chart_rule` | 2.36 % of TRAIN houses (480), 2.04 % of VAL (52); always gives exactly lx = ceil(ly/2) ∈ {9, 10} |

The analysis docstring already pre-registers the per-cell caveat ("per-house accuracy on each arm's native raster"). The table above gives the numbers to quote with it. Note that in this design the per-axis pixel scale of the two arms is not a free choice: the long axis is identical and the short axis is finer by L / l_short. m1 is the way to remove it from the conclusion.

## Method (all run, not inferred)

1. Read v378 against v371/v365 line by line, `diff train_v376 train_v378` and `diff gen_hx_cache_v375 gen_hx_cache_v378` (only the world switch), plus the lineage diff v375 → v376 (evaluation-only `--oracle-union`).
2. Ran `test_v378_rect.py` (exit 0, 7 s).
3. `check_rays.py`: independent exact metric traversal, from the sorted analytic boundary parameters, compared beam by beam with `make_rays_aniso`. I also checked metric reach and metric angle, and isotropic equality with `make_rays` for every chart side L = 8…20 m.
4. `check_physical.py` and `check_inject.py`: the same metric poses observed on both charts, compared on a common 2.5 cm metric grid, with injected axis bugs as a sensitivity control. `check_worst.py` inspected the worst agents.
5. `check_poses.py`: continuous point-to-contour clearance, point-in-polygon, separation fallback statistics, chart statistics over 246 scenes, and a brute-force EDT check. A separate pass computed the chart-rule statistics over the whole index.
6. `check_doors.py`: synthetic metric doorways and corridors, both axes, with cx ≠ cy.
7. Extended `square=True` == v371 equivalence on 156 scenes (all six fixture arrays, `extent_m` and `world_id`).
8. Full `gen_hx_cache_v378.py --world houseexpo_rect --per-stratum 2400` (7,200 scenes) on 1 worker, and the trainer's cache path on it.
9. CPU smoke of `train_v378.py --world houseexpo_rect --smoke` (in-memory path), end to end through load, DT, max-fusion rollout, loss, evaluation and `node_metrics`.
10. `pose_noise.py` and `pose_noise_paired.py`: pose-draw noise with a fixed checkpoint (M1); `paired_patch_test.py` and `paired_feasibility.py` for the proposed patch.

## Checks that pass

**1. Sensor = the same physical sensor: PASS.**
- *Beams:* 256 beams uniform in **metric** angle (`dr = sin a, dc = cos a`, with metric boundary spacing `cy/|sin|` and `cx/|cos|`). Row maps to y and col to x correctly. The last visited cell of beams 0/16/32/48/64 lies at metric angles 0 / 22.9 / 45.5 / 68.1 / 90° on a 14 × 8 chart, which is uniform up to cell quantisation.
- *Range:* strict 5 m. A cell is entered only if its boundary crossing is at t ≤ 5 m from the snapped centre, and the metric disk mask on cell centres around the continuous pose (`(ax[None]-x)² + (ay[:,None]-y)² ≤ 25`) is then applied as in v293. The maximum visited-centre distance per beam is 4.92-5.07 m, within 5 m + half a cell diagonal before the disk mask.
- *Traversal:* identical to an independent exact traversal on 8,192/8,192 beams.
- *First hit and corners:* the unchanged sealed `raycast_observe` handles first hit. Corner ties are conservative (m7).
- *Rays leaving the chart:* they are killed by the in-bounds `alive` mask. In practice they never get there, because every house has ≥ 0.5 m of occupied exterior on each axis.
- *Pose snapping and borders:* poses snap with `rc = (floor(y/cy), floor(x/cx))`. The bounds check is per axis (`x < lx`, `y < ly`). Free poses are enforced.
- *End to end:* with the same poses, the known-area IoU is 0.978 (chart-changed scenes 0.970), label agreement on both-known cells is 0.993, and the metric observed-area ratio is 0.995. Injected bugs give IoU 0.905 (cx/cy swapped in the rays) and 0.889 (isotropic rays with the max cell), with minima around 0.6, so the check is sensitive. The worst correct agents (IoU 0.88) are small-footprint agents whose differences are boundary quantisation and fans through openings; I inspected them in `worst_iou.png`.

**2. Poses: PASS.**
- *Clearance:* `distance_transform_edt(fine, sampling=(cy, cx))` uses the correct (row, col) order. It equals the brute-force metric distance with 0 error on 183,947 cells; the swapped order would be off by up to 0.52 m.
- *Candidates and deltas:* candidate centres are `((col+.5)·cx, (row+.5)·cy)`. The deltas are ±0.1 m metric.
- *Separation fallback (1.0 → 0.5 → 0):* metric. 1.0 m held in 99.2 % of rect scenes and 0.5 m in the rest; 0 was never needed.
- *Inside the contour:* every pose lies inside the contour.
- *Radio:* the 8 m radio is computed on metric positions, through the sealed `radius_adjacency` and `select_forest`.

**3. Opening mask: PASS.** Runs along x use cx for width and cy for thickness, and runs along y the reverse. Synthetic doorways of 0.9 / 1.8 / 0.3 / 1.2 / 0.45 m are classified correctly on both axes; swapping the two cell sizes misclassifies 3 of the 4 cases. 1.0 m corridors along either axis are not flagged.

**4. Chart rule: PASS.** `Lx = max(8, ceil(ext_x + 1))`, and the same for y, then the aspect cap. Aspect ≤ 2 is guaranteed; the maximum over the index is exactly 2.0. Every house lies inside its chart with ≥ 0.5 m margin per axis. The long side always equals the v371 side L.

**5. `square=True` == v371: PASS.**
- The test suite's 6 scenes and my 156 scenes (96 VAL/fresh, and 60 TRAIN up to index 2280) are bit-identical in private, positions, trajectory, adjacency, walls, door mask, `extent_m` and `world_id`.
- Isotropic `make_rays_aniso == make_rays` holds for every L = 8…20.
- The `cx != cy` branch keeps the exact v371 EDT expression in square mode.

**6. Trainer and evaluation path: PASS (semantic items above).**
- `train_v378` = `train_v376` + the import switch. The training loop is unchanged since the reviewed v375; v376 adds only the evaluation-only oracle.
- Nothing on the path reads `cell_size_m` or `extent_m`. The v293 `observe_trajectory` (square / 20 m asserts) and `bank.expected_frame` are not used by HouseExpo worlds, because `--dynamic` is asserted off.
- The bank's `forest_targets` and `select_forest` read only `communication_radius_m`.
- The CPU smoke completes. The chain O CLI equals the batch-N line (d2400) and the v372 line (h600) apart from world and cache, so it is the same recipe.

**7. Split hygiene, determinism and robustness: PASS.**
- *Splits:* `world_spec` delegates to `v365.world_spec`, which raises for anything but TRAIN/VALIDATION; the index holds no TEST rows; `scene_plan` is v365's. The same TRAIN houses (indices 0..K−1 per stratum) and the same VAL panel are used in both arms.
- *Seeds:* `sha256('v378:<SPLIT>:<house>:poses:<N>')`, with no global RNG. Without M1 the prefix change re-samples poses, as intended by the author but confounding; see M1.
- *Full k2400 cache:* `gen_hx_cache_v378.py --world houseexpo_rect --per-stratum 2400 --workers 1` finished **7,200/7,200 scenes with 0 exceptions** in 1,348 s on one core (about 0.19 s per scene, the same as `houseexpo_fit`). It is 42 MB, contains only TRAIN, and every file name is unique.
  - 24 random rows fully regenerated in-process match exactly: private, positions, adjacency, edges, reached[k] and door_mask.
  - The trainer's cache path (`--train-cache ... --epochs 6 --expect-updates 43200 --max-updates 2`, CPU) passes every assert: exact plan, index sha, data-module sha, and regeneration of the first and last 4 rows. It then trains 2 updates with finite loss 2.787.
  - Reference rows digest for the **current, unpaired** module (sha256 over the rows of `json.dumps([file, spec, private_sha256], sort_keys=True)`): `44a6bab6ef83482f5be818ba6b15e6281719242dfd74f8545e129d92b44b2fc8`. It changes with M1.

## Answer to "is rect vs square a fair one-change comparison?"

- **Same in both arms:** the houses (TRAIN indices 0..K−1 per stratum, the same VAL and fresh panels); N; the metric sensor, radio, clearance and opening semantics; the trainer code path; the recipe flags; the seeds and initial weights (the model init does not depend on data); and the update count.
- **Different by design (the treatment):** the chart, and with it the block geometry (Lx/8 × Ly/8 m) and the raster (finer along the short axis only).
- **Remaining confounds:**
  - **(a) The pose draw.** It is shared by both seeds, about 0.8-1.0 pp SD in the R8 difference, and of the order of the thresholds. It **requires M1**.
  - **(b) The evaluation pixel scale per axis.** It is inseparable from the treatment on native rasters. Disclose it, and remove it from the conclusion with the m1 common-grid secondary.
  - **(c) Dilution.** 27 % of panel houses have an identical chart. Report the chart-changed subset.
  - **(d) The cell-unit DT loss.** It is part of the treatment; disclose it.
  - **(e) The control's identity.** It is unchecked, and depends on the batch-N chart fallback. It **requires M2**.
- (a) and (e) are the only items serious enough to change before launch, and both are small code changes.
