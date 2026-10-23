VERDICT: LAUNCH OK WITH CHANGES

# Pre-launch code review: v375 data-scale study (`--train-cache`, streamed targets/DT)

Reviewer: independent Claude subagent, 2026-10-23, CPU only (no GPU). Nothing under `diag/`, `handoff_src/` or `houseexpo_src/` was modified.
Scratch scripts and outputs: `/tmp/claude-0/-home-claude/4bad606d-6bc2-5614-a287-aa8ba2b79cab/scratchpad/review375/`.

## Files reviewed (sha256)

| file | sha256 |
|---|---|
| `/home/claude/sheafgbp/diag/train_v375.py` (**the trainer this verdict covers**) | `1e98ffb484c46d6f21209842606db1802337240718d5ec82ecc06692be841248` |
| `/home/claude/sheafgbp/diag/train_v374.py` (reviewed base) | `57dd08c8a6dc17bfabbc4fb239047c5ef9f336670111d2369c0a71f41b3a47b6` |
| `/home/claude/sheafgbp/diag/gen_hx_cache_v375.py` | `b3a0792d40ffe7bc9bf255ce51a48a886887ce81bf33353ab5c36a527115eee5` |
| `/home/claude/sheafgbp/diag/test_v375_stream.py` | `cc2307327fdc94c21fbc56a058723ceead7b6733d0f655545a3a47d7e7e3835a` |
| `/home/claude/sheafgbp/diag/mapping_houseexpo_data_v365.py` | `e1e6b5aab5105190b0b4e2f7cf40c42ac71504867ac84ecf789634787aa68b91` |
| `/home/claude/sheafgbp/diag/mapping_houseexpo_data_v371.py` | `6abdc221f01c323b968409b1281d0f1c6d2b03526901419088cd33aea915e009` |
| `/home/claude/sheafgbp/diag/chain_day4_N.sh` (launch script, read for context) | `387899ac769b727085ab13f43d12f304df311da2a8fb33380a595ae0cb6aa35e` |
| HouseExpo index used locally | `7b4730cc52689c9044228a6f9bf8e1b4785cb5f6f40f54a422e603c6000101c1` |

## Summary

The code change is correct. A training update on a cached scene computes the same thing as an update on an in-memory scene. On CPU it gives the bit-identical loss, targets and DT. The pinned / `non_blocking` path has no race. The cache holds TRAIN only, TEST is never touched, and generation is deterministic. I generated the full 14,400-scene k4800 cache for both worlds locally, with no failure.

Three things need fixing before the result can be read. None of them is a code bug in the update path:

- **M1. Completion gate.** The runs can stop at the wall-clock deadline and still produce the "final VAL R8" that the pre-registered rule reads. Nothing checks that they completed. The stream path adds about 8% more kernel launches per update.
- **M2. Cache validation.** The manifest checks can silently accept a cache with the wrong K, or one built from a different index or base module.
- **M3. Near-duplicate analysis.** HouseExpo has many near-duplicate layouts. Going from 600 to 7,200 / 14,400 TRAIN houses triples the number of VAL-panel houses that have a near-copy in training. This can move the headline by roughly the size of the 2 pp decision threshold. It needs a stratified analysis, written down before the results are seen.

## Method (all run, not inferred)

1. `diff train_v374.py train_v375.py`. I also diffed the lineage `train_v365 → v371 → v374`, which contains only the world switch and the `--panel` diagnostic. So the v367 and v372 controls ran the same training loop.
2. **Full cache generation with the real generator:** `gen_hx_cache_v375.py --per-stratum 4800 --workers 2` for `houseexpo` and `houseexpo_fit`, 14,400 scenes each. Both exited 0. They took 1,212 s (houseexpo) and 1,548 s (houseexpo_fit) on 2 shared cores, and each cache is about 62 MB on disk.
3. **Checks on each full cache:**
   - I timed the trainer's `load_cached` (copied verbatim) and measured peak RSS.
   - I checked that the manifest rows equal `scene_plan('TRAIN', size=128, train_per_stratum=4800)` exactly.
   - I spot-checked 42 random rows plus the first and last row against an in-process regeneration (`make_fixture`, `select_forest`, `forest_targets`). The compared fields were private, positions, edges with exact types, door_mask, reached[k], `stream_targets == forest_targets` and the truncated DT (T=4), all as exact equality.
   - I computed order-sensitive digests for cross-machine comparison.
4. **One full training update on CPU, in-memory path vs stream path.** I used the real `SheafModel`, max-fusion rollout, `stage_loss` and clipping, on an N=16 and an N=8 HouseExpo scene. I compared the losses and counted leaf aten ops with `torch.profiler`; on GPU these are about one kernel launch each.
5. Re-ran `test_v375_stream.py`: exit 0, 30 equality checks plus the trainer smoke.
6. **Near-duplicate screen** of the 48 VALIDATION panel houses (indices 0-15 per stratum, the val and fresh panels) against TRAIN indices 0..K-1:
   - Pass 1: free-mask IoU at 64×64.
   - Pass 2 (layout level): the conservative H128 raster exactly as `v365.rasterize`, so thin interior walls survive. On the wall-boundary cells I computed F1 with a 1-cell (0.156 m) tolerance, maximised over the 8 dihedral transforms, on the 30 best IoU candidates.
   - Control group: 48 never-used TRAIN houses (indices 4800-4815).
   - Scripts: `neardup.py`, `neardup2.py`; output in `neardup2.json`.

## Required changes

**M1. Gate the comparison on completed training, and budget the lease for the slower update.**
- *What `chain_day4_N.sh` does now:*
  - It launches 3 co-scheduled trainers with `DE = START + 6600`.
  - It applies "D2400 final VAL R8 ≥ control + 5 pp on both seeds; < 2 pp ⇒ not data-limited" to `final.json['val']`.
  - The trainer writes `final.json` even when `--deadline-epoch` cut training short (L633, L669-689).
- *Why that biases the answer:* a truncated run has seen fewer than 43,200 updates and stopped mid-cosine (lr not annealed). It is therefore biased low, i.e. towards "not data-limited".
- *What already checks completion:* the chain checks `epochs_completed == 72` only for the batch-M runs, and only to pick the chart. It checks nothing for the v375 arms or for the v367 control.
- *Why truncation is plausible:* the stream path rebuilds targets and the DT for 5 hops on every update. Measured cost:
  - N=16: **+710 leaf ops per update on top of 10,902 (+6.5 %)**.
  - N=8: +710 on top of 5,941 (+12 %).
  - With the 50/50 N mix, that is about **+8.4 % launches per update**.
  - These eager-mode trainers are launch-bound, so expect roughly +5-9 % wall time per update.
  - Fewer validations (6 or 3 instead of 12) give part of that back.
- *Required:*
  - (a) Treat any arm or control with `final.json['updates'] != 43200` or `epochs_completed != --epochs` as INCONCLUSIVE, and write that into the chain and the analysis.
  - (b) Before launch, project the time from the control's own history: `sum(epoch_seconds)` × 1.09, plus 6 × (or 3 ×) the validation time, plus load time. If this is above about 6,300 s, lengthen `DE`/`DT`; the deadline is not part of the recipe unless it binds.
- *Recommended alongside:* a 200-update CUDA preflight with `--max-updates 200` on the cache path, under the same 3-way co-schedule. It measures s/update. It also runs the pinned-memory / `non_blocking` / CUDA int8 `amax` path for the first time: the test suite is CPU-only, and with `--device cpu` the `pin` lambda is the identity.

**M2. Make a wrong cache fail fast; the current manifest checks are not sufficient.**
- *What is checked now:* `world`, `size`, `hops`, `sha(data module file)`, and `split == 'TRAIN'` per row, plus a per-file `private` hash.
- *What can still pass silently:*
  - (i) **A wrong K.** Neither `per_stratum` nor `len(rows)` is checked against anything. A k2400 cache passed to the 3-epoch arm, or a k4800 cache in the "k2400" directory, gives 21,600 or 86,400 updates and a different cosine.
  - (ii) **A different index file.** `index_sha256` is recorded but never compared with `$HOUSEEXPO_INDEX`. Leakage is impossible, because the split is a hash of the contour. But a different index changes which house sits at which index, and so breaks "indices 0-199 = the control's houses".
  - (iii) **A changed base module for `houseexpo_fit`.** Only `mapping_houseexpo_data_v371.py` is hashed, but `world_spec`, `scene_plan`, `narrow_mask` and `_seed` come from the imported v365 module, which is not hashed.
- *Required (a few lines in the `elif args.train_cache` block):*
  - Assert `args.epochs * len(train) == expected`: add `--expect-updates 43200`, or check `man['per_stratum']` against the `-k` value in the chain.
  - Assert `[r['spec'] for r in man['rows']] == data.scene_plan('TRAIN', size=size, train_per_stratum=man['per_stratum'])`. This held on both of my caches, and it implies TRAIN-only, unique and complete.
  - Assert `man['index_sha256'] == sha_file(os.environ['HOUSEEXPO_INDEX'])`.
  - For `houseexpo_fit`, also record and compare the sha256 of `mapping_houseexpo_data_v365.py`.
- *Cheap cross-check with the control (recommended):*
  - The cache's `rows[0:8].private_sha256` (rooms 1, indices 0-7) must equal the control's `run.json['train_private_sha256']`.
  - The v375 `run.json['initial_weights_sha256']` must equal the same-seed control's.
  - The values from my local houseexpo cache are listed in the Reference section.

**M3. Pre-register a near-duplicate-stratified secondary analysis before reading the results.** This is analysis-only and does not block the launch.
- *The issue:* splits are exact-contour-hash only, and near or rotated copies are not detected, as `v365` L12-13 states. The near-copy rate is not small.
- *Layout-level screen* (wall-boundary F1 with 1-cell tolerance, max over D4; a lower bound, because of the 30-candidate prefilter):

  | 48 VAL panel houses (idx 0-15 × 3 strata) with a TRAIN match | ≥ 0.90 | ≥ 0.95 |
  |---|---|---|
  | in the 600-house control set (k200) | 3 | 1 |
  | in the 7,200-house set (k2400) | 10 | 5 |
  | in the 14,400-house set (k4800) | 14 | 7 |
  | reference: 48 never-used TRAIN houses vs k200 / k2400 / k4800 | 2 / 6 / 9 | 2 / 3 / 6 |

- *Where it concentrates:* the headline val panel (indices 0-7), mostly in rooms stratum 1. There, matches ≥ 0.90 go from 2/8 houses (k200) to 5/8 (k2400) and 6/8 (k4800). In rooms 2 they go from 0 to 1 to 2 of 8.
- *Flagged val-panel houses (VALIDATION rooms/index):*
  - k200: r1/i1, r1/i3.
  - k2400: r1/i0-4, r2/i2.
  - k4800: r1/i0-5, r2/i2, r2/i3.
  - Fresh-panel (8-15) houses are flagged in `neardup2.json`.
- *Possible size of the effect:* the R8 headline is an equal-weight mean over 6 strata, and rooms 1 carries 2 of them. If a near-copy in training lifts R8 on a house by Δ (the v374 seen-vs-unseen TRAIN gap is about 15 pp), then D2400 vs control can gain about 0.17·Δ ≈ 1.5-2.5 pp from near-copies alone. That is the size of the "< 2 pp" threshold and about 40 % of the 5 pp one.
- *What it is not:* this is not a split leak. The reference TRAIN houses show the same rates, so it is a property of HouseExpo (SUNCG templates and generic small layouts), and TEST will behave the same.
- *Why it matters anyway:* it changes what "data-limited prior" means.
- *Required:* fix the matching rule before the results exist (e.g. this screen at ≥ 0.90 against each arm's own training set), and report the D2400 / D4800 − control gain separately on near-copy-free houses.
  - The per-scene rows in `validation/final-val.json` and `final-fresh.json` are enough for this, so no re-run is needed.
  - If the gain on near-copy-free houses is below the threshold while the headline is above it, the conclusion is "gain is (partly) near-duplicate memorisation", not "data-limited prior".

## Minor

- **m1. Validation cadence is not matched.**
  - The control validates every 6 epochs: 12 points, every 3,600 updates.
  - D2400 with `--val-every 1` validates at 6 points (every 7,200 updates); D4800 at 3 points (every 14,400).
  - `best.pt` is therefore selected from 12 / 6 / 3 candidates. The pre-registered metric uses final weights (`final.json['val']`), so this is fine as long as nobody compares `best.pt` or "best epoch" numbers across arms.
- **m2. The v374 `--panel train` gap diagnostic no longer works for v375 models.** Its "never-trained TRAIN" panel (indices 200-215) is inside every v375 training set. For a gap measurement on v375 models, use indices ≥ K. Rooms stratum 1 has only 5,383 TRAIN houses, so for K = 4800 use 4800-4815.
- **m3. Resources (measured, k4800, CPU here):**
  - Load and per-file hash of 14,400 npz: **30.6 s** (houseexpo) / 29.1 s (houseexpo_fit), against about 2-5 min to regenerate 600 fixtures in the control.
  - Peak RSS +2.92 GiB during load.
  - Private maps 2.64 GiB.
  - Pinned estimate about 2.69 GiB per k4800 process and 1.35 GiB per k2400 process, so **about 5.4 GiB pinned for the three co-scheduled arms**. Check this against the 5090 host's RAM.
  - The CUDA caching host allocator rounds up to a power of two; private maps are exactly 2^17 or 2^18 bytes, so nothing is wasted there.
  - 28,800 separate `cudaHostAlloc` calls (2 per scene), probably seconds to tens of seconds. One contiguous pinned buffer would be faster but is not needed.
  - `door_mask` (225 MiB) and the numpy `reached` stay in host RAM for training items without being used.
  - `z['private']` is decompressed twice per file. Harmless.
- **m4. Generator robustness and partial caches.**
  - Partial failure is fail-safe: there is no manifest until the end, `mkdir(exist_ok=False)` prevents mixing, and a corrupt npz raises a CRC error on load.
  - One exception aborts the whole pool run, and there is no resume. My full scans of both worlds at K = 4800 hit 0 exceptions, so this is not expected to bite.
  - The manifest is written in place rather than tmp+rename, and it contains `seconds`, so its sha is not reproducible across runs. Use the rows digest (Reference) for cross-machine identity.
  - The generator script's own sha256 and the sealed-module hashes are not recorded.
- **m5. Scheduling.** Generating the four caches on the 5090 host with 8 workers while batch M is still training takes CPU away from launch-bound trainers. That can push batch M into its deadline. Through `epochs_completed == 72` this then flips the chart rule, and it also degrades v372 itself. Generate with fewer workers or with `nice`, or before or after batch M.
- **m6. Cosmetic.** `--smoke` does not truncate the cache path's train set, and `--epochs 0` still loads the cache. The test suite never runs `--device cuda`; this is covered by the M1 preflight.

## Checks that pass

**1. Equivalence: PASS.**
- **Targets.** `stream_targets(own, reached[k])` equals `bank.forest_targets(inputs, k)[0].reshape(n, -1)`, exactly and as int8, on the 30 test cases and on the 42+2 spot-checked rows per world.
  - Why it is exact: `reached` always contains the diagonal (`np.eye`), and -1 is the minimum of the ternary alphabet. So filling the unreached sources with -1 and taking `amax` gives exactly `own[s].max(0)`.
- **DT.** The per-update truncated DT (T=4) equals the load-time DT exactly. It is the same function on the same int8 tensor.
- **Full update.** One complete update gives the bit-identical loss on both paths (CPU, N=16: 2.318997859954834; N=8: 2.947779893875122).
- **Inputs.**
  - Edges: list of `(int, int, float)` tuples, sorted, equal to `select_forest`'s, with empty forest handled (`reshape(-1, 3)` gives `[]`).
  - Positions: float64, equal.
  - `own_t`: int8, contiguous.
  - `reached`: `np.bool_` becomes `torch.bool`, and only the `torch.where` condition reads it.
  - `torch.full((), -1, int8)` broadcasts correctly.
  - int8 `amax` is supported on CUDA (`max_values` dispatch covers `AT_ALL_TYPES`, which includes kChar). It is not yet exercised on GPU (see M1).
- **Pinned memory and `non_blocking`.** The source host tensors are persistent, never written after pinning, and never freed. The H2D copies and their consumers run on the same stream. There is no race.
- **`del sc['private']`.** It runs after the only training-side read (`n = ...`, L528). Training items are never evaluated. The training loop reads only `own_t`, `tgt`, `edges`, `positions_m` and `'dt' in sc`; stream items lack `dt`, so L646 computes it.
- **Bypassed keys.** `sc.get('host')` is absent for stream items, so they correctly take the stream branch. `train[si]` is never mutated (`dict(sc, ...)` copy).

**2. Split hygiene: PASS (with the M2 gaps).**
- Both manifests contain only `split == 'TRAIN'`, and the rows equal `scene_plan('TRAIN')`.
- TEST cannot be generated: `world_spec` and `scene_plan` raise, and the index holds no TEST rows.
- VALIDATION still comes from in-memory `load()` and never enters the cache.
- 0 house-id overlap between the 48 VAL-panel houses and the 14,400 TRAIN houses. Exact contour duplicates share a split by construction.
- TRAIN per stratum is 5,383 / 8,533 / 6,450, all ≥ 4,800, so there is no `IndexError` and no wrap-around.
- Near duplicates: see M3.

**3. Matched control: PASS (with M1 and m1).**
- `total_updates = epochs × len(train)` is 6 × 7,200 = 3 × 14,400 = 72 × 600 = 43,200.
- The LambdaLR (200-update warm-up, cosine to 0.05 over `total_updates`), AdamW lr 3e-4 with wd 0, clip 5.0, and HOPS/WEIGHTS are untouched.
- Model init uses `fork_rng` + `manual_seed(seed)`, so it is identical to the same-seed control. `order_rng` uses the same seed; the different permutation length is inherent to the design.
- The CLI flags in `chain_day4_N.sh` equal the v367/v372 lines apart from `--epochs`, `--val-every` and `--train-cache`; `--dt-trunc`, `--lr`, `--precision`, `--consistency` and `--wire` keep their defaults.
- Stratum and N balance are identical: 1/3 per stratum, N=8/16 50/50 (7,200 / 7,200 at k4800).
- Indices 0-199 per stratum are the control's 600 houses, because the per-stratum list order is a hash of the house id and does not depend on K.

**4. Determinism: PASS.**
- Every RNG is a `default_rng` seeded from `sha256(world_id + ':poses:N')`. There is no global RNG, `lru_cache` or ordering dependency.
- Multiprocessing output (fork, `imap`, order-preserving) equals in-process regeneration on all spot-checked rows.
- File names `r{rooms}-i{index:05d}.npz` are unique, because the plan has a single size and N is set by parity.

## Reference digests (local regeneration, index sha256 `7b4730cc…`, data modules as in the table above)

Rows digest = sha256 over the rows in order of `json.dumps([file, spec, private_sha256], sort_keys=True)`. Content digest = the same over the `array_hash` of every array in every npz.

| cache | rows | rows digest | content digest (all arrays) |
|---|---|---|---|
| houseexpo k4800 | 14,400 | `96a86d4d52bcf564a2809ba41fa3bd59d791ba24bb04b8b7558f39b558b81cb2` | `c23a833dd42328f472ed731be5e79e6fdab0d00e6044ad049d0b9f53805a8b13` |
| houseexpo k2400 (rows idx < 2400 of the above) | 7,200 | `ffb298b4729088d51a81f4a87836740bcaba9f7704730628cfbea258428996d7` | – |
| houseexpo_fit k4800 | 14,400 | `a9c186d0ed23280ebf7f6829983be606946d9c0e0d982e26ebefe89900ade261` | `ee62a7ca7c57de6e141f5bfe04da9c6a55b9deaf92317043993aec6f57ba90e6` |
| houseexpo_fit k2400 | 7,200 | `7275dc0b5974ada7a82eb76be8819af268551a5372d3397ef89bc6273f7e9a1d` | – |

The first 8 `private_sha256` values of the houseexpo cache (rooms 1, idx 0-7) should equal v367 `run.json['train_private_sha256']`:
`4c358df8… 34b51248… 583aeece… 4f6f2ba8… a0d4dfff… 26d387d7… 1a4407b0… f2234bc9…`
(full values in `review375/cache-houseexpo-k4800/manifest.json`). For houseexpo_fit, compare against v372 instead: `531701c5… 368d47be… 140647fe… db3d9f76… d85b3140… 72be1d87… 7517c1b4… 591bf4b3…`.
