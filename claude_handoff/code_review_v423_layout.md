# Code review: batch 3S-C (train_3d_v423 --layout 16x16x1), 2026-10-25 04:08-04:24 UTC

**Verdict: LAUNCH OK WITH CHANGES.** The trainer patch is correct. The block layout is the only behavioural change, and `--layout 8x8x4` is v420 bit-for-bit. With 16x16x1, every use of BX/BY/BZ/D is consistent: the encoder unfold, the coverage pooling and the decoder reshape all use the same x-major order. I checked this on an off-diagonal column, which the test does not guarantee (S8). The chain carries the 3S-W fixes and no wait pattern matches itself.

Two things must change before launch:
- M1: the verdict rule labels a clearly *negative* layout effect "the layout is not a lever".
- M2: D8-5 is not yet in the OVERNIGHT_REPORT, although the chain cites it as the pre-registration.

**Scope.** Local files:
- `scripts/claude_v342/{train_3d_v420,train_3d_v423,test_3d_v423,diag_3sa_v420,diag_interior_v420,diag_storey_v423,diag_interior_v423}.py`
- `chain_day6_3SW.sh`, `chain_day6_3SC.sh`
- the scratchpad `make_v423.py`, `make_chain_3sc.py`, `d8_5.md`, `d8_234.md`
- OVERNIGHT_REPORT D4-6 and D8-0..D8-4, DESIGN_3D_STOREY_V420.md section 4, code_review_v422_decoder.md

On uk-linux I ran read-only commands only:
- `ps`, `nvidia-smi` and `ls`;
- `sha256sum` of every pinned file;
- a Python snippet that ran the chain's `load()` on the control's JSON files and logs and printed the result, writing nothing.

Locally:
- `bash -n` on the chain, and in-memory compilation of its Python heredocs;
- CPU-only tensor checks with `python -B` on random maps, writing no files;
- a coverage count on the 3S-A diag slices already in the scratchpad.

I launched nothing. The only file I wrote is this review.

## Must fix before launch

**M1. A clearly negative result is labelled "the layout is not a lever", and the null label over-claims.**
- Where: chain_day6_3SC.sh:129. Also :17 (header), :85 (lease `expectation`), d8_5.md:20 (criterion) and :31 (next step).
- The null branch is `max(d("R8_surface_f1")) < .01 and max(d("R8_var_occ_f1")) < .02`. It has no lower bound.
- Failure scenario:
  - Suppose 16x16x1 is clearly worse, say Δvar −6 pp and Δsurface −2 pp on both seeds. The chain writes "the layout is not a lever".
  - D8-5:31 then sends the search to "encoder and training length".
  - But that outcome means the layout matters a lot, and 8x8x4 is the better one.
- Unlike decoder width (3S-W), a negative effect is plausible here:
  - Each column's single 8-bit scalar must now also carry the scene's ceiling height. That height is U(2.4, 3.0) m, a range of 6 voxels, and in v420 the top layer could carry it alone. Slabs are about 41 % of the R8 surface target (3S-W review M2).
  - The same scalar must also carry the furniture's vertical structure: table tops over free space, wall cabinets, lintels.
  - The decoder's horizontal context shrinks (S4).
- "Not a lever" also generalises from one alternative layout to all layouts. 3S-W worded its null narrowly ("加宽 2 倍的解码器不是杠杆"), and this one should be equally narrow.
- Fix: change it before launch, because it is the pre-registered reading.
  ```python
  if min(d("R8_var_occ_f1")) >= .03 and min(d("R8_surface_f1")) >= .01: verdict = "the code layout is a lever (confirmed): 16x16x1 better than 8x8x4"
  elif max(d("R8_var_occ_f1")) <= -.02 or max(d("R8_surface_f1")) <= -.01: verdict = "16x16x1 is worse than 8x8x4 (the layout matters; 16x16x1 rejected)"
  elif max(d("R8_surface_f1")) < .01 and max(d("R8_var_occ_f1")) < .02: verdict = "16x16x1 is not better than 8x8x4 (no layout effect detected)"
  else: verdict = "ambiguous"
  ```
  - `max(...) <= -x` means both seeds are at or below −x. The thresholds mirror the null thresholds.
  - With the "worse" branch placed before the null, the null cannot fire on a large negative effect.
  - Mirror the change in d8_5.md:19-21, in the next steps at :29-31 (add a "worse" branch: 8x8x4 stays the baseline), and in the chain at :15-18 and :85.

**M2. D8-5 is not in the OVERNIGHT_REPORT yet, although the chain says it is.**
- Where: chain_day6_3SC.sh:2-3 ("Pre-registered in OVERNIGHT REPORT D8-5").
- Local `claude_handoff/OVERNIGHT_REPORT_20261021.md` (mtime 04:04 UTC) ends at D8-4 (line 960). `grep D8-5` finds nothing.
- The only copy of D8-5 is the scratchpad file `d8_5.md`, written at 04:07 UTC.
- Failure: if the chain is queued as it stands, the record claims a pre-registration that was not in the report at launch. Adding it after the fact is a post-hoc edit.
- Fix:
  1. Append the revised D8-5 to the report before deploying the chain. It should include M1 and whichever S items you adopt, and the real revision time from `date -u`.
  2. Change the chain header to "written 04:07 UTC, revised hh:mm UTC after code_review_v423_layout.md".
  3. Only the chain sha changes, unless you also take S1.

## Should fix or note

**S1. A wrong `--trainer` flag gives a silent mis-configuration in the diags.**
- Where: diag_storey_v423.py:19 and :36-41; diag_interior_v423.py:16 and :40-45.
- The state_dict keys and shapes are identical in both layouts. So a 16x16x1 checkpoint loads into an 8x8x4 model without any error.
- The layout is set only `if hasattr(T, 'LAYOUTS')`. With the default `--trainer train_3d_v420`, or with `train_3d_v422`, a 3S-C checkpoint is scored as 8x8x4. The numbers are wrong and nothing fails.
- The chain does pass `--trainer train_3d_v423` (:94-95), so the chain itself is correct. The risk is any later ad-hoc re-run.
- The docstring says "the same script scores v420 / v422 / v423 checkpoints". That holds only with the matching `--trainer`. A v422 checkpoint under `--trainer train_3d_v423` hits the width assert, which fails loudly, and that is fine.
- Fix:
  1. After line 40 (storey) and line 44 (interior), add `assert list(ck['meta'].get('blocks', [8, 8, 4])) == [getattr(T, 'BX', 8), getattr(T, 'BY', 8), getattr(T, 'BZ', 4)], (ck['meta'].get('blocks'), args.trainer)`.
  2. Re-upload, and re-pin GSHA/ISHA (chain :41-42).
  3. 3S-W does not pin these files.

**S2. The diag's 16x16x1 path has never run on a real checkpoint.**
- Where: chain :92-97; test_3d_v423.py:66-71.
- Test part C only greps the source text. For 3S-W, review M1 required a dry run on a probe checkpoint; no v423 checkpoint exists yet.
- My trace says the path works: meta `blocks` (a list) → module globals → model build → strict `load_state_dict` → `T.rollout_max` with D = 256. But D8-5 calls the interior-wall recall "the most specific readout".
- Fix (cheap):
  - After the first preflight passes FIT (:79), run `timeout -k 30 900 $P scripts/diag_interior_v423.py --source $S --run $OUT/pf-$TS2-s$seed --trainer train_3d_v423 2>&1 | tail -1 > $OUT/pf-$TS2-s$seed-diag-interior.json` once. The preflight directory already has `checkpoints/final.pt`.
  - This runs before START, so outside the lease, and takes about 2-3 min of GPU. Log a failure; do not abort on it.
- Alternative: accept the risk. A report-only diag can be re-run later into a new directory without touching history.
- Also call `check_shas` once more just before the diag loop (:92). At the moment the diag pins are checked only before training.

**S3. The coverage-mask granularity is part of the variable, but it opens mechanisms other than the one D8-5 proposes.**
- Where: train_3d_v423.py:128 (mask per coordinate), :154 (decoder reads log(1/P)), :197-198 and :208 (join per coordinate).
- What I measured on the 6 own maps (target at K0) saved by the 3S-A diag (`scratchpad/diag3sa/slices-*-K0.npz`):
  - Mask fraction: 8x8x4 .27-.57 (mean .38), 16x16x1 .21-.43 (mean .29), a ratio of about .76. The control's mean over 24 VAL storeys × 8 agents is .311 (diag per_scene).
  - In v420, a covered L/8 column has 3.86-3.95 of its 4 z-blocks covered. Vertical coverage therefore carries almost no information, which supports the 2.5-D premise for coverage.
- What the layout changes through the mask:
  1. **Content density.** Each covered column gets its own scalar. This is the D8-5 hypothesis.
  2. **Outline resolution.** The decoder reads log(1/P), so the known/unknown outline it receives is twice as fine. That alone can raise boundary-wall recall (boundary walls are .71 of the R8 wall target, and D8-3 already explains them by the outline), the known-BCE and the FPR, without any change in what mu carries.
  3. **Join granularity.** The max-precision join picks one winning agent per coordinate. Smaller blocks let different agents win smaller areas, which can shrink the −5 pp fusion gap independently of (1).
  4. **Fewer evidence-carrying coordinates per agent.** About .76× in the sample, e.g. 59 vs 82 of 256.
- The mask is defined per coordinate, so it cannot be separated from the layout. It is part of the variable, not a confound for "the layout is a lever".
- It *is* a confound for the mechanism D8-5 states: twice the horizontal resolution lets the code carry interior walls.
- Fix, before launch:
  - Pre-register a mechanism reading on the readout D8-5 already names. "Δ K0 interior-wall recall ≥ +5 pp on both seeds → content mechanism supported; < +2 pp on both → any gain came through outline or join granularity."
  - Use K0, not R8. The control's seed spread is 2.6 pp at K0 (.095 / .121) but 15 pp at R8 (.092 / .246).
  - Also report:
    - Δ K0 against Δ R8 (the join-granularity effect);
    - Δ boundary-wall recall (the outline effect);
    - the mean mask_fraction and q_mean of both arms. Both are in diag.json per_scene; add the means to `diag_arm_vs_control` at :143.

**S4. Defining the trunk and decoder on the code grid changes their effective architecture along with the layout.**
- Where: train_3d_v423.py:95-115 and :136-159.
- **Trunk on 4x4x16 patches.**
  - Conv outputs are (4,4,16) → (4,4,16) → (2,2,8) → (1,1,4) → (1,1,2). For v420 they are (8,8,4) → … → (1,1,1). I checked both.
  - This is sensible: the receptive field is 17 cells, so it covers the whole column, and the activation counts per layer are equal except the last (2 vs 1).
  - In v420 the final map is 1x1x1, so mean == max. I checked this: v420's `proj` sees a duplicated vector. v423 pools over 2 z positions, so mean ≠ max.
  - The last layer in v423 sees only the centre xy tap.
  - All benign; record it.
- **Decoder.**
  - `pre` (two 3x3x3 convs) now runs on a 16x16x1 grid. Two thirds of its z taps only ever see zero padding, so those weights are dead.
  - Its reach in metres halves: ±2 cells × L/16 instead of ±2 × L/8.
  - Roughly, the theoretical horizontal receptive field of an output voxel falls from about 1.0 L to about 0.66 L. The number of code coordinates that can influence a voxel falls from essentially all 256 to about 110.
  - The step 16x16x1 → 16x16x8 is the identity in x and y and a pure copy along z (`F.interpolate` checked numerically). Height information comes only from the public z plane at `lift` and the later stages. That is as intended.
- None of this is a bug; it is part of "layout" as implemented. But D8-5:5-9 ("唯一变量 / 不变的东西") does not mention it, and a null or negative result could come from the shorter decoder context.
- Fix: one line in D8-5, and mention it in the null and "worse" readings.

**S5. The gate is sound, but it logs too little.**
- Where: chain :52-54.
- Logic:
  - The chain skips only if the verdict starts with "decoder width is a lever", the exact string of chain_day6_3SW.sh:126.
  - A missing analysis.json, a missing key, "INCONCLUSIVE (…)", "ambiguous" or "not limited by decoder width x2" all lead to run.
  - This matches D8-4's next steps (lever → W = 2 budget; not a lever or ambiguous → 3S-C at W = 1) and D8-5:17.
- Soundness: 3S-C's control is 3S-A in every branch, so its reading does not depend on 3S-W. The gate is a priority rule, not a logical dependency. Fine.
- The analysis.json file is written inside the 3S-W chain before that chain exits, and the gate runs only after the chain has gone. There is no race.
- The problem: the log records only `run` / `skip`, so a missing file and an INCONCLUSIVE 3S-W look the same.
- Fix: log the verdict string (or "missing") and the sha of `runs/v422-decoder/analysis.json`, and also write it into the lease JSON.

**S6. The waits have no self-match, but three gaps remain. Part of 3S-W review S4 was never implemented.**
- Where: chain :49-51, :73, :88.
- No self-match:
  - `chain_day6_3SW[.]sh` and `chain_day6_3S[.]sh` do not match `chain_day6_3SC.sh`.
  - `^$P scripts/train_` is anchored, and pgrep excludes itself.
  - The running 3S-W chain is `bash scripts/chain_day6_3SW.sh` (pid 720119, ppid 1), and the pattern matches it. Its `$P scripts/train_3d_v422.py` child matches too. GPU app 720723 is using 7.8 GB.
- Gap (a): the pattern also matches any process whose command line contains that string.
  - Examples: `less scripts/chain_day6_3SW.sh`, or `tail -f` on it, would block 3S-C until closed.
  - A launcher such as `bash -c '…chain_day6_3SW.sh…; bash scripts/chain_day6_3SC.sh'` would make 3S-C wait on its own parent forever.
  - Fix: launch with a plain `nohup setsid bash scripts/chain_day6_3SC.sh > runs/chain_day6_3SC.out 2>&1 < /dev/null &`. Optionally add `while kill -0 720119 2>/dev/null; do sleep 120; done`.
- Gap (b): the GPU-free check exists only inside `busy()`. Neither chain re-checks before each preflight or main run, although 3S-W review S4 asked for exactly that and D8-4 says it was done.
  - Fix: put `[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { log "ABORT INCONCLUSIVE (GPU not free): seed $seed"; continue; }` before :73 and before :88.
- Gap (c): local `chain_day6_3SB.sh:22` (not deployed) waits only for chain_day6_3S.sh, the v421 cache and `train_` processes.
  - If 3SB is ever queued, it can start in the ~1 s gap between a 3S-C preflight and its main run, or during the 3S-C diags.
  - Fix: add chain_day6_3SW/3SC to 3SB's wait before deploying it, and add `chain_day6_3SB[.]sh` to 3S-C's `busy()`.
- Note: there is no cap on the wait. A GPU process that never exits blocks the chain forever. This harms no history.

**S7. The analysis lacks cheap one-variable asserts.**
- Where: chain :103-121.
- Layout does not affect any parameter shape, so the arm's `initial_weights_sha256` must equal the control's for the same seed. The test shows identical tensors, and my CPU check gives the same hash for both layouts.
- The control's values are `85061f9f2a1a2ad4…` (342101) and `d18249febbdd6632…` (342202).
- Fix: also return `m` from `load()`, and at :119 assert three things:
  1. the initial-weights hashes are equal;
  2. both `cache.manifest_sha256` equal `864b1fe3…`;
  3. both `parameters` are `{'trunk': 610368, 'head': 16770, 'decoder': 468376}`.

  A mismatch, for example after a torch upgrade, would be informative rather than silent.

**S8. Test part B's locality check catches a block-order bug only if the chosen column is off-diagonal, and the decoder order is untested.**
- Where: test_3d_v423.py:49-55.
- The test uses `k` = the first covered coordinate of agent 0. If ix == iy, a trunk that ordered blocks y-major would still pass. The mask assert (:48) checks only the head's own pooling.
- My local CPU check, on a random own map with column (3, 11), i.e. k = 59 (the transposed index would be 179):
  - Perturbing only that column changes only coordinate 59.
  - A single known voxel in that column sets only mask[59].
  - Perturbing post coordinate 59 changes the decoder's `pre` output only in the 5x5 window centred on (3, 11). For column (12, 2) the window is centred on (12, 2).
- So the code is right; only the test is weak.
- Fix: choose `k` explicitly off-diagonal (the first covered k with ix ≠ iy, and assert one exists), and add the decoder-order check. This changes only the test's sha, which nothing pins.

**S9. D8-5's motivation leaves out the counter-evidence from its own D8-3, and also the strongest supporting evidence.**
- Where: d8_5.md:4.
- Counter-evidence:
  - D8-5 says "户型本身只剩 8×8，和 2D 线同一密度". D8-3 used the same fact the other way.
  - The 2D line reaches K0 .92-.94 with 64 scalars on an 8x8 grid (BEST_RESULTS).
  - 8x8x4 has 4 scalars per L/8 column.
  - So D8-3 concluded that the code is not too small.
- Supporting evidence D8-5 does not cite: D4-6, the 2D ceiling protocol on a fixed 20 m chart.
  - Results: 8x8x1 .694; 4x4x4 .623 (the same 64 dimensions, with more scalars per coarser block); 16x16x1 .918.
  - D4-6's conclusion: what decides is the number of independent spatial cells, not the total dimension.
  - Caveat: that was a fixed 20 m chart. With fitted charts, 8x8 reached .93.
- My coverage count (S3) supports "the vertical is nearly redundant" for coverage.
- D8-3's height-band recall (.34-.57) shows the decoder does not treat walls as extrusions. That is the opening this batch tests.
- Fix: two sentences in D8-5:4.

**S10. Minor.**
- train_3d_v423.py:92 and :133: the PatchTrunk3D / Decoder3D docstrings still say "8x8x4". This is cosmetic; editing them changes the trainer sha and needs a re-pin, so skip it unless you change other things.
- chain :33: the lock is taken before `check_shas`. A failed launch (say, a file not uploaded) leaves `runs/v423-layout.lock` behind, and the operator must `rmdir` it. It is an empty directory, not history. Say so in the launch note.
- The analysis `delta` includes `decoder_width` (0.0). Cosmetic, as in 3S-W.
- D8-5:28 expects about 0.12 s/update. That is plausible: per-layer activation counts are identical apart from trunk layer 4. But no CUDA probe was run; the preflight will measure it. Expect about 2,100-2,300 s per seed against the 6,000 s FIT cap (3S-A: 2,095 s including load).

## Checked, no issue
- **Diff v420 → v423** (make_v423.py, verified by `diff`): the changes are a prepended docstring, the `LAYOUTS` table, the `global CHART_M, BX, BY, BZ` line, the `--layout` argument, and `BX, BY, BZ = LAYOUTS[args.layout]; assert BX * BY * BZ == D`. There are no `__future__` imports, so the second string literal is harmless.
- **16x16x1 consistency:**
  - Trunk block size (4, 4, 16); head pooling kernel (8, 8, 32); the divisibility asserts hold for 128,128,32.
  - unfold → permute(0,2,3,4,1,…) is x-major, `avg_pool3d(...).flatten(1)` is x-major, and `post[:, :D].reshape(b,1,BX,BY,BZ)` is x-major (S8 check).
  - `rollout_max` uses only D.
  - meta records `blocks=[16,16,1]` and `args.layout`. The layout is set at :334, before the model at :369 and the meta at :374.
  - Spawned fixture workers re-import the module with the default layout, but `_fixture` does not use it.
- **One variable at the start of training:** no parameter shape depends on the layout. Initial tensors are identical, and the CPU hashes match for both layouts. Data order (`order_rng`), cache and panels are the same. The trainer flags in COMMON and :88 reproduce the control's `run.json` args (lr, val-every 4000, val-per-stratum 8, threads 4, gen-workers 8, dims).
- **test_3d_v423 A and B are real checks.**
  - A compares state_dicts for both seeds, plus source, join and decoder logits.
  - B: the mask is compared against an independent reshape, `outside`-randomisation is checked against "the perturbation is real", and the shapes and the loss decrease are asserted.
- **Diags:**
  - v420 checkpoints (`blocks` [8,8,4]; v420/v422 trainers have no `LAYOUTS`) are configured correctly.
  - v422 checkpoints with `--trainer train_3d_v422` (`decoder_width` from meta) are correct.
  - v423 checkpoints with `--trainer train_3d_v423` are correct (but see S1).
  - The computation is identical to the control's diag (the diff is only the docstring and the config lines), so arm and control diag numbers are comparable.
  - The control's `diag-a-s*/diag.json` has all 7 keys the analysis reads, and `diag-interior-a-s*.json` has all 4.
- **Pins:** local and uk-linux shas are identical:
  - train_3d_v423 `a5540187…`
  - data module `4793a327…`
  - train_v378 `db7bcc35…`
  - diag_storey_v423 `d89ada23…`
  - diag_interior_v423 `f27dd0a3…`
  - manifest `864b1fe3…` (it also equals the control's `run.json`)

  The chain is not yet on uk-linux (local sha `05ac560f…`). `check_shas` runs before each seed and includes the manifest and both diags.
- **Guards:** the atomic lock, the never-overwrite guard (`a-s*`, `lease_s*.json`, `analysis.json`), and the control-presence check.
- **Budget and lease:**
  - The rule is identical to 3S-W, with 16,000 updates hard-coded.
  - START comes after the preflight; DE = START + 6600 and DT = START + 7150. `timeout -k 30` makes the SIGKILL land at ≤ START + 7180 < 7200.
  - Anything short of 16,000 updates gives INCONCLUSIVE.
- **Analysis on the real control:** run read-only on uk-linux. Both seeds load with `blocks` [8,8,4], `decoder_width` defaulting to 1.0, 16,000 updates, 24 VAL specs, and clip share 0/320.
- **Syntax:** `bash -n` passes, both `<<'PY'` heredocs and the gate's `-c` compile, and the unquoted lease heredoc contains only the intended `$` variables.
- **Isolation:** the TEST split is never generated (no "TEST" in any v423 file). Writes go only under `/mnt/datasets/...` (`runs/v423-layout`, the shared append-only log).
- **3S-W is not disturbed:** 3S-C adds only new files. It reads, but never writes, the data module, the k-bit wire and the manifest that 3S-W also pins.
- **Timeline:** 3S-W was at upd3950 at 04:21 UTC (0.206 s/update), so the 3S-W chain should end around 06:15-06:30 UTC. 3S-C would then take about 2 × 40 min plus diags, finishing around 08:00 UTC.
