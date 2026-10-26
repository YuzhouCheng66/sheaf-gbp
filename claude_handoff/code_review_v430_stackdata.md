# Code review: batch 3S-Y (`scripts/claude_v342/chain_day6_3SY.sh`, D8-28: the full-TRAIN cache on the saturated stack recipe)

Reviewer: Claude (independent), 2026-10-26 12:46-12:55 UTC (`date -u` on the local machine and on uk-linux agree to the second). Nothing was launched, edited or deleted; the only writes were this file and the CPU smoke test under `runs/review-smoke/3sy/` on uk-linux.

**Verdict: LAUNCH OK WITH CHANGES** - one must-fix (M1: the pre-registered early stop after cycle 1 would, in the likely case, end the batch before the saturated regime it is meant to test is ever reached; drop it, in the chain and in D8-28, before launch). Everything else checked is sound: gate, pins, budget, leases, waits, diag phase, analysis (smoke-tested on uk-linux: no crash with the arm absent, full path verified on a synthetic arm), hand-off.

## What was verified (facts, not assumptions)

- `runs/v429-stack/analysis.json` (real file): `verdict = "lever (cycles 2 x 25500)"`, `next_baseline_dir = "runs/v429-stack"`, `next_baseline_recipe = {d3, W 2.0, blocks [32,32,1], wire [8,8], learned, uniform, cache_dir .../train-r1-k1600}`. The gate at chain_day6_3SY.sh:50-60 passes on it.
- `runs/v429-stack/c2-s342101/run.json`: D 1024, raw_code_bytes 2048, layout 32x32x1, d3 W 2.0, learned, uniform, wire [8,8], lr 1e-4, updates 25,500, `warm_start` present (keys init_from / init_file_sha256 / source_updates / source_weights_sha256 / source_seed), cache k1600 with manifest sha 864b1fe3... = the pinned MSHA (line 40). `c1-s*` : lr 3e-4, 25,500, no warm start, `checkpoints/initial.pt` present (line 66 check passes). So line 68's "control is a two-cycle run" assert passes and CUPD = 25,500, CLR = 0.0003.
- k5383 manifest (real file): version 420, dims [128,128,32], per_stratum 5383, scenes 15,812, data_module sha = the pinned DSHA; sha256 ab336090... (unchanged since 3S-S: `runs/v430-data/a-s*/run.json` record the same manifest sha). Line 62's validity check passes.
- Initial-tensor identity (analysis line 176-178, 195): `runs/v430-data/a-s342101` (v430 lazy, k5383, 32x32x1, seed 342101) has `initial_weights_sha256` f62b09c3... = `runs/v429-stack/c1-s342101` (eager, k1600); seed 342202: 1e52c38f... = 1e52c38f.... So the init does not depend on the cache or on lazy loading (train_3d_v430.py:468 seeds before the model is built; :491 builds the lazy row list without consuming RNG); the "EVERY initial tensor equal" assert will hold.
- VAL panel: the 24 `final-val.json` specs of the lazy k5383 run equal the control's (`specs` assert, line 194, will hold).
- Pins: all six sha256 values at lines 34-39 match the files on uk-linux now; `train_3d_v430.py` and `diag_*_v429.py` unchanged since the previous review; the diag scripts were verified to load 32x32x1 checkpoints in that review and ran in 3S-X on exactly the control runs (runs/v429-stack/diag-a-s*).
- Budget: 3S-S measured the lazy k5383 run at 0.2000 s/update, load 45 s, val 30 s (chain_day3.log 04:09 / 05:06 UTC); 3S-X (eager, same layout) 0.2000-0.2010, load 62-63 s. Fit check (line 96): 1.05 x (45 + 25,500 x 0.200 + 90) = 5,497 s <= 6,300 s; it passes up to 0.230 s/update. Each 3S-X run took 5,190 s of training and ~86 min lease-to-lease; expect ~1.5 h per run, ~6 h for four, inside the one 7,200 s inclusive lease each (train deadline 6,600 s, total 7,150 s, `timeout -k 30` at DT, preflight `timeout -k 60 1800`).
- GPU / waits: `nvidia-smi --query-compute-apps` is empty, no chain_day6 / train_3d process exists (3S-X finished 12:35 UTC), so `busy()` (line 46) is false and the chain starts after the 60 s re-check. `busy()` waits on 3SX/3SS/3SD/3ST/3SQ/3SR/3SP/3SL/3SC/RS chains, any `$P scripts/train_*`, `$P scripts/diag_*`, `owned_ceiling` and any GPU compute app; the patterns cannot match the chain itself (`3SY` is not in the list) and `python -` heredocs are not matched. Lock (`mkdir runs/v430-stackdata.lock`, line 44) comes after `check_shas` (line 43) and before any write; `runs/v430-stackdata*` does not exist on uk-linux (never-overwrite check at line 45 passes). All writes are under /mnt/datasets (`runs/v430-stackdata`, `runs/chain_day3.log`); TEST is not touched (VAL / fresh panels only).
- run_seed (lines 86-109): UPD = CUPD = 25,500 (not the preflight's fitted count), BUD = 6,300; cycle 1 `--lr 0.0003` (the control's), cycle 2 `--init-from $OUT/c1-s$seed/checkpoints/final.pt --lr 1e-4`; `--val-every 6375`; `check_shas` re-run before every run (line 87) and before the diag phase (line 129). The early-stop heredoc (lines 111-125) correctly reads the ARM's cycle 1 from `runs/v430-stackdata/c1-s*` (the path edit from the 3S-X template is correct) against `runs/v429-stack/c1-s*`, and treats a deadline-cut cycle 1 as "incomplete".
- Diag phase (lines 130-140): `a-s* -> c2-s*` (or `c1-s*` on early stop), diag_storey / diag_interior (GPU, 1,800 s timeouts) and diag_veto (CPU) with `--trainer train_3d_v430`, on the arm and on `runs/v429-stack/c2-s*`.
- Analysis heredoc (lines 141-246), smoke-tested on uk-linux (CPU, `nice -n 19`, `CUDA_VISIBLE_DEVICES=`, OUT redirected to `runs/review-smoke/3sy/{absent,synth,synth1}`, control = the real `runs/v429-stack` c1/c2):
  - A, arm absent: `INCONCLUSIVE ('incomplete seeds')`, analysis.json written, exit 0 - no crash.
  - B, synthetic two-cycle arm = the control's own c1/c2 files with run.json cache fields rewritten to k5383 / 15,812 / lazy: all asserts pass (two-cycle control, recipe, cycle-1 recipe/count, warm-start chain, "the only variable is the cache", VAL panel, params + identity on the real initial.pt tensors); deltas 0.0 -> `not better (cycles 2 x 25500)`, guards all True, mechanism string and fusion gap produced, hand-off `next_baseline_dir = runs/v429-stack`, prefix `c2-s`, updates 51,000, cycles {2 x 25,500}, recipe incl. cache_dir (k1600 when not adopted; the arm's k5383 when adopted).
  - C, cycle-1-only arm: the `cycles == 1` branch (lines 210-212) yields `not better (early stop after cycle 1 ...)`; reachable and crash-free.
  - Verdict thresholds match D8-28 exactly: lever = min dVar >= +.03 AND min dSurf >= +.02; worse = max dSurf <= -.01 OR max dVar <= -.02; not better = max dSurf < .01 AND max dVar < .02; else ambiguous; guards FPR +.003 / slab -.02 / K0 var -.02; adoption needs all three; fusion-gap widening > 2 pp flagged on either seed; final targets per seed; bytes 2,048 recorded.
- Pre-registration D8-28 (OVERNIGHT_REPORT line 1528, "12:45 UTC") vs the chain header ("12:46 UTC", file mtime 12:46): consistent with the real clock (12:46:52 UTC at review start, no training started); one variable, control, thresholds, guards, mechanism readings, budget (4 leases, ~1.5 h) and the disclosure that the data order differs by construction (different manifest) are all stated in both.

## Must fix

### M1 - the early stop after cycle 1 defeats the batch's own question (chain_day6_3SY.sh:14, :104, :111-127, :210-212; OVERNIGHT_REPORT D8-28 line 1532 "周期 1 后对对照周期 1 的早停规则同 3S-X")

Scenario. The question is whether data quantity helps once training saturates - i.e. in cycle 2, where the control's curve is flat (3S-X c2 trajectory: .8736 -> .8721 -> .8765 from 12,750 updates on). Cycle 1 of the arm sees each of the 15,812 houses ~1.6x (the control ~5.4x), and 3S-S already showed +/-0.8 pp at equal updates. The pre-registered early stop (line 122: stop if both seeds show dR8 surface < +1 pp AND dR8 var < +2 pp after cycle 1, vs the control's c1 .8595/.858 surface, .7784/.7745 var) therefore fires in precisely the outcome the hypothesis predicts (equal at cycle 1, gain in cycle 2). Expected result: two leases, `batch 3S-Y: cycle 2 not run (stop ...)`, verdict `not better (early stop after cycle 1 vs the control's cycle 1) (cycles 1 x 25500)` - a true statement about cycle 1 that says nothing about saturation, which is what D8-28 asks. That rule was right for 3S-X (a lever expected to show at cycle 1) and is wrong here.

Fix (before launch, with a fresh `date -u` timestamp in the header and in D8-28, since it changes the pre-registration):
1. Line 122: delete the `elif all(d[0] < .01 and d[1] < .02 ...): print("stop", ...)` branch, so cycle 2 runs whenever both cycle-1 runs are complete (`incomplete` still skips it). Keep the cycle-1 deltas in the log line (line 126) as a report-only reading.
2. Lines 210-212: with no early stop, a missing cycle 2 is always `INCONCLUSIVE (cycle 2 missing)`; drop the `es` computation (or keep it report-only under a different key).
3. Line 14 and the lease `expectation` string at line 104 ("early stop after cycle 1 if ..."): say "no early stop: cycle 2 always runs (the question is the saturated regime); cycle-1 deltas reported only".
4. D8-28: replace "周期 1 后对对照周期 1 的早停规则同 3S-X" with the same statement; budget stays 4 leases (~6 h), as already pre-registered.
If the user prefers to keep the early stop to save two leases, then at minimum the verdict label and D8-28 must say "not better at cycle 1 (saturation untested)" so the result cannot be read as an answer to the saturation question.

## Should fix / notes

### S1 - the arm's manifest is recorded but not pinned (chain_day6_3SY.sh:62-63)
`check_shas` pins the k1600 manifest (MSHA) but the k5383 manifest is only validity-checked (version/dims/per_stratum/scenes) and its sha put in the lease. 3S-S's arm ran on manifest sha ab3360905fcf9f7963b819d52921e466d5b640b3552fec50559dd837740830a0, which is what the file has now. Add `[ "$AMSHA" = "ab33609...0830a0" ] || { log "ABORT: full-TRAIN manifest differs from 3S-S's"; exit 2; }` after line 63 so a regenerated cache cannot slip in silently between the gate and the runs.

### S2 - the lock is taken before the gate (chain_day6_3SY.sh:44 vs :50-70)
If any gate check aborts (exit 2), `runs/v430-stackdata.lock` stays and a corrected relaunch aborts with "already launched" until someone `rmdir`s it. Inherited from the template; either move `mkdir $OUT.lock` after line 70, or document the `rmdir runs/v430-stackdata.lock` step. Also: the script is not yet on uk-linux (`scripts/chain_day6_3SY.sh` absent); copy it with LF endings (the generator writes LF) and launch from the work dir (`cd` at line 20 is unconditional, but line 30's self-sha and the lock path assume it).

### S3 - cosmetic log / lease fields for cycle 1 (chain_day6_3SY.sh:89, :102, :106)
`INIT="--lr 0.0003"` is non-empty for cycle 1, so `${INIT:+, warm start lr 1e-4}` prints "warm start lr 1e-4" for cycle 1 as well (visible in 3S-X's log: "start c1 seed 342101 (... warm start lr 1e-4)"), and the lease's `"warm_start"` field holds `--lr 0.0003` for cycle 1. Use `[ "$CYC" = 2 ]` for the suffix and write `"warm_start": null` for cycle 1. No effect on the verdict (the analysis reads run.json, not the lease).

### S4 - the shell recipe equality is tautological (chain_day6_3SY.sh:67-69)
DEC/WID/PREC/KW (line 67) and CDEC/CWID/CPREC/CKW (line 68) are both read from `runs/v429-stack/c2-s342101/run.json`, so line 69 can never fail. The real guards are the gate's `blocks == [32,32,1]` on analysis.json and the analysis asserts (lines 185, 193, 195), which are sufficient; optionally read line 67 from `next_baseline_recipe` instead so the two sources are actually cross-checked.

### S5 - hand-off recipe lacks a lazy flag (chain_day6_3SY.sh:242)
On adoption `next_baseline_recipe.cache_dir` becomes `.../train-r1-k5383`; the follow-up chain (3S-Z) must derive `--cache-lazy` from the directory name, as make_chain_3SY.py did by hand. Add `cache_lazy` (from run.json) to the recipe dict so the next generator can read it.

### S6 - control diagnostics recomputed (chain_day6_3SY.sh:137-139)
`runs/v429-stack/diag-a-s*`, `diag-interior-a-s*.json`, `veto-a-s*.json` are already the diagnostics of exactly `runs/v429-stack/c2-s*` (the control). The chain recomputes them under `runs/v430-stackdata/diag-control-s*` (~12 min of GPU after training). Harmless and self-contained; could be symlinked instead.

### S7 - timing expectation for the report
Launch at ~13:00 UTC -> c1-s342101 done ~14:30, c1-s342202 ~16:00, c2-s342101 ~17:30, c2-s342202 ~19:00, diag + analysis ~19:15 UTC (the lazy rate is the same as the eager one, 0.200 s/update; the fit check has margin to 0.230). If the preflight ever reports `nan`, the run aborts INCONCLUSIVE before the lease is written (line 94), as designed.

## Not findings (checked and fine)
- The lazy loader's RAM footprint: 125 GB RAM, 72 GB page cache; 3S-S ran two lazy 16k runs without incident.
- `rate()` (lines 73-84) and the preflight regexes are unchanged from the chains that ran today.
- `load()` requires `f["updates"] == m["updates"]` and >= 24,000 for every run it reads (arm c1/c2 and control c1/c2), so a deadline-cut run can only produce INCONCLUSIVE, never a verdict.
- Guards and mechanism are computed inside try/except and cannot crash the verdict; `absolute` and `fusion_gap` are report-only.
- The 3S-X review's M1-M3 / S1-S7 fixes are all carried over (v430 trainer for the diags, cycle-1-vs-cycle-1 reading when cycle 2 is missing, 6,300 s budget, two-cycle hand-off, actual cache in the lease, control lr for cycle 1, three guards for adoption, data-order disclosure).
