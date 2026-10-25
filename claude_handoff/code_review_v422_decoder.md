# Code review: batch 3S-W (train_3d_v422 --decoder-width 2.0), 2026-10-25 ~03:56 UTC

**Verdict: LAUNCH OK WITH CHANGES.** The trainer patch is correct. Decoder width is the only behavioural change, W = 1.0 is v420, and trunk/head init is unchanged. The budget fits with a wide margin. Before launch, fix three things: the remote diag script is stale (M1), the negative verdict rule ignores the metric the hypothesis is about (M2), and the pre-registration timestamps are later than the real clock (M3).

Scope: local `scripts/claude_v342/{train_3d_v420,train_3d_v422,test_3d_v422,diag_3sa_v420}.py` and `chain_day6_3SW.sh`, plus the D8-2..D8-4 text. On uk-linux I only read files: shas, logs, run.json/final.json/diag.json, the probe run, and the process list. I also did a CPU-only, read-only tensor comparison of two `initial.pt` files.

## Must fix before launch

**M1. The remote `diag_3sa_v420.py` is the old version, so the chain's report-only diag step will fail. The script is also not pinned.**
- Where: chain_day6_3SW.sh:80 and :31-36.
- Remote `scripts/diag_3sa_v420.py` has sha `91069a4f…` (03:34 UTC). It is the version that produced the control's `diag-a-s*`, and it has no `--trainer` or `--decoder-width`. The local copy has sha `2c69149b…` (03:44) and adds them (local diag_3sa_v420.py:16, :21-23, :33). The chain is not on uk-linux yet.
- Failure: if only the chain is uploaded, line 80 exits with argparse "unrecognized arguments". The pre-registered D8-3 breakdown on the new checkpoints (D8-4 line 64) is then missing. The verdict is not affected.
- Even after upload, `check_shas` does not pin the diag, so nothing proves which diag code produced the arm's numbers.
- I diffed the two versions. With default arguments the new one is behaviourally identical for the control (it only adds the importlib import, the width override and the meta assertion). So the control diag outputs stay comparable.
- Fix:
  1. Upload the local diag.
  2. Add `sha256sum scripts/diag_3sa_v420.py` = `2c69149b3877e685988ded2f1b2215e3e29d96be0a8abc53a7146dc5f490a0ef` to `check_shas`.
  3. Dry-run it once on the existing W = 2 probe checkpoint (`runs/v422-probe/pf-w2-s342101`, which has `final.pt`) with `--trainer train_3d_v422 --decoder-width 2.0`, writing to a scratch out dir. This is about 2 min of GPU. It exercises the only untested path: W = 2 model build, meta check and strict `load_state_dict`.

**M2. The "not decoder-limited" rule uses only Δ R8 surface F1. That metric is diluted by slabs, while the hypothesis concerns walls, so the rule can return a negative verdict even when walls improve clearly.**
- Where: d8_234.md:61, chain_day6_3SW.sh:107 and :72.
- From the reference predictors on this panel:
  - Slab-only R8 surface F1 .585 = 2s/(1+s), so slabs are s ≈ .41 of the R8 target surface.
  - Furniture is .18, so walls and lintels are ≈ .41.
  - A wall-recall gain of +2-3 pp therefore moves R8 surface F1 by less than 1 pp. The variable-structure F1 (no slabs) carries the wall signal.
- I ran the chain's analysis heredoc in memory with fake final.json files:
  - Δsurface +0.5 pp with Δvar +5 pp on both seeds returns **"not decoder-limited at this budget"**.
  - Δsurface +0.8 pp with Δvar +2.5 pp returns the same.
  - So the batch can label as negative exactly the wall-specific gain that D8-3 predicts.
- This can only be amended before launch.
- Fix: the negative label should require Δ R8 surface < +1 pp **and** Δ R8 var < +2 pp on both seeds; anything else is ambiguous. +2 pp is about 1.25 × SD(Δvar), using the 3S-A var seed spread of 1.6 pp. Change line 107 to `elif max(d("R8_surface_f1")) < .01 and max(d("R8_var_occ_f1")) < .02:`, and change the D8-4 text and the lease `expectation` string to match.

**M3. The pre-registration and chain header timestamps are later than the real clock.**
- Where: d8_234.md:50 ("写于发射前，2026-10-25 04:15 UTC"), chain_day6_3SW.sh:2 ("written ~04:00 UTC") and :19 ("Probe 04:05 UTC").
- uk-linux and local `date -u` agree. It was 03:56 UTC when I checked; d8_234.md mtime is 03:46 UTC; the chain is 03:45; the probe log is 03:43 (run 03:40-03:43).
- Failure: if launch happens before 04:15, the record says the pre-registration was written after launch. Fixing it after launch would mean editing a pre-registration post hoc.
- Fix: take the real times from file mtimes or `date -u` and correct all three before launch.

## Should fix or note

**S1. The D8-3 reading over-claims that the bottleneck is the decoder.**
- Where: d8_234.md:45-47, :40.
- "墙的信息很可能不是码装不下，而是解码器画不出来" and "解码器容量（证据最直接）" do not follow from the evidence:
  - Low wall recall even at 2.1-3.2 m argues, at most, against code capacity.
  - Nothing in D8-3 tells the encoder (block-local trunk → 1 scalar per block), the decoder, and training length apart. Training length: 16k updates against 57.6k for the 2D line, and the loss was still falling (D8-2).
  - "4.5 channels per voxel" is a heuristic, not evidence. 3D kernels have 27 taps and 2D kernels 9.
- The new report-only `diag_interior_v420.py` (control, 03:50/03:51 UTC) sharpens this:
  - Interior walls, which only the code can place: K0 recall .095 / .121 and R8 .092 / .246.
  - Boundary walls, which the known/unknown outline implies: ≈ .51-.53.
  - This pattern fits an encoder/code failure as well as a decoder failure.
- W = 2 still gives 72 / 8 = 9 channels per voxel against 32 per pixel in 2D. A null result therefore supports at most "not limited by decoder width ×2", not "不是解码器的问题".
- A positive result shows decoder width is a lever; it does not show that the encoder/code is adequate.
- D8-3 also says the 56-59 % of false positives far from structure are invented furniture or walls off by 2 or more. But `near` (diag:48) dilates only struct | slab, so jitter around *real* furniture also lands in that bucket.
- "3D-A / 3D-B 规则世界对解码器要求低" is a conjecture written as fact.
- Fix:
  - Reword the D8-3 reading and the negative label as above.
  - Add `diag_interior_v420.py --trainer train_3d_v422 --decoder-width 2.0` for the arm as a pinned, report-only step after line 80. Interior-wall recall is the most specific readout of the hypothesis.
  - Consider a decoder-ceiling (autodecoder) test later to separate encoder from decoder.

**S2. Power and pairing.**
- Where: d8_234.md:59-62, :65-68.
- What same-seed pairing shares and does not share:
  - Shared: trunk/head init and data order (`order_rng`).
  - Not shared: decoder init, since the shapes differ.
  - Neither trainer sets cuDNN deterministic flags, so GPU nondeterminism is uncontrolled in both arms.
- From the 3S-A spreads (surface .5 pp, var 1.6 pp, each an estimate with 1 df), SD(Δ) ≈ 0.5 pp for surface and ≈ 1.6 pp for var.
- The +3 pp var threshold is about 1.9 SD. A true +3 pp gain passes on both seeds only about 25 % of the time, so "lever confirmed" is conservative, which is fine.
- But "ambiguous → 3S-C" means a real moderate decoder gain would be dropped.
- The decoder width to use for 3S-C is not specified.
- Fix: state the width for 3S-C in D8-4. W = 1 keeps 3S-A as the single-variable control. Note the power limitation in the text.

**S3. "One variable": the coupled effects are small but should be recorded.**
- Where: train_3d_v422.py:435; D8-4 line 52.
- Wall-clock: 0.12 → 0.20-0.22 s/update. Metrics are unaffected because the schedule counts updates, and the budget fits (see the last section).
- Global gradient clipping at norm 5 over *all* parameters:
  - In 3S-A it never engaged: 0 of 320 logged updates had total norm > 5; max 4.57 and 2.35.
  - The probe's decoder gradient norm is similar (≈ 1.0 at upd50).
  - If the 4× larger decoder makes clipping engage, trunk/head steps shrink too. That is a coupling, not decoder capacity.
  - Fix: have the analysis count logged updates with total norm > 5 for both arms (they are in the logs) and report it.
- CPU contention: `gen_3d_cache_v421.py` (8 workers on 14 cores, running since 02:12 UTC) is not waited for. This affects timing only. The probe was measured under the same load.

**S4. The waits have race windows.**
- Where: chain_day6_3SW.sh:40.
- No pattern matches the chain itself. `chain_day6_3S[.]sh` does not match `3SW.sh`.
- Missed cases:
  1. Ad-hoc `bash -c 'for s …; do timeout … $P scripts/diag_…; done'` loops, such as the diag_interior loop that ran 03:48-03:51. Between iterations no `^$P scripts/diag_` process exists, so a 120 s poll can land in that gap.
  2. Any GPU user not matching the patterns.
  3. Local `chain_day6_3SB.sh:22` waits on `chain_day6_3S[.]sh`, which does not match 3SW. If 3SB is launched mid-batch, only `^$P scripts/train_` holds it, and it can slip into the roughly 1 s gaps between 3SW's preflight and main run (lines 62→75) or between seeds.
  4. There is no lock: a second 3SW instance passes the never-overwrite guard (line 38) while the first is still waiting.
- Fix:
  - Before each preflight and main run, require `nvidia-smi --query-compute-apps=pid --format=csv,noheader` to be empty.
  - Take an atomic `mkdir runs/v422-decoder.lock || exit 2` at start.
  - Add `chain_day6_3SW[.]sh` to 3SB's wait list.

**S5. The cache manifest is not re-checked before the second seed.**
- Where: chain_day6_3SW.sh:31-36 and :42-43; d8_234.md:69.
- `check_shas`, run at line 60 per seed, covers the trainer, the data module and the k-bit wire. MSHA is checked once, at line 43. D8-4 line 69 says all four pins, the manifest included, are re-checked before the second seed. That is untrue for the manifest.
- Fix: move the MSHA test into `check_shas`, along with the diag pin from M1.

**S6. The hard kill can land 10 s outside the inclusive lease.**
- Where: chain_day6_3SW.sh:68, :75.
- DT = START + 7150 and `timeout -k 60` sends SIGKILL at START + 7210. `_sigterm` only sets STOP, so a run caught in final val/fresh keeps going until the KILL.
- Expected runtime is about 3,800 s, so this will not trigger. 3S-A had the same bound.
- Fix: `-k 40`, or DT = START + 7140.

**S7. Part C of the test is a text grep, and some printed claims are not asserted.**
- Where: test_3d_v422.py:56-59, :44.
- Part C asserts that three strings are present and that `DECODER_WIDTH` occurs exactly 7 times. It does not assert that the global is set before the model is built, although line 59 prints that claim. It does not assert what run.json records.
- The decoder parameter counts are printed, not asserted.
- The claims do hold:
  - By reading, line 337 (set) comes before line 372 (build).
  - On uk-linux, the probe's run.json has `decoder_width: 2.0` and decoder parameters 1,832,704.
  - All 14 trunk/head tensors in the probe's `initial.pt` are `torch.equal` to `runs/v420-storey/a-s342101/checkpoints/initial.pt`, with no extra keys.
- Fix:
  - `assert src.index('DECODER_WIDTH = args.decoder_width') < src.index('model = SheafModel3D(')`
  - `assert (p1, p2) == (468376, 1832704)`

**S8. Analysis details.**
- Where: chain_day6_3SW.sh:66, :76, :84-99.
- (a) The analysis does not check that the arm and control VAL panels are the same scene list (the specs in `validation/final-val.json`), or that both run.json files carry manifest sha `864b1fe3…`. Both are cheap asserts.
- (b) A preflight crash is logged as "INCONCLUSIVE (budget)".
- (c) The trainer's exit code is not logged (line 76).
- (d) `delta` includes `decoder_width` (1.0) as if it were a metric.
- (e) `fresh_R8_furn_f1`, which the 3S-A analysis reported, is dropped.

## Checked, no issue
- **Diff v420 → v422:** the only changes are the docstring, `DECODER_WIDTH`, the channel scaling in `Decoder3D`, the CLI argument and `decoder_width` in meta. The training loop, loss, schedule, data and RNG use are identical.
- **W = 1.0:** gives c0 = 64 and chans [48, 36], i.e. v420.
- **Channel arithmetic:** pre 2→c0→c0; lift c0+4→c0; blocks c0+4→96→96 and 96+4→72→72; output 72→16. Recomputed parameter counts: 468,376 and 1,832,704, matching D8-4.
- **Init and RNG:** the decoder is built after trunk and head inside `fork_rng`. No torch RNG is used later. Data order comes from the numpy `order_rng`, so it is the same in both arms.
- **Test A/B:** these are real checks. A compares state_dict for both seeds, then source, join and logits at seed 342202. B pairs the same seed with v420.
- **Budget:**
  - Probe (150 updates): load 118 s; 0.20 s/update over upd50→100 and 0.22 over upd100→150.
  - The chain's `rate()` gives 0.20, load 118, val 30, so FIT = 1.05 × 3,408 = 3,578 ≤ 6,000.
  - 3S-A ran about 7 % slower than its preflight, so expect about 3,700-3,900 s per seed against the 6,600 s train deadline. The break-even rate is 0.34 s/update.
  - 16,000 updates are hard-coded; anything short of 16,000 gives INCONCLUSIVE (simulated).
- **Pins on uk-linux:** the trainer, data module and train_v378 shas match the pins, and the manifest matches the control's run.json and lease. No shadowing copy of the data module exists in the sealed source.
- **Analysis:**
  - It loads the control correctly: updates 16,000, `decoder_width` defaulting to 1.0.
  - Both heredocs compile. Verdict branches match D8-4 (simulated): exactly +3/+3 gives confirmed; one seed at +2.9 var gives ambiguous; one seed missing gives INCONCLUSIVE.
  - Absolute-criteria logic matches 3S-A.
  - The control's `diag-a-s342101` and `diag-a-s342202` exist on uk-linux, with all 7 keys read for K0 and K8.
  - Heredoc quoting is correct: `<<'PY'` is literal, and the unquoted lease JSON contains no `$` or backticks besides the intended variables.
- **TEST split:** never generated. VALIDATION and TRAIN only, in both the trainer and the diag.
