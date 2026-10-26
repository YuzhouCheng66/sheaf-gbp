# Code review: batch 3S-Z2 (`scripts/claude_v342/chain_day6_3SZ2.sh`, D8-32: 64x64x1 on the two-cycle stack)

Reviewer: Claude (independent), 2026-10-26 ~20:05-20:20 UTC. Read-only on uk-linux.

**Verdict: LAUNCH OK WITH CHANGES** - two must-fix items (M1 governing verdict, M2 eval-script pin). Gate, recipe, budget and the v420 analysis path are sound.

## Verified

- **Diff vs 3S-Y**: only the intended edits.
- **Gate (:56-65)**: 3S-Z's heredoc writes `next_baseline_arm="z1"`, dir `runs/v431-rung` and prefix `z1-s` only when adopted, meaning a lever with all three guards True (chain_day6_3SZ.sh:188,195-197,208). Anything else exits 0. The trap (:50) removes the lock on every exit after :49. `busy()` matches the live `bash scripts/chain_day6_3SZ.sh 2026-10-26T14:27Z` (pid 904910), not itself.
- **Recipe**: the control's run.json args equal the arm's COMMON except the layout: same cache path and manifest 864b1fe3…, d3 W 2.0, learned, uniform, 8,8, val_every 6,375, 25,500 updates, lr 3e-4 then 1e-4, seeds. The control trainer was v430. v431 adds only the layout and the prior floor, which is inactive at D 1,024. `--layout 64x64x1 --allow-non256` is set (:90).
- **Pins**: all nine local shas equal the pins. On uk-linux all match except `eval_decision_v431.py`, which is **absent**.
- **Budget** (live `z1-s342101.log`): 0.2056 s/update from upd100 to upd1000, 0.2086 including VAL, gen 62 s, VAL ~20 s. GPU memory is 8,540 of 32,607 MiB.
  - Training per lease: about 62 + 25,500 x 0.209 ≈ 5,390 s, which is under DE 6,600. The final val and fresh evaluations add about 45 s, under DT 7,150.
  - FIT check ≈ 1.05 x (62 + 5,243 + 90) ≈ 5,670, under the 6,300 limit. It fails only above about 0.229 s/update (it counts 3 of the 6 evaluations: 60 s or less).
  - Batch ≈ 6.4 h + ~40 min diag/eval.
- **Analysis**:
  - The asserts hold against the real control fields.
  - The decision merge (:206-217) is in its own try and cannot alter the v420 verdict. Prefixed keys do not collide; deltas are automatic.
  - The rule at :228-231 matches D8-32 (≥+3/≥+2 lever; ≤-1/≤-2 worse; <+1/<+2 not better). One cycle → INCONCLUSIVE. Adoption needs all three guards.
  - next_baseline_* follows the 3S-Y schema.
- **eval/diag calls** (:137-145):
  - The `a-s` symlink resolves; `--device cuda` is set; never-overwrite holds in the chain and the script.
  - Diags use v431 for the arm and v430 for the control. The eval's v431 default is bit-identical for the control at D 1,024.

## Must fix

**M1 - The governing calibrated verdict is never computed (:17-19, :226-231, :257-266).**
- D8-32 says: if 3S-DR adopts the calibrated rule, the same reading under that rule is the formal verdict.
- The heredoc derives `verdict`, `adopt` and `next_baseline_dir/prefix/recipe` from v420 only. Analyst C says calibration helps blunter models more, so the rules can disagree. Then `analysis.json`, which the next chain reads by machine, names the wrong baseline, and the formal verdict becomes a hand computation done after the data are seen.
- **Fix**:
  - After the v420 block, compute `verdict_calibrated` with the same thresholds on `d("calibrated_val_R8_var_occ_f1")` and `d("calibrated_val_R8_surface_f1")`.
  - Use guards on `calibrated_val_R8_unknown_known_fpr` (≤ control + .003), `calibrated_val_R8_slab_recall` (≥ control - .02) and `calibrated_val_K0_var_occ_f1` (Δ ≥ -.02). A missing key gives INCONCLUSIVE.
  - Take `governing_rule` from a chain argument (`$1` = v420|calibrated, also written into each lease) or from a 3S-DR result file named now and read at analysis time.
  - Derive adopt and next_baseline_* from the governing verdict, and write both verdicts.
  - If the rule is undetermined and the two verdicts disagree, keep the control as the next baseline.

**M2 - The eval_decision pin will abort the batch (:35, :39; `check_shas` runs at :48, before every lease at :92, and after training at :133).**
- The file is not deployed on uk-linux. `code_review_v431_decision.md` (20:05 UTC) has a must-fix that edits it, so `c7ed51da…` is already stale.
- Launching now aborts at :48; deploying the revision later aborts a lease or the post-training phase (exit 2), losing training or the analysis.
- **Fix**: apply the decision-review fix, deploy it, re-pin, then launch. If v420 governs, move the ESHA check from `check_shas` to just before :144-145 and log "n/a" instead of exiting. If calibrated may govern (M1), keep the pin but freeze the file first.

## Should fix

- **S1 (:109)** - The lease `expectation` still describes an early stop: "or cycle 1 alone on early stop … early stop after cycle 1 if Δ R8 surface < +1pp". The chain has no such stop, so the provenance record contradicts D8-32. Replace the text with "no early stop; missing cycle 2 => INCONCLUSIVE".
- **S2 (:205)** - `identity_enc` sits outside the inner try. A `torch.load` error in this report-only reading makes a valid seed INCONCLUSIVE. Wrap it and record "n/a".
- **S3 (:203)** - "The only variable is the layout" drops the parameter-count check that 3S-Z had (chain_day6_3SZ.sh:164). Add `a["params"]["trunk"] == c["params"]["trunk"] and a["params"]["head"] == c["params"]["head"]`. These are counts, not tensors, so the "not asserted" disclosure in D8-32 still holds.
- **S4 (:236, :176-177)** - The slab guard reads only `diag_storey`. A timeout or crash there makes `slab_ok` "n/a" and blocks a real lever. Fall back to `v420_val_R8_slab_recall` from the decision JSON (same pooled definition).
- **S5 (:4, :107)** - The 14:27 approval quoted in D8-29 names "3S-Z", yet the lease asserts "user-approved" for four more 8 KB leases. Confirm with the user, or take the approval as `$1` the way chain_day6_3SZ.sh:26 does.
