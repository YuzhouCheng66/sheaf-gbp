# Code review: batch 3S-DR, decision rule (eval_decision_v431.py + test, D8-31)

**Verdict: LAUNCH OK WITH CHANGES.** One must-fix: the code does not compute the fallback rule that D8-31 promises to report. I found no defect that would make the verdict wrong. Both arms come from the same logits, so the paired Δ is exact.

Claude subagent, 2026-10-26 ~20:05 UTC, read-only; no GPU, `runs/` untouched. Lines refer to `eval_decision_v431.py`. The uk-linux trainer v431 (`9097801f…`), data module (`4793a327…`, in `scripts/`) and `train_v378.py` match local. The eval script is not deployed yet.

## Verified

1. **Panels.**
   - The split is `index < 8` / `>= 8` of `scene_plan('VALIDATION', 16, R1)`. None is skipped and the order matches trainer L479-493.
   - Both c2 runs have `val_per_stratum 8` and `skipped_panel_specs []`.
   - TRAIN-24 is (rooms 1-3, index 0-7), the same set as `probe_C_gen.py`. The manifest strata are exactly {1,2,3}, and k5383 selects the same houses.
   - `scene_plan` raises on TEST.
2. **Rules.** `decide` matches `predict`, with int8 output and `[n, V]` shapes. The calibrated branch is exactly the rule in D8-31. I checked the test vectors by hand.
3. **Summaries.**
   - L103 is the same mean of per-scene `agg` with None skipped as `evaluate()` L543.
   - `final.json` holds `val`/`fresh` keyed `'0'…'8'`.
   - C's CPU rebuild matched final.json to about 3e-5, so 2e-3 is about 60× the drift.
4. **Slab guard.** L97 is `diag_storey_v429` `recall_slab` (also C's `recall_occ_slab`), pooled over scenes and agents.
5. **Config and prior.**
   - The c2 meta is blocks [32,32,1], D 1024, d3, W 2.0, learned, uniform, 8+8, agents 8. `configure` sets every knob, and a layout or width error fails strict loading.
   - v430 → v431 changes only LAYOUTS and the floor. 1e-4/(8·1024) = 1.22e-8 > 1e-8, so the result is bit-identical.
   - n is always 8. z1 was trained with v431, so its floor is consistent too.
6. **Sweep.** The argmax is taken by value, not by string order, and ties go to the lowest threshold.
7. **Leakage.** None: the constants are literals, and VAL/fresh are only scored and compared for equality.

## Must fix

**M1 — fallback rule missing (L25-31, L79).**
- D8-31 says: if only the FPR guard fails, report "o ≥ 1.5, k ≥ 0".
- The margin is thin. C's VAL ΔFPR is +0.22/+0.24 pp against a +0.30 guard, and the fresh control FPR (.0220/.0210) is higher than VAL's.
- Fix: add `'occ_only': where(k >= 0, (o >= OCC_T).to(int8), -1)` to `decide` and to `rules` for val/fresh.
- Test that `(pred == -1) == (k < 0)`, which makes FPR identical by construction.
- Without this, that branch needs a post-hoc code change and a rerun.

## Should fix

**S1 — the sanity check can silently not run (L58, L112).**
- When `--ckpt final.pt`, assert that `val` and `fresh` are in `final`.
- After L57, assert `model.weights_hash() == ck['weights_sha256'] == final['weights_sha256']`.

**S2 — check per scene (L112-116).** A mismatch in one scene moves the mean by only 1/24 of its error.
- Compare against the rows of `validation/final-{val,fresh}.json` (both runs have them), matched on (rooms, index), all metrics, K0/K8.
- Set the tolerance to 5e-4.

**S3 — independent check of the calibrated rule.** On VAL it must reproduce `probe_C_summary.json` `split_rule_fpr_guard.val` to 1e-4 rounding:
- K0 .9287/.9300, R8 .9075/.9100, var .8443/.8476;
- the train24 sweep must match `occ_sweep` (s1 R8 .9139/.9155/.9126 at 1.0/1.5/2.0).

Record the diffs. C's histogram code is an independent implementation.

**S4 — the assert runs before the write (L116 vs L118).** Dump the JSON with `sanity_ok: false` first, then exit non-zero.

**S5 — missing guards (L34-38, L62-66).**
- Assert `meta` dims, D, agents and version against `W` / `T`.
- For `--train-sweep`, assert every row has split TRAIN and the manifest `data_module_sha256` matches `W.__file__` (as trainer L477-478 does).

**S6 — the mechanism readout is ambiguous (L111).** The code gives a per-hop argmax; D8-31 names one threshold.
- Also report C's `occ_only_mean`: the argmax of (K0 + R8)/2.
- Say in D8-31, before the run, which readout falsifies. Suggested: any of K0/R8/mean outside [1.0, 2.0].

**S7 — runtime (L98-101).** The sweep runs the full `metrics3d` 34 times per scene, including the tol-1 and furniture dilations it does not need.
- My estimate (from C's ~7 s/scene for ~4 calls): about 15-20 min per checkpoint, so about 1 h for both against the 30 min budget.
- Either compute only the surface F1 (`_f1(po & ~((t==1)&~s), s)` with one `surface_of` per hop), or amend the budget.

**S8 — no verdict script.** Add a short script that reads both seeds and computes, per seed on fresh:
- ΔR8 var and ΔR8 surface;
- the guards against **this script's own v420 control** (not the diag's VAL-only numbers);
- the paired per-house deltas.

It should print the D8-31 branch, so the verdict is not computed by hand.
