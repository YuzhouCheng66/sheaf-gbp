# Independent code review: v342 matched pair (owned-block vs dense global64), pre-launch

Reviewer: independent Claude Code reviewer, 2026-10-21. Scope: `/home/claude/sheafgbp/diag/train_v342_owned.py` (469 lines, read fully) and `/home/claude/sheafgbp/diag/test_v342_math.py` (59 lines, read fully). Sealed dependencies read where the trainer calls them: `mapping_multiroom_forest_v295.py` (initialize/advance/quantize/scalar_belief/central_reference, select_forest), `mapping_native_bank_v306.py` (select_forest L87-107, forest_targets L110-122), `mapping_recognition_metrics_v277.py` (node_metrics), `mapping_multiroom_data_v293.py` (make_fixture), `train_mapping_support_native_v306.py` (scene_plan L118-121, draw_plan L107-115), `mapping_recognition_hops_v277.py` (`_edges`, LATENT_PRIOR, v277 `factors` semantics L90-118), `mapping_implicit_model_v192.py` (latent_quantize L55-60, codec L37-46), `mapping_recurrent_gbp_v213.py` (initialize L13-24), `mapping_context_codec_v121.py` (ContextDecoder L149-183, for the "v121-shaped" comparison). Context: `reports/BRIEF_COMMON.md`, README sections 3, 6, 7, 9, and the pre-registration `diag/mapping_owned_pair_v342.json`.

Method: static reading plus CPU-only mechanics checks run from a scratchpad (no GPU, no network, no file under review or under `handoff_src` modified; the provided `test_v342_math.py` was executed as-is and prints ALL PASS). Side effect disclosure: importing the trainer refreshed `diag/__pycache__/train_v342_owned.cpython-313.pyc` (bytecode cache only; a cache directory already existed there).

Empirical checks performed (scratchpad `check_v342.py`, `check2_v342.py`, `check3_v342.py`), results quoted inline below:

| check | result |
|---|---|
| OwnedHead `observed` index for a single known block at (row 2, col 5) / (row 7, col 0) | `[21]` / `[56]` (row-major, as required) |
| PatchTrunk rows that change when only block 21 (resp. 56) is non-empty | `[21]` / `[56]` |
| Perturb one pixel inside block 21 (interior and at the block's top-left corner) | only coordinate 21 of (mu, q) changes; exactly zero change elsewhere |
| OwnedDecoder reshape of feature index 21 | lands at grid `[2,5]` |
| `truncated_dt` vs brute-force truncated Euclidean DT (T=4, random 20x20) | max abs err 4.8e-8 |
| `forest_targets(inp,0)[0] == inp['private']` on a real fixture | True |
| `SheafModel.latent_quantize` vs sealed v192 `latent_quantize` on 1000 random values (incl. out of range) | `torch.equal` True |
| Trunk initial weights identical across arms at the same seed | True |
| Scheduler factor at u = 0 / 199 / 200 / 1776 / 3551 / 3552 (total 3552) | 0.005 / 0.993 / 0.993 / 0.525 / 0.05 / 0.05 |
| Precision envelope with sigmoid saturated to exactly 1.0 (float32) | q = 4.0 exactly; precision == bound, `>` check does not fire (N=8,16); prior 1.95e-7 / 9.8e-8 > profile-4 low 1e-8; 15 saturated sources give cavity 60.0 < 128 |
| Full forward + backward on a real VALIDATION fixture (N=8, H128) for both arms | shapes OK; `predict` returns int8 [8,128,128] in {-1,0,1}; `node_metrics` accepts it |
| Parameter counts (trunk / head / decoder) | owned 218,976 / 16,770 / 630,498; dense 218,976 / 8,520,832 / 792,034 |
| Scene counts at H128 | train 592 (296 per N, 192/206/194 per rooms), val 48, fresh 48; fixture + 5 forest_targets = 0.03-0.05 s per scene (data phase about 40 s) |
| `test_v342_math.py --source <sealed root>` | 9/9 PASS, ALL PASS |
| Refinement closed form (L362-364) vs actual R0 rollout features, owned arm | mean mismatch on mask==0 coordinates 1.57e-2 at init (100% of mask==0 coordinates have zq != 0); 0.0 on mask==1 coordinates; log-var mismatch 6e-8 |
| d(loss)/d(decoder input) at init | max 6e-10 (owned), 1.1e-9 (dense): deep default-init CNN attenuates |
| 60-update CPU liveness probe (one scene, constant lr 3e-4, no warmup) | trunk/head grad norms rise from 1e-9/1e-10 to about 1e-3/1e-2 by update 20 in both arms; dense head drifts |dW| = 15.5 and saturates (|mu| std 1.00, mean q 3.94, trunk gnorm back to 5e-6 by update 60); owned head |dW| = 0.41, q mean 1.95, no saturation |

## Item-by-item findings

### 1. No ground-truth / peer-raw leakage into model inputs: PASS (one disclosure note)

- Encoder input is the agent's own ternary map plus public xy only: `PatchTrunk.forward` L98-107 builds `[known, free, wall, x, y]` from `private` and `public(...)[0,:2]`; nothing else enters `self.net`.
- `OwnedHead.forward` L119-125: `observed` (L124) is computed from `private >= 0` (own map) and is returned as the third output only; `SheafModel.source` L186-189 returns it as `mask`, which is consumed only by `rollout` (L212 `evidence_q = mask * q_learned`), by the refinement variant (L363), and by diagnostics (L381, L384). It is never concatenated into decoder input.
- Decoder input is exactly `[hard8 posterior mean 64, log posterior variance 64]`: `features()` L218-220; `OwnedDecoder.forward` L149-154 and `DenseDecoder.forward` L165-171 consume only `features`.
- Targets reach only `stage_loss` (L246-253) and `node_metrics` (L375); `dt` is derived from targets (L328) and used only inside the loss. `door_mask` is used only in `node_metrics`. `predict` (L256-260) touches no target.
- Disclosure note (not a leak, but must be stated in the write-up): in the owned arm the posterior variance channel at R0 equals `1/(mask_k q_k + prior)`, so the decoder can read the agent's own 64-bit block-support map exactly from the variance channel, and at R>0 it reads the reached-union block support. This is the declared mechanism of the intervention and it passes through the declared 64+64 bottleneck (README section 3 item 3 permits the decoder to read posterior (q,h)); it is not a raster own-mask or GT. The pre-registration text "never a decoder input" is literally true of the indicator but the information is recoverable from the variance channel; say so plainly.

### 2. Stage targets / weights / edges: PASS

- Targets: `load()` L313-314 calls `bank.forest_targets(inp, k)` for k in HOPS; `forest_targets` (bank L110-122) propagates reached sets over `select_forest(...)['edges']` for exactly `rounds` synchronous steps and takes the per-cell max (union; conflicts rejected). Edges for the rollout come from `bank.select_forest(positions_m, adjacency, communication_radius_m)` at L315 with the same arguments as inside `forest_targets` L117, so rollout edges == target edges. The trainer does not use `fixture['evaluation']['targets']` (all-radius-edge hop targets), which is correct per README section 6.
- Weights: `WEIGHTS = {0:.5, 1:.125, 2:.125, 4:.125, 8:.125}` L31; applied at L415.
- R0 target == own: `reached = eye` at rounds=0 so `own[s].max(0)` is the own map; verified `np.array_equal` True on a real fixture. `node_metrics` additionally enforces own subset of target (metrics L61-62).
- Reached-set semantics match GBP hops: r rounds of `advance` propagate sources r forest-hops; test (5) confirms belief == `central_reference` (which uses `reachability`) within wire quantisation.

### 3. Rollout reproduces sealed v295 semantics: PASS

- `make_rollout` L207-228: FP64 `q = diag_embed(mask*q_learned + prior)[None]`, `h = (mask*q_learned*mu)[None]` (L212-215), `forest.initialize(q, h, edges)` (L216) which re-validates FP64/diagonal/positivity/envelope (forest L131-142); `advance(state, 1, quantized=True, trace=False)` per round (L224) = profile-4 quantised wire with STE (forest L174-194, L197-223); `scalar_belief` for mean/variance (L219); `prior = LATENT_PRIOR/(n*64)` (L211) matches v277 `factors` (hops L115-118) with LATENT_PRIOR = 1e-4 (hops L31; trainer L111).
- Source means are hard-8 with STE: `SheafModel.source` L189 applies `latent_quantize` (identical values and identical STE form to v192, verified). Posterior means fed to the decoder are hard-8 via the sealed v192 `latent_quantize` (L220, L292, L301).
- No posterior is reused as evidence: `state['q']/state['h']` are set once in `initialize` and never replaced; the refinement variant (L355-369) only re-optimises the SOURCE mean `z` against the agent's own observation and then re-runs the rollout from scratch (L370), i.e. it is local computation L, not reinjection.
- Envelope: q in [1e-4, 4] (L123, L137, sigmoid) and mask in {0,1}; at exact float32 saturation precision equals 4 + prior and the strict `>` check in forest L141 does not fire (verified above). Cavity precision stays within profile-4 [1e-8, 128] for N <= 16.
- `test_v342_math.py` exercises exactly these primitives through `make_rollout` and passes 9/9 on this machine, including complementary owned sources fusing to the union within one hard8 step (max err 7.87e-3) and the dense contrast (mixing 0.913).

### 4. Arms matched: PASS on the controlled factors; two asymmetries to disclose (RISK)

Matched: same `PatchTrunk` constructed first under `fork_rng(seed)` (L178-180; identical initial trunk weights verified), same `CnnTrunk` decoder family with 2 output channels (known logit + DT channel) in both decoders (L147, L163), same seeded data order (`order_rng = default_rng(args.seed)` L401, `perm` L403), same AdamW/LambdaLR/lr/epochs/clip/seed (L332-334, L418), same loss (L246-253), same rollout, same deadlines. The only code paths that differ are `OwnedHead+OwnedDecoder` vs `DenseHead+DenseDecoder` (L181-184).

Asymmetries that could confound:

- (a) Head capacity and fan-in: `DenseHead` is `Linear(8192,1024)+Linear(1024,128)` = 8.52 M parameters, 508x the `OwnedHead` (16.8 K) and 65x the anchor's single `Linear(2048,64)`-style head. Same nominal lr does not mean matched effective step size: in the 60-step CPU probe the dense head moved |dW| = 15.5 vs 0.41 and saturated (|mu| -> 1, q -> 3.94, trunk gradient collapsing to 5e-6) while the owned head did not. With warmup and 592 diverse scenes this will be milder, but it is a plausible path for the control to under-train or binarise early. Defensibility: the dense head follows the pre-registered anchor-family description ("flatten -> MLP"), and the registration already has G4 (absolute comparison against v310 E12 numbers) which protects against an under-trained control. Recommendation: do not change the architecture now (it is registered); log saturation fractions (patch C) and treat G2 as insufficient on its own if dense K0 own-wall-F1 lands far below the anchor's 83.1 %.
- (b) Decoder spatial prior: `OwnedDecoder` hard-wires variable k to grid cell (k//8, k%8) (L151), while `DenseDecoder` must learn placement through `Linear(128,2048)` (L162, L166-169). The intervention therefore bundles two things: (i) q_k = 0 abstention / union-composing fusion semantics and (ii) a decoder-side spatial inductive bias. This is inherent to "public ownership", but the write-up should separate them using numbers the script already produces: R0 (no fusion) differences reflect (ii) plus per-block encoder capacity; the R8 - R0 change in `peer_wall_f1`/`union_accuracy` reflects (i). Decoder parameter counts are close (630 K vs 792 K, dense larger) and decoder input normalisation is identical (`/8.` on log-variance, L151, L166), so the dense arm is not handicapped at the decoder.
- Update counts: both arms are matched only if both complete 6 epochs before `--deadline-epoch`; `final.json['updates']` records the actual count (L445). Compare the two before applying G2.

### 5. Loss and prediction: PASS

- Known BCE over all pixels per agent (L249, `.mean(-1)` over H*W, then mean over agents L253).
- DT target: `truncated_dt` (L232-243) is the exact truncated Euclidean distance to the nearest wall cell with truncation T = 4 (verified to 5e-8 against brute force), computed from the stage target `tgt[k] == 1` at L328, per stage.
- L1 on known cells only with near-wall weight: `weight = known * (1 + 4*[dt <= 1.5])` L251 (dt in {0, 1, sqrt2} get weight 5), normalised per agent L252.
- Prediction: `wall = softplus(wall_ch) < .5` inside `known_logit >= 0`, else -1 (L259-260); int8 ternary, reshaped to [N,H,W] at L374; row-major flatten of the decoder output (L154, L171) matches `targets.reshape(n,-1)` (L327). Verified end-to-end through `node_metrics`.

### 6. Evaluation integrity: PASS

- Metrics via sealed `node_metrics` on raw `predict` output (L374-375), no post-processing.
- Aggregation: per-scene node means (L377-379), then per-(rooms, N) stratum means (L389-392), then unweighted mean over strata for `all` (L393): equal weight per stratum as the README requires.
- `fresh` = VALIDATION indices 8-15 (`index + 8`, L306); generated by the sealed generator, distinct worlds (checked), outside the v306 bank's VALIDATION range [0,7] and outside the earlier diagnostics' idx 0-3; evaluated only once, on final weights (L451-455), never used for selection.
- `best.pt` selection uses only the VAL panel's R8 `wall_f1` (L433-436). Reported final numbers come from final weights (L446-460), consistent with the registration ("final weights also reported"). Note `best.pt` is never evaluated on fresh, which is the conservative choice.
- Refinement variant: objective uses `sc['tgt'][0]` and `sc['dt'][0]` (L365), i.e. the agent's own observation only (R0 target == own, verified), and is written to `final-val-refined.json` / `final['val_refined']` separately (L457-460). But see item 8 (bug): the inner closed form is mis-specified for the owned arm.

### 7. Safety / budget: PASS with two RISKs

- Output directory must be fresh (`exist_ok=False`, L294); every write (`run.json`, `checkpoints/*.pt`, `validation/*.json|npz`, `history.json`, `final.json`) is under `--out` (L339-341, L343-347, L383, L431, L439, L449, L454, L459, L461-463). No `os.remove`/`shutil` anywhere. Only side effect outside `--out`: Python bytecode caches next to the imported sources (set `PYTHONDONTWRITEBYTECODE=1` if the sealed tree must stay byte-identical).
- Deadlines: epoch deadline checked before every update (L406) and after each epoch (L427); total deadline guards each final evaluation (L446, L451, L456). SIGTERM/SIGINT set `STOP` (L35-37, L283) and are honoured at the next update boundary, followed by validation and `latest.pt` (L428-437).
- Checkpoints: `initial.pt` (weights + meta, L341), `best.pt`/`latest.pt`/`final.pt` with optimizer, scheduler, torch/cuda/numpy/python RNG states and weight hash (L343-347). `nonfinite gradient` raises (L419-420), leaving `latest.pt` from the previous epoch.
- RISK 7a: after `STOP` (or after the training deadline) the script still runs up to three full evaluations before `final.pt` is written (L446-461); the final evaluations are guarded by `deadline_total` but not by `STOP`, and an evaluation that starts just before `deadline_total` can overrun it (the refined variant is about 48 scenes x 50 decoder forward/backward passes). If a supervisor sends SIGKILL after a grace period, `final.pt`/`final.json` may never appear. Patch B moves `save('final.pt')` before the evaluations (zero effect on results). Budget about 5 minutes of headroom between `train_deadline_seconds` (5400) and `total_deadline_seconds` (7000) for the three evaluations; the registration already leaves 1600 s, which is sufficient.
- RISK 7b: device cache size. `sc['dt']` is float32 [N,16384] x 5 hops per scene (L328): about 2.6 GB for 688 scenes plus about 0.8 GB of int8 maps/targets, per arm. Fine on a 32 GB card, also if both arms run concurrently; mention in the run notes.

### 8. Bugs: one real bug (refinement variant), several minor RISKs

- BUG (FIX-FIRST, patch A): the local-refinement closed form at L362-364 feeds the decoder `cat((zq, log var))` with `zq = latent_quantize(z)` on ALL 64 coordinates, but the true R0 posterior mean is `mask*q*mu/(mask*q+prior)`, i.e. exactly 0 on every block with `mask == 0` (h = 0 there). For the owned arm (mask fraction about 0.18 at H128) the refinement therefore optimises observed-block codes in a context where unobserved blocks show arbitrary `zq` values (measured mismatch 1.57e-2 at init; unbounded after training) that the subsequent real rollout (L370) replaces with 0. The refined owned codes are thus tuned to a decoder input the system never produces, biasing `val_refined` against the owned arm. The dense arm (mask == 1) is unaffected. Fix: multiply by `mask` (one token). Training, selection, `val` and `fresh` panels are unaffected.
- Block ordering `PatchTrunk` (L103-104, `unfold(2).unfold(3)` then `permute(0,2,3,1,4,5)`) vs `OwnedHead.observed` (`avg_pool2d(..., p).flatten(1)`, L124) vs `OwnedDecoder.reshape(b,1,8,8)` (L151): all row-major (row = y block, col = x block) and mutually consistent (verified with single-block probes at indices 21 and 56; decoder places 21 at [2,5]). PASS.
- Strict block locality of the owned arm: patches are separate batch elements, so conv zero-padding is inside each 16x16 patch; perturbing an interior pixel or a corner pixel of block 21 changes only coordinate 21 (exact zero elsewhere). PASS.
- Scheduler lambda (L334): warmup 200 updates, cosine to 0.05 at `total_updates`, called with the step count (0 at construction, after `opt.step()` per update): values verified; no step-order warning. PASS.
- Gradient path: STE forms are correct (L194; v192 L60; forest L194); `mask`/`active` are intentionally non-differentiable constants; q receives gradient through both h and precision (L212-215) and through `var.log()` (L220); decoder receives gradient from all five stages. No wrong detach found. PASS.
- Liveness (README section 9): at initialisation the gradient reaching trunk/head is about 1e-9 to 1e-10 (attenuation through the default-init 13-conv decoder; d loss/d decoder-input max 6e-10). This is the same construction as the anchor's `ContextDecoder` (v121 L149-162), and with Adam the trunk/head gradient norms rise to about 1e-3 / 1e-2 within 20 updates in both arms (CPU probe), so it is not a broken graph; but because Adam's eps = 1e-8 exceeds these initial gradients, the encoder is effectively frozen for the first few dozen updates. The script already logs per-group gradient norms every 50 updates (L417, L424): confirm in the first log lines that `trunk`/`head` norms are >= 1e-4 by update 100-200 in BOTH arms (the preflight `--max-updates 200` is the right tool). RISK, monitor.
- Sync overheads (RISK, performance only): `stage_loss` returns `float(known_bce.mean())`, `float(wall_term.mean())` (L253) = 10 host syncs per update; `gnorm` (L417) does one `float()` per parameter tensor (about 40) on every update although it is printed every 50; plus `float(norm)` and `float(total)`. Estimated cost a few ms per update, tens of seconds per run; not a correctness issue. Patch C defers `gnorm` to the logging branch and adds saturation fractions (README section 9 asks for saturation logging; the dense-head saturation seen in the probe is exactly the signal to watch).
- Minor: in the refinement loop the decoder parameters accumulate `.grad` across 50 x 48 backward passes (L366) because only `z` is zeroed; harmless (post-training, `state_dict` does not contain grads) but wasteful; `torch.autograd.grad(loss, z)` would avoid it. Not patched.
- Minor: `final-val` (L447) repeats the epoch-6 validation on identical weights (duplicate about 1 minute and duplicate panels); harmless.
- Minor: `--deadline-epoch` help text says "monotonic" but the comparison uses `time.time()` consistently (L406, L427); only the help text is wrong.
- Minor: `fmt` (L397) and the `score > best[0]` comparison (L435) would raise `TypeError` if an aggregated `wall_f1`/`unknown_known_fpr` were `None`; this requires all strata to be `None`, which cannot happen at H128 with these panels.
- Checked and found correct: all tensors created on the right device (`public(...)` uses `grid.device`, `truncated_dt` uses `wall.device`, forest tables on `p.device`); `_edges` accepts the bank's `(int, int, 1.0)` tuples; `LambdaLR.state_dict()` with a plain lambda pickles fine (lambdas are skipped); `torch.save` of RNG states fine; `np.savez_compressed` filenames contain no path separators; json payloads contain only Python scalars/None.

### 9. Cheating / unfair handicap: PASS

- Owned arm: no cross-block information path exists in the encoder (separate batch elements; padding is intra-patch; verified exactly). The per-block head MLP acts on a single block's feature vector (L117, L121). The only cross-block channel is the GBP posterior itself, which is the declared message. The variance channel reveals block support (see item 1 note) through the declared bottleneck, which is the hypothesis under test, not a side channel.
- Dense arm: not handicapped by the code. Same trunk, larger decoder, far larger head; `active` is always 1 (every agent has sensor evidence per generator assertion, data L135), reproducing v277 semantics (hops L113). The dense decoder is v299-shaped (tanh mean planes + log1p(q) planes at 16x16, L166-169). The one way the control can fall short is its own optimisation dynamics (item 4a), which is observable in the logs and bounded by gate G4.

## Minimal patches (unified diffs against the current file)

Patch A (required before the `val_refined` panel can be reported as a fair comparison; no effect on training or on the main panels):

```diff
--- a/diag/train_v342_owned.py
+++ b/diag/train_v342_owned.py
@@ -361,7 +361,7 @@
                         o.zero_grad(set_to_none=True)
                         zq = model.latent_quantize(z)
                         var = 1. / (mask * q + prior)
-                        feats = torch.cat((zq, var.log()), -1)
+                        feats = torch.cat((zq * mask, var.log()), -1)   # R0 posterior mean is exactly 0 where mask == 0 (h = mask*q*mu)
                         loss, _, _ = stage_loss(model.decoder, feats, sc['tgt'][0], sc['dt'][0])
                         loss.backward(); o.step()
                         with torch.no_grad():
```

Patch B (safety: never lose the final weights to a late SIGKILL; identical weights, identical results):

```diff
--- a/diag/train_v342_owned.py
+++ b/diag/train_v342_owned.py
@@ -443,6 +443,7 @@
 
     # ---- final: fresh panel + refinement variants on the final weights (not selected on)
     final = dict(arm=args.arm, updates=updates, epochs_completed=len(history), weights_sha256=model.weights_hash(), seconds=time.time() - t0)
+    save('final.pt', updates, len(history))
     if time.time() < args.deadline_total:
         ev_val = evaluate(val, 'final-val')
         final['val'] = ev_val['summary']; print(f"[{args.arm}] FINAL val: {fmt(ev_val['summary'])}", flush=True)
@@ -458,7 +459,6 @@
         final['val_refined'] = ev_ref['summary']; print(f"[{args.arm}] FINAL val + local refinement({args.refine_steps}): {fmt(ev_ref['summary'])}", flush=True)
         with open(out / 'validation' / 'final-val-refined.json', 'w') as f:
             json.dump(ev_ref, f, indent=1)
-    save('final.pt', updates, len(history))
     with open(out / 'final.json', 'w') as f:
         json.dump(final, f, indent=1)
     print(f'[{args.arm}] done {time.time()-t0:.0f}s', flush=True)
```

Patch C (optional, recommended: fewer host syncs, and the saturation fractions README section 9 asks for; does not change any computed gradient or update):

```diff
--- a/diag/train_v342_owned.py
+++ b/diag/train_v342_owned.py
@@ -414,14 +414,16 @@
                 l, kn, wl = stage_loss(model.decoder, feats[k], sc['tgt'][k], sc['dt'][k])
                 total = total + WEIGHTS[k] * l; parts[k] = (kn, wl)
             total.backward()
-            gnorm = {name: math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in ps if p.grad is not None)) for name, ps in groups.items()}
+            logging = (updates + 1) % 50 == 0
+            if logging:   # pre-clip per-group norms, computed only when printed
+                gnorm = {name: math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in ps if p.grad is not None)) for name, ps in groups.items()}
             norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
             if not math.isfinite(float(norm)):
                 raise FloatingPointError('nonfinite gradient')
             opt.step(); sched.step(); updates += 1
             losses.append(float(total))
-            if updates % 50 == 0:
-                print(f"[{args.arm}] ep{epoch} upd{updates} loss {np.mean(losses[-50:]):.4f} k0(kn,wl)={parts[0][0]:.3f},{parts[0][1]:.3f} k8={parts[8][0]:.3f},{parts[8][1]:.3f} grad {gnorm} lr {sched.get_last_lr()[0]:.2e} mask {float(mask.mean()):.2f} {time.time()-t0:.0f}s", flush=True)
+            if logging:
+                sat_mu, sat_q = float((mu.detach().abs() > .99).float().mean()), float((q.detach() > 3.99).float().mean())
+                print(f"[{args.arm}] ep{epoch} upd{updates} loss {np.mean(losses[-50:]):.4f} k0(kn,wl)={parts[0][0]:.3f},{parts[0][1]:.3f} k8={parts[8][0]:.3f},{parts[8][1]:.3f} grad {gnorm} lr {sched.get_last_lr()[0]:.2e} mask {float(mask.mean()):.2f} sat|mu|>.99 {sat_mu:.2f} q>3.99 {sat_q:.2f} {time.time()-t0:.0f}s", flush=True)
         model.eval()
```

(If patch C is applied, the two `float()` calls in `stage_loss` L253 can also be deferred by returning `known_bce.mean().detach()` / `wall_term.mean().detach()` and wrapping with `float()` only in the print; this is purely a speed matter and is left optional.)

## Launch notes (no code change)

1. Run the preflight on the GPU with `--smoke --max-updates 200` for both arms and confirm: (i) `trunk`/`head` gradient norms climb from about 1e-9 to >= 1e-4 by update 100-200 in both arms; (ii) the dense arm's `q`/`|mu|` saturation fractions are not already near 1 at update 200 (if they are, the control is binarising before warmup ends and this must be stated next to G2); (iii) first validation runs end-to-end and writes `validation/epoch-0001.json`.
2. Both arms must reach the same `updates` (6 x 592 = 3552) before `--deadline-epoch`; compare `final.json['updates']` of the two arms before applying G1-G4. Launch concurrently or give each arm its own deadlines; estimated per-arm wall-clock at H128 is 15-30 minutes including the three final evaluations, well inside the 5400/7000 s deadlines.
3. In the report, separate (a) R0 differences (encoder capacity + decoder spatial prior) from (b) R8 minus R0 changes in `peer_wall_f1` / `union_accuracy` / `unknown_known_fpr` (fusion semantics), and state the parameter counts recorded in `run.json['parameters']` (owned 0.87 M total vs dense 9.53 M total, almost all in the dense head).
4. State explicitly that the owned arm's variance channel carries the 64-bit (reached) block-support map through the declared bottleneck.

## Verdict

FIX-FIRST (minor). The training core, data/target construction, GBP rollout, loss, prediction, metrics, selection and output discipline are correct and matched; the sealed math test passes; no leakage or cross-block side channel exists. The one genuine bug is confined to the separately reported local-refinement evaluation variant (patch A, one token) and does not touch training, `best.pt` selection, or the `val`/`fresh` panels. Patch B is a safety improvement with no effect on results; patch C is optional logging/perf. None of the three patches changes the training trajectory of either arm, so no re-validation of the math test is needed after applying them (re-running `test_v342_math.py` takes seconds anyway).

If the launch window cannot wait: LAUNCH as-is is acceptable for the main matched comparison (G1-G4), with the understanding that the owned arm's `val_refined` numbers are invalid and must be recomputed afterwards from `final.pt` with a patched evaluator, and that `final.pt` is only written after the final evaluations.
