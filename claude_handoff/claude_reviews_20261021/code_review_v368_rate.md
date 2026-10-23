# Pre-launch code review: train_v368 (lattice join + own-q + entropy model / rate term), batch J

Reviewer: independent Claude subagent, 2026-10-22, CPU only, no GPU, no network. Time-boxed (30 min).
Nothing under `diag/` or `handoff_src/` was modified. Scratch: `/tmp/claude-0/-home-claude/4bad606d-6bc2-5614-a287-aa8ba2b79cab/scratchpad/review368/`.

**Verdict: LAUNCH OK WITH CHANGES.** There are no BLOCKERs. M1 should be fixed before launch: it is a small change, and without it the four rate arms spend their first ~4k updates on a mis-directed rate gradient. M2 can be handled after the runs, but it has to be handled before anyone quotes a figure in bytes per message.

## Scope
- `/home/claude/sheafgbp/diag/train_v368.py` (sha256 `649f0a43f14c5891…`), compared with `train_v365.py` (`7beb10e8…`). Generator: `make_train_v368.py`.
- `/home/claude/sheafgbp/diag/test_v368_rate.py`, `eval_lexjoin_v364.py` (tests-only mode), `kbit_bytes_v348.py`, `chain_day3_J.sh`. For the warm-start check: `train_v347_maxsel.py` (the trainer behind `v354-endpoint-long`; `chain_day2_AB.sh` lines 115-121).
- Context read: report D3-6 and D4-1, and the declared semantics in README sections 3/8/9/10.

## Method (every check below was run, not inferred)
1. Ran `make_train_v368.py train_v365.py` into scratch and compared with `cmp`: the output is **byte-identical** to `train_v368.py`. Read the full `diff train_v365.py train_v368.py` (143 lines).
2. `check_codes.py`: compared `EntropyModel.hard_codes` with an independent re-implementation of the `kbit_wire_quantize` code computation, on the rollout's own factor (p = mask·q + prior, eta = mask·q·mu_hard, prior = 1e-4/(64N)). Inputs: 400 random trials with N = 2..12 and q drawn uniform, near the floor or near the top (175,936 coordinates); a sweep of 4.4M float32 q values; and constructed codebook-edge points. The re-implementation is cross-checked against the function's actual p_hat output.
3. `check_misc.py`: checked state_dict key, shape and init equality between v347 and v368 `SheafModel`, and swept mean-index agreement over all 255 levels × q ∈ [1e-4, 1e-3] for N = 2, 3, 8, 16.
4. `check_step.py`: a synthetic 6-agent scene through the real `SheafModel`, `make_rollout(fusion='max')` with `WIRE='kbit'`, and the sealed forest. Checks: K0 equals the own encoded factor; R8 on a tree equals the centralised join; gradient finiteness and gradient flow; lambda-term and fit-term routing; the exact `total.backward(); r_fit.backward()` ordering; the LR multiplier on both param groups over 21,312 updates.
5. `check_fit.py`: simulated how fast the freshly initialised entropy model fits, under the trainer's exact optimiser and schedule (AdamW 1e-3 × LambdaLR warm-up and cosine, ~120 observed coordinates per update), for two plausible code distributions.
6. End-to-end CPU smoke of the real trainer: `--smoke --max-updates 4 --wire kbit --fusion max --rate-model --rate-lambda 0.03`. It runs; `summary['rate']` is written to final.json; `final.pt` has `entropy_model` (6 tensors); the optimizer has 2 param groups.
7. Ran `test_v368_rate.py`: **ALL PASS (8/8)**. Ran `eval_lexjoin_v364.py --tests-only --trainer train_v368.py`: **ALL PASS (4/4)**. Note that these 4 tests exercise the evaluator's own `rollout_lex` on the sealed wire, not the trainer's `rollout_max`. That is why I added check 4 against the trainer itself.
8. Simulated the chain's admissibility and replication branch in a mock directory, where `final.json` has no `val` key.

## Findings

### MAJOR
**M1 — The entropy model is fresh at warm start and fits slowly, but the rate gradient into the encoder is on from update 0. The LR comment is wrong.** Lines 581, 601-604 and 697-702.
- The v347 checkpoint has no `entropy_model`, so `em` starts from its init. The precision mixture is centred at log q ≈ −2 (components −3..−1, scale 1). The mean mixture is centred at 0.
- The em param group shares the model's LambdaLR. I verified the resulting LR: 5e-6 at u=0, 1e-3 at u=199, 8.8e-4 at u=5000, 5e-5 at the end. The comment at line 602 says "constant-ish"; it is not.
- Simulated fit lag, as model bits minus empirical entropy per coordinate:

  | Code distribution | u=500 | u=1000 | u=2000 | u=4000 | u=6000 |
  |---|---|---|---|---|---|
  | Saturating-q (H_mu+H_q = 11.43) | +3.3 | +2.9 | +1.7 | +0.4 | +0.2 |
  | Spread-q (13.35) | +1.4 | +1.1 | +0.5 | +0.2 | — |

  The 1e-3 Adam step needs thousands of steps to move the mixture means by the ~3 units required.
- Meanwhile, for lambda > 0 the encoder (LR 1e-4 after update 200) receives `lambda * rate(detach_params=True)` against this mis-fit density. The rate gradient on log q (2e-2 per observed coordinate at lambda=1, from an untrained density) is ~40× its gradient on mu_c, and it points toward the init's mass at q ≈ e^-2.
- q is also the max-selection key, so for the first ~2-4k of 21,312 updates the rate gradient pulls observed precision down toward an arbitrary init location. Because the density then fits the codes that have moved, the result can be path-dependent: the arms' rate-distortion points partly measure the init rather than the codes.
- **Fix (minutes):**
  1. Before the loop, pre-fit `em` on detached TRAIN codes of the warm-start encoder. For example, 2-3k fit-only Adam steps at lr 1e-2 on `rate(..., detach_inputs=True)`; this is the trunk/head forward only and takes seconds to minutes. Alternatively, start `lambda` at 0 and ramp it in after the fit has converged.
  2. Take the em group out of the cosine schedule by passing LambdaLR a list of two lambdas, the second constant.
  3. Fix the line 602 comment.

  The lambda=0 control is unaffected either way.

**M2 — "Bits per observed own coordinate" is not message bytes. Relayed codes are a selection-biased subset and are not measured.** Lines 624-628 and 668-675, against `kbit_bytes_v348.py`.
- The rate (training and evaluation) is computed on each agent's **own** codes of its observed blocks. Under max fusion, a message carries a 64-bit presence mask plus, for every non-prior coordinate of the sender's subtree, the code of the **max-precision winner**: on average 30.5/64 coordinates, against 15% own-observed (D3-6).
- Winners are biased toward high-q codes, so their code length under the shared model differs from the own-code mean. A lambda arm that lowers entropy mostly in low-precision codes, which rarely win, would shrink own-code bits more than message bytes.
- `kbit_bytes_v348.py` uses fixed 2k bits per non-prior coordinate. The step "bytes = 64 + bits × 30.5" is therefore an approximation, not an exact count.
- Admissibility is pre-registered on own-code bits, so the gate is fine as defined. Everything needed for the exact count is saved: `final.pt` plus `entropy_model`. Relays forward codebook values bit-exactly, so the indices of a message can be recovered exactly from (p_hat, e_hat).
- **Required before any B/message claim:** a post-hoc evaluator that sums `64 + Σ_{non-prior coords of each R8 message} (b_mu + b_q)` under the arm's own model. It should also add arithmetic-coder termination overhead (≤ ~2 bits/message), and should report the max-selection tie rate per arm (see N3).

### MINOR
**m1 — chain_day3_J.sh lines 36-52: an admissibility crash launches a broken batch K.**
- If any arm's `final.json` lacks `val` (the total deadline passed before final-val, line 749), `g()` raises KeyError, and stdout is empty. Then `BEST=''`, `ARM=''` and `LAM=''`, and `[ "$ARM" = none ]` is false.
- I verified this in a mock directory: batch K then starts the l0 replication, and a second job `--out runs/v369-rate/-s342101 --rate-lambda --init-from …` (argparse error). It also writes invalid JSON to `lease_K.json`.
- Fix: wrap `g()` in try/except returning None; require `final['updates'] == 21312` (or equal across arms); and test `[ -z "$ARM" ] || [ "$ARM" = none ]`.

**m2 — The REVIEW_OK gate is not bound to the reviewed code.** Lines 19 and 25.
- The gate only checks that the file exists. The trainer sha is recorded in the lease but never compared, and `$(cat REVIEW_OK)` is pasted into a JSON string, so a quote or newline in it breaks `lease.json`.
- Suggest: put `649f0a43f14c58915a4f5843303c4a92364d73b1f42cbbcc9498a4df72a1db31` (or the sha after the M1 fix) in REVIEW_OK, and abort if `sha256sum scripts/train_v368.py` differs.

**m3 — The precision index is not computed by the same expression as the wire** (lines 135 vs 192).
- `hard_codes` uses `floor((L-lo)/s)` with `s=(hi-lo)/(nq-1)`; the wire uses `floor((L-lo)/(hi-lo)*(nq-1))`.
- I found 0 mismatches in 4.4M swept and 176k random float32 q, but 13-19 of 118 constructed exact-codebook-edge doubles differ by one code. In practice this is measure-zero, but the code is not bit-for-bit by construction.
- Suggest a shared helper that returns the wire's integer codes, with `hard_codes` calling it.

**m4 — The mean index is taken from mu_c, not from what the wire sends** (line 191 vs 138).
- The wire quantises eta/p = mu_hard·q/(q+prior); hard_codes uses round(mu_c·127). These differ by one level when |k|·prior/(q+prior) ≥ 0.5.
- Mismatches: N=2 gave 13,846/510k and N=3 gave 2,256/510k, both near q ≈ 1e-4. **N=8 and N=16 gave 0.** Those are the only agent counts in the v293 TRAIN (296/296) and VALIDATION (24/24) plans, so batch J is unaffected.
- This is a latent foot-gun for other N. The same shared-helper fix as m3 removes it.

**m5 — Warm-start provenance.** Lines 591-595.
- The meta check passes for v347 only because absent keys default to the CLI value. I verified that the state_dict is strictly compatible: same keys, shapes and init values.
- `init_report` does not record that the source wire was `sealed` and the target `kbit`. Add `source_wire=payload['meta'].get('wire','sealed')`.
- The sealed→kbit swap is acceptable. All four arms incl. the lambda=0 control share it, and the per-coordinate shift in decoder input is small: the floor codebook moves log-var by ≤ 0.092. D4-1 measured own-q at −0.10/−0.13 pp K0 on profile-4.

**m6 — `initial.pt` (line 611) omits `entropy_model`; em gradients are not checked for finiteness** (lines 719-721 check model params only). In lambda>0 arms a NaN em would surface through the model gradient check. In the lambda=0 arm it would only show up as NaN in the final rate.

### NOTE (verified correct)
- **N1 — Diff discipline.** Only the declared changes are present: `select` (380-389), own-q (408-409), EntropyModel/LogisticMixture (143-203), CLI flags and asserts (492-493, 579-581), em load/save (589-590, 616), opt group (601-602), rate in eval (624-628, 668-675) and train (694-702, 715-716), and the log line (726).
  - Loss, stage weights, data plans, the metric set, selection (best on VAL R8 wall F1), deadlines and the schedule formula are unchanged.
  - The eval and train encoder call `head(trunk(·))` + `latent_quantize`, which is exactly `model.source`.
- **N2 — Rate term.**
  - Both pmfs sum to 1 to within 1e-12 over 255 mean symbols and 254 reachable precision symbols (the top code nq-1 ⇔ p ≥ 128 is unreachable since q ≤ 4; the clamp to nq-2 is correct).
  - The end bins absorb the tails. The prior is included with the scene's own N.
  - The noisy relaxation matches both quantisers: round with grid-point centres for mu; floor with bin-midpoint centres for log p, so y+U(−s/2,s/2) spans exactly the floor bin.
  - Gradient routing, verified: the lambda term gives zero decoder gradient, nonzero head and trunk gradient, and no em gradient. The fit term changes no model gradient and fills all em gradients.
  - `total.backward()` then `r_fit.backward()` is safe because the graphs are disjoint. `clip_grad_norm_` sees model params only, so the lambda=0 arm's model trajectory is identical to running without the rate model. The extra `rand_like` draws in lambda>0 arms have no other RNG consumer (data order uses numpy `default_rng`).
  - Per-observed-coordinate normalisation is a sound proxy, since the presence mask is a hard function of the observation and the encoder cannot change the count.
- **N3 — Lattice join in training.**
  - K0 belief equals the own k-bit-encoded factor exactly (max diff 0).
  - R8 on a connected tree equals the centralised lex join at every node, bitwise.
  - R8 features alone carry finite encoder gradient through `gather` and the STE (head-grad norm 339).
  - No NaN: the own p_wire ≥ the 1e-8 code > 0 always (prior ≥ 1e-4/(64·16) = 9.8e-8), so the selected P is > 0. The tie key is computed under no_grad.
  - Interpretation caveat: rate pressure on q can collapse precision codes and increase precision ties. Ties resolve to the **larger mean** (a deterministic but semantic bias). Baseline tie rate is 4.7% of multi-source blocks (D4-1). Report it per arm, which M2's evaluator can do.
- **N4 — chain_day3_J.sh otherwise correct.** Arms and flags match the lease. "Largest admissible" is implemented as the last passing arm in increasing lambda. The thresholds match the pre-registration (bits ≤ 0.8·l0, R8 ≥ l0−1 pp, FPR ≤ l0+0.3 pp, VAL). The batch-I marker exists (`chain_day3_H.sh:73`). The pgrep patterns are scoped. Throughput (21,312 updates × 4 arms in 6,300 s) could not be verified here; m1's `updates` check covers truncation.

## Required changes before launch
1. M1: pre-fit or warm up the entropy model before the encoder rate term, and give the em group a constant LR.
2. m1 and m2: admissibility robustness (try/except, empty-ARM test, updates check) and binding REVIEW_OK to the sha.
3. Recommended, not required: the m3/m4 shared code helper, m5 provenance, and m6. M2 is post-hoc but mandatory before any B/message claim.

Scratch scripts and outputs: `/tmp/claude-0/-home-claude/4bad606d-6bc2-5614-a287-aa8ba2b79cab/scratchpad/review368/check_codes.py`, `/tmp/claude-0/-home-claude/4bad606d-6bc2-5614-a287-aa8ba2b79cab/scratchpad/review368/check_step.py`, `/tmp/claude-0/-home-claude/4bad606d-6bc2-5614-a287-aa8ba2b79cab/scratchpad/review368/check_fit.py` (output `/tmp/claude-0/-home-claude/4bad606d-6bc2-5614-a287-aa8ba2b79cab/scratchpad/review368/fit.txt`), `/tmp/claude-0/-home-claude/4bad606d-6bc2-5614-a287-aa8ba2b79cab/scratchpad/review368/check_misc.py`, `/tmp/claude-0/-home-claude/4bad606d-6bc2-5614-a287-aa8ba2b79cab/scratchpad/review368/smoke_l03.log`, `/tmp/claude-0/-home-claude/4bad606d-6bc2-5614-a287-aa8ba2b79cab/scratchpad/review368/lexjoin_tests.txt`.
