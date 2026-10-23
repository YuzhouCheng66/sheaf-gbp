# Independent code review: v348 k-bit code study (item 1, extreme coding), pre-launch

Reviewer: independent Claude Code reviewer, 2026-10-22. Scope: `/home/claude/sheafgbp/diag/train_v348_kbit.py` (619 lines, read fully) diffed against the already-reviewed `/home/claude/sheafgbp/diag/train_v347_maxsel.py` (566 lines). Sealed dependencies read where the new code touches them: `mapping_multiroom_forest_v295.py` (`quantize` L174-194, `advance` L197-223, `initialize`, `scalar_belief`, `validate_forest`, `MAX_NODES=16`), `mapping_implicit_model_v192.py` (`latent_quantize` L55-60, `latent_codes` L37-39), `train_mapping_support_native_v306.py` (`AGENTS=(8,16)` L29, `scene_plan` L118-121). Batch under review: arms b8/b6/b5/b4 = `--wire kbit --mu-bits b --q-bits b --fusion max --readout posterior_only --arm owned --size 128 --epochs 36 --seed 342101`; gate relative to b8 (K0 F1 >= b8-1pp, R8 F1 >= b8-1pp, FPR <= b8+0.3pp).

Method: `diff` of the two trainers, static reading, and CPU-only checks run from the scratchpad (`review348/check_v348.py`, `review348/check2_v348.py`; no GPU, no network, nothing under `diag/` or `handoff_src/` modified; bytecode writing disabled). The trainer module was imported with `make_rollout`, `kbit_wire_quantize` and `SheafModel.latent_quantize` driven directly, with the module globals `MU_BITS/Q_BITS/WIRE` set before each `make_rollout` call, exactly as `main()` does.

## 1. Diff v347 -> v348: only the k-bit machinery changed (PASS)

The complete diff is: new globals + `kbit_wire_quantize` (L112-127); `levels = 2**(MU_BITS-1)-1` in `SheafModel.latent_quantize` (L229, L231: 127 at 8 bits, identical to v347); `wire_q` selector (L245); new `rollout_sum_kbit` (L252-279) and its dispatch at the top of `rollout_sum` (L282-283); `rollout_max` calls `wire_q` instead of `forest.quantize` (L323); CLI flags (L406-408); globals set from argparse (L413) and a guard (L414); checkpoint/run metadata fields (L480). Loss (`stage_loss` L367-374), `truncated_dt`, `predict`, data loading (L434-459), seeds (L429, L543), optimiser/scheduler (L474-476), gradient clipping, validation cadence, model-selection score (R8 wall F1 on VALIDATION, L584), deadlines, `evaluate` and `node_metrics` are byte-identical to v347. With `--fusion max` the new `rollout_sum_kbit` is dead code, so the b8 arm is v347-max with the wire swapped and nothing else.

## 2. `kbit_wire_quantize` (L115-127) (PASS, one MINOR)

Empirical results (CPU):

| check | result |
|---|---|
| distinct precision levels on a 200,001-point log sweep of [1e-8,128], k=4/5/6/8 | 15 / 31 / 63 / 255 = 2^k-1 (plus exact 0 for p=0) |
| distinct mean symbols on a 200,001-point sweep of [-1,1], k=4/5/6/8 | 15 / 31 / 63 / 255 = 2(2^(k-1)-1)+1 <= 2^k |
| p = 0 / 1e-8 / 1e-8(1-1e-12) / 128 / 128(1+1e-12) | finite; 0 stays exactly 0 with eta_hat 0; endpoints map to 1e-8 and 128 (FP64 ulp); below/above range clamp to the end level, no NaN/inf |
| prior-only coordinate (p = 1e-4/(64N), eta = 0), N=8/16 | finite; mean symbol exactly 0; precision level listed in finding M1 below |
| STE: d p_hat/dp, d eta_hat/d eta, d eta_hat/dp | exactly 1, exactly 1, exactly 0 (identical structure to sealed `quantize` L194) |
| k=8 precision codebook vs sealed profile-4 `geomspace(1e-8,128,255)` | max relative difference 5.5e-8 (sealed casts through float32 at L193; kbit does not) |
| encoder alphabet (`SheafModel.latent_quantize`) at k=4/5/6/8 | exactly the multiples of 1/7, 1/15, 1/31, 1/127 in [-1,1] (2^k-1 symbols) |
| wire mean vs encoder code at worst case (q = 1e-4 floor, mu = +-1), N=8 and N=16, all k | after the belief path's `latent_quantize` the difference is exactly 0 (raw FP64 difference 2.5e-8..3.0e-8 = float32 representation of the same grid point, not a re-rounding) |
| shrinkage prior/(q+prior) at q = 1e-4 vs half code step | N=8: 1.95e-3 vs 3.94e-3 (k=8, margin x2.0), 1.61e-2 (k=6, x8.3), 3.33e-2 (k=5, x17), 7.14e-2 (k=4, x37); N=16 margins double. The K0 code is never re-rounded on the wire for any (k, N) in the plan. Note the k=8 margin is only x2: it depends on N >= 4 (for N <= 3 and q at the floor the 8-bit code would round down one step); the plan has N in {8,16} only (`AGENTS=(8,16)`). |

The precision codebook is `low*(high/low)^(code/(nq-1))`, code = round(...) (L122-123), i.e. the same geometric grid as the sealed profile-4 (L189-191) when nq = 255, so the b8 precision path is the sealed codebook. The mean is `round(eta/p * L)/L` (L125-126) computed from the raw p (not p_hat), so the transmitted mean is exactly the source's grid point; `eta_hat = p_hat * mu_hat` (L127). Clamp to [-1,1] can only act on FP noise since every cavity mean is a convex combination of grid points.

## 3. Bit accounting on the wire with `--fusion max` (PASS)

- Every message is `(p_hat, p_hat*mu_hat)` with p_hat one of 2^Q_BITS values (2^k-1 levels + the never-emitted exact-zero code; p_own = q*mask + prior > 0 always) and mu_hat one of 2^MU_BITS-1 values: representable as (mean index, precision index) in MU_BITS + Q_BITS bits per coordinate; log2(2^k-1) < k so "2k bits" is a slight over-count, which is the conservative direction. There is no block exponent or side information (unlike the sealed packet).
- Origin quantisation once, relays bit-exact: `rollout_max` quantises each source factor once (L323) and every subsequent message is a per-coordinate `select` (L303-308, gather) among `p_wire[i]` and incoming codebook values (L340-343); nothing is re-rounded or averaged. Verified on an 8-node tree with disjoint 8-coordinate ownership: all 448 relayed belief precisions are codebook values and all relayed belief means are `lq127` of grid points, at k=4 and k=8.
- Leak audit: the only non-codebook quantities inside `rollout_max` are `p_own[j]`, `eta_own[j]` used in `belief()` (L327-328), i.e. the node's own factor, which is node-local and identical to v347. The decoder reads `[lq127(E/P), log(1/P)]` only (L333, `OwnedDecoder` with `posterior_only` uses the posterior planes only, L182-183). `mask`, raw q and raw mu never cross an edge; targets reach only `stage_loss`/`node_metrics` (unchanged from v347).
- Union exactness: complementary owned sources on an 8-node path, max fusion, k-bit wire: every node's R8 mean equals `lq127(union)` with error exactly 0 at k=8, 6, 5 and 4 (the v347 sealed wire gives 2.36e-2 on the same construction because it codes eta/sqrt(p) with a shared block exponent). So b8-kbit is a *better*-behaved reference than v347-max on the mean path and identical on the precision path.

## 4. Equivalence of `rollout_sum_kbit` with the sealed `forest.advance` (PASS; not used by the batch)

- Structure (E1): with the sealed `forest.quantize` swapped in for the k-bit quantiser (the lambda at L245 resolves the global at call time, so this is a pure monkeypatch), `rollout_sum_kbit` reproduces the `forest.initialize/advance/scalar_belief` path bit-for-bit at stages 0/1/2/4/8 (max abs diff 0.00e+00 on an 8-node, 7-edge tree with random 50 % masks). The explicit cavity sums (L273-274) are therefore the sealed synchronous cavity rule.
- Wire (E2): with the real k=8 wire the two paths differ only through the mean codec: log-variance differs by 2.4e-7 (the float32 cast), means by up to 4.7e-2. Against an *unquantised* float GBP reference (`advance(quantized=False)`) the k-bit mean error is 2.6e-2 / 2.7e-2 / 1.9e-2 / 1.9e-2 at stages 1/2/4/8 versus the sealed wire's 2.4e-2 / 4.7e-2 / 4.7e-2 / 2.8e-2; log-variance errors are identical for both (4.5e-2 .. 1.16e-1). "Agreement up to wire quantisation" holds, with the sealed wire being the coarser of the two on the mean.
- Gradients through `rollout_max` with the k-bit wire are finite and non-zero w.r.t. source mean and precision (E3).

## 5. Is b8 a faithful reference? (PASS, two NOTEs)

- Ordering: `main()` sets `MU_BITS/Q_BITS/WIRE` at L413, before `make_rollout` (L432) and before `SheafModel(...)` (L461); `kbit_wire_quantize` and `SheafModel.latent_quantize` read the globals at call time, so no import-time capture. The `wire_q` closure is fixed at L245 inside `make_rollout`, after the globals are set.
- `evaluate()` (L491-536) uses the same `rollout` closure (L512) and the same wire; the refinement variant uses `model.latent_quantize` (L504, L511), which is MU_BITS-aware, and then re-runs the same rollout.
- Metadata: `run.json`/all checkpoints carry `mu_bits`, `q_bits`, `wire`, `bits_per_coordinate_on_wire` (L480).
- Training signal: identical to v347-max except the wire (section 1). The wire at k=8 is *not* bit-identical to the sealed profile-4 wire (different mean codec, section 3/4), so v347-max numbers are not a drop-in b8; the batch's own b8 arm is the correct reference and the gate is defined that way.

## Findings

**No BLOCKER. No MAJOR.**

- **MINOR M1 (L122, precision rounding direction).** `round` to the nearest log-codebook level can map a prior-only factor to a level *above* the receiver's own prior. Measured p_hat/prior for a prior-only coordinate: k=4: 1.42 (N=8), 0.54 (N=16); k=5: 1.14 (N=8), 1.05 (N=16); k=6: 1.03 (N=8), 0.97 (N=16); k=8: 0.96 (N=8), 1.012 (N=16; identical in v347 since the codebook is the sealed one). Under max-selection a no-information message (mean exactly 0) then wins every never-observed coordinate, shifting the decoder's R>=1 log-variance channel on those coordinates by -0.35 (k=4,N=8), -0.13 (k=5), -0.03 (k=6,N=8), -0.012 (k=8,N=16) relative to R0 (measured on an all-unobserved 8-node path; mean unaffected, exactly 0). The decoder is trained under it and it is a constant per-(k,N) offset on a channel already divided by 8, so it does not threaten the gate; but it is a k-dependent artefact unrelated to code length and breaks the clean statement "a prior-only message never overrides own prior". Options: disclose as is, or change `torch.round` to `torch.floor` at L122 (conservative codebook, p_hat <= p always; then applied to all four arms including b8, and the k=8 codebook is one level below the sealed one). Not a launch condition.
- **NOTE N1 (L423/L432, belief-side mean alphabet).** The belief mean fed to the decoder is quantised by the sealed 127-level `latent_quantize` at every k (L266, L294, L333), not by the k-bit alphabet. The k-grid to 127-grid map is injective (grid spacing >= 1/31 vs 1/127), node-local and off the wire, so it carries no extra information; the decoder sees each k-bit code at its nearest 127-level point (offset <= 1/254). Record this in the write-up when describing "the decoder reads the k-bit code".
- **NOTE N2 (L122 clamp).** The k-bit quantiser clamps silently where the sealed `quantize` raises (verified: p=200 -> sealed `ValueError`, kbit returns 128). Unreachable in this batch (prior >= 9.77e-8 > 1e-8 for N <= 16; cavity precision <= 60 < 128 for sum, <= 4 for max), but a future change of N or of the q envelope would be clipped without error. Negative p is silently zeroed as well.
- **NOTE N3 (L480).** `bits_per_coordinate_on_wire=16` for `--wire sealed` ignores the per-64 block exponent and the 172-byte packet framing (21.5 bits/coordinate including header); it is a label, not a measurement.
- **NOTE N4 (L414).** `--wire sealed --q-bits != 8` is accepted and recorded in metadata although q_bits is unused on the sealed wire.
- **NOTE N5 (L245-251).** The `wire_q` assignment was inserted before the `make_rollout` docstring, so the docstring is now a no-op string statement. Cosmetic.
- **NOTE N6 (design, k=4).** At k=4 the precision codebook step is a factor 5.2, so about 6-7 levels cover [1e-4,4]; overlapping sources (block-boundary blocks seen by two agents) will tie far more often than at k=8 and max-selection then resolves by candidate order (own `p_wire` first, then directed-edge order). Deterministic and inherent to the study, but worth stating when interpreting b4/b5.

## Verdict: LAUNCH OK

The diff is confined to the k-bit wire; the b8 arm is v347-max with the wire swapped; codebooks have exactly 2^k-1 precision levels and 2^k-1 mean symbols; STE paths match the sealed quantiser; K0 codes round-trip exactly for all (k, N) in the plan; relayed messages are codebook values and the decoder reads only the posterior through the bottleneck; the k-bit union is exact at every k; the explicit sum rollout reproduces the sealed `advance` bit-for-bit when given the sealed quantiser. The one MINOR (precision rounds up for prior-only factors at some (k,N)) should be disclosed or fixed by flooring in all four arms, but does not invalidate the comparison.
