# Analyst C: fusion, training objective and furniture (round 2, 2026-10-26)

**Bottom line.** The biggest single gap is the decision rule. The occupancy loss weights surface voxels 1 + w = 5 times, so σ(occ_logit) is not P(occupied): thresholding it at 0.5 calls a voxel occupied once the calibrated probability passes 1/6, and occupied outputs come out one voxel too thick. The rule matching the loss is `occ_logit >= log 5 = 1.61`; the TRAIN-chosen threshold is +1.5 for both seeds and transfers to VAL. With no retraining and no GPU, 3S-X goes R8 .877 → .908/.910, var .805 → .844/.848, K0 .901 → .929/.930, FPR inside the guard. **Recommended batch: one evaluation-only change to `predict`.** The fusion gap is half harder targets, half source ranking, not a decoder distribution shift. For furniture, the occupancy head is the bottleneck, not the veto.

## 0. Material
- Checkpoints: 3S-X `runs/v429-stack/c2-s{342101,342202}/checkpoints/final.pt`; replication on L1 `runs/v429-ladder/l1-s*` (16k).
- Houses: **24 TRAIN** (first 8 per room stratum of `cache3d-v420/train-r1-k1600`, regenerated for slab/object/structure masks; all 24 private-map hashes match the cache) and the **24-house VAL panel**; all 8 agents; K0 and R8. Thresholds always chosen on TRAIN, reported on VAL. Fresh panel and TEST untouched.
- Scripts next to this memo: `probe_C_gen.py` (fixtures), `probe_C_hist.py` (per-agent 2-D logit histograms in 12 voxel classes; join re-run with the winning source tracked; coverage-keyed join; copy-winner counterfactual; ~7 s/scene on 4 CPU threads), `probe_C_union.py` (union-encode oracle), `probe_C_analyze.py` / `probe_C_guard.py` (offline scoring). Summaries `probe_C_summary.json`, `probe_C_summary_L1.json`; uk-linux outputs in `analysis_20261026/C/`.
- Sanity: the default rule rebuilt from the histograms reproduces `metrics3d` exactly (K0 .90135/.90203, R8 .87655/.87735 = final.json).

## 1. Measurements

### 1a. Decision rule (VAL; thresholds picked on TRAIN; s1/s2)

| rule | K0 surface | R8 surface | R8 var | R8 furniture F1 | R8 FPR |
|---|---|---|---|---|---|
| default: occupied if k>=0 and o>=0 | .901/.902 | .877/.877 | .805/.805 | .572/.563 | .0201/.0204 |
| o >= +1.5, k >= 0 (FPR unchanged by construction) | .918/.919 | .898/.900 | .830/.832 | .562/.562 | .0201/.0204 |
| **split: occupied if o>=+1.5 and k>=−0.5; free if k>=0** | **.929/.930** | **.908/.910** | **.844/.848** | .587/.589 | .0223/.0228 |
| product σ(k)σ(o) >= .525 | .909/.910 | .886/.888 | .814/.817 | .564/.560 | .0201/.0204 |
| ceiling, inadmissible: no veto, o>=+1.5 | .947/.947 | .922/.924 | .870/.873 | .624/.629 | ≥ .09 |

- **Optimum.** The occupancy-threshold optimum is +1.5 on TRAIN and on VAL, both seeds, both hops; between +1.0 and +2.0 it stays within 0.35 pp of the best, a range that brackets log 5.
- **L1 replication** (weaker checkpoint): TRAIN optimum +1.25/+1.5; split rule R8 +4.8/+4.6 pp, var +5.3/+4.9, K0 +3.8/+3.5, R8 FPR +0.27/+0.29 pp. As expected, the less sharp model gains more.
- **Guards under the split rule:** slab recall R8 .960 → .975; occupied precision K0 .910 → .973, R8 .862 → .952; wall-zone recall R8 .895 → .887; furniture-region recall R8 .70 → .53/.54.
- **FPR trade-off:** each −0.25 on the occupied-output known threshold buys ~+0.4 pp K0 and +0.3 pp R8 for ~+0.12 pp R8 FPR. At k ≥ −1.5, K0 .941/.942 but R8 FPR +0.75 pp.

### 1b. Veto or occupancy head (VAL, default rule; % of target-surface voxels lost; s1/s2)

| class | K0 lost to veto | K0 lost to occupancy head | R8 lost to veto | R8 lost to occupancy head |
|---|---|---|---|---|
| slab | 3.8/3.7 | 0.1/0.0 | 2.7/2.5 | 0.2/0.0 |
| wall zone | 6.2/6.0 | 1.1/1.3 | 6.5/6.2 | 2.7/2.3 |
| furniture region | 15.5/15.5 | 13.0/12.5 | 12.7/13.0 | 17.3/16.8 |

At K0 the occupancy head had 13.3/13.5 of the 15.5 vetoed furniture points as occupied, but only at the biased threshold. Under the calibrated rule, furniture-region recall is .55/.56 (K0, k ≥ 0) with similar precision, so furniture-region F1 stays ~.56–.59 under every admissible rule.

### 1c. Fusion (VAL, R8 surface F1, s1/s2)

| readout | R8 surface F1 |
|---|---|
| K0 (own code to own map) | .901/.902 |
| union-encode oracle: R8 union map encoded as own, no join | .889/.887 |
| fused R8, learned-precision join (actual) | .877/.877 |
| fused R8, join re-keyed by exact block coverage (evaluation only) | .889/.890 (var .821/.822) |
| copy the winner's own K0 decoding | .848/.848 |
| fused R8 scored only on voxels its winning source observed | .895/.897 |

Each R8 target-surface voxel falls into one class:

| class | definition | share | recall | precision |
|---|---|---|---|---|
| S | own code wins, observed | 18 % | .94 | .905 |
| O | another agent's code wins; that agent observed the voxel | 68 % | .91 | .870 |
| M | the winning source never observed the voxel | 14 % | .81/.82 | .73/.72 |

M holds 28–29 % of the false positives; keying the join by coverage cuts the M share to 6 %.

## 2. Diagnosis

1. **Calibration: yes, the objective does not match the metric (confidence high).** The offset log(1 + w) is predicted from first principles and confirmed on two seeds, two checkpoints and TRAIN → VAL transfer. ~1.7–2.3 pp is free at zero FPR cost, ~2.8–3.3 pp within the FPR guard. Fat occupied outputs are the pack's "one-voxel misplacement": occupied precision goes .91 → .97 once calibrated. The split rule already meets R8 ≥ .90. K0 stays 2 pp short; the no-veto ceiling (.947) shows its limiter is the known head's border at surfaces, held in place only by the FPR guard.
2. **Fusion gap −2.5 pp (confidence medium-high).**
   - ~1.2–1.5 pp **target difficulty**: even the union oracle loses it. Same "content per code coordinate" limit as the house-size effect, so a layout question (3S-Z).
   - ~1.0–1.3 pp **source ranking**: the coverage-keyed join reaches the union oracle.
   - **No mosaic penalty:** the decoder beats copy-winner by 2.8–3.0 pp and is only 0.6 pp below K0 on winner-observed voxels.
   - Caveat: the oracle's encoder never saw union maps, so the oracle may understate the ceiling.
3. **Furniture (confidence medium).** The occupancy head is the bottleneck; the veto's apparent share comes from the biased threshold. Calibrated furniture-region recall is .55–.60 (K0) / .50–.54 (R8) with similar precision: missing information in code and decoder (the layout ladder). No admissible rule lifts furniture F1 past ~.59.

## 3. Recommended batch: 3S-DR, decision rule matched to the loss (evaluation only)

- **Exact change.** A flag `--decision {v420, calibrated}` on the evaluation/diagnostic path; default `v420` = current `predict`, bit for bit. `calibrated`:
  ```
  occ   = (occ_logit >= 1.5) & (known_logit >= -0.5)
  known = (known_logit >= 0) | occ
  pred  = where(occ, 1, where(known, 0, -1))
  ```
  Both constants frozen from the 24 TRAIN houses (FPR guard applied on TRAIN). Weights, data, loss and metric definitions unchanged.
- **Cost.** **No GPU, no lease** (vs 0.20 s/update, 7,200 s). CPU ~7 s/scene/checkpoint (measured): VAL + fresh, two seeds, ~12 min; can run alongside 3S-Z.
- **Control.** The same `runs/v429-stack/c2-s*` checkpoints under the default rule, paired by seed and house.
- **Pre-registered prediction**, scored on the untouched **fresh 24-house panel**, reported with VAL. Estimates from VAL: ΔR8 surface +3.2 ± 0.6 pp, ΔR8 var +4.1 ± 0.8, ΔK0 +2.8 ± 0.6, ΔR8 FPR +0.23 ± 0.06 pp. Lever rule: ΔR8 var ≥ +3 and ΔR8 surface ≥ +1 pp on both seeds. Guards: R8 FPR ≤ control + 0.3 pp, slab recall ≥ control − 2, K0 var ≥ control − 2.
- **Falsified if:** on fresh, ΔR8 surface < +1 or Δvar < +3 pp on either seed (not a lever); FPR guard broken (fall back to `k >= 0`: FPR identical by construction, VAL R8 +2.2/+2.3, var +2.5/+2.7); mechanism falsified if any w = 4 checkpoint (e.g. 3S-Z) has its TRAIN-optimal occupancy threshold outside [+1.0, +2.0].
- **For coordinator and user.** This changes how every reported number is produced; re-score all paired comparisons under one rule. Sharper models gain less (3S-X +3.2 vs L1 +4.7 pp R8), so the rule will **shrink** gaps between recipes; score the 3S-Z verdict against L1 under both rules.
- **Not recommended now:** retraining with w = 0 (the threshold calibrates for free and keeps the surface-gradient focus); hop reweighting (neither part of the fusion gap is a training-signal problem the decoder can fix); coverage-keyed selection, +1.2 pp (needs a key on the wire, breaking the max-precision join, or 3S-Q's coverage precision, worse in training). Next GPU candidate if needed: the known-head border at surfaces (~1.8 pp of K0 to the no-veto ceiling).

## 4. Orthogonality with 3S-Z (64x64x1)

Fully orthogonal and stackable: the rule is applied after training to any checkpoint's logits, and its offset comes from the loss weight (w = 4), which 3S-Z keeps. 3S-Z's TRAIN optimum should again be ~+1.25 to +1.5, with a positive but smaller gain if 3S-Z is sharper (estimate: R8 +2 to +3.5 pp). 3S-Z attacks what the rule cannot: the target-difficulty half of the fusion gap, and furniture.
