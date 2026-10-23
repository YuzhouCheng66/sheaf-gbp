# Day-5 first-principles round: brief for the independent analysts (Claude, 2026-10-23 15:30 UTC)

Goal set by the user: find the **structurally new design most likely to raise HouseExpo accuracy by a large margin** (not another
+1 pp), under the hard constraints, then launch it as a pre-registered experiment. Five analysts work independently from this brief and
the files listed below; the main agent synthesises and picks ONE change plus its matched control.

## Hard constraints (README section 3; non-negotiable)
1. Each agent's map is **64 Gaussian variables + 64 diagonal precisions**. No extra private rasters, tokens or side channels into fusion
   or readout; the encoder may be wide internally, but everything sample-dependent that reaches fusion/readout passes this bottleneck.
2. Fusion stays Sheaf-GBP with a declared energy / message semantics. Current: per-coordinate **max-precision selection (lattice join)**,
   which is exact on any graph. Precision must remain a confidence (it decides the join); it cannot smuggle geometry.
3. The decoder reads only the bottleneck posterior (q, h) (plus public frame information that is identical for all agents and disclosed).
4. TRAIN fits / VALIDATION selects / TEST sealed (HouseExpo TEST houses are not even in the index).
5. One batch = one inclusive 7,200 s lease on one RTX 5090; at most two training jobs in parallel; a from-scratch run of the current model
   does 57,600 updates in ~5,400 s with two jobs; the decoder-width-2 arm (batch T, running now) is slower.
6. Pre-registration: hypothesis, single intervention, matched control, criterion (two seeds must both pass), written before launch.

## Where we are (all numbers verified from run files; see BEST_RESULTS_20261023.md and OVERNIGHT_REPORT D4/D5)
- Best HouseExpo model (v384, three cosine cycles): VAL R8 wall F1 **.893 / .901**, K0 **.918 / .920**, fresh R8 .871 / .876;
  8+-room houses R8 .82–.87 (weakest), 2–4-room .92–.94. Accuracy .974–.977, unknown-FPR .011–.013.
- Synthetic v293: R8 .94 (three seeds). HouseExpo zero-shot from v293 models: R8 ~.42 (walls vanish) -> v293 numbers are in-distribution.
- Progression on HouseExpo, one variable at a time: fixed 20 m chart + 600 houses: R8 .64/.65 -> house-fitted square chart (+10 pp,
  "code capacity density": blocks on the house 24 -> 45 of 64) -> 7,200 houses (+5 to +10 pp, decoder prior data-limited; 14,400 saturates)
  -> 4 epochs (+2.5/+1.1) -> 2nd cosine cycle (+2.4/+2.4) -> 3rd cycle (+1.0/+1.3, halving).
- Things that did NOT help: 16 bits per coordinate instead of 8 (ceiling unchanged => "one scalar per block" is the limit, not bits);
  rectangular charts (+0.25/+2.6, ambiguous, placebo noise -5.9..+2.1); consistency/homomorphism loss (max fusion already loses <= 0.4 pp vs
  ideal union encoding); sum fusion (worse than max); anchored readout (worse than posterior-only); SDF head, free-vs-wall head (v293).
- Ceiling diagnostics (free codes fitted per agent, decoder trained on TRAIN with free codes, 4,000 steps; fixed 20 m chart unless noted):
  8x8x1 (64 dims) K0 .694 (house-fitted chart: .819); 4x4x4 (64 dims, 4 scalars per 5 m block) .623; 16x16x1 (256 dims) .918.
  Note: the trained v384 model (K0 .92) now EXCEEDS the .819 free-code ceiling, so that protocol under-estimates what a jointly trained
  encoder/decoder reaches; treat ceiling numbers as relative, not absolute.
- Information content (lossless adaptive-context coding, VAL medians by stratum): whole-house wall map 203 / 284 / 424 bits; axis-aligned
  polygon description 264 / 449 / 731 bits (27 / 48 / 79 vertices, 100 % axis-aligned); K0 target 333-362 bits; R8 target 347-799 bits.
  The code carries 64 x 8 = 512 bits of means + 64 x 8 bits of precisions per agent. So raw capacity is NOT the limit; the fixed
  one-scalar-per-block semantics is.
- Residual anatomy (figures/houseexpo_best_v383_panels.png; reviews error_anatomy_*): missed short wall stubs beside doorways, narrow
  openings, thin internal walls in large houses; free space and outer walls are nearly perfect.
- Run-to-run noise ~ +-1-2 pp; placebo groups up to -5.9..+2.1 pp => any claimed gain must clear +1 pp on both seeds.
- Runtime per agent per step on a 5090 (H128): encode 0.44 ms, decode 0.73 ms, 50-step Adam local refinement 122 ms, join 0.02 ms.

## Current architecture (scripts/claude_v342/train_v378.py; lines 53-260 are the model)
- Public chart: house-fitted square, L = max(8, ceil(bbox + 1 m)); 8x8 public blocks of L/8 m; raster H = 128 (cell L/128).
- Encoder PatchTrunk: strictly block-local CNN on each 8x8 block (inputs: known, free, wall planes + xy); OwnedHead -> mu_k in [-1,1]
  (hard-8 quantised via STE), q_k in [1e-4, 4] (learned, times the indicator "block k has >= 1 own-known cell").
- Wire: sealed profile-4 quantiser (8 + 8 bits per coordinate); messages are (precision, precision x mean) per coordinate.
- Fusion: max-precision join per coordinate over own factor + incoming messages, 8 synchronous rounds on the radio forest.
- Decoder OwnedDecoder: input 2 planes at 8x8 (posterior hard-8 mean, log posterior variance / 8) -> 2 convs (64 ch) -> bilinear to 16x16
  -> CnnTrunk (lift 128, stages 96/64/48/32 with bilinear upsampling to 128x128, + xy/scale planes at every stage) -> 2 channels:
  known logit and a truncated distance-to-wall field (wall := DT < 0.5). Loss: known BCE + weighted L1 on DT (stages K0, R1, R2, R4, R8).
  Decoder = 630k of 866k parameters. The decoder does not know the chart size L (pixel scale varies per house).
- Training: AdamW lr 3e-4, 200-update warm-up + cosine, 14,400 TRAIN houses (cache), 4 epochs per cycle.

## Experiments in flight (do not duplicate)
- Batch T (v386): decoder width x2 (2.49 M params) vs batch-Q control at equal 57,600 updates. Result in ~2 h.
- v387 diagnostic (CPU): 50-step Adam refinement of each agent's own code through the frozen v384 decoder (analysis-by-synthesis encoding).
  Pre-registered reading: refined K0 >= +3 pp => the feed-forward encoder is a bottleneck; <= +1 pp => decoder/representation is.

## Files to read (local paths relative to the project root)
claude_handoff/README.md (sections 3, 4, 7), claude_handoff/HANDOFF_20261023.md, claude_handoff/BEST_RESULTS_20261023.md,
claude_handoff/OVERNIGHT_REPORT_20261021.md (sections D3-3, D4-3..D4-21, D5-0..D5-2; the file is long, grep for the section headers),
claude_handoff/claude_reviews_20261021/{encoder_information.md, energy_restriction_gbp.md, design_outline_post_accuracy.md,
lit_review_v342_representation.md, error_anatomy_v347max_s342202.json, code_review_v375_datascale.md},
scripts/claude_v342/train_v378.py, scripts/claude_v342/mapping_houseexpo_data_v371.py, scripts/claude_v342/owned_ceiling_v371.py,
claude_handoff/figures/houseexpo_best_v383_panels.png (view it).

## What each analyst must deliver (write to claude_handoff/claude_reviews_20261021/fp_day5_<role>.md, in Chinese, <= 1,500 words)
1. 第一性原理：从你的角度，HouseExpo 上 R8 .89–.90 的残差到底由什么决定（给出推理链和引用的证据）。
2. 候选结构（最多 3 个）：机制、为什么它能带来 >= +3 pp（给出数量级估计和依据）、它如何满足全部硬约束（逐条）、实现代价（改哪些模块）、
   一个 7,200 s 租约内能否完成训练、匹配对照是什么、预注册判据（两个种子都要过的阈值）、什么证据会否定它。
3. 反证：你最推荐的候选最可能怎样失败；已有哪些结果与它相悖。
4. 排序：top-1、top-2，和你明确不推荐的方向（说明原因）。
Rules: cite line numbers or section names for every factual claim; separate evidence from conjecture; do not propose anything that
relaxes the 64 + 64 constraint or lets the decoder see private inputs; do not propose "more of the same" (longer training, more data,
wider decoder) as top-1 since those are measured or in flight.
