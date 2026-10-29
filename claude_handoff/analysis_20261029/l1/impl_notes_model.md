# L1 model side: implementation notes (Claude, 2026-10-29)

Pre-registration D8-82 (`claude_handoff/DESIGN_LLM_LATENT_SHEAF_GBP.md`, sha `8c85c2f0...`). Status: implemented and unit-tested on CPU
with tiny random models; NOT run on the real models, NOT launched. Nothing committed; no GPU, no downloads, no ssh.

## Files (all under `scripts/claude_v342/`)

| file | role |
|---|---|
| `l1_model.py` | SenderAdapter, Reader, Agent, Pair (the arms), tiny models, shared training loop, evaluation (`eval_plain`, `eval_latent`, `absence_curve`), readouts |
| `l1_train.py` | CLI for B0 / B4 / U / S1 / S1L1: train one epoch, evaluate VAL (+ absence curve) and fresh, write `summary.json`, `trainable.pt`, `log.txt` |
| `l1_text.py` | CLI for T (sender LoRA, two-stage receiver LoRA, budget sweep, absence with the empty message) and T0 (frozen base sender + T's receivers) |
| `chain_day9_L1.sh` | the uk-linux chain: one launch = one 7,200 s lease (`--lease N`), stage 0 -> P0 gate -> stage 1 in a fixed order, `FILL_` sha pins, lock dir, SOLO GPU wait, `timeout -k 30`, never pkill |
| `test_l1_model.py` | 19 pytest tests (CPU, no network): tiny LlamaConfig models (hidden 32 / 48, 2 layers; tiny-b without BOS) + WordLevel tokenizers built in the test |
| (read only) `l1_fusion.py` (coordinator), `l1_data.py`, `l1_aggregate.py` (other agent) | fusion math; prompts / metrics / jsonl; verdict and `summary_schema()` / `validate_summary()` |

Test command and result (venv `C:\Users\27118\AppData\Local\Temp\claude\venv-l1`: torch 2.15.0+cpu, transformers 4.52.4, peft 0.19.1, tokenizers 0.21.4):

```
CUDA_VISIBLE_DEVICES= python -B -m pytest scripts/claude_v342/test_l1_model.py scripts/claude_v342/test_l1_fusion.py -q
30 passed in 44.86s
```

(19 tests of `test_l1_model.py` + 11 of `test_l1_fusion.py`.) The `--tiny` subprocess runs of `l1_train.py` (S1 hetero, B0 same, B4 hetero) and
`l1_text.py` (T then T0) are part of the suite; every summary they write passes `l1_aggregate.validate_summary` with no problem.

## How the arms map to code

All arms share `Agent` (frozen base + peft LoRA r 16 / alpha 32 / dropout .05 on q,k,v,o,gate,up,down; gradient checkpointing, non-reentrant)
and `Agent.loss` (pass 2: `inputs_embeds = [BOS?] prefix | K soft tokens | suffix | answer EOS`, right padding, CE on the answer tokens + EOS
only; logits are computed only at the answer positions through `decoder` + `lm_head`, not for the whole sequence).

| arm | `Pair.step` | evaluation |
|---|---|---|
| B0 | each agent: `loss(pieces(no_comm_prompt(ex, i)))`, no soft tokens | `eval_plain` |
| B4 | one agent: `loss(pieces(full_prompt(ex)))`; **one run trains every model of the pair one after the other** (hetero: Qwen3 then Llama) and writes one summary with `val.em_model` per model (see Decisions 1) | `eval_plain` per model, merged by `l1_train.merge_models` |
| U | `encode_both` -> z (2, B, K, D_e); `soft_i = Reader_i(z_j)`; training: each directed message zeroed with p = .3 per example (`soft_from_train`); adapters `fixed_lam=True` (no precision head), no w | `eval_latent`: `soft_from_eval` zeroes dropped messages |
| S1 | `x = fuse_direct(z, lam, [(0, 1)], w)`, `soft_i = Reader_i(x_i)`, w = softplus(w_raw), w = 1 at init; no dropout | `eval_latent`: `gbp_sync(z, lam, edges, w, rounds = tree_diameter = 1, drop, wire_dtype = fp16)`; C assertion on the first batch |
| S1L1 | as S1 with `fixed_lam=True` (lam = 1 everywhere, no precision parameters: S1 - S1L1 = exactly the two `lam_head` Linear layers) | as S1 |
| T | `l1_text.py`: sender LoRA per model (one epoch, loss on `text_sender_target` + EOS); messages greedy, `max_new_tokens = budget` 64; receiver LoRA per model (own Agent object): stage 1 one epoch with the OTHER agent's gold message (truncated to 64 tokens with the sender model's tokenizer), stage 2 625 steps on a fixed seeded 5,000-example subset with the trained senders' messages | messages for VAL / fresh by the trained senders; budgets {16, 64, 256}; absence curve with the empty message; sender latency = wall time / example + decode steps |
| T0 | no sender training: `Agent(lora=False)` (frozen base) writes <= 64 tokens from `text_sender_prompt`; receivers = T's `trainable.pt['receiver']` loaded by model name | as T, arm `"T0"` |

Training loss = mean of the two agents' answer losses (B4: the one agent). Pass 1 is under grad: `test_s1_loss_finite_and_gradients_reach_everything_on_both_agents`
checks non-zero grads on both LoRAs, both adapters, both readers and `w_raw`, and that base weights get no grad.

`summary.json` (l1_train): `arm, pair, seed, model (None unless --model), models, val, fresh, absence ({"0.0", "0.3", "0.6", "1.0"} on VAL; latent arms),
c_assert (S1 / S1L1 only), w, complete, steps, agents, params, train (loss history, step time), timings, config, scripts_sha (l1_model / l1_train /
l1_fusion / l1_data)`. `val` / `fresh` blocks: `em, f1, em_agent, f1_agent, em_model, f1_model, n, em_by_owner {own, other, none, all: {em, n}},
em_by_owner_agent, em_by_type, disagreement {mean, std, corr_with_em}, lam [per agent: mean_log_lam, frac_at_lower, frac_at_upper, mean_lam,
corr_mean_lam_entropy, corr_mean_log_lam_entropy], first_token_entropy_mean, wire_bytes_per_example, wire_bytes_per_message, sender_latency_s_per_example,
answer_gen_s_per_example, answer_decode_steps_mean, truncation, c_assert_detail, absence_p1_detail, n_messages_dropped`. EM / F1 in percentage points.

## Decisions / ambiguities (the simplest reading was implemented)

1. **B4 = one run per (pair, seed) holding every model.** `l1_aggregate.py` reads P0 from the single `B4-hetero-s<seed>/summary.json`
   (`val.em_model[model]` for both models) and names run dirs `<arm>-<pair>-s<seed>`; a per-model B4 run (as the brief said) could not feed P0.
   So `l1_train.py --arm B4` trains Qwen3 then Llama sequentially (one Pair at a time, the first is freed before the second loads), merges the two
   evaluations (`em_agent` = [Qwen3, Llama], `em` = their mean, full per-model readouts under `val.per_model`). `--model` still restricts a run to one
   model (diagnostics only). Consequence for the chain: B4 jobs get their own estimate `--est-b4` (default 50 min; two trainings) instead of 25.
2. **`c_assert` is written only for S1 / S1L1** (true or false, never skipped: `test_c_assert_false_is_written_not_skipped`). The key is absent for
   B0 / B4 / U / T / T0 because `validate_summary` treats any non-True value as "c_assert false" -> INCOMPLETE, and the schema says "optional for U".
3. **Pass 1 input** = the agent's own latent prompt without soft tokens (`prefix + suffix` = own 5 paragraphs + question, BOS first if the tokenizer
   has one); `hidden_states[L]` with `output_hidden_states=True`, L = floor(0.6667 x n_layers) = 18 / 28 (Qwen3-1.7B), 10 / 16 (Llama-3.2-1B).
4. **SenderAdapter**: queries (K, D_e) ~ N(0, .02^2); q / k / v / o projections; residual on the queries + LayerNorm; MLP 4 x D_e + LayerNorm;
   final LayerNorm -> z (the trivial-solution guard); precision head = one Linear(D_e, D_e) on the pre-LN features, zero-initialised so lam = 1
   at step 0, clamped to [ln 1e-3, ln 1e3] (clamping stops the gradient at the bounds; the readout reports the clamped fractions). The adapter and
   the reader run in float32 with autocast disabled (z, lam feed the float32 solve); the soft tokens are cast to the base dtype (bf16 on CUDA).
5. **Loss** = mean over all answer tokens (+ EOS) of the batch (token mean, HF style), not a per-example mean. Per-example losses and the
   teacher-forced first-token entropy are returned in `info` but not used for training.
6. **Generation**: greedy, `num_beams=1`, left padding, `model.generate(inputs_embeds, attention_mask)` returns only the new tokens (verified
   in `test_generate_left_padded_batch_matches_single_and_returns_new_tokens_only`); the prediction is the first line of the decoded text, stripped.
   The first answer token's entropy comes from `output_logits=True` (step 0) = the same distribution as the teacher-forced one.
7. **U**: adapters are built with `fixed_lam=True` (parameter count = S1 - the two precision heads - w; U has no `w_raw`). The training dropout
   draws one Bernoulli(.3) per directed message per example from the global torch RNG (seeded). Wire bytes for U = K x D_e x 2 B per message
   (means only, fp16); S1 / S1L1 = `wire_bytes(K, D_e)` = 32 KB per message (means + precisions); `wire_bytes_per_example` = 2 messages.
8. **Absence curve**: per evaluation pass a `torch.Generator` seeded with `seed * 1009 + round(100 p)` draws one drop per directed message per
   example; the (2, B) mask is expanded to (2, B, K, D_e) for `gbp_sync` (its reshape accepts (2E, 1, 1, 1) or the full shape, not (2E, B, 1, 1)).
   U: dropped message = zero vector. The p = 1 check of the design ("S1 equals B0 of the same network") is recorded as `absence_p1_detail`
   (max |soft_i - Reader_i(z_i)|, `equal` within 1e-5 relative), not raised: the trained B0 arm is a different network, so "the same network
   with its own z" is the implementable statement.
9. **C assertion**: first VAL batch of the p = 0 pass, float32, `fuse_direct` vs `gbp_sync(rounds = 1, no drop, no wire)`, tolerance
   1e-4 (1 + max |x|); the details (max diff, scale, w, batch size) are in `val.c_assert_detail`.
10. **Readouts**: disagreement = `edge_residual(z)` (Frobenius over K x D_e) per example, Pearson with the mean EM of the two agents on that
    example; lam statistics per agent per example (mean log lam, fractions within 1e-6 of the bounds), Pearson of mean lam (and of mean log lam)
    with the first-token entropy; Pearson returns null when fewer than 3 points or zero variance. Owner buckets: own / other / none relative to the
    answering agent (B4: "all"). `em_by_type` (bridge / comparison) is an extra.
11. **`--max-ctx-tokens 900`**: if `[BOS] prefix suffix` exceeds it, every paragraph of the shard the prompt reads (B4: all 10) is cut to
    floor(f x its tokens) with f = (P - over - 2 per paragraph) / P and re-tokenised; if still over, the prefix ids are tail-cut (code 2). Counts
    of codes 0 / 1 / 2 are in `train.truncation_train` and `val.truncation`. (`all_paragraphs` / `shards` are assumed to be lists of
    `[title, sentences]`, as `l1_data.shard_example` writes them.)
12. **Optimiser**: AdamW betas (.9, .999), eps 1e-8, weight decay 0 (the design gives none); LoRA params = names containing `lora_`, everything
    else trainable (adapter, reader, w_raw) at the adapter lr; warm-up 100 then cosine to 0 at `--steps`; grad clipping 1.0 over all trainable
    params; bf16 autocast on CUDA only (float32 on CPU). `--steps` beyond one epoch re-shuffles per epoch (seed, epoch).
13. **Timing**: step times are kept only from step 51 (`TIMING_WARMUP_STEPS = 50`): `step_time_post_warmup_s` is null when a run has fewer
    than 51 steps; the log prints "rate: warm-up" before that. Sender latency = pass 1 + fusion wall time per example with
    `torch.cuda.synchronize`; text sender latency = message generation wall time per example and the mean decode steps (incl. EOS).
14. **Deadline**: `--deadline-epoch` (the chain passes lease end - 60 s) raises `Deadline` between steps / batches -> `INCOMPLETE.json`,
    exit 3, no `summary.json` (the aggregator then reports the arm as missing). A run never resumes; the chain never re-runs into a dir that
    exists without a summary.
15. **Text arm details**: the two senders are trained jointly (one AdamW over both LoRAs, loss = mean over the two agents' sender losses on the
    same batch; the same pair = one shared sender seeing both shards); the receivers likewise. Stage 1 and stage 2 of the receivers use separate
    AdamW instances and schedules (stage 2: warm-up 100 + cosine over 625 at the same lr 2e-4). Gold messages are `text_sender_target` of the
    other agent cut to the budget with the SENDER model's tokenizer; the stage-2 subset is the first 5,000 of a seeded permutation (seed + 7)
    of TRAIN and its messages are generated by the trained senders before the senders are freed. Budget 256 messages are generated at
    evaluation only; 16 = the 64-token messages cut to 16 sender tokens. `message_f1_vs_gold` (token F1 of the generated message against the
    gold target) and `message_chars_mean` are extra readouts. T's `trainable.pt` = `{'sender': {model: LoRA state}, 'receiver': {...}}`;
    T0 refuses a checkpoint of another pair or LoRA config. T's absence curve replaces a dropped message by the empty string (design 5).
16. **Chain** (`chain_day9_L1.sh D8-82 "<approval UTC>" --lease N [--est-base 25 --est-b4 50 --est-latent 50 --est-text 50 --est-t0 25]`):
    one launch = one lease; `--lease 1` runs stage 0 only (B0-hetero, B4-hetero x both seeds, seed-major) and then `l1_aggregate.py --runs
    runs/l1 --stage0`; the chain reads the printed `P0: PASS / FAIL / INCOMPLETE` line (the aggregator always exits 0) and writes
    `runs/l1/STAGE0_PASS` or `STAGE0_FAIL` (FAIL -> the chain exits 0 and later leases do nothing). Later leases first finish any missing stage-0
    job, then run stage 1 in the fixed order T, U, S1, S1L1, T0 (hetero, seed-major per arm), then same-pair B0, B4, T, S1; a job whose
    `summary.json` exists is skipped, a job whose dir exists without one is skipped with a log line (never overwrite), a job whose estimate does
    not fit the remaining lease time stops the lease. Every job: `timeout -k 30 <lease end>` and `--deadline-epoch`. The aggregator's
    `aggregate.json / .md` (written into runs/l1 and overwritten each call) are copied to `runs/l1/aggregate/<tag>.*`. HF offline
    (`HF_HUB_OFFLINE=1`): the two models must already be in `$HF_HUB_CACHE` (the chain aborts otherwise and never downloads). `FILL_` pins:
    the six scripts and the three jsonl splits.
17. **l1_data API differences from the brief** (adapted): `sup` is `[sentences of the gold paragraph in shard 0, ... in shard 1]` (not
    `[title, idx]` pairs) and `gold_title` is a 2-list; `exact_match` returns 1.0 / 0.0 (cast to int in the records). The fallback data API
    inside `l1_model.py` (`_FallbackData`) follows these; it is used only when `l1_data.py` cannot be imported and `l1_train.py` / `l1_text.py`
    refuse it outside `--tiny` / `--allow-fallback-data`.
18. Tokenizers: BOS is prepended manually (`add_special_tokens=False` everywhere) only when `tokenizer.bos_token_id` exists (Llama-3.2 yes,
    Qwen3 no); a missing pad token is set to EOS (Llama-3.2); EOS of the answer = `tokenizer.eos_token_id` (Qwen3 `<|im_end|>`, Llama
    `<|eot_id|>`). Generation passes `do_sample=False` explicitly (Qwen3's generation_config defaults to sampling; transformers only warns).
19. `R2_ROUNDS = None` in `l1_model.py` is the TODO flag for the deferred R = 2 readout: any other value makes `Pair.__init__` raise
    (`test_r2_flag_errors_when_set`). The same-pair L sweep (design 3, "实现前只在同模型臂上扫 L") is not in the chain; `--layer-frac` exists.

## Not implemented as written

- Per-model B4 runs with `--model` as the brief described -> replaced by Decision 1 (aggregator contract). The brief's chain job list
  "B4 for both hetero models" is therefore 2 runs (one per seed), not 4.
- The chain's per-job defaults: B4 50 min (two trainings) instead of 25; T is expected to need ~90-110 min on the 5090 (two sender epochs,
  two receiver stages, ~14k generated messages incl. the 256-token sweep): pass `--est-text 110` after the first measurement, otherwise the
  fit rule will not start T in a lease that already ran another job.
- `c_assert` for U is not written (nothing is fused in U; the aggregator calls it optional and rejects any non-True value).
- R = 2 (flag only), the L sweep, Gemma-3-1B-it: not implemented.

## Memory / time expectations (5090, bf16, to be re-measured; never quote pre-warm-up rates)

- Latent arm (hetero): Qwen3-1.7B bf16 ~3.4 GB + Llama-3.2-1B ~2.5 GB + LoRA/adapters (~30 M params, fp32 master + Adam ~0.5 GB) + activations
  at batch 8 x ~950 tokens with checkpointing and `output_hidden_states` (~0.9 GB per pass for Qwen3): ~12-18 GB peak. Per step 2 x (pass 1 +
  pass 2) with backward through both: 0.8-1.5 s -> 2,500 steps 35-60 min; evaluation = 4 absence passes on VAL + fresh = 5 x 1,000 examples x 2
  agents (pass 1 + 16-token greedy) ~ 8-12 min. The 50-min default estimate is tight; re-estimate from the first S1 run's
  `timings.step_time_post_warmup_s` and `timings.eval_s`.
- B0: ~0.4-0.6 s/step (two forwards) -> ~20-25 min + ~4 min eval. B4: one model at a time, ~1,200 tokens, ~0.4-0.6 s/step -> ~20-25 min each,
  ~45-50 min per run.
- T: senders 2,500 steps (~15-20 min), messages for 5,000 + 2 x 1,000 examples at 64 tokens plus 2 x 1,000 at 256 (~25-35 min at eval batch 16),
  receivers 2,500 + 625 steps (~25 min), evaluation 3 budgets x 2 splits + 3 absence passes (~10 min): ~90-110 min. T0: ~15-20 min.
- `output_hidden_states=True` in pass 1 keeps all 29 / 17 layer outputs; if memory is short, a forward hook on layer L - 1 would halve it
  (not done: the brief asked for `hidden_states[L]`).
