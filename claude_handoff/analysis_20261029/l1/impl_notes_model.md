# L1 model side: implementation notes (Claude, 2026-10-29; revised after the 5090 smoke numbers)

Pre-registration D8-82 (`claude_handoff/DESIGN_LLM_LATENT_SHEAF_GBP.md`, sha `8c85c2f0...`). Status: implemented and unit-tested on CPU
with tiny random models; the coordinator's smoke runs on the real models produced the timings quoted below; NOT launched. Nothing
committed; no GPU, no downloads, no ssh from this side.

## Files (all under `scripts/claude_v342/`)

| file | role |
|---|---|
| `l1_model.py` | SenderAdapter, Reader, Agent, Pair (the arms), tiny models, shared training loop, evaluation (`eval_plain`, `eval_latent`, `absence_curve`), readouts |
| `l1_train.py` | CLI for B0 / B4 / U / S1 / S1L1: train, save, evaluate in the criterion-first order, `summary.json`; `--eval-only DIR` (readouts only); `--merge-b4 DIR_A DIR_B --out DIR` (CPU merge of two per-model B4 runs) |
| `l1_text.py` | T in phases (`--phase sender` / `receiver` / `sweep256` / `all`) and T0 |
| `chain_day9_L1.sh` | the uk-linux chain: one launch = one 7,200 s lease (`--lease N`), stage 0 (B0, B4l, B4q, merge) -> P0 gate -> stage 1 in a fixed order, `FILL_` sha pins, lock dir, SOLO GPU wait, `timeout -k 30`, never pkill |
| `test_l1_model.py` | 22 pytest tests (CPU, no network): tiny LlamaConfig models (hidden 32 / 48, 2 layers; tiny-b without BOS) + WordLevel tokenizers built in the test |
| (read only) `l1_fusion.py` (coordinator), `l1_data.py`, `l1_aggregate.py` (other agent) | fusion math; prompts / metrics / jsonl; verdict, `summary_schema()`, `validate_summary()` |

Test command and result (venv `C:\Users\27118\AppData\Local\Temp\claude\venv-l1`: torch 2.15.0+cpu, transformers 4.52.4, peft 0.19.1, tokenizers 0.21.4):

```
CUDA_VISIBLE_DEVICES= python -B -m pytest scripts/claude_v342/test_l1_model.py scripts/claude_v342/test_l1_fusion.py scripts/claude_v342/test_l1_data.py scripts/claude_v342/test_l1_aggregate.py -q
66 passed in 135.59s (0:02:15)
```

Review items of `code_review_l1.md` taken in this revision: M2 / M3 through the coordinator's caps and estimates (`CTX_FULL=2048`, the new
`--est-*` defaults); S2 (`Pair.soft_from_train` runs `fuse_direct` with autocast disabled on float32 inputs, so the Laplacian matmul is never
bf16 under the training autocast); S3 (`make_pieces` re-shrinks the paragraphs up to four times with a growing margin before the code-2 tail
cut, so the question / "Answer:" of the empty-suffix prompts survive; tested); S6 (`--keep-preds` in the chain's `COMMON`: `preds_val.json` /
`preds_fresh.json` / `preds_val_p*.json` per run). NOT taken: M1 (the C assertion in float64) needs `l1_fusion.py` to stop casting its inputs
to float32 (coordinator-owned); `eval_latent` already records `c_assert_detail` (max diff, scale, tolerance) and the switch on the model
side is one line (`.double()` in place of `.float()` at the C check) once the fusion keeps float64. S7 (T0's frozen Qwen3 may not stop at
`<|im_end|>`, so its messages tend to run to the budget) is a reporting note, not a code change.

(22 of `test_l1_model.py` + 11 of `test_l1_fusion.py` + the other agent's 33.) The `--tiny` subprocess runs in the suite: `l1_train.py` S1 hetero
(incl. a deadline injected in the readouts, then `--eval-only`, and a deadline during training), B0 same, B4 hetero both models, B4 hetero
per model x 2 + `--merge-b4` (+ three refused merges), and `l1_text.py` T phase all -> T0, T sender -> receiver -> sweep256 -> T0 (+ the
refused `--phase receiver` without `--sender-from`). Every summary.json they write passes `l1_aggregate.validate_summary`.

## How the arms map to code

All arms share `Agent` (frozen base + peft LoRA r 16 / alpha 32 / dropout .05 on q,k,v,o,gate,up,down; gradient checkpointing, non-reentrant)
and `Agent.loss` (pass 2: `inputs_embeds = [BOS?] prefix | K soft tokens | suffix | answer EOS`, right padding, CE on the answer tokens + EOS
only; logits are computed only at the answer positions through `decoder` + `lm_head`, not for the whole sequence).

| arm | `Pair.step` / job | evaluation |
|---|---|---|
| B0 | each agent: `loss(pieces(no_comm_prompt(ex, i)))`, no soft tokens | `eval_plain` |
| B4 | one agent: `loss(pieces(full_prompt(ex)))`. Hetero: two per-model jobs `B4l` (`--model Llama`) and `B4q` (`--model Qwen3`), merged by `--merge-b4` into `B4-hetero-s<seed>` (em_model with both keys; Decision 1); same pair: one job (Qwen3) | `eval_plain` per model; `merge_models` |
| U | `encode_both` -> z (2, B, K, D_e); `soft_i = Reader_i(z_j)`; training: each directed message zeroed with p = .3 per example; adapters `fixed_lam=True` (no precision head), no w | `eval_latent`: `soft_from_eval` zeroes dropped messages |
| S1 | `x = fuse_direct(z, lam, [(0, 1)], w)`, `soft_i = Reader_i(x_i)`, w = softplus(w_raw), w = 1 at init; no dropout | `eval_latent`: `gbp_sync(z, lam, edges, w, rounds = tree_diameter = 1, drop, wire_dtype = fp16)`; C assertion on the first batch |
| S1L1 | as S1 with `fixed_lam=True` (lam = 1 everywhere, no precision parameters: S1 - S1L1 = exactly the two `lam_head` Linear layers) | as S1 |
| T | three jobs per (pair, seed): `Tsend` = `--phase sender` (sender LoRA per model, one epoch, loss on `text_sender_target` + EOS; greedy 64-token messages for the stage-2 subset, VAL, fresh -> `trainable.pt` (senders), `messages.json`, `summary_sender.json`); `T` = `--phase receiver --sender-from Tsend` (receiver LoRA per model: stage 1 one epoch with the other agent's gold message cut to 64 sender tokens, stage 2 625 steps on the stored subset messages; budgets {16, 64}; absence with the empty message -> `summary.json`, `trainable.pt` (receivers)); `Tsweep` = `--phase sweep256 --sender-from --receiver-from` (256-token messages, evaluated -> `summary_sweep256.json`) | `eval_text`; sender latency = message-generation wall time / example + decode steps |
| T0 | no sender training: `Agent(lora=False)` writes <= 64 tokens zero-shot; receivers = T's `trainable.pt['receiver']`; `--budgets 16,64` from the chain | as T, arm `"T0"` |

Training loss = mean of the two agents' answer losses (B4: the one agent). Pass 1 is under grad: `test_s1_loss_finite_and_gradients_reach_everything_on_both_agents`
checks non-zero grads on both LoRAs, both adapters, both readers and `w_raw`, and that base weights get no grad.

### Evaluation order and the soft deadline (coordinator request, 5090 smoke: S1 = 95 min training + ~15 min evaluation vs the 118-min job deadline)

`l1_train.py`: train -> **save `trainable.pt`** -> VAL at p = 0 (the criterion metric, carries the C assertion) -> fresh at p = 0 -> the
readouts (absence curve p in {.3, .6, 1} on VAL). A `Deadline` during training or the VAL / fresh passes -> `INCOMPLETE.json`, exit 3, no
summary.json. A `Deadline` during the readouts is caught: summary.json is written with every complete key plus
`"readouts_incomplete": "absence p=0.3: deadline ... passed during eval_latent"` (and `absence` holding only the complete p's), exit 0.
`--eval-only DIR` rebuilds the Pair from `DIR/trainable.pt` (+ its stored config: arm, pair, seed, k, d_e, layer_frac, LoRA, max_ctx, data
paths unless overridden) and recomputes only the readouts into `DIR/readouts.json` (`log_readouts.txt`); summary.json is never touched
(the test checks its sha before / after). B0 / B4 have no readouts beyond VAL / fresh: `--eval-only` writes the note only.
`l1_text.py` (receiver phase, all, T0): save `trainable.pt` -> VAL at the main budget (64) -> fresh at 64 -> readouts: budget 16 (VAL,
fresh), absence p in {.3, .6, 1}; the same soft deadline (`readouts_incomplete`, `budgets_done`). `--debug-deadline-at readouts` (tests only,
both CLIs) moves the deadline into the past right before the readouts.

`summary.json` (l1_train): `arm, pair, seed, model (--model or None), models, val, fresh, absence, c_assert (S1 / S1L1 only), w, complete,
steps, readouts_incomplete, agents, params, train, timings, config, scripts_sha (l1_model / l1_train / l1_fusion / l1_data), trainable_sha256`.
`val` / `fresh` blocks: `em, f1, em_agent, f1_agent, em_model, f1_model, n, em_by_owner {own, other, none, all: {em, n}}, em_by_owner_agent,
em_by_type, disagreement {mean, std, corr_with_em}, lam [per agent: mean_log_lam, frac_at_lower, frac_at_upper, mean_lam, corr_mean_lam_entropy,
corr_mean_log_lam_entropy], first_token_entropy_mean, wire_bytes_per_example, wire_bytes_per_message, sender_latency_s_per_example,
answer_gen_s_per_example, answer_decode_steps_mean, truncation, c_assert_detail, absence_p1_detail, n_messages_dropped`. EM / F1 in percentage points.
Merged B4 (`--merge-b4`): `val` / `fresh` = `merge_models` over the two per-model blocks (`em_model` both keys, `em_agent` in model order, `em` =
mean over the models, `per_model` = the full per-model readouts), `merged: true`, `parts {model: {dir, summary_sha256}}`, `config {'parts':
{model: part config}}`, `train` / `timings` / `params` per model, `scripts_sha` (asserted identical between the parts), `merged_from.json`.
The merge refuses: a part that is not a single-model B4 run, two parts of the same model, different pair / seed, different scripts_sha, an
existing summary.json.

## Decisions / ambiguities (the simplest reading was implemented)

1. **Hetero B4 = two per-model jobs + a CPU merge.** `l1_aggregate.py` reads P0 from the single `B4-hetero-s<seed>/summary.json`
   (`val.em_model[model]` for both models). One hetero B4 job is 111-139 min on the 5090 (cap 1,536 / 2,048) = too long for a lease, so
   the chain runs `B4l-hetero-s<seed>` (Llama, est 55 min) and `B4q-hetero-s<seed>` (Qwen3, est 100 min) with `--model`, then
   `l1_train.py --merge-b4 B4q B4l --out B4-hetero-s<seed>` (CPU, nice 10, timeout 600) for every seed whose two parts exist and whose merged
   summary does not. Without `--model` a B4 run still trains all models of the pair sequentially (same pair: Qwen3 only, one job).
2. **`c_assert` is written only for S1 / S1L1** (true or false, never skipped: `test_c_assert_false_is_written_not_skipped`). The key is absent for
   B0 / B4 / U / T / T0 because `validate_summary` treats any non-True value as "c_assert false" -> INCOMPLETE, and the schema says "optional for U".
3. **Pass 1 input** = the agent's own latent prompt without soft tokens (`prefix + suffix` = own 5 paragraphs + question, BOS first if the tokenizer
   has one); `hidden_states[L]` with `output_hidden_states=True`, L = floor(0.6667 x n_layers) = 18 / 28 (Qwen3-1.7B), 10 / 16 (Llama-3.2-1B).
4. **SenderAdapter**: queries (K, D_e) ~ N(0, .02^2); q / k / v / o projections; residual on the queries + LayerNorm; MLP 4 x D_e + LayerNorm;
   final LayerNorm -> z (the trivial-solution guard); precision head = one Linear(D_e, D_e) on the pre-LN features, zero-initialised so lam = 1
   at step 0, clamped to [ln 1e-3, ln 1e3] (clamping stops the gradient at the bounds; the readout reports the clamped fractions). The adapter and
   the reader run in float32 with autocast disabled (z, lam feed the float32 solve); the soft tokens are cast to the base dtype (bf16 on CUDA).
5. **Loss** = mean over all answer tokens (+ EOS) of the batch (token mean, HF style), not a per-example mean.
6. **Generation**: greedy, `num_beams=1`, left padding, `model.generate(inputs_embeds, attention_mask)` returns only the new tokens (verified
   in `test_generate_left_padded_batch_matches_single_and_returns_new_tokens_only`); the prediction is the first line of the decoded text, stripped.
   The first answer token's entropy comes from `output_logits=True` (step 0) = the same distribution as the teacher-forced one.
7. **U**: adapters are built with `fixed_lam=True` (parameter count = S1 - the two precision heads - w; U has no `w_raw`). The training dropout
   draws one Bernoulli(.3) per directed message per example from the global torch RNG (seeded). Wire bytes for U = K x D_e x 2 B per message
   (means only, fp16); S1 / S1L1 = `wire_bytes(K, D_e)` = 32 KB per message (means + precisions); `wire_bytes_per_example` = 2 messages.
8. **Absence curve**: per evaluation pass a `torch.Generator` seeded with `seed * 1009 + round(100 p)` draws one drop per directed message per
   example; the (2, B) mask is expanded to (2, B, K, D_e) for `gbp_sync` (its reshape accepts (2E, 1, 1, 1) or the full shape, not (2E, B, 1, 1)).
   U: dropped message = zero vector. The p = 1 check of the design ("S1 equals B0 of the same network") is recorded as `absence_p1_detail`
   (max |soft_i - Reader_i(z_i)|, `equal` within 1e-5 relative), not raised: the trained B0 arm is a different network.
9. **C assertion**: first VAL batch of the p = 0 pass, float32, `fuse_direct` vs `gbp_sync(rounds = 1, no drop, no wire)`, tolerance
   1e-4 (1 + max |x|); the details (max diff, scale, w, batch size) are in `val.c_assert_detail`.
10. **Readouts**: disagreement = `edge_residual(z)` (Frobenius over K x D_e) per example, Pearson with the mean EM of the two agents on that
    example; lam statistics per agent per example (mean log lam, fractions within 1e-6 of the bounds), Pearson of mean lam (and of mean log lam)
    with the first-token entropy; Pearson returns null when fewer than 3 points or zero variance. These come from the VAL p = 0 pass itself
    (no extra forward), so they are never cut by the soft deadline; only the p > 0 absence passes (and, for T, budget 16) are.
    Owner buckets: own / other / none relative to the answering agent (B4: "all"). `em_by_type` (bridge / comparison) is an extra.
11. **`--max-ctx-tokens`** (chain: `CTX_SHARD=900` for the shard arms, `CTX_FULL=2048` for B4; the smoke: 97 % of B4's 10-paragraph prompts
    exceed 900 tokens, mean 1,500; 22 % of the 5-paragraph shards, mean 764): if `[BOS] prefix suffix` exceeds the cap, every paragraph of the
    shard the prompt reads (B4: all 10) is cut to floor(f x its tokens) with f = (P - over - 2 per paragraph) / P and re-tokenised; if still
    over, the prefix ids are tail-cut (code 2). Counts of codes 0 / 1 / 2 are in `train.truncation_train` and `val.truncation`.
12. **Optimiser**: AdamW betas (.9, .999), eps 1e-8, weight decay 0 (the design gives none); LoRA params = names containing `lora_`, everything
    else trainable (adapter, reader, w_raw) at the adapter lr; warm-up 100 then cosine to 0 at `--steps`; grad clipping 1.0 over all trainable
    params; bf16 autocast on CUDA only (float32 on CPU). `--steps` beyond one epoch re-shuffles per epoch (seed, epoch). Gradient
    checkpointing stays on everywhere (`--no-grad-ckpt` OOMs the 5090 even for B0 at cap 1,280; the chain's `NOCKPT_SHARD` stays "").
13. **Timing**: step times are kept only from step 51 (`TIMING_WARMUP_STEPS = 50`): `step_time_post_warmup_s` is null when a run has fewer
    than 51 steps; the log prints "rate: warm-up" before that. Sender latency = pass 1 + fusion wall time per example with
    `torch.cuda.synchronize`; text sender latency = message generation wall time per example and the mean decode steps (incl. EOS).
14. **Deadline**: `--deadline-epoch` (the chain passes lease end - 60 s) raises `Deadline` between steps / batches; see "Evaluation order and
    the soft deadline" above for what is INCOMPLETE and what is `readouts_incomplete`. A run never resumes; the chain never re-runs into a dir
    that exists without its done marker.
15. **Text arm details**: the two senders are trained jointly (one AdamW over both LoRAs, loss = mean over the two agents' sender losses on the
    same batch; the same pair = one shared sender seeing both shards); the receivers likewise. Stage 1 and stage 2 of the receivers use separate
    AdamW instances and schedules (stage 2: warm-up 100 + cosine over 625 at the same lr 2e-4). Gold messages are `text_sender_target` of the
    other agent cut to the budget with the SENDER model's tokenizer (in the receiver phase = the receiver model's tokenizer, same base model);
    the stage-2 subset is the first 5,000 of a seeded permutation (seed + 7) of TRAIN; `messages.json` stores, per split and budget, the
    texts, the sender token ids, the token counts and the timing, plus the subset indices and example ids (the receiver phase verifies them
    against its `--train`). Budget 16 = the stored 64-token ids cut to 16 (exact, not re-tokenised); 256 = generated in the sweep256 phase
    (`--sweep-budget`). `message_f1_vs_gold` (token F1 of the generated message against the gold target) and `message_chars_mean` are extra
    readouts. The receiver summary records the sender run's `trainable.pt` and `messages.json` sha256 and copies its timing block
    (`sender_from.summary_sender`). T0 refuses a receiver checkpoint of another pair / seed / LoRA config. T's absence curve replaces a
    dropped message by the empty string (design 5). `--phase all` keeps the single-run behaviour (the tiny tests' path; budgets [16, 64, 256],
    `sweep256: "included"`).
16. **Chain** (`chain_day9_L1.sh D8-82 "<approval UTC>" --lease N [--est-base 62 --est-b4l 55 --est-b4q 100 --est-b4 100 --est-latent 115
    --est-tsend 90 --est-text 85 --est-t0 15 --est-sweep 35]`): one launch = one lease; `--lease 1` runs stage 0 only, in the order B0 s1,
    B4l s1, B0 s2, B4l s2, B4q s1, B4q s2, then the B4 merges, then `l1_aggregate.py --runs runs/l1 --stage0`; the chain reads the printed
    `P0: PASS / FAIL / INCOMPLETE` line (the aggregator always exits 0) and writes `runs/l1/STAGE0_PASS` or `STAGE0_FAIL` (FAIL -> the chain
    exits 0 and later leases do nothing). Later leases first finish any missing stage-0 job, then run stage 1 in the fixed order S1 s1, S1 s2,
    Tsend s1, T s1, Tsend s2, T s2, U s1, U s2, S1L1 s1, S1L1 s2, T0 s1, T0 s2 (hetero); B0, B4 (one job), Tsend / T per seed, S1 (same pair);
    Tsweep hetero s1, s2, same s1, s2 at the end. A job is skipped when its done marker exists (`summary.json`; Tsend `summary_sender.json`;
    Tsweep `summary_sweep256.json`), skipped with a log line when its dir exists without the marker (never overwrite), and the lease stops
    starting jobs when an estimate does not fit the remaining time. T needs Tsend's marker, Tsweep both T and Tsend, T0 the T summary. Every
    job: `timeout -k 30 <lease end>` and `--deadline-epoch <lease end - 60 s>`; shard arms get `--max-ctx-tokens $CTX_SHARD $NOCKPT_SHARD`,
    B4 `--max-ctx-tokens $CTX_FULL`. The aggregator's `aggregate.json / .md` are copied to `runs/l1/aggregate/<tag>.*`. HF offline
    (`HF_HUB_OFFLINE=1`): the two models must already be in `$HF_HUB_CACHE`. `FILL_` pins: the six scripts and the three jsonl splits.
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

- Per-model B4 runs are now the hetero default (Decision 1) because of the lease length; `B4-hetero-s<seed>` is a merged, not a trained, dir
  (`merged: true`, `merged_from.json`).
- The T arm is three jobs (Tsend / T / Tsweep) instead of one (the single job measured ~180 min on the 5090).
- `c_assert` for U is not written (nothing is fused in U; the aggregator calls it optional and rejects any non-True value).
- R = 2 (flag only), the L sweep, Gemma-3-1B-it: not implemented.

## Measured on the 5090 (coordinator's smokes, 2026-10-29; gradient checkpointing on) and what the chain assumes

- B0 at cap 900: 1.303 s/step (both models per step) = 54 min + ~4 min evaluation -> `--est-base 62`.
- B4 at cap 2,048: Qwen3 92 min, Llama 47 min of training -> `--est-b4q 100`, `--est-b4l 55`; same-pair B4 (Qwen3 only) `--est-b4 100`.
- S1 at cap 900: 2.29 s/step = 95 min + ~15 min evaluation (4 absence passes + fresh) -> `--est-latent 115`; the readouts (p = .3 / .6 / 1)
  are the part the soft deadline may cut; `--eval-only` recomputes them later.
- T: sender training 1.39 s/step = 58 min + messages 0.216 s/example at budget 64 (5,000 + 2,000 examples = 25 min) -> `--est-tsend 90`;
  receiver stage 1 1.31 s/step = 55 min + stage 2 ~14 min + evaluation -> `--est-text 85`; the 256-token sweep ~27 min of generation ->
  `--est-sweep 35`; T0 `--est-t0 15`.
- `--no-grad-ckpt` OOMs (30.9 GiB in use even for B0 at cap 1,280): checkpointing stays on; `NOCKPT_SHARD=""`.
- `output_hidden_states=True` in pass 1 keeps all 29 / 17 layer outputs; a forward hook on layer L - 1 would halve that memory (not done: the
  brief asked for `hidden_states[L]`).
