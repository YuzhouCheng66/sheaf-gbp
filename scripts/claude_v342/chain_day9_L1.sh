#!/bin/bash
# Day-9 batch L1 "LLM 隐态协作：sheaf 共识 vs 文字交换" (Claude). Pre-registration: OVERNIGHT_REPORT D8-82 (design
# claude_handoff/DESIGN_LLM_LATENT_SHEAF_GBP.md; criteria P0-P3 are judged by l1_aggregate.py, not here). Trainers: l1_train.py (B0 / B4 /
# U / S1 / S1L1) and l1_text.py (T in three phases, T0); model side l1_model.py, fusion l1_fusion.py, prompts l1_data.py. Structure and safety
# rules = chain_day8_E3L5.sh (uk-linux 5090: sha pins, FILL_ check, one lock dir, never overwrite, SOLO GPU wait, ONE 7,200 s lease per
# launch, timeout -k on our own child PIDs only, never pkill / killall, CPU aggregate after the GPU work). No renewal, no resume: a job that
# does not fit the remaining lease time is NOT started; a job killed by its deadline leaves INCOMPLETE.json (no done marker) and is never
# re-run into the same directory (the coordinator removes or renames the run dir first). A job whose deadline falls into its readouts
# (l1_train.py / the T receiver phase) still writes summary.json with "readouts_incomplete" and exits 0; the readouts can be recomputed
# later with `l1_train.py --eval-only <run dir>` (readouts.json; summary.json untouched).
#
# ONE LAUNCH = ONE LEASE (--lease N, N = 1, 2, ...). Jobs run in a FIXED order, each skipped when its done marker exists
# (summary.json; Tsend: summary_sender.json; Tsweep: summary_sweep256.json):
#   stage 0 (lease 1; a later lease continues it if not every stage-0 summary exists yet), in this order:
#       B0-hetero-s342101, B4l-hetero-s342101, B0-hetero-s342202, B4l-hetero-s342202, B4q-hetero-s342101, B4q-hetero-s342202, where
#       B4l / B4q = `l1_train.py --arm B4 --pair hetero --model $LLAMA / $QWEN --max-ctx-tokens $CTX_FULL` (the hetero B4 as ONE job is
#       ~111-139 min on the 5090 = too close to / over the job deadline). After the jobs, for each seed whose B4q and B4l summaries exist and
#       B4-hetero-s<seed>/summary.json does not: `l1_train.py --merge-b4 B4q B4l --out B4-hetero-s<seed>` (CPU, nice 10, timeout 600) = the
#       B4 summary l1_aggregate reads (em_model with both models). Then `l1_aggregate.py --runs runs/l1 --stage0` (it always exits 0 and
#       prints one line "P0: PASS" / "P0: FAIL" / "P0: INCOMPLETE"; the chain reads that line: PASS -> runs/l1/STAGE0_PASS; FAIL ->
#       runs/l1/STAGE0_FAIL and this chain exits 0 without training anything else; INCOMPLETE / anything else -> nothing else is started).
#       Its aggregate.json / aggregate.md (written into runs/l1, overwritten by every call) are copied to runs/l1/aggregate/stage0_lease<N>.*.
#       Lease 1 never starts a stage-1 job.
#   stage 1 (leases 2..N, only with STAGE0_PASS), hetero: S1 s1, S1 s2, Tsend s1, T s1, Tsend s2, T s2, U s1, U s2, S1L1 s1, S1L1 s2,
#       T0 s1, T0 s2; then the same pair: B0 s1, B0 s2, B4 s1, B4 s2 (one model, one job), Tsend s1, T s1, Tsend s2, T s2, S1 s1, S1 s2; then at the END
#       Tsweep hetero s1, s2, Tsweep same s1, s2. The T arm is three jobs per (pair, seed) because one job does not fit a lease
#       (5090 smoke: ~180 min): Tsend-<pair>-s<seed> = l1_text.py --phase sender (senders + the 64-token messages), T-<pair>-s<seed> =
#       --phase receiver --sender-from $RUNS/Tsend-<pair>-s<seed> (only when its summary_sender.json exists; this is the dir the aggregator
#       reads), Tsweep-<pair>-s<seed> = --phase sweep256 (only when both T and Tsend are done). T0 needs the T run of the same seed and
#       passes --budgets 16,64.
#   Before every job: its estimate (--est-base B0 / --est-b4l / --est-b4q / --est-b4 (same pair) / --est-latent / --est-tsend / --est-text /
#   --est-t0 / --est-sweep minutes; defaults 62 / 55 / 100 / 100 / 115 / 90 / 85 / 15 / 35 from the 5090 smokes of 2026-10-29: B0 1.303 s/step
#   at cap 900 = 54 min + ~4 min eval; B4 at cap 2,048 Qwen3 92 min + Llama 47 min; S1 2.29 s/step at cap 900 = 95 min + ~15 min eval;
#   RE-ESTIMATE after the first full runs, D8-82: 实现后按实测重估) must fit the remaining lease time, else the lease stops starting jobs.
#   Every job gets --deadline-epoch (lease end - 60 s) and runs under timeout -k 30 (lease end). After the last job: `l1_aggregate.py
#   --runs runs/l1` on everything that exists (CPU only, nice 10); its files are copied to runs/l1/aggregate/lease<N>.*; the verdict is the
#   last lease's.
# Context caps (chain variables near the pins; the 5090 smoke: 97 % of B4's 10-paragraph prompts exceed 900 tokens, mean 1,500; 22 % of the
#   5-paragraph shards, mean 764): CTX_SHARD (B0, U, S1, S1L1, Tsend, T, T0: --max-ctx-tokens $CTX_SHARD) and CTX_FULL (B4). NOCKPT_SHARD
#   ("" = gradient checkpointing stays on everywhere: --no-grad-ckpt OOMs the 5090 even for B0 at cap 1,280; if ever set to "--no-grad-ckpt"
#   it is appended to every shard-arm job, never to B4).
# Writes only under /mnt/datasets (runs/l1/..., runs/l1/chain_day9.log). The data splits and the HF cache are read, never written
# (HF_HUB_OFFLINE=1: the models must already be in $HF_HUB_CACHE; the chain never downloads).
#
# The coordinator fills every FILL_ (after the review and after the user confirms the GPU; the chain refuses FILL_):
#   MODEL_SHA   scripts/l1_model.py      TRAIN_SHA  scripts/l1_train.py      TEXT_SHA  scripts/l1_text.py
#   FUSION_SHA  scripts/l1_fusion.py     DATA_SHA   scripts/l1_data.py       AGG_SHA   scripts/l1_aggregate.py
#   TRAINJ_SHA  $DATA/train.jsonl        VALJ_SHA   $DATA/val.jsonl          FRESHJ_SHA $DATA/fresh.jsonl   (splits_l1, seed 342101)
#   APPROVAL    the UTC time of the user's GPU confirmation -- passed as $2, written into lease_<N>.json
# Launch (never from an interactive shell that may die; one chain per ssh command whose text names only this chain):
#   timeout 60 ssh uk-linux 'cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 && (setsid nohup bash scripts/chain_day9_L1.sh "D8-82" "<APPROVAL UTC>" --lease 1 > runs/chain_day9_L1_lease1.out 2>&1 < /dev/null &)'
#   ... later leases: --lease 2, --lease 3, ... (one launch per lease, after the previous chain process has exited).
# IF THE COORDINATOR MOVES NON-BLOCKING ARMS (T0, Tsweep, the L sweep) TO BIGBOY (notes only; no second chain is written):
#   cd / work dir   -> $HOME/claude-sheaf-gbp/work ; python -> $HOME/miniconda3/envs/mlpsp/bin/python (never modify envs);
#   HF_HUB_CACHE    -> a bigboy cache holding the two models; DATA -> a copy of hotpot/splits_l1 (re-check the jsonl shas);
#   --expect-gpu is not used by these trainers; keep <= 2 GB of outputs on /home (bigboy is ~96 % full).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
PREREG="$1"; APPROVAL="$2"
[ "$PREREG" = "D8-82" ] && [ -n "$APPROVAL" ] || { echo "usage: chain_day9_L1.sh D8-82 <user GPU approval UTC> --lease N [--est-base 62] [--est-b4l 55] [--est-b4q 100] [--est-b4 100] [--est-latent 115] [--est-tsend 90] [--est-text 85] [--est-t0 15] [--est-sweep 35]"; exit 2; }
shift 2
LEASE=""; EST_BASE=62; EST_B4L=55; EST_B4Q=100; EST_B4=100; EST_LATENT=115; EST_TSEND=90; EST_TEXT=85; EST_T0=15; EST_SWEEP=35
while [ $# -gt 0 ]; do
  case "$1" in
    --lease) LEASE="$2"; shift 2;;
    --est-base) EST_BASE="$2"; shift 2;;
    --est-b4l) EST_B4L="$2"; shift 2;;
    --est-b4q) EST_B4Q="$2"; shift 2;;
    --est-b4) EST_B4="$2"; shift 2;;
    --est-latent) EST_LATENT="$2"; shift 2;;
    --est-tsend) EST_TSEND="$2"; shift 2;;
    --est-text) EST_TEXT="$2"; shift 2;;
    --est-t0) EST_T0="$2"; shift 2;;
    --est-sweep) EST_SWEEP="$2"; shift 2;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
case "$LEASE" in ''|*[!0-9]*) echo "--lease N (positive integer) is required"; exit 2;; esac
[ "$LEASE" -ge 1 ] || { echo "--lease must be >= 1"; exit 2; }
export HF_HUB_CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/hf-cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude
mkdir -p $TMPDIR
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
WORK=/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021
DATA=$WORK/hotpot/splits_l1                                               # train.jsonl / val.jsonl / fresh.jsonl (CC BY-SA 4.0; never in the repo)
RUNS=$WORK/runs/l1
LOG=runs/l1/chain_day9.log
QWEN=Qwen/Qwen3-1.7B
LLAMA=meta-llama/Llama-3.2-1B-Instruct
SEEDS="342101 342202"
# ---------------------------------------------------------------- pinned by the coordinator (after the review)
MODEL_SHA=FILL_MODEL_SHA                                                   # scripts/l1_model.py
TRAIN_SHA=FILL_TRAIN_SHA                                                   # scripts/l1_train.py
TEXT_SHA=FILL_TEXT_SHA                                                     # scripts/l1_text.py
FUSION_SHA=FILL_FUSION_SHA                                                 # scripts/l1_fusion.py
DATA_SHA=FILL_DATA_SHA                                                     # scripts/l1_data.py
AGG_SHA=FILL_AGG_SHA                                                       # scripts/l1_aggregate.py
TRAINJ_SHA=FILL_TRAINJ_SHA                                                 # $DATA/train.jsonl
VALJ_SHA=FILL_VALJ_SHA                                                     # $DATA/val.jsonl
FRESHJ_SHA=FILL_FRESHJ_SHA                                                 # $DATA/fresh.jsonl
# ---------------------------------------------------------------- context caps and checkpointing (set by the coordinator from the smoke runs)
CTX_SHARD=900                                                              # B0, U, S1, S1L1, Tsend, T, T0: --max-ctx-tokens (5-paragraph shards)
CTX_FULL=2048                                                              # B4: --max-ctx-tokens (10 paragraphs; hetero B4 = two per-model jobs + merge)
NOCKPT_SHARD=""                                                            # "" = gradient checkpointing on everywhere; "--no-grad-ckpt" would be appended to shard arms only
mkdir -p runs/l1
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day9_L1 started pid $$ host $(hostname) lease $LEASE sha256 $(sha256sum scripts/chain_day9_L1.sh | cut -c1-16) prereg $PREREG approval $APPROVAL estimates base $EST_BASE b4l $EST_B4L b4q $EST_B4Q b4 $EST_B4 latent $EST_LATENT tsend $EST_TSEND text $EST_TEXT t0 $EST_T0 sweep $EST_SWEEP min; caps shard $CTX_SHARD full $CTX_FULL; nockpt '$NOCKPT_SHARD'"
if grep -q -e "FILL_" <<<"$MODEL_SHA $TRAIN_SHA $TEXT_SHA $FUSION_SHA $DATA_SHA $AGG_SHA $TRAINJ_SHA $VALJ_SHA $FRESHJ_SHA"; then log "ABORT: unfilled FILL_ placeholders (coordinator)"; exit 2; fi
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
check_shas(){
  [ "$(sha scripts/l1_model.py)" = "$MODEL_SHA" ] || { log "ABORT: l1_model.py sha changed ($(sha scripts/l1_model.py))"; exit 2; }
  [ "$(sha scripts/l1_train.py)" = "$TRAIN_SHA" ] || { log "ABORT: l1_train.py sha changed ($(sha scripts/l1_train.py))"; exit 2; }
  [ "$(sha scripts/l1_text.py)" = "$TEXT_SHA" ] || { log "ABORT: l1_text.py sha changed ($(sha scripts/l1_text.py))"; exit 2; }
  [ "$(sha scripts/l1_fusion.py)" = "$FUSION_SHA" ] || { log "ABORT: l1_fusion.py sha changed ($(sha scripts/l1_fusion.py))"; exit 2; }
  [ "$(sha scripts/l1_data.py)" = "$DATA_SHA" ] || { log "ABORT: l1_data.py sha changed ($(sha scripts/l1_data.py))"; exit 2; }
  [ "$(sha scripts/l1_aggregate.py)" = "$AGG_SHA" ] || { log "ABORT: l1_aggregate.py sha changed ($(sha scripts/l1_aggregate.py))"; exit 2; }
  [ "$(sha $DATA/train.jsonl)" = "$TRAINJ_SHA" ] || { log "ABORT: $DATA/train.jsonl sha changed or missing"; exit 2; }
  [ "$(sha $DATA/val.jsonl)" = "$VALJ_SHA" ] || { log "ABORT: $DATA/val.jsonl sha changed or missing"; exit 2; }
  [ "$(sha $DATA/fresh.jsonl)" = "$FRESHJ_SHA" ] || { log "ABORT: $DATA/fresh.jsonl sha changed or missing"; exit 2; }
}
check_shas
mkdir $RUNS.lock 2>/dev/null || { log "ABORT: a chain_day9_L1 lease is already running (lock $RUNS.lock exists)"; exit 2; }
trap 'rmdir $RUNS.lock 2>/dev/null' EXIT
[ -e $RUNS/lease_$LEASE.json ] && { log "ABORT: $RUNS/lease_$LEASE.json exists (never overwrite; use the next lease number)"; exit 2; }
FREE_MB=$(df -Pm /mnt/datasets | awk 'NR==2{print $4}')
if [ "$FREE_MB" -lt 5000 ]; then log "ABORT: only ${FREE_MB} MB free on /mnt/datasets (< 5,000)"; exit 2; fi
[ -d $HF_HUB_CACHE/models--Qwen--Qwen3-1.7B ] && [ -d $HF_HUB_CACHE/models--meta-llama--Llama-3.2-1B-Instruct ] || { log "ABORT: the two models are not in $HF_HUB_CACHE (the chain never downloads)"; exit 2; }
# ---------------------------------------------------------------- before the lease: import self-check (CPU)
SELF=$(CUDA_VISIBLE_DEVICES= timeout -k 30 300 $P -B -c "import sys; sys.path.insert(0, 'scripts'); import l1_model, l1_train, l1_text, l1_fusion, l1_data, l1_aggregate, torch, transformers, peft; print('import self-check ok', torch.__version__, transformers.__version__, peft.__version__)" 2>&1 | tail -1)
case "$SELF" in "import self-check ok"*) log "L1 $SELF";; *) log "ABORT: import self-check failed: $SELF"; exit 2;; esac
# 5090 SOLO: no other chain, no project trainer / diag, no compute process on the GPU (wait, never preempt)
busy(){ pgrep -af "bash scripts/chain_" | grep -v "chain_day9_L1" | grep -q . || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || pgrep -f "^$P scripts/g4_" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
check_shas
# ================================================================ the lease (7,200 s inclusive)
START=$(date +%s); DT=$((START+7150)); DL=$((DT-60))
cat > $RUNS/lease_$LEASE.json <<JSON
{"lease": $LEASE, "lease_start_epoch": $START, "total_deadline_epoch": $DT, "job_deadline_epoch": $DL, "inclusive_seconds": 7200, "batch": "L1", "prereg": "$PREREG",
 "user_gpu_approval_utc": "$APPROVAL", "seeds": [342101, 342202],
 "estimates_min": {"base": $EST_BASE, "b4l": $EST_B4L, "b4q": $EST_B4Q, "b4": $EST_B4, "latent": $EST_LATENT, "tsend": $EST_TSEND, "text": $EST_TEXT, "t0": $EST_T0, "sweep": $EST_SWEEP},
 "ctx_shard": $CTX_SHARD, "ctx_full": $CTX_FULL, "nockpt_shard": "$NOCKPT_SHARD",
 "models": {"hetero": ["$QWEN", "$LLAMA"], "same": ["$QWEN"]}, "data": "$DATA", "data_sha256": {"train": "$TRAINJ_SHA", "val": "$VALJ_SHA", "fresh": "$FRESHJ_SHA"},
 "scripts_sha256": {"l1_model.py": "$MODEL_SHA", "l1_train.py": "$TRAIN_SHA", "l1_text.py": "$TEXT_SHA", "l1_fusion.py": "$FUSION_SHA", "l1_data.py": "$DATA_SHA", "l1_aggregate.py": "$AGG_SHA"},
 "chain_sha256": "$(sha scripts/chain_day9_L1.sh)", "hf_hub_cache": "$HF_HUB_CACHE", "offline": true, "renewal": "none", "resume": "none"}
JSON
log "L1 lease $LEASE start $START (job deadline +7090, kill at +7150)"
STOP=0
COMMON="--train $DATA/train.jsonl --val $DATA/val.jsonl --fresh $DATA/fresh.jsonl --hf-cache $HF_HUB_CACHE --keep-preds"   # preds_*.json per run (a few MB) for the per-question analysis
SHARD="--max-ctx-tokens $CTX_SHARD $NOCKPT_SHARD"                          # every shard arm (B0, U, S1, S1L1, Tsend, T, T0)
FULL="--max-ctx-tokens $CTX_FULL"                                          # B4 only (never NOCKPT_SHARD)
TSEND="--max-ctx-tokens $((CTX_SHARD+16)) $NOCKPT_SHARD"                   # review M2: the text prompts carry an instruction line (sender, ~13 tokens) or
TRECV="--max-ctx-tokens $((CTX_SHARD+72)) $NOCKPT_SHARD"                   # "Message from the other agent: " + <= 64 message tokens (receiver, T0); the cap is
TSWEEP="--max-ctx-tokens $((CTX_SHARD+264)) $NOCKPT_SHARD"                 # raised by that much so the DOCUMENTS get the same CTX_SHARD budget as the latent arms
job(){                                                                            # $1 = run name, $2 = estimate (min), $3 = done marker file, $4.. = script and arguments
  local name=$1 est=$2 marker=$3; shift 3
  local dir=$RUNS/$name
  [ $STOP = 1 ] && return 1
  if [ -f $dir/$marker ]; then log "L1 $name: $marker exists -> skipped (done)"; return 0; fi
  if [ -e $dir ]; then log "L1 $name: $dir exists without $marker (partial / INCOMPLETE run; never overwrite) -> skipped, the coordinator decides"; return 0; fi
  local n=0; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $n -lt 20 ]; do log "L1 $name: GPU busy before start; waiting"; sleep 30; n=$((n+1)); done
  local now rem; now=$(date +%s); rem=$((DL-now))                              # review S4: the fit check uses the time left AFTER the GPU wait
  if [ $((est*60)) -gt $rem ]; then log "L1 lease $LEASE stops before $name: estimate $est min > remaining $((rem/60)) min"; STOP=1; return 1; fi
  check_shas
  log "L1 $name start (estimate $est min, remaining $((rem/60)) min): $*"
  now=$(date +%s)
  timeout -k 30 $((DT-now)) $P -B "$@" --out $dir --deadline-epoch $DL > $RUNS/$name.log 2>&1
  local x=$?
  log "L1 $name done (exit $x): $(tail -1 $RUNS/$name.log | cut -c1-300)"
  [ $x -eq 124 ] && log "L1 $name TIMED OUT (lease end); no $marker -> the arm is missing until re-run elsewhere"
  return 0
}
tsend(){ job Tsend-$1-s$2 $EST_TSEND summary_sender.json scripts/l1_text.py --arm T --phase sender --pair $1 --seed $2 $COMMON $TSEND; }
trecv(){                                                                          # the T-<pair>-s<seed> dir the aggregator reads; needs the sender phase
  if [ -f $RUNS/Tsend-$1-s$2/summary_sender.json ]; then job T-$1-s$2 $EST_TEXT summary.json scripts/l1_text.py --arm T --phase receiver --pair $1 --seed $2 --sender-from $RUNS/Tsend-$1-s$2 $COMMON $TRECV
  else [ $STOP = 1 ] || log "L1 T-$1-s$2: Tsend-$1-s$2 has no summary_sender.json yet -> skipped"; fi
}
tsweep(){                                                                         # the 256-token sweep (report only); needs both T phases
  if [ -f $RUNS/Tsend-$1-s$2/summary_sender.json ] && [ -f $RUNS/T-$1-s$2/summary.json ]; then job Tsweep-$1-s$2 $EST_SWEEP summary_sweep256.json scripts/l1_text.py --arm T --phase sweep256 --pair $1 --seed $2 --sender-from $RUNS/Tsend-$1-s$2 --receiver-from $RUNS/T-$1-s$2 $COMMON $TSWEEP
  else [ $STOP = 1 ] || log "L1 Tsweep-$1-s$2: Tsend-$1-s$2 / T-$1-s$2 not both done -> skipped"; fi
}
t0(){                                                                             # zero-shot text (report only); needs T's receivers
  if [ -f $RUNS/T-$1-s$2/summary.json ]; then job T0-$1-s$2 $EST_T0 summary.json scripts/l1_text.py --arm T0 --pair $1 --seed $2 --receiver-from $RUNS/T-$1-s$2 --budgets 16,64 $COMMON $TRECV
  else [ $STOP = 1 ] || log "L1 T0-$1-s$2: T-$1-s$2 has no summary.json yet -> skipped"; fi
}
merge_b4(){                                                                       # CPU only: B4q + B4l per-model summaries -> B4-hetero-s<seed>/summary.json (what l1_aggregate reads)
  for s in $SEEDS; do
    if [ -f $RUNS/B4q-hetero-s$s/summary.json ] && [ -f $RUNS/B4l-hetero-s$s/summary.json ] && [ ! -f $RUNS/B4-hetero-s$s/summary.json ]; then
      CUDA_VISIBLE_DEVICES= timeout -k 30 600 nice -n 10 $P -B scripts/l1_train.py --merge-b4 $RUNS/B4q-hetero-s$s $RUNS/B4l-hetero-s$s --out $RUNS/B4-hetero-s$s > $RUNS/B4-hetero-s$s.merge.log 2>&1
      log "L1 B4-hetero-s$s merge of B4q + B4l (exit $?): $(tail -1 $RUNS/B4-hetero-s$s.merge.log | cut -c1-300)"
    fi
  done
}
stage0_jobs(){
  for s in $SEEDS; do
    job B0-hetero-s$s $EST_BASE summary.json scripts/l1_train.py --arm B0 --pair hetero --seed $s $COMMON $SHARD
    job B4l-hetero-s$s $EST_B4L summary.json scripts/l1_train.py --arm B4 --pair hetero --model $LLAMA --seed $s $COMMON $FULL
  done
  for s in $SEEDS; do job B4q-hetero-s$s $EST_B4Q summary.json scripts/l1_train.py --arm B4 --pair hetero --model $QWEN --seed $s $COMMON $FULL; done
  merge_b4
}
stage0_complete(){
  for s in $SEEDS; do for r in B0-hetero B4-hetero; do [ -f $RUNS/$r-s$s/summary.json ] || return 1; done; done
  return 0
}
aggregate(){                                                                      # $1 = tag (stage0_lease<N> / lease<N>), $2.. = extra flags; CPU only
  local tag=$1; shift
  mkdir -p $RUNS/aggregate
  CUDA_VISIBLE_DEVICES= timeout -k 30 1800 nice -n 10 $P -B scripts/l1_aggregate.py --runs $RUNS "$@" > $RUNS/aggregate/$tag.log 2>&1
  local x=$?
  cp -f $RUNS/aggregate.json $RUNS/aggregate/$tag.json 2>/dev/null; cp -f $RUNS/aggregate.md $RUNS/aggregate/$tag.md 2>/dev/null
  log "L1 aggregate $tag (exit $x): $(grep -a -E '^(P0|VERDICT):' $RUNS/aggregate/$tag.log | tr '\n' ' ')"
  return $x
}
stage1_jobs(){
  for s in $SEEDS; do job S1-hetero-s$s $EST_LATENT summary.json scripts/l1_train.py --arm S1 --pair hetero --seed $s $COMMON $SHARD; done
  for s in $SEEDS; do tsend hetero $s; trecv hetero $s; done
  for s in $SEEDS; do job U-hetero-s$s $EST_LATENT summary.json scripts/l1_train.py --arm U --pair hetero --seed $s $COMMON $SHARD; done
  for s in $SEEDS; do job S1L1-hetero-s$s $EST_LATENT summary.json scripts/l1_train.py --arm S1L1 --pair hetero --seed $s $COMMON $SHARD; done
  for s in $SEEDS; do t0 hetero $s; done
  for s in $SEEDS; do job B0-same-s$s $EST_BASE summary.json scripts/l1_train.py --arm B0 --pair same --seed $s $COMMON $SHARD; done
  for s in $SEEDS; do job B4-same-s$s $EST_B4 summary.json scripts/l1_train.py --arm B4 --pair same --seed $s $COMMON $FULL; done
  for s in $SEEDS; do tsend same $s; trecv same $s; done
  for s in $SEEDS; do job S1-same-s$s $EST_LATENT summary.json scripts/l1_train.py --arm S1 --pair same --seed $s $COMMON $SHARD; done
  for s in $SEEDS; do tsweep hetero $s; done
  for s in $SEEDS; do tsweep same $s; done
}
# ---------------------------------------------------------------- stage 0 and the P0 gate
if [ ! -f $RUNS/STAGE0_PASS ] && [ ! -f $RUNS/STAGE0_FAIL ]; then
  stage0_jobs
  if stage0_complete; then
    log "L1 stage 0 complete: l1_aggregate.py --stage0 (P0 gate; CPU)"
    aggregate stage0_lease$LEASE --stage0
    P0=$(grep -a -E '^P0: ' $RUNS/aggregate/stage0_lease$LEASE.log | tail -1)
    case "$P0" in
      "P0: PASS") date -u +%FT%TZ > $RUNS/STAGE0_PASS; log "L1 P0 passed -> STAGE0_PASS";;
      "P0: FAIL") date -u +%FT%TZ > $RUNS/STAGE0_FAIL; log "L1 P0 FAILED -> STAGE0_FAIL; no other arm is trained (D8-82: 不过就停，改分片规则)";;
      *) log "L1 stage 0 aggregate did not decide ('$P0'): nothing else is started this lease";;
    esac
  else
    log "L1 stage 0 not complete yet (lease $LEASE); stage 1 waits for a later lease"
  fi
fi
# ---------------------------------------------------------------- stage 1 (leases 2..N only, only with STAGE0_PASS)
if [ -f $RUNS/STAGE0_FAIL ]; then
  log "L1 STAGE0_FAIL present: stage 1 is not run"
elif [ "$LEASE" -eq 1 ]; then
  log "L1 lease 1 = stage 0 only (D8-82); stage 1 starts with --lease 2"
elif [ -f $RUNS/STAGE0_PASS ]; then
  stage1_jobs
else
  log "L1 no STAGE0_PASS: stage 1 is not run"
fi
END=$(date +%s)
echo "{\"lease\": $LEASE, \"lease_end_epoch\": $END, \"lease_seconds\": $((END-START)), \"stopped_for_time\": $STOP}" > $RUNS/lease_end_$LEASE.json
log "L1 lease $LEASE GPU work over after $((END-START)) s; the GPU is not used again by this chain"
# ---------------------------------------------------------------- after the GPU work: the aggregate of everything that exists (CPU only)
check_shas
aggregate lease$LEASE
log "L1 disk: $RUNS $(du -sm $RUNS | cut -f1) MB; /mnt/datasets free $(df -Pm /mnt/datasets | awk 'NR==2{print $4}') MB"
log "chain_day9_L1 lease $LEASE finished"
