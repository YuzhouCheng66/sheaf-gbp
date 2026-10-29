#!/bin/bash
# Day-9 batch L1 "LLM 隐态协作：sheaf 共识 vs 文字交换" (Claude). Pre-registration: OVERNIGHT_REPORT D8-82 (design
# claude_handoff/DESIGN_LLM_LATENT_SHEAF_GBP.md; criteria P0-P3 are judged by l1_aggregate.py, not here). Trainers: l1_train.py (B0 / B4 /
# U / S1 / S1L1) and l1_text.py (T / T0); model side l1_model.py, fusion l1_fusion.py, prompts l1_data.py. Structure and safety rules =
# chain_day8_E3L5.sh (uk-linux 5090: sha pins, FILL_ check, one lock dir, never overwrite, SOLO GPU wait, ONE 7,200 s lease per launch,
# timeout -k on our own child PIDs only, never pkill / killall, CPU aggregate after the GPU work). No renewal, no resume: a job that does
# not fit the remaining lease time is NOT started; a job killed by its deadline leaves INCOMPLETE.json (no summary.json) and is never re-run
# into the same directory (the coordinator removes or renames the run dir first).
#
# ONE LAUNCH = ONE LEASE (--lease N, N = 1, 2, ...). Jobs are run in a FIXED order, each skipped when its summary.json exists:
#   stage 0 (lease 1; a later lease continues it if not every stage-0 summary exists yet):
#       B0-hetero-s342101, B4-hetero-s342101 (ONE B4 run trains Qwen3 then Llama and holds both in em_model, as l1_aggregate reads it),
#       then the same for seed 342202, then `l1_aggregate.py --runs runs/l1 --stage0` (it always exits 0 and prints one line
#       "P0: PASS" / "P0: FAIL" / "P0: INCOMPLETE"; the chain reads that line: PASS -> runs/l1/STAGE0_PASS; FAIL -> runs/l1/STAGE0_FAIL and
#       this chain exits 0 without training anything else; INCOMPLETE / anything else -> nothing else is started). Its aggregate.json /
#       aggregate.md (written into runs/l1, overwritten by every call) are copied to runs/l1/aggregate/stage0_lease<N>.*.
#       Lease 1 never starts a stage-1 job.
#   stage 1 (leases 2..N, only with STAGE0_PASS): hetero T s1, T s2, U s1, U s2, S1 s1, S1 s2, S1L1 s1, S1L1 s2, T0 s1, T0 s2
#       (T0 needs the T run of the same seed: skipped with a log line when it is missing); then same-pair B0, B4, T, S1 for both seeds.
#   Before every job: estimate (--est-base B0 / --est-b4 / --est-latent / --est-text / --est-t0 minutes; defaults 25 / 50 / 50 / 50 / 25 --
#   RE-ESTIMATE after the first runs, D8-82: 实现后按实测重估; T trains two senders, two receivers twice and generates ~14k messages, so
#   expect ~90-110 min and pass --est-text accordingly) must fit the remaining lease time, else the lease stops starting jobs. Every job
#   gets --deadline-epoch (lease end - 60 s) and runs under timeout -k 30 (lease end). After the last job: `l1_aggregate.py --runs runs/l1`
#   on everything that exists (CPU only, nice 10); its files are copied to runs/l1/aggregate/lease<N>.*; the verdict is the last lease's.
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
# IF THE COORDINATOR MOVES NON-BLOCKING ARMS (T0, the budget sweep, the L sweep) TO BIGBOY (notes only; no second chain is written):
#   cd / work dir   -> $HOME/claude-sheaf-gbp/work ; python -> $HOME/miniconda3/envs/mlpsp/bin/python (never modify envs);
#   HF_HUB_CACHE    -> a bigboy cache holding the two models; DATA -> a copy of hotpot/splits_l1 (re-check the jsonl shas);
#   --expect-gpu is not used by these trainers; keep <= 2 GB of outputs on /home (bigboy is ~96 % full).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
PREREG="$1"; APPROVAL="$2"
[ "$PREREG" = "D8-82" ] && [ -n "$APPROVAL" ] || { echo "usage: chain_day9_L1.sh D8-82 <user GPU approval UTC> --lease N [--est-base 25] [--est-b4 50] [--est-latent 50] [--est-text 50] [--est-t0 25]"; exit 2; }
shift 2
LEASE=""; EST_BASE=25; EST_B4=50; EST_LATENT=50; EST_TEXT=50; EST_T0=25
while [ $# -gt 0 ]; do
  case "$1" in
    --lease) LEASE="$2"; shift 2;;
    --est-base) EST_BASE="$2"; shift 2;;
    --est-b4) EST_B4="$2"; shift 2;;
    --est-latent) EST_LATENT="$2"; shift 2;;
    --est-text) EST_TEXT="$2"; shift 2;;
    --est-t0) EST_T0="$2"; shift 2;;
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
mkdir -p runs/l1
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day9_L1 started pid $$ host $(hostname) lease $LEASE sha256 $(sha256sum scripts/chain_day9_L1.sh | cut -c1-16) prereg $PREREG approval $APPROVAL estimates base $EST_BASE b4 $EST_B4 latent $EST_LATENT text $EST_TEXT t0 $EST_T0 min"
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
 "user_gpu_approval_utc": "$APPROVAL", "seeds": [342101, 342202], "estimates_min": {"base": $EST_BASE, "b4": $EST_B4, "latent": $EST_LATENT, "text": $EST_TEXT, "t0": $EST_T0},
 "models": {"hetero": ["$QWEN", "$LLAMA"], "same": ["$QWEN"]}, "data": "$DATA", "data_sha256": {"train": "$TRAINJ_SHA", "val": "$VALJ_SHA", "fresh": "$FRESHJ_SHA"},
 "scripts_sha256": {"l1_model.py": "$MODEL_SHA", "l1_train.py": "$TRAIN_SHA", "l1_text.py": "$TEXT_SHA", "l1_fusion.py": "$FUSION_SHA", "l1_data.py": "$DATA_SHA", "l1_aggregate.py": "$AGG_SHA"},
 "chain_sha256": "$(sha scripts/chain_day9_L1.sh)", "hf_hub_cache": "$HF_HUB_CACHE", "offline": true, "renewal": "none", "resume": "none"}
JSON
log "L1 lease $LEASE start $START (job deadline +7090, kill at +7150)"
STOP=0
COMMON="--train $DATA/train.jsonl --val $DATA/val.jsonl --fresh $DATA/fresh.jsonl --hf-cache $HF_HUB_CACHE"
job(){                                                                            # $1 = run name, $2 = estimate (min), $3.. = script and arguments
  local name=$1 est=$2; shift 2
  local dir=$RUNS/$name
  [ $STOP = 1 ] && return 1
  if [ -f $dir/summary.json ]; then log "L1 $name: summary.json exists -> skipped (done)"; return 0; fi
  if [ -e $dir ]; then log "L1 $name: $dir exists without summary.json (partial / INCOMPLETE run; never overwrite) -> skipped, the coordinator decides"; return 0; fi
  local now rem; now=$(date +%s); rem=$((DL-now))
  if [ $((est*60)) -gt $rem ]; then log "L1 lease $LEASE stops before $name: estimate $est min > remaining $((rem/60)) min"; STOP=1; return 1; fi
  local n=0; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $n -lt 20 ]; do log "L1 $name: GPU busy before start; waiting"; sleep 30; n=$((n+1)); done
  check_shas
  log "L1 $name start (estimate $est min, remaining $((rem/60)) min): $*"
  now=$(date +%s)
  timeout -k 30 $((DT-now)) $P -B "$@" --out $dir --deadline-epoch $DL > $RUNS/$name.log 2>&1
  local x=$?
  log "L1 $name done (exit $x): $(tail -1 $RUNS/$name.log | cut -c1-300)"
  [ $x -eq 124 ] && log "L1 $name TIMED OUT (lease end); no summary.json -> the arm is missing until re-run elsewhere"
  return 0
}
stage0_jobs(){
  for s in $SEEDS; do
    job B0-hetero-s$s $EST_BASE scripts/l1_train.py --arm B0 --pair hetero --seed $s $COMMON
    job B4-hetero-s$s $EST_B4 scripts/l1_train.py --arm B4 --pair hetero --seed $s $COMMON
  done
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
  for s in $SEEDS; do job T-hetero-s$s $EST_TEXT scripts/l1_text.py --arm T --pair hetero --seed $s $COMMON; done
  for s in $SEEDS; do job U-hetero-s$s $EST_LATENT scripts/l1_train.py --arm U --pair hetero --seed $s $COMMON; done
  for s in $SEEDS; do job S1-hetero-s$s $EST_LATENT scripts/l1_train.py --arm S1 --pair hetero --seed $s $COMMON; done
  for s in $SEEDS; do job S1L1-hetero-s$s $EST_LATENT scripts/l1_train.py --arm S1L1 --pair hetero --seed $s $COMMON; done
  for s in $SEEDS; do
    if [ -f $RUNS/T-hetero-s$s/summary.json ]; then job T0-hetero-s$s $EST_T0 scripts/l1_text.py --arm T0 --pair hetero --seed $s --receiver-from $RUNS/T-hetero-s$s $COMMON
    else [ $STOP = 1 ] || log "L1 T0-hetero-s$s: T-hetero-s$s has no summary.json yet -> skipped"; fi
  done
  for s in $SEEDS; do job B0-same-s$s $EST_BASE scripts/l1_train.py --arm B0 --pair same --seed $s $COMMON; done
  for s in $SEEDS; do job B4-same-s$s $EST_B4 scripts/l1_train.py --arm B4 --pair same --seed $s $COMMON; done
  for s in $SEEDS; do job T-same-s$s $EST_TEXT scripts/l1_text.py --arm T --pair same --seed $s $COMMON; done
  for s in $SEEDS; do job S1-same-s$s $EST_LATENT scripts/l1_train.py --arm S1 --pair same --seed $s $COMMON; done
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
