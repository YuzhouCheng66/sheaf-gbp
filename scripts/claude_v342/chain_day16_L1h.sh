#!/bin/bash
# Day-16 batch L1h "对齐读取的发送端（同模型对）" (Claude). Pre-registration: OVERNIGHT_REPORT D8-91 (criteria G1h / P2h are judged by l1h_aggregate.py,
# not here). Trainer l1h_train.py (arms Hu / Hs; model side l1h_model.py on top of l1f / l1e / l1d / l1c and the PINNED L1 files, all pinned here again).
# The receiver AND the sender's model of every run is the FROZEN B0 checkpoint of the same seed (runs/l1/B0-hetero-s<seed>/trainable.pt, pinned
# by sha; read only). Structure and safety rules = chain_day11_L1c.sh (uk-linux 5090: sha pins, FILL_ check, one lock dir, never overwrite, SOLO
# GPU wait, ONE 7,200 s lease per launch, timeout -k on our own child PIDs only, never pkill / killall, CPU aggregate after the GPU work).
# No renewal, no resume: a job that does not fit the remaining lease time is NOT started; a job killed by its deadline leaves INCOMPLETE.json (no
# summary.json) and is never re-run into the same directory. A job whose deadline falls into its readouts still writes summary.json.
#
# ONE LAUNCH = ONE LEASE (--lease N; L1h uses lease 19 = the first of the two B leases, per the user's "做" on 2026-11-01). Jobs in a FIXED order, each skipped when its summary.json exists,
# several per lease when they fit (--est-l1h minutes each, default 60; RE-ESTIMATE from the 5090 smoke):
#   Hu-same2-s342101 -> the G1h gate (CPU: l1h_aggregate.py --gate 342101 prints "G1h(s342101): PASS | FAIL | INCOMPLETE"; PASS -> runs/l1h/L1H_G1_PASS,
#                        FAIL -> runs/l1h/L1H_G1_FAIL and nothing else is ever started by this chain, INCOMPLETE -> nothing else this lease)
#   (no second seed this lease: D8-91 section 4, the second seed needs a lease of its own; Hs is NOT run: D8-89 section 3c, inherited by D8-91 section 2)
#   Lease 19: the user's "做" (2026-11-01 ~01:5x UTC) to the proposal "at most one more narrow test (1 lease)", D8-91.
#   Every job gets --deadline-epoch (lease end - 60 s) and runs under timeout -k 30 (lease end). After the last job:
#   l1h_aggregate.py --l1 runs/l1 --l1h runs/l1h (CPU, nice 10) -> runs/l1h/aggregate/lease<N>.*; the verdict is the last lease's.
# Writes only under /mnt/datasets (runs/l1h/..., runs/l1h/chain_day16.log). The data splits, the HF cache and runs/l1 are read, never written.
#
# The coordinator fills every FILL_ (after the review and after the user confirms the GPU; the chain refuses FILL_):
#   L1H_MODEL_SHA scripts/l1h_model.py   L1H_TRAIN_SHA scripts/l1h_train.py   L1H_AGG_SHA scripts/l1h_aggregate.py   L1F_MODEL_SHA / L1E_MODEL_SHA / L1D_MODEL_SHA / L1C_MODEL_SHA (imported)
#   B0_CKPT_SHA_342101 / B0_CKPT_SHA_342202   runs/l1/B0-hetero-s<seed>/trainable.pt (the frozen receivers / senders)
#   MODEL_SHA / TRAIN_SHA / FUSION_SHA / DATA_SHA / TRAINJ_SHA / VALJ_SHA / FRESHJ_SHA = the L1 pins (already filled)
#   APPROVAL the UTC time + words of the user's GPU confirmation -- passed as $2, written into lease_<N>.json
# Launch: one lease per launch, via scripts/l1h_leases.sh, after the previous chain process has exited.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
PREREG="$1"; APPROVAL="$2"
[ "$PREREG" = "D8-91" ] && [ -n "$APPROVAL" ] || { echo "usage: chain_day16_L1h.sh D8-91 <user GPU approval UTC> --lease N [--est-l1h 55]"; exit 2; }
shift 2
LEASE=""; EST=70
while [ $# -gt 0 ]; do
  case "$1" in
    --lease) LEASE="$2"; shift 2;;
    --est-l1h) EST="$2"; shift 2;;
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
DATA=$WORK/hotpot/splits_l1
RUNS=$WORK/runs/l1h
L1RUNS=$WORK/runs/l1                                                      # B0 (the frozen receivers / senders) / T of batch L1 (read only)
LOG=runs/l1h/chain_day16.log
SEEDS="342101 342202"
GATE_SEED=342101
# ---------------------------------------------------------------- pinned by the coordinator (after the review)
L1H_MODEL_SHA=FILL_L1H_MODEL_SHA                                           # scripts/l1h_model.py
L1H_TRAIN_SHA=FILL_L1H_TRAIN_SHA                                           # scripts/l1h_train.py
L1H_AGG_SHA=FILL_L1H_AGG_SHA                                               # scripts/l1h_aggregate.py
L1F_MODEL_SHA=FILL_L1F_MODEL_SHA                                           # scripts/l1f_model.py (imported by l1h_model)
L1E_MODEL_SHA=FILL_L1E_MODEL_SHA                                           # scripts/l1e_model.py (imported by l1f_model)
L1D_MODEL_SHA=FILL_L1D_MODEL_SHA                                           # scripts/l1d_model.py (imported by l1e_model)
L1C_MODEL_SHA=FILL_L1C_MODEL_SHA                                           # scripts/l1c_model.py (imported by l1d_model)
B0_CKPT_SHA_342101=FILL_B0_CKPT_SHA_342101                                 # runs/l1/B0-hetero-s342101/trainable.pt
B0_CKPT_SHA_342202=FILL_B0_CKPT_SHA_342202                                 # runs/l1/B0-hetero-s342202/trainable.pt
MODEL_SHA=aa3a4eec71ea38b11b034282cd550c671c6358d037c2a7179c80ca4d438f5b86                                                   # scripts/l1_model.py (L1 pin)
TRAIN_SHA=5f8eee7833fd53851650ff4aa2249269e3a7fbab76b8de92695c403f23b9f33a                                                   # scripts/l1_train.py (L1 pin; imported by l1d_train)
FUSION_SHA=a0f56eb6de811a4247fa784f701c23ef43119626ff1e9e6821846d379fa6a474                                                 # scripts/l1_fusion.py (L1 pin)
DATA_SHA=41d0db811e710579ddcdab76c8c622358984423da54a8e792716fa76c75263dd                                                     # scripts/l1_data.py (L1 pin)
TRAINJ_SHA=4b6af5e2bdb78d23d57ba83592c71944f9dbbb6568db4e9bdfb2ffd1b7e0caa3                                                 # $DATA/train.jsonl (L1 pin)
VALJ_SHA=b1e240bbe70873e1f2ac7586171caf6d3620f269c0e02807033a5fded5f68ecf                                                     # $DATA/val.jsonl (L1 pin)
FRESHJ_SHA=985610d8e092745c9edc37b7624ab7e502a9e94da9b0c7a280cd4f68d00acc60                                                 # $DATA/fresh.jsonl (L1 pin)
# ---------------------------------------------------------------- the L1h knobs (D8-91; defaults of l1h_train.py, passed explicitly for the record)
CTX_SHARD=900
L1H_FLAGS="--steps 2000 --k 8 --eval-batch 32 --absence-ps 0.0,1.0 --tau 1.0 --n-blocks 2"
mkdir -p runs/l1h
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day16_L1h started pid $$ host $(hostname) lease $LEASE sha256 $(sha256sum scripts/chain_day16_L1h.sh | cut -c1-16) prereg $PREREG approval $APPROVAL est l1h $EST min; cap $CTX_SHARD; flags $L1H_FLAGS"
if grep -q -e "FILL_" <<<"$L1H_MODEL_SHA $L1H_TRAIN_SHA $L1H_AGG_SHA $L1F_MODEL_SHA $L1E_MODEL_SHA $L1D_MODEL_SHA $L1C_MODEL_SHA $B0_CKPT_SHA_342101 $B0_CKPT_SHA_342202 $MODEL_SHA $TRAIN_SHA $FUSION_SHA $DATA_SHA $TRAINJ_SHA $VALJ_SHA $FRESHJ_SHA"; then log "ABORT: unfilled FILL_ placeholders (coordinator)"; exit 2; fi
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
check_shas(){
  [ "$(sha scripts/l1h_model.py)" = "$L1H_MODEL_SHA" ] || { log "ABORT: l1h_model.py sha changed ($(sha scripts/l1h_model.py))"; exit 2; }
  [ "$(sha scripts/l1h_train.py)" = "$L1H_TRAIN_SHA" ] || { log "ABORT: l1h_train.py sha changed ($(sha scripts/l1h_train.py))"; exit 2; }
  [ "$(sha scripts/l1h_aggregate.py)" = "$L1H_AGG_SHA" ] || { log "ABORT: l1h_aggregate.py sha changed ($(sha scripts/l1h_aggregate.py))"; exit 2; }
  [ "$(sha scripts/l1f_model.py)" = "$L1F_MODEL_SHA" ] || { log "ABORT: l1f_model.py sha changed ($(sha scripts/l1f_model.py))"; exit 2; }
  [ "$(sha scripts/l1e_model.py)" = "$L1E_MODEL_SHA" ] || { log "ABORT: l1e_model.py sha changed ($(sha scripts/l1e_model.py))"; exit 2; }
  [ "$(sha scripts/l1d_model.py)" = "$L1D_MODEL_SHA" ] || { log "ABORT: l1d_model.py sha changed ($(sha scripts/l1d_model.py))"; exit 2; }
  [ "$(sha scripts/l1c_model.py)" = "$L1C_MODEL_SHA" ] || { log "ABORT: l1c_model.py sha changed ($(sha scripts/l1c_model.py))"; exit 2; }
  [ "$(sha scripts/l1_model.py)" = "$MODEL_SHA" ] || { log "ABORT: l1_model.py sha changed ($(sha scripts/l1_model.py))"; exit 2; }
  [ "$(sha scripts/l1_train.py)" = "$TRAIN_SHA" ] || { log "ABORT: l1_train.py sha changed ($(sha scripts/l1_train.py))"; exit 2; }
  [ "$(sha scripts/l1_fusion.py)" = "$FUSION_SHA" ] || { log "ABORT: l1_fusion.py sha changed ($(sha scripts/l1_fusion.py))"; exit 2; }
  [ "$(sha scripts/l1_data.py)" = "$DATA_SHA" ] || { log "ABORT: l1_data.py sha changed ($(sha scripts/l1_data.py))"; exit 2; }
  [ "$(sha $DATA/train.jsonl)" = "$TRAINJ_SHA" ] || { log "ABORT: $DATA/train.jsonl sha changed or missing"; exit 2; }
  [ "$(sha $DATA/val.jsonl)" = "$VALJ_SHA" ] || { log "ABORT: $DATA/val.jsonl sha changed or missing"; exit 2; }
  [ "$(sha $DATA/fresh.jsonl)" = "$FRESHJ_SHA" ] || { log "ABORT: $DATA/fresh.jsonl sha changed or missing"; exit 2; }
  [ "$(sha $L1RUNS/B0-hetero-s342101/trainable.pt)" = "$B0_CKPT_SHA_342101" ] || { log "ABORT: B0-hetero-s342101/trainable.pt sha changed or missing"; exit 2; }
  [ "$(sha $L1RUNS/B0-hetero-s342202/trainable.pt)" = "$B0_CKPT_SHA_342202" ] || { log "ABORT: B0-hetero-s342202/trainable.pt sha changed or missing"; exit 2; }
}
check_shas
for s in $SEEDS; do [ -f $L1RUNS/B0-hetero-s$s/summary.json ] || { log "ABORT: $L1RUNS/B0-hetero-s$s/summary.json missing"; exit 2; }; done
mkdir $RUNS.lock 2>/dev/null || { log "ABORT: a chain_day16_L1h lease is already running (lock $RUNS.lock exists)"; exit 2; }
trap 'rmdir $RUNS.lock 2>/dev/null' EXIT
for d in $RUNS $L1RUNS $WORK/runs/l1b $WORK/runs/l1c $WORK/runs/l1d $WORK/runs/l1e $WORK/runs/l1f $WORK/runs/l1g; do [ -e $d/lease_$LEASE.json ] && { log "ABORT: $d/lease_$LEASE.json exists (lease numbers continue L1 .. L1g; use the next number)"; exit 2; }; done
FREE_MB=$(df -Pm /mnt/datasets | awk 'NR==2{print $4}')
if [ "$FREE_MB" -lt 5000 ]; then log "ABORT: only ${FREE_MB} MB free on /mnt/datasets (< 5,000)"; exit 2; fi
[ -d $HF_HUB_CACHE/models--Qwen--Qwen3-1.7B ] && [ -d $HF_HUB_CACHE/models--meta-llama--Llama-3.2-1B-Instruct ] || { log "ABORT: the two models are not in $HF_HUB_CACHE (the chain never downloads)"; exit 2; }
# ---------------------------------------------------------------- before the lease: import self-check (CPU)
SELF=$(CUDA_VISIBLE_DEVICES= timeout -k 30 300 $P -B -c "import sys; sys.path.insert(0, 'scripts'); import l1h_model, l1h_train, l1h_aggregate, l1f_model, l1e_model, l1d_model, l1c_model, l1_model, l1_train, l1_fusion, l1_data, torch, transformers, peft; print('import self-check ok', torch.__version__, transformers.__version__, peft.__version__)" 2>&1 | tail -1)
case "$SELF" in "import self-check ok"*) log "L1h $SELF";; *) log "ABORT: import self-check failed: $SELF"; exit 2;; esac
# 5090 SOLO: no other chain, no project trainer / diag, no compute process on the GPU (wait, never preempt)
busy(){ pgrep -af "bash scripts/chain_" | grep -v "chain_day16_L1h" | grep -q . || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "scripts/diag_" >/dev/null || pgrep -f "^$P scripts/g4_" >/dev/null || pgrep -f "scripts/l1_" >/dev/null || pgrep -f "scripts/l1b_" >/dev/null || pgrep -f "scripts/l1c_" >/dev/null || pgrep -f "scripts/l1d_" >/dev/null || pgrep -f "scripts/l1e_" >/dev/null || pgrep -f "scripts/l1f_" >/dev/null || pgrep -f "scripts/l1g_" >/dev/null || pgrep -f "scripts/l1h_train" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }   # never "scripts/l1h_" (matches the queue -> waits on itself); the chain's own l1h_train child is started after this wait
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
check_shas
# ================================================================ the lease (7,200 s inclusive)
START=$(date +%s); DT=$((START+7150)); DL=$((DT-60))
cat > $RUNS/lease_$LEASE.json <<JSON
{"lease": $LEASE, "lease_start_epoch": $START, "total_deadline_epoch": $DT, "job_deadline_epoch": $DL, "inclusive_seconds": 7200, "batch": "L1h", "prereg": "$PREREG",
 "user_gpu_approval_utc": "$APPROVAL", "seeds": [342101, 342202], "gate_seed": $GATE_SEED, "estimates_min": {"l1h": $EST}, "ctx_shard": $CTX_SHARD, "flags": "$L1H_FLAGS",
 "models": {"same2": ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B"], "lora_source": "the hetero B0 checkpoint, agent 0 (Qwen), on both instances", "sender": "AlignedSender (D8-91)"}, "data": "$DATA", "l1_runs": "$L1RUNS",
 "receivers": {"342101": "$L1RUNS/B0-hetero-s342101/trainable.pt", "342202": "$L1RUNS/B0-hetero-s342202/trainable.pt"},
 "receiver_sha256": {"342101": "$B0_CKPT_SHA_342101", "342202": "$B0_CKPT_SHA_342202"},
 "data_sha256": {"train": "$TRAINJ_SHA", "val": "$VALJ_SHA", "fresh": "$FRESHJ_SHA"},
 "scripts_sha256": {"l1h_model.py": "$L1H_MODEL_SHA", "l1h_train.py": "$L1H_TRAIN_SHA", "l1h_aggregate.py": "$L1H_AGG_SHA", "l1f_model.py": "$L1F_MODEL_SHA", "l1e_model.py": "$L1E_MODEL_SHA", "l1d_model.py": "$L1D_MODEL_SHA", "l1c_model.py": "$L1C_MODEL_SHA", "l1_model.py": "$MODEL_SHA", "l1_train.py": "$TRAIN_SHA", "l1_fusion.py": "$FUSION_SHA", "l1_data.py": "$DATA_SHA"},
 "chain_sha256": "$(sha scripts/chain_day16_L1h.sh)", "hf_hub_cache": "$HF_HUB_CACHE", "offline": true, "renewal": "none", "resume": "none"}
JSON
log "L1h lease $LEASE start $START (job deadline +7090, kill at +7150)"
STOP=0
COMMON="--train $DATA/train.jsonl --val $DATA/val.jsonl --fresh $DATA/fresh.jsonl --hf-cache $HF_HUB_CACHE --keep-preds --max-ctx-tokens $CTX_SHARD $L1H_FLAGS"
job(){                                                                            # $1 = run name, $2 = estimate (min), $3.. = trainer arguments
  local name=$1 est=$2; shift 2
  local dir=$RUNS/$name
  [ $STOP = 1 ] && return 1
  if [ -f $dir/summary.json ]; then log "L1h $name: summary.json exists -> skipped (done)"; return 0; fi
  if [ -e $dir ]; then log "L1h $name: $dir exists without summary.json (partial / INCOMPLETE run; never overwrite) -> skipped, the coordinator decides"; return 0; fi
  local n=0; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $n -lt 20 ]; do log "L1h $name: GPU busy before start; waiting"; sleep 30; n=$((n+1)); done
  local now rem; now=$(date +%s); rem=$((DL-now))
  if [ $((est*60)) -gt $rem ]; then log "L1h lease $LEASE stops before $name: estimate $est min > remaining $((rem/60)) min"; STOP=1; return 1; fi
  check_shas
  log "L1h $name start (estimate $est min, remaining $((rem/60)) min): $*"
  now=$(date +%s)
  timeout -k 30 $((DT-now)) $P -B scripts/l1h_train.py "$@" --out $dir --deadline-epoch $DL > $RUNS/$name.log 2>&1
  local x=$?
  log "L1h $name done (exit $x): $(tail -1 $RUNS/$name.log | cut -c1-300)"
  [ $x -eq 124 ] && log "L1h $name TIMED OUT (lease end); no summary.json -> the arm is missing until re-run elsewhere"
  return 0
}
aggregate(){                                                                      # $1 = tag, $2.. = extra flags (e.g. --gate 342101); CPU only
  local tag=$1; shift
  mkdir -p $RUNS/aggregate
  CUDA_VISIBLE_DEVICES= timeout -k 30 1800 nice -n 10 $P -B scripts/l1h_aggregate.py --l1 $L1RUNS --l1h $RUNS --out $RUNS/aggregate/$tag "$@" > $RUNS/aggregate/$tag.log 2>&1
  local x=$?
  log "L1h aggregate $tag (exit $x): $(grep -a -E '^(G1h|VERDICT)' $RUNS/aggregate/$tag.log | tr '\n' ' ')"
  return $x
}
gate(){                                                                           # the G1h gate on the gate seed (D8-91 section 4 / 5); idempotent
  [ -f $RUNS/L1H_G1_FAIL ] && { log "L1h gate: L1H_G1_FAIL present -> nothing else is started"; STOP=1; return 1; }
  [ -f $RUNS/L1H_G1_PASS ] && return 0
  [ -f $RUNS/Hu-same2-s$GATE_SEED/summary.json ] || { log "L1h gate: Hu-same2-s$GATE_SEED not done yet -> nothing else is started this lease"; STOP=1; return 1; }
  aggregate gate_lease$LEASE --gate $GATE_SEED
  local G; G=$(grep -a -E "^G1h\(s$GATE_SEED\): " $RUNS/aggregate/gate_lease$LEASE.log | tail -1)
  case "$G" in
    "G1h(s$GATE_SEED): PASS") date -u +%FT%TZ > $RUNS/L1H_G1_PASS; log "L1h G1h passed on seed $GATE_SEED -> L1H_G1_PASS";;
    "G1h(s$GATE_SEED): FAIL") date -u +%FT%TZ > $RUNS/L1H_G1_FAIL; log "L1h G1h FAILED on seed $GATE_SEED -> L1H_G1_FAIL; no other L1h job is started (D8-91: CHANNEL-NO-6-same-aligned)"; STOP=1; return 1;;
    *) log "L1h gate did not decide ('$G'): nothing else is started this lease (the coordinator decides)"; STOP=1; return 1;;
  esac
}
# ---------------------------------------------------------------- the jobs
if [ -f $RUNS/L1H_G1_FAIL ]; then
  log "L1h L1H_G1_FAIL present: nothing is started"
else
  job Hu-same2-s$GATE_SEED $EST --arm Hu --pair same2 --seed $GATE_SEED --receiver-from $L1RUNS/B0-hetero-s$GATE_SEED $COMMON
  if gate; then
    : # no second seed this lease (D8-91 section 4): the second seed needs its own lease
    : # Hs is not run in this batch (D8-89 section 3c, inherited): only Hu s1 -> gate are reachable this lease
  fi
fi
END=$(date +%s)
echo "{\"lease\": $LEASE, \"lease_end_epoch\": $END, \"lease_seconds\": $((END-START)), \"stopped_for_time\": $STOP}" > $RUNS/lease_end_$LEASE.json
log "L1h lease $LEASE GPU work over after $((END-START)) s; the GPU is not used again by this chain"
# ---------------------------------------------------------------- after the GPU work: the aggregate of everything that exists (CPU only)
check_shas
aggregate lease$LEASE
log "L1h disk: $RUNS $(du -sm $RUNS | cut -f1) MB; /mnt/datasets free $(df -Pm /mnt/datasets | awk 'NR==2{print $4}') MB"
log "chain_day16_L1h lease $LEASE finished"
