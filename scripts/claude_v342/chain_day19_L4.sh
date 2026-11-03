#!/bin/bash
# Day-19 ladder L4 "修好选句头（V3）、对数概率空间的 sheaf 共识（S3）对两轮文字（T2）" (Claude). Pre-registration: OVERNIGHT_REPORT D8-94 (the per-job criteria are
# judged by l2_gate.py (V3) and l3_gate.py (S3a / S3b / T2), which write the marker files GATE_<job>_<verdict>). Trainers: l2_train.py (V3, on
# l2_model.py) and l3_train.py (S3a / S3b / T2, on l3_model.py), on top of the PINNED L1h / L1 files. The frozen B0 checkpoint of the seed is every
# run's 'default' LoRA. Structure and safety rules = chain_day17_L2.sh (uk-linux 5090: sha pins, FILL_ check, one lock dir, never overwrite, SOLO
# GPU wait, ONE 7,200 s lease per launch, timeout -k on our own child PIDs only, never pkill / killall, CPU gates after the GPU work). No renewal, no resume.
#
# ONE LAUNCH = ONE LEASE = ONE JOB (--lease N selects the job from the FIXED table of D8-94 section 4; a lease whose condition is not met starts
# nothing and exits 0 so the queue moves on):
#   45 V3-same2-s342101 (runs/l2)   46 V3-same2-s342202 (runs/l2)   47 S3a-same2-s342101 [45 done]   48 S3b-same2-s342101 [47 done]
#   49 T2-same2-s342101 [47 done]   50 S3a-same2-s342202 [46 done]   51 S3b-same2-s342202 [50 done]   52 T2-same2-s342202 [50 done]
#   53+ nothing (reserved for the coordinator)
#   GPU approval: the user's "ok，继续推进突破性实验" (2026-11-03) under the standing "顺序拉起、不监控" instruction, passed as $2 into every lease_<N>.json.
# Writes only under /mnt/datasets (runs/l4/..., runs/l2/V2-*, runs/l4/chain_day19.log). Data, HF cache, runs/l1, runs/l2/V1-* are read, never written.
#
# The coordinator fills every FILL_ (after the review and the smokes; the chain refuses FILL_):
#   L3_MODEL_SHA / L3_TRAIN_SHA / L3_GATE_SHA   L2_MODEL_SHA / L2_TRAIN_SHA / L2_GATE_SHA   L1H_MODEL_SHA   L1F / L1E / L1D / L1C_MODEL_SHA   B0_CKPT_SHA_342101 / 342202
#   MODEL_SHA / TRAIN_SHA / FUSION_SHA / DATA_SHA / TRAINJ_SHA / VALJ_SHA / FRESHJ_SHA = the L1 pins (already filled)
# Launch: one lease per launch, via scripts/l4_leases.sh, after the previous chain process has exited.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
PREREG="$1"; APPROVAL="$2"
[ "$PREREG" = "D8-94" ] && [ -n "$APPROVAL" ] || { echo "usage: chain_day19_L4.sh D8-94 <user GPU approval UTC> --lease N [--est-v2 75 --est-s2a 95 --est-s2b 80 --est-t2 75]"; exit 2; }
shift 2
LEASE=""; EST_V2=75; EST_S2A=95; EST_S2B=80; EST_T2=75
while [ $# -gt 0 ]; do
  case "$1" in
    --lease) LEASE="$2"; shift 2;;
    --est-v2) EST_V2="$2"; shift 2;;
    --est-s2a) EST_S2A="$2"; shift 2;;
    --est-s2b) EST_S2B="$2"; shift 2;;
    --est-t2) EST_T2="$2"; shift 2;;
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
RUNS=$WORK/runs/l4
L2RUNS=$WORK/runs/l2                                                      # V1 (read) and V2 (written here, gated by l2_gate.py)
L1HRUNS=$WORK/runs/l1h                                                    # read only (l2_gate's --l1h)
L1RUNS=$WORK/runs/l1                                                      # B0 (read only)
LOG=runs/l4/chain_day19.log
# ---------------------------------------------------------------- pinned by the coordinator (after the review and the smokes)
L3_MODEL_SHA=FILL_L3_MODEL_SHA                                             # scripts/l3_model.py
L3_TRAIN_SHA=FILL_L3_TRAIN_SHA                                             # scripts/l3_train.py
L3_GATE_SHA=FILL_L3_GATE_SHA                                               # scripts/l3_gate.py
L2_MODEL_SHA=FILL_L2_MODEL_SHA                                             # scripts/l2_model.py (V2 added)
L2_TRAIN_SHA=FILL_L2_TRAIN_SHA                                             # scripts/l2_train.py
L2_GATE_SHA=FILL_L2_GATE_SHA                                               # scripts/l2_gate.py
L1H_MODEL_SHA=FILL_L1H_MODEL_SHA                                           # scripts/l1h_model.py (pinned in lease 19)
L1F_MODEL_SHA=FILL_L1F_MODEL_SHA                                           # scripts/l1f_model.py
L1E_MODEL_SHA=FILL_L1E_MODEL_SHA                                           # scripts/l1e_model.py
L1D_MODEL_SHA=FILL_L1D_MODEL_SHA                                           # scripts/l1d_model.py
L1C_MODEL_SHA=FILL_L1C_MODEL_SHA                                           # scripts/l1c_model.py
B0_CKPT_SHA_342101=FILL_B0_CKPT_SHA_342101                                 # runs/l1/B0-hetero-s342101/trainable.pt
B0_CKPT_SHA_342202=FILL_B0_CKPT_SHA_342202                                 # runs/l1/B0-hetero-s342202/trainable.pt
MODEL_SHA=aa3a4eec71ea38b11b034282cd550c671c6358d037c2a7179c80ca4d438f5b86                                                   # scripts/l1_model.py (L1 pin)
TRAIN_SHA=5f8eee7833fd53851650ff4aa2249269e3a7fbab76b8de92695c403f23b9f33a                                                   # scripts/l1_train.py (L1 pin)
FUSION_SHA=a0f56eb6de811a4247fa784f701c23ef43119626ff1e9e6821846d379fa6a474                                                 # scripts/l1_fusion.py (L1 pin)
DATA_SHA=41d0db811e710579ddcdab76c8c622358984423da54a8e792716fa76c75263dd                                                     # scripts/l1_data.py (L1 pin)
TRAINJ_SHA=4b6af5e2bdb78d23d57ba83592c71944f9dbbb6568db4e9bdfb2ffd1b7e0caa3                                                 # $DATA/train.jsonl (L1 pin)
VALJ_SHA=b1e240bbe70873e1f2ac7586171caf6d3620f269c0e02807033a5fded5f68ecf                                                     # $DATA/val.jsonl (L1 pin)
FRESHJ_SHA=985610d8e092745c9edc37b7624ab7e502a9e94da9b0c7a280cd4f68d00acc60                                                 # $DATA/fresh.jsonl (L1 pin)
# ---------------------------------------------------------------- the knobs (D8-94 section 3; defaults of the trainers, passed explicitly for the record)
CTX_SHARD=900
L2_FLAGS="--eval-batch 32 --tau 1.0 --n-blocks 2"
L3_FLAGS="--eval-batch 32"
mkdir -p runs/l4 runs/l2
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day19_L4 started pid $$ host $(hostname) lease $LEASE sha256 $(sha256sum scripts/chain_day19_L4.sh | cut -c1-16) prereg $PREREG approval $APPROVAL est v2 $EST_V2 s2a $EST_S2A s2b $EST_S2B t2 $EST_T2 min; cap $CTX_SHARD"
if grep -q -e "FILL_" <<<"$L3_MODEL_SHA $L3_TRAIN_SHA $L3_GATE_SHA $L2_MODEL_SHA $L2_TRAIN_SHA $L2_GATE_SHA $L1H_MODEL_SHA $L1F_MODEL_SHA $L1E_MODEL_SHA $L1D_MODEL_SHA $L1C_MODEL_SHA $B0_CKPT_SHA_342101 $B0_CKPT_SHA_342202 $MODEL_SHA $TRAIN_SHA $FUSION_SHA $DATA_SHA $TRAINJ_SHA $VALJ_SHA $FRESHJ_SHA"; then log "ABORT: unfilled FILL_ placeholders (coordinator)"; exit 2; fi
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
check_shas(){
  [ "$(sha scripts/l3_model.py)" = "$L3_MODEL_SHA" ] || { log "ABORT: l3_model.py sha changed ($(sha scripts/l3_model.py))"; exit 2; }
  [ "$(sha scripts/l3_train.py)" = "$L3_TRAIN_SHA" ] || { log "ABORT: l3_train.py sha changed ($(sha scripts/l3_train.py))"; exit 2; }
  [ "$(sha scripts/l3_gate.py)" = "$L3_GATE_SHA" ] || { log "ABORT: l3_gate.py sha changed ($(sha scripts/l3_gate.py))"; exit 2; }
  [ "$(sha scripts/l2_model.py)" = "$L2_MODEL_SHA" ] || { log "ABORT: l2_model.py sha changed ($(sha scripts/l2_model.py))"; exit 2; }
  [ "$(sha scripts/l2_train.py)" = "$L2_TRAIN_SHA" ] || { log "ABORT: l2_train.py sha changed ($(sha scripts/l2_train.py))"; exit 2; }
  [ "$(sha scripts/l2_gate.py)" = "$L2_GATE_SHA" ] || { log "ABORT: l2_gate.py sha changed ($(sha scripts/l2_gate.py))"; exit 2; }
  [ "$(sha scripts/l1h_model.py)" = "$L1H_MODEL_SHA" ] || { log "ABORT: l1h_model.py sha changed ($(sha scripts/l1h_model.py))"; exit 2; }
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
for s in 342101 342202; do [ -f $L1RUNS/B0-hetero-s$s/summary.json ] || { log "ABORT: $L1RUNS/B0-hetero-s$s/summary.json missing"; exit 2; }; done
mkdir $RUNS.lock 2>/dev/null || { log "ABORT: a chain_day19_L4 lease is already running (lock $RUNS.lock exists)"; exit 2; }
trap 'rmdir $RUNS.lock 2>/dev/null' EXIT
for d in $RUNS $WORK/runs/l3 $L2RUNS $L1RUNS $L1HRUNS $WORK/runs/l1b $WORK/runs/l1c $WORK/runs/l1d $WORK/runs/l1e $WORK/runs/l1f $WORK/runs/l1g; do [ -e $d/lease_$LEASE.json ] && { log "ABORT: $d/lease_$LEASE.json exists (lease numbers continue L1 .. L2; use the next number)"; exit 2; }; done
[ -e $L2RUNS.lock ] || [ -e $WORK/runs/l3.lock ] && { log "ABORT: runs/l2.lock or runs/l3.lock exists (another chain is running)"; exit 2; }
FREE_MB=$(df -Pm /mnt/datasets | awk 'NR==2{print $4}')
if [ "$FREE_MB" -lt 5000 ]; then log "ABORT: only ${FREE_MB} MB free on /mnt/datasets (< 5,000)"; exit 2; fi
[ -d $HF_HUB_CACHE/models--Qwen--Qwen3-1.7B ] || { log "ABORT: Qwen3-1.7B is not in $HF_HUB_CACHE (the chain never downloads)"; exit 2; }
# ---------------------------------------------------------------- before the lease: import self-check (CPU)
SELF=$(CUDA_VISIBLE_DEVICES= timeout -k 30 300 $P -B -c "import sys; sys.path.insert(0, 'scripts'); import l3_model, l3_train, l3_gate, l2_model, l2_train, l2_gate, l1h_model, l1f_model, l1e_model, l1d_model, l1c_model, l1_model, l1_train, l1_fusion, l1_data, torch, transformers, peft; print('import self-check ok', torch.__version__, transformers.__version__, peft.__version__)" 2>&1 | tail -1)
case "$SELF" in "import self-check ok"*) log "L3 $SELF";; *) log "ABORT: import self-check failed: $SELF"; exit 2;; esac
# 5090 SOLO: no other chain, no project trainer / diag, no compute process on the GPU (wait, never preempt)
busy(){ pgrep -af "bash scripts/chain_" | grep -v "chain_day19_L4" | grep -q . || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "scripts/diag_" >/dev/null || pgrep -f "^$P scripts/g4_" >/dev/null || pgrep -f "scripts/l1_" >/dev/null || pgrep -f "scripts/l1b_" >/dev/null || pgrep -f "scripts/l1c_" >/dev/null || pgrep -f "scripts/l1d_" >/dev/null || pgrep -f "scripts/l1e_" >/dev/null || pgrep -f "scripts/l1f_" >/dev/null || pgrep -f "scripts/l1g_" >/dev/null || pgrep -f "scripts/l1h_" >/dev/null || pgrep -f "scripts/l2_train" >/dev/null || pgrep -f "scripts/l2_diag" >/dev/null || pgrep -f "scripts/l3_train" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }   # never "scripts/l3_" (matches the queue -> waits on itself)
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
check_shas
# ---------------------------------------------------------------- the job of this lease (D8-94 section 4)
cond_done(){ [ -f $1/summary.json ]; }
NAME=""; KIND=""; OUT=""; EST=0; ARGS=""; WHY=""
case "$LEASE" in
  45) NAME=V3-same2-s342101; KIND=l2; OUT=$L2RUNS/$NAME; EST=$EST_V2; ARGS="--arm V3 --pair same2 --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --stage1-steps 1000 --stage2-steps 1000";;
  46) NAME=V3-same2-s342202; KIND=l2; OUT=$L2RUNS/$NAME; EST=$EST_V2; ARGS="--arm V3 --pair same2 --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202 --stage1-steps 1000 --stage2-steps 1000";;
  47) NAME=S3a-same2-s342101; KIND=l3; OUT=$RUNS/$NAME; EST=$EST_S2A; ARGS="--arm S3a --pair same2 --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --v2-from $L2RUNS/V3-same2-s342101 --stage3-steps 1500"
      cond_done $L2RUNS/V3-same2-s342101 || WHY="needs runs/l2/V3-same2-s342101/summary.json";;
  48) NAME=S3b-same2-s342101; KIND=l3; OUT=$RUNS/$NAME; EST=$EST_S2B; ARGS="--arm S3b --pair same2 --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --v2-from $L2RUNS/V3-same2-s342101 --init-from $RUNS/S3a-same2-s342101 --stage4-steps 1000"
      cond_done $RUNS/S3a-same2-s342101 || WHY="needs runs/l4/S3a-same2-s342101/summary.json";;
  49) NAME=T2-same2-s342101; KIND=l3; OUT=$RUNS/$NAME; EST=$EST_T2; ARGS="--arm T2 --pair same2 --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --v2-from $L2RUNS/V3-same2-s342101 --init-from $RUNS/S3a-same2-s342101 --stage4-steps 1000"
      cond_done $RUNS/S3a-same2-s342101 || WHY="needs runs/l4/S3a-same2-s342101/summary.json";;
  50) NAME=S3a-same2-s342202; KIND=l3; OUT=$RUNS/$NAME; EST=$EST_S2A; ARGS="--arm S3a --pair same2 --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202 --v2-from $L2RUNS/V3-same2-s342202 --stage3-steps 1500"
      cond_done $L2RUNS/V3-same2-s342202 || WHY="needs runs/l2/V3-same2-s342202/summary.json";;
  51) NAME=S3b-same2-s342202; KIND=l3; OUT=$RUNS/$NAME; EST=$EST_S2B; ARGS="--arm S3b --pair same2 --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202 --v2-from $L2RUNS/V3-same2-s342202 --init-from $RUNS/S3a-same2-s342202 --stage4-steps 1000"
      cond_done $RUNS/S3a-same2-s342202 || WHY="needs runs/l4/S3a-same2-s342202/summary.json";;
  52) NAME=T2-same2-s342202; KIND=l3; OUT=$RUNS/$NAME; EST=$EST_T2; ARGS="--arm T2 --pair same2 --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202 --v2-from $L2RUNS/V3-same2-s342202 --init-from $RUNS/S3a-same2-s342202 --stage4-steps 1000"
      cond_done $RUNS/S3a-same2-s342202 || WHY="needs runs/l4/S3a-same2-s342202/summary.json";;
  *) WHY="no job is assigned to lease $LEASE (D8-94 section 4: 53+ reserved)";;
esac
if [ -n "$WHY" ]; then
  log "L3 lease $LEASE: nothing started ($WHY)"
  echo "{\"lease\": $LEASE, \"job\": \"${NAME:-none}\", \"skipped\": true, \"why\": \"$WHY\", \"utc\": \"$(date -u +%FT%TZ)\"}" > $RUNS/lease_${LEASE}_skipped.json
  log "chain_day19_L4 lease $LEASE finished (skipped)"
  exit 0
fi
if [ -f $OUT/summary.json ]; then log "L3 lease $LEASE: $NAME summary.json exists -> nothing started"; exit 0; fi
if [ -e $OUT ]; then log "L3 lease $LEASE: $OUT exists without summary.json (partial / INCOMPLETE run; never overwrite) -> nothing started, the coordinator decides"; exit 0; fi
COMMON="--train $DATA/train.jsonl --val $DATA/val.jsonl --fresh $DATA/fresh.jsonl --hf-cache $HF_HUB_CACHE --keep-preds --max-ctx-tokens $CTX_SHARD"
if [ "$KIND" = l2 ]; then TRAINER=scripts/l2_train.py; FLAGS="$L2_FLAGS"; else TRAINER=scripts/l3_train.py; FLAGS="$L3_FLAGS"; fi
# ================================================================ the lease (7,200 s inclusive)
START=$(date +%s); DT=$((START+7150)); DL=$((DT-60))
cat > $RUNS/lease_$LEASE.json <<JSON
{"lease": $LEASE, "lease_start_epoch": $START, "total_deadline_epoch": $DT, "job_deadline_epoch": $DL, "inclusive_seconds": 7200, "batch": "L4", "prereg": "$PREREG",
 "user_gpu_approval_utc": "$APPROVAL", "job": "$NAME", "kind": "$KIND", "out": "$OUT", "estimate_min": $EST, "ctx_shard": $CTX_SHARD, "flags": "$FLAGS", "args": "$ARGS",
 "data": "$DATA", "l1_runs": "$L1RUNS", "l2_runs": "$L2RUNS",
 "receiver_sha256": {"342101": "$B0_CKPT_SHA_342101", "342202": "$B0_CKPT_SHA_342202"},
 "data_sha256": {"train": "$TRAINJ_SHA", "val": "$VALJ_SHA", "fresh": "$FRESHJ_SHA"},
 "scripts_sha256": {"l3_model.py": "$L3_MODEL_SHA", "l3_train.py": "$L3_TRAIN_SHA", "l3_gate.py": "$L3_GATE_SHA", "l2_model.py": "$L2_MODEL_SHA", "l2_train.py": "$L2_TRAIN_SHA", "l2_gate.py": "$L2_GATE_SHA", "l1h_model.py": "$L1H_MODEL_SHA", "l1f_model.py": "$L1F_MODEL_SHA", "l1e_model.py": "$L1E_MODEL_SHA", "l1d_model.py": "$L1D_MODEL_SHA", "l1c_model.py": "$L1C_MODEL_SHA", "l1_model.py": "$MODEL_SHA", "l1_train.py": "$TRAIN_SHA", "l1_fusion.py": "$FUSION_SHA", "l1_data.py": "$DATA_SHA"},
 "chain_sha256": "$(sha scripts/chain_day19_L4.sh)", "hf_hub_cache": "$HF_HUB_CACHE", "offline": true, "renewal": "none", "resume": "none"}
JSON
log "L3 lease $LEASE start $START (job deadline +7090, kill at +7150): $NAME ($KIND, estimate $EST min)"
STOP=0
now=$(date +%s); rem=$((DL-now))
if [ $((EST*60)) -gt $rem ]; then
  log "L3 lease $LEASE stops before $NAME: estimate $EST min > remaining $((rem/60)) min"; STOP=1
else
  n=0; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $n -lt 20 ]; do log "L3 $NAME: GPU busy before start; waiting"; sleep 30; n=$((n+1)); done
  check_shas
  log "L3 $NAME start: $TRAINER $ARGS $COMMON $FLAGS"
  now=$(date +%s)
  timeout -k 30 $((DT-now)) $P -B $TRAINER $ARGS $COMMON $FLAGS --out $OUT --deadline-epoch $DL > $RUNS/$NAME.log 2>&1
  x=$?
  log "L3 $NAME done (exit $x): $(tail -1 $RUNS/$NAME.log | cut -c1-300)"
  [ $x -eq 124 ] && log "L3 $NAME TIMED OUT (lease end); no summary.json -> the job is missing until re-run by the coordinator"
fi
END=$(date +%s)
echo "{\"lease\": $LEASE, \"job\": \"$NAME\", \"lease_end_epoch\": $END, \"lease_seconds\": $((END-START)), \"stopped_for_time\": $STOP}" > $RUNS/lease_end_$LEASE.json
log "L3 lease $LEASE GPU work over after $((END-START)) s; the GPU is not used again by this chain"
# ---------------------------------------------------------------- after the GPU work (CPU only): the gate of this job (+ the S2b re-gate after a T2 job)
check_shas
mkdir -p $RUNS/gates $L2RUNS/gates
gate_l3(){ CUDA_VISIBLE_DEVICES= timeout -k 30 600 nice -n 10 $P -B scripts/l3_gate.py --job $1 --runs $RUNS --l2 $L2RUNS --out $RUNS/gates > $RUNS/gates/$1.log 2>&1; log "L3 gate $1 (exit $?): $(grep -a '^GATE' $RUNS/gates/$1.log | tr '\n' ' ')"; }
if [ "$KIND" = l2 ]; then
  CUDA_VISIBLE_DEVICES= timeout -k 30 600 nice -n 10 $P -B scripts/l2_gate.py --job $NAME --runs $L2RUNS --l1h $L1HRUNS --out $L2RUNS/gates > $L2RUNS/gates/$NAME.log 2>&1
  log "L2 gate $NAME (exit $?): $(grep -a '^GATE' $L2RUNS/gates/$NAME.log | tr '\n' ' ')"
else
  gate_l3 $NAME
  case "$NAME" in T2-*) S3B=${NAME/T2-/S3b-}; [ -f $RUNS/$S3B/summary.json ] && gate_l3 $S3B;; esac
fi
log "L3 disk: $RUNS $(du -sm $RUNS | cut -f1) MB; /mnt/datasets free $(df -Pm /mnt/datasets | awk 'NR==2{print $4}') MB"
log "chain_day19_L4 lease $LEASE finished"
