#!/bin/bash
# Day-17 ladder L2 "把对齐读取的隐态信道推向文本水平，再到异构" (Claude). Pre-registration: OVERNIGHT_REPORT D8-92 (the per-job criteria are judged
# by l2_gate.py, which writes the marker files GATE_<job>_<verdict> the later leases' conditions read; the same2 Hu / Hs runs are also fed to
# l1h_aggregate.py). Trainers: l2_train.py (arms R1 / V1 / Hu on l2_model.py, on top of l1h_model.py and the PINNED L1 files) and l1h_train.py
# (the pinned L1h trainer, for the same2 arms Hs / Hu-d32 / Hu-L9 / Hu s2). The frozen B0 checkpoint of the seed is every run's 'default' LoRA.
# Structure and safety rules = chain_day16_L1h.sh (uk-linux 5090: sha pins, FILL_ check, one lock dir, never overwrite, SOLO GPU wait, ONE
# 7,200 s lease per launch, timeout -k on our own child PIDs only, never pkill / killall, CPU gates after the GPU work). No renewal, no resume.
#
# ONE LAUNCH = ONE LEASE = ONE JOB (--lease N selects the job from the FIXED table of D8-92 section 4; a lease whose condition is not met starts
# nothing and exits 0 so the queue moves on):
#   20 R1-same2-s342101        21 V1-same2-s342101        22 Hs-same2-s342101 (runs/l1h)   23 Hu-d32-same2-s342101   24 Hu-L9-same2-s342101
#   25 Hu-same2-s342202 (runs/l1h)   26 R1-same2-s342202 [GATE_R1-same2-s342101_PASS and Hu-same2-s342202 done]
#   27 V1-same2-s342202 [GATE_V1-same2-s342101_PASS]   28 Hs-same2-s342202 (runs/l1h) [GATE_Hs-same2-s342101_ADDS]
#   29 Hu-hetero-s342101   30 R1-hetero-s342101 [Hu-hetero-s342101 done]   31 Hu-hetero-s342202 [GATE_Hu-hetero-s342101_PASS]
#   32 R1-hetero-s342202 [GATE_R1-hetero-s342101_PASS and Hu-hetero-s342202 done]   33+ nothing (reserved for the coordinator)
#   GPU approval: the user's "一次性实验顺序拉起 … 至少 40 小时" (2026-11-01 ≈14:55 / 15:05 UTC), passed as $2 into every lease_<N>.json.
#   Every job gets --deadline-epoch (lease end - 60 s) and runs under timeout -k 30 (lease end); a job that does not fit its estimate is not started.
# Writes only under /mnt/datasets (runs/l2/..., runs/l1h/<same2 L1h runs>, runs/l2/chain_day17.log). Data, HF cache, runs/l1 and the L1h Hu s1
# checkpoint are read, never written.
#
# The coordinator fills every FILL_ (after the review, the smokes and E0; the chain refuses FILL_):
#   L2_MODEL_SHA / L2_TRAIN_SHA / L2_GATE_SHA   scripts/l2_{model,train,gate}.py      L1H_MODEL_SHA / L1H_TRAIN_SHA / L1H_AGG_SHA   the pinned L1h files
#   L1F_MODEL_SHA / L1E_MODEL_SHA / L1D_MODEL_SHA / L1C_MODEL_SHA (imported)      B0_CKPT_SHA_342101 / B0_CKPT_SHA_342202
#   L1H_HU_S1_CKPT_SHA   runs/l1h/Hu-same2-s342101/trainable.pt (the frozen sender / decoder of R1 s1)      SEL_LAYER   the E0c layer (D8-92 section 3)
#   MODEL_SHA / TRAIN_SHA / FUSION_SHA / DATA_SHA / TRAINJ_SHA / VALJ_SHA / FRESHJ_SHA = the L1 pins (already filled)
# Launch: one lease per launch, via scripts/l2_leases.sh, after the previous chain process has exited.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
PREREG="$1"; APPROVAL="$2"
[ "$PREREG" = "D8-92" ] && [ -n "$APPROVAL" ] || { echo "usage: chain_day17_L2.sh D8-92 <user GPU approval UTC> --lease N [--est-r1 90 --est-v1 80 --est-l1h 70 --est-hetero 75 --est-hetero-r1 95]"; exit 2; }
shift 2
LEASE=""; EST_R1=90; EST_V1=80; EST_L1H=70; EST_HET=75; EST_HET_R1=95
while [ $# -gt 0 ]; do
  case "$1" in
    --lease) LEASE="$2"; shift 2;;
    --est-r1) EST_R1="$2"; shift 2;;
    --est-v1) EST_V1="$2"; shift 2;;
    --est-l1h) EST_L1H="$2"; shift 2;;
    --est-hetero) EST_HET="$2"; shift 2;;
    --est-hetero-r1) EST_HET_R1="$2"; shift 2;;
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
RUNS=$WORK/runs/l2
L1HRUNS=$WORK/runs/l1h                                                    # the L1h runs (Hu s1 = the frozen sender of R1 s1; Hs / Hu s2 are written here)
L1RUNS=$WORK/runs/l1                                                      # B0 (the frozen LoRA of every agent) / T of batch L1 (read only)
LOG=runs/l2/chain_day17.log
# ---------------------------------------------------------------- pinned by the coordinator (after the review, the smokes and E0)
L2_MODEL_SHA=FILL_L2_MODEL_SHA                                             # scripts/l2_model.py
L2_TRAIN_SHA=FILL_L2_TRAIN_SHA                                             # scripts/l2_train.py
L2_GATE_SHA=FILL_L2_GATE_SHA                                               # scripts/l2_gate.py
L1H_MODEL_SHA=FILL_L1H_MODEL_SHA                                           # scripts/l1h_model.py (pinned in lease 19)
L1H_TRAIN_SHA=FILL_L1H_TRAIN_SHA                                           # scripts/l1h_train.py (pinned in lease 19)
L1H_AGG_SHA=FILL_L1H_AGG_SHA                                               # scripts/l1h_aggregate.py (pinned in lease 19)
L1F_MODEL_SHA=FILL_L1F_MODEL_SHA                                           # scripts/l1f_model.py (imported by l1h_model)
L1E_MODEL_SHA=FILL_L1E_MODEL_SHA                                           # scripts/l1e_model.py (imported by l1f_model)
L1D_MODEL_SHA=FILL_L1D_MODEL_SHA                                           # scripts/l1d_model.py (imported by l1e_model)
L1C_MODEL_SHA=FILL_L1C_MODEL_SHA                                           # scripts/l1c_model.py (imported by l1d_model)
B0_CKPT_SHA_342101=FILL_B0_CKPT_SHA_342101                                 # runs/l1/B0-hetero-s342101/trainable.pt
B0_CKPT_SHA_342202=FILL_B0_CKPT_SHA_342202                                 # runs/l1/B0-hetero-s342202/trainable.pt
L1H_HU_S1_CKPT_SHA=FILL_L1H_HU_S1_CKPT_SHA                                 # runs/l1h/Hu-same2-s342101/trainable.pt (R1 s1 --init-from)
SEL_LAYER=FILL_SEL_LAYER                                                   # E0c: the attention layer of the V1 span selection
MODEL_SHA=aa3a4eec71ea38b11b034282cd550c671c6358d037c2a7179c80ca4d438f5b86                                                   # scripts/l1_model.py (L1 pin)
TRAIN_SHA=5f8eee7833fd53851650ff4aa2249269e3a7fbab76b8de92695c403f23b9f33a                                                   # scripts/l1_train.py (L1 pin)
FUSION_SHA=a0f56eb6de811a4247fa784f701c23ef43119626ff1e9e6821846d379fa6a474                                                 # scripts/l1_fusion.py (L1 pin)
DATA_SHA=41d0db811e710579ddcdab76c8c622358984423da54a8e792716fa76c75263dd                                                     # scripts/l1_data.py (L1 pin)
TRAINJ_SHA=4b6af5e2bdb78d23d57ba83592c71944f9dbbb6568db4e9bdfb2ffd1b7e0caa3                                                 # $DATA/train.jsonl (L1 pin)
VALJ_SHA=b1e240bbe70873e1f2ac7586171caf6d3620f269c0e02807033a5fded5f68ecf                                                     # $DATA/val.jsonl (L1 pin)
FRESHJ_SHA=985610d8e092745c9edc37b7624ab7e502a9e94da9b0c7a280cd4f68d00acc60                                                 # $DATA/fresh.jsonl (L1 pin)
# ---------------------------------------------------------------- the knobs (D8-92 section 3; defaults of the trainers, passed explicitly for the record)
CTX_SHARD=900
L1H_FLAGS="--steps 2000 --k 8 --eval-batch 32 --absence-ps 0.0,1.0 --tau 1.0 --n-blocks 2"
L2_FLAGS="--eval-batch 32 --tau 1.0 --n-blocks 2"
mkdir -p runs/l2 runs/l1h
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day17_L2 started pid $$ host $(hostname) lease $LEASE sha256 $(sha256sum scripts/chain_day17_L2.sh | cut -c1-16) prereg $PREREG approval $APPROVAL est r1 $EST_R1 v1 $EST_V1 l1h $EST_L1H hetero $EST_HET hetero-r1 $EST_HET_R1 min; cap $CTX_SHARD; sel_layer $SEL_LAYER"
if grep -q -e "FILL_" <<<"$L2_MODEL_SHA $L2_TRAIN_SHA $L2_GATE_SHA $L1H_MODEL_SHA $L1H_TRAIN_SHA $L1H_AGG_SHA $L1F_MODEL_SHA $L1E_MODEL_SHA $L1D_MODEL_SHA $L1C_MODEL_SHA $B0_CKPT_SHA_342101 $B0_CKPT_SHA_342202 $L1H_HU_S1_CKPT_SHA $SEL_LAYER $MODEL_SHA $TRAIN_SHA $FUSION_SHA $DATA_SHA $TRAINJ_SHA $VALJ_SHA $FRESHJ_SHA"; then log "ABORT: unfilled FILL_ placeholders (coordinator)"; exit 2; fi
case "$SEL_LAYER" in ''|*[!0-9]*) log "ABORT: SEL_LAYER '$SEL_LAYER' is not an integer"; exit 2;; esac
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
check_shas(){
  [ "$(sha scripts/l2_model.py)" = "$L2_MODEL_SHA" ] || { log "ABORT: l2_model.py sha changed ($(sha scripts/l2_model.py))"; exit 2; }
  [ "$(sha scripts/l2_train.py)" = "$L2_TRAIN_SHA" ] || { log "ABORT: l2_train.py sha changed ($(sha scripts/l2_train.py))"; exit 2; }
  [ "$(sha scripts/l2_gate.py)" = "$L2_GATE_SHA" ] || { log "ABORT: l2_gate.py sha changed ($(sha scripts/l2_gate.py))"; exit 2; }
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
  [ "$(sha $L1HRUNS/Hu-same2-s342101/trainable.pt)" = "$L1H_HU_S1_CKPT_SHA" ] || { log "ABORT: runs/l1h/Hu-same2-s342101/trainable.pt sha changed or missing"; exit 2; }
}
check_shas
for s in 342101 342202; do [ -f $L1RUNS/B0-hetero-s$s/summary.json ] || { log "ABORT: $L1RUNS/B0-hetero-s$s/summary.json missing"; exit 2; }; done
mkdir $RUNS.lock 2>/dev/null || { log "ABORT: a chain_day17_L2 lease is already running (lock $RUNS.lock exists)"; exit 2; }
trap 'rmdir $RUNS.lock 2>/dev/null' EXIT
for d in $RUNS $L1RUNS $L1HRUNS $WORK/runs/l1b $WORK/runs/l1c $WORK/runs/l1d $WORK/runs/l1e $WORK/runs/l1f $WORK/runs/l1g; do [ -e $d/lease_$LEASE.json ] && { log "ABORT: $d/lease_$LEASE.json exists (lease numbers continue L1 .. L1h; use the next number)"; exit 2; }; done
[ -e $L1HRUNS.lock ] && { log "ABORT: runs/l1h.lock exists (an L1h chain is running)"; exit 2; }
FREE_MB=$(df -Pm /mnt/datasets | awk 'NR==2{print $4}')
if [ "$FREE_MB" -lt 5000 ]; then log "ABORT: only ${FREE_MB} MB free on /mnt/datasets (< 5,000)"; exit 2; fi
[ -d $HF_HUB_CACHE/models--Qwen--Qwen3-1.7B ] && [ -d $HF_HUB_CACHE/models--meta-llama--Llama-3.2-1B-Instruct ] || { log "ABORT: the two models are not in $HF_HUB_CACHE (the chain never downloads)"; exit 2; }
# ---------------------------------------------------------------- before the lease: import self-check (CPU)
SELF=$(CUDA_VISIBLE_DEVICES= timeout -k 30 300 $P -B -c "import sys; sys.path.insert(0, 'scripts'); import l2_model, l2_train, l2_gate, l1h_model, l1h_train, l1h_aggregate, l1f_model, l1e_model, l1d_model, l1c_model, l1_model, l1_train, l1_fusion, l1_data, torch, transformers, peft; print('import self-check ok', torch.__version__, transformers.__version__, peft.__version__)" 2>&1 | tail -1)
case "$SELF" in "import self-check ok"*) log "L2 $SELF";; *) log "ABORT: import self-check failed: $SELF"; exit 2;; esac
# 5090 SOLO: no other chain, no project trainer / diag, no compute process on the GPU (wait, never preempt)
busy(){ pgrep -af "bash scripts/chain_" | grep -v "chain_day17_L2" | grep -q . || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "scripts/diag_" >/dev/null || pgrep -f "^$P scripts/g4_" >/dev/null || pgrep -f "scripts/l1_" >/dev/null || pgrep -f "scripts/l1b_" >/dev/null || pgrep -f "scripts/l1c_" >/dev/null || pgrep -f "scripts/l1d_" >/dev/null || pgrep -f "scripts/l1e_" >/dev/null || pgrep -f "scripts/l1f_" >/dev/null || pgrep -f "scripts/l1g_" >/dev/null || pgrep -f "scripts/l1h_" >/dev/null || pgrep -f "scripts/l2_train" >/dev/null || pgrep -f "scripts/l2_diag" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }   # never "scripts/l2_" (matches the queue -> waits on itself); our own trainer child is started after this wait
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
check_shas
# ---------------------------------------------------------------- the job of this lease (D8-92 section 4)
cond_marker(){ [ -f $RUNS/GATE_$1 ]; }                                            # $1 = <job>_<verdict>
cond_done(){ [ -f $1/summary.json ]; }
NAME=""; KIND=""; OUT=""; EST=0; ARGS=""; WHY=""
case "$LEASE" in
  20) NAME=R1-same2-s342101; KIND=l2; OUT=$RUNS/$NAME; EST=$EST_R1; ARGS="--arm R1 --pair same2 --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --init-from $L1HRUNS/Hu-same2-s342101 --stage2-steps 1500";;
  21) NAME=V1-same2-s342101; KIND=l2; OUT=$RUNS/$NAME; EST=$EST_V1; ARGS="--arm V1 --pair same2 --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --sel-layer $SEL_LAYER --stage1-steps 1000 --stage2-steps 1000";;
  22) NAME=Hs-same2-s342101; KIND=l1h; OUT=$L1HRUNS/$NAME; EST=$EST_L1H; ARGS="--arm Hs --pair same2 --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101";;
  23) NAME=Hu-d32-same2-s342101; KIND=l1h; OUT=$RUNS/$NAME; EST=$EST_L1H; ARGS="--arm Hu --pair same2 --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --d-e 32";;
  24) NAME=Hu-L9-same2-s342101; KIND=l1h; OUT=$RUNS/$NAME; EST=$EST_L1H; ARGS="--arm Hu --pair same2 --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --layer-frac 0.34";;
  25) NAME=Hu-same2-s342202; KIND=l1h; OUT=$L1HRUNS/$NAME; EST=$EST_L1H; ARGS="--arm Hu --pair same2 --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202";;
  26) NAME=R1-same2-s342202; KIND=l2; OUT=$RUNS/$NAME; EST=$EST_R1; ARGS="--arm R1 --pair same2 --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202 --init-from $L1HRUNS/Hu-same2-s342202 --stage2-steps 1500"
      cond_marker R1-same2-s342101_PASS && cond_done $L1HRUNS/Hu-same2-s342202 || WHY="needs GATE_R1-same2-s342101_PASS and runs/l1h/Hu-same2-s342202/summary.json";;
  27) NAME=V1-same2-s342202; KIND=l2; OUT=$RUNS/$NAME; EST=$EST_V1; ARGS="--arm V1 --pair same2 --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202 --sel-layer $SEL_LAYER --stage1-steps 1000 --stage2-steps 1000"
      cond_marker V1-same2-s342101_PASS || WHY="needs GATE_V1-same2-s342101_PASS";;
  28) NAME=Hs-same2-s342202; KIND=l1h; OUT=$L1HRUNS/$NAME; EST=$EST_L1H; ARGS="--arm Hs --pair same2 --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202"
      cond_marker Hs-same2-s342101_ADDS || WHY="needs GATE_Hs-same2-s342101_ADDS";;
  29) NAME=Hu-hetero-s342101; KIND=l2; OUT=$RUNS/$NAME; EST=$EST_HET; ARGS="--arm Hu --pair hetero --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --stage1-steps 2000";;
  30) NAME=R1-hetero-s342101; KIND=l2; OUT=$RUNS/$NAME; EST=$EST_HET_R1; ARGS="--arm R1 --pair hetero --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --init-from $RUNS/Hu-hetero-s342101 --stage2-steps 1500"
      cond_done $RUNS/Hu-hetero-s342101 || WHY="needs runs/l2/Hu-hetero-s342101/summary.json";;
  31) NAME=Hu-hetero-s342202; KIND=l2; OUT=$RUNS/$NAME; EST=$EST_HET; ARGS="--arm Hu --pair hetero --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202 --stage1-steps 2000"
      cond_marker Hu-hetero-s342101_PASS || WHY="needs GATE_Hu-hetero-s342101_PASS";;
  32) NAME=R1-hetero-s342202; KIND=l2; OUT=$RUNS/$NAME; EST=$EST_HET_R1; ARGS="--arm R1 --pair hetero --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202 --init-from $RUNS/Hu-hetero-s342202 --stage2-steps 1500"
      cond_marker R1-hetero-s342101_PASS && cond_done $RUNS/Hu-hetero-s342202 || WHY="needs GATE_R1-hetero-s342101_PASS and runs/l2/Hu-hetero-s342202/summary.json";;
  *) WHY="no job is assigned to lease $LEASE (D8-92 section 4: 33+ reserved for the coordinator)";;
esac
if [ -n "$WHY" ]; then
  log "L2 lease $LEASE: nothing started ($WHY)"
  echo "{\"lease\": $LEASE, \"job\": \"${NAME:-none}\", \"skipped\": true, \"why\": \"$WHY\", \"utc\": \"$(date -u +%FT%TZ)\"}" > $RUNS/lease_${LEASE}_skipped.json
  log "chain_day17_L2 lease $LEASE finished (skipped)"
  exit 0
fi
if [ -f $OUT/summary.json ]; then log "L2 lease $LEASE: $NAME summary.json exists -> nothing started"; exit 0; fi
if [ -e $OUT ]; then log "L2 lease $LEASE: $OUT exists without summary.json (partial / INCOMPLETE run; never overwrite) -> nothing started, the coordinator decides"; exit 0; fi
COMMON="--train $DATA/train.jsonl --val $DATA/val.jsonl --fresh $DATA/fresh.jsonl --hf-cache $HF_HUB_CACHE --keep-preds --max-ctx-tokens $CTX_SHARD"
if [ "$KIND" = l2 ]; then TRAINER=scripts/l2_train.py; FLAGS="$L2_FLAGS"; else TRAINER=scripts/l1h_train.py; FLAGS="$L1H_FLAGS"; fi
# ================================================================ the lease (7,200 s inclusive)
START=$(date +%s); DT=$((START+7150)); DL=$((DT-60))
cat > $RUNS/lease_$LEASE.json <<JSON
{"lease": $LEASE, "lease_start_epoch": $START, "total_deadline_epoch": $DT, "job_deadline_epoch": $DL, "inclusive_seconds": 7200, "batch": "L2", "prereg": "$PREREG",
 "user_gpu_approval_utc": "$APPROVAL", "job": "$NAME", "kind": "$KIND", "out": "$OUT", "estimate_min": $EST, "ctx_shard": $CTX_SHARD, "flags": "$FLAGS", "args": "$ARGS",
 "sel_layer": $SEL_LAYER, "data": "$DATA", "l1_runs": "$L1RUNS", "l1h_runs": "$L1HRUNS",
 "receiver_sha256": {"342101": "$B0_CKPT_SHA_342101", "342202": "$B0_CKPT_SHA_342202"}, "l1h_hu_s1_ckpt_sha256": "$L1H_HU_S1_CKPT_SHA",
 "data_sha256": {"train": "$TRAINJ_SHA", "val": "$VALJ_SHA", "fresh": "$FRESHJ_SHA"},
 "scripts_sha256": {"l2_model.py": "$L2_MODEL_SHA", "l2_train.py": "$L2_TRAIN_SHA", "l2_gate.py": "$L2_GATE_SHA", "l1h_model.py": "$L1H_MODEL_SHA", "l1h_train.py": "$L1H_TRAIN_SHA", "l1h_aggregate.py": "$L1H_AGG_SHA", "l1f_model.py": "$L1F_MODEL_SHA", "l1e_model.py": "$L1E_MODEL_SHA", "l1d_model.py": "$L1D_MODEL_SHA", "l1c_model.py": "$L1C_MODEL_SHA", "l1_model.py": "$MODEL_SHA", "l1_train.py": "$TRAIN_SHA", "l1_fusion.py": "$FUSION_SHA", "l1_data.py": "$DATA_SHA"},
 "chain_sha256": "$(sha scripts/chain_day17_L2.sh)", "hf_hub_cache": "$HF_HUB_CACHE", "offline": true, "renewal": "none", "resume": "none"}
JSON
log "L2 lease $LEASE start $START (job deadline +7090, kill at +7150): $NAME ($KIND, estimate $EST min)"
STOP=0
now=$(date +%s); rem=$((DL-now))
if [ $((EST*60)) -gt $rem ]; then
  log "L2 lease $LEASE stops before $NAME: estimate $EST min > remaining $((rem/60)) min"; STOP=1
else
  n=0; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $n -lt 20 ]; do log "L2 $NAME: GPU busy before start; waiting"; sleep 30; n=$((n+1)); done
  check_shas
  log "L2 $NAME start: $TRAINER $ARGS $COMMON $FLAGS"
  now=$(date +%s)
  timeout -k 30 $((DT-now)) $P -B $TRAINER $ARGS $COMMON $FLAGS --out $OUT --deadline-epoch $DL > $RUNS/$NAME.log 2>&1
  x=$?
  log "L2 $NAME done (exit $x): $(tail -1 $RUNS/$NAME.log | cut -c1-300)"
  [ $x -eq 124 ] && log "L2 $NAME TIMED OUT (lease end); no summary.json -> the job is missing until re-run by the coordinator"
fi
END=$(date +%s)
echo "{\"lease\": $LEASE, \"job\": \"$NAME\", \"lease_end_epoch\": $END, \"lease_seconds\": $((END-START)), \"stopped_for_time\": $STOP}" > $RUNS/lease_end_$LEASE.json
log "L2 lease $LEASE GPU work over after $((END-START)) s; the GPU is not used again by this chain"
# ---------------------------------------------------------------- after the GPU work (CPU only): the gate of this job (+ the L1h aggregate for the same2 L1h runs)
check_shas
mkdir -p $RUNS/gates
CUDA_VISIBLE_DEVICES= timeout -k 30 600 nice -n 10 $P -B scripts/l2_gate.py --job $NAME --runs $RUNS --l1h $L1HRUNS --out $RUNS/gates > $RUNS/gates/$NAME.log 2>&1
log "L2 gate $NAME (exit $?): $(grep -a '^GATE' $RUNS/gates/$NAME.log | tr '\n' ' ')"
if [ "$KIND" = l1h ] && [ "$OUT" != "${OUT#$L1HRUNS/}" ]; then
  mkdir -p $L1HRUNS/aggregate
  CUDA_VISIBLE_DEVICES= timeout -k 30 1800 nice -n 10 $P -B scripts/l1h_aggregate.py --l1 $L1RUNS --l1h $L1HRUNS --out $L1HRUNS/aggregate/l2_lease$LEASE > $L1HRUNS/aggregate/l2_lease$LEASE.log 2>&1
  log "L1h aggregate after lease $LEASE (exit $?): $(grep -a -E '^(G1h|VERDICT)' $L1HRUNS/aggregate/l2_lease$LEASE.log | tr '\n' ' ')"
fi
log "L2 disk: $RUNS $(du -sm $RUNS | cut -f1) MB; /mnt/datasets free $(df -Pm /mnt/datasets | awk 'NR==2{print $4}') MB"
log "chain_day17_L2 lease $LEASE finished"
