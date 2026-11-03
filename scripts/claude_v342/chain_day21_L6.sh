#!/bin/bash
# Day-21 ladder L6 "训练验证头：候选并集上的因子从生成似然换成验证后验，sheaf-GBP 求和" (Claude). Pre-registration: OVERNIGHT_REPORT D8-96 (the per-job
# criteria are judged by l6_gate.py, which writes the marker files GATE_<job>_<verdict>). Runner: l6_verifier.py on top of the PINNED l5_belief /
# l2_model / L1h / L1 files; inputs = the frozen V1 pair of the seed (runs/l2), the B0 checkpoint of the seed as the 'default' LoRA (runs/l1),
# the L5 B1t records of the seed (runs/l5: the evaluation candidates) and, for the VER jobs, the GEN candidates of this chain (runs/l6).
# Structure and safety rules = chain_day20_L5.sh (uk-linux 5090: sha pins, FILL_ check, one lock dir, never overwrite, SOLO GPU wait, ONE 7,200 s
# lease per launch, timeout -k on our own child PIDs only, never pkill / killall, CPU gates after the GPU work). No renewal, no resume.
#
# ONE LAUNCH = ONE LEASE = ONE JOB (--lease N selects the job from the FIXED table of D8-96 section 4; a lease whose condition is not met starts
# nothing and exits 0 so the queue moves on):
#   57 GEN-same2-s342101 [runs/l2/V1-same2-s342101 done]   58 GEN-same2-s342202 [V1 s2 done]
#   59 VER-same2-s342101 [GEN s1 + runs/l5/B1t-same2-s342101 preds]   60 VER-same2-s342202 [GEN s2 + L5 B1t s2 preds]
#   61+ nothing (reserved for the coordinator)
#   GPU approval: the user's "批准，做" (2026-11-03, D8-96 leases 57-60), passed as $2 into every lease_<N>.json.
# Writes only under /mnt/datasets (runs/l6/..., runs/l6/chain_day21.log). Data, HF cache, runs/l1, runs/l2, runs/l4, runs/l5 are read, never written.
#
# The coordinator fills every FILL_ (after the tests and the smokes; the chain refuses FILL_):
#   L6_VERIFIER_SHA / L6_GATE_SHA / L5_BELIEF_SHA   L2_MODEL_SHA   L1H_MODEL_SHA   L1F / L1E / L1D / L1C_MODEL_SHA   B0_CKPT_SHA_342101 / 342202
#   MODEL_SHA / TRAIN_SHA / FUSION_SHA / DATA_SHA / TRAINJ_SHA / VALJ_SHA / FRESHJ_SHA = the L1 pins (already filled)
# Launch: one lease per launch, via scripts/l6_leases.sh, after the previous chain process has exited.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
PREREG="$1"; APPROVAL="$2"
[ "$PREREG" = "D8-96" ] && [ -n "$APPROVAL" ] || { echo "usage: chain_day21_L6.sh D8-96 <user GPU approval UTC> --lease N [--est-gen 40 --est-ver 100]"; exit 2; }
shift 2
LEASE=""; EST_GEN=40; EST_VER=100
while [ $# -gt 0 ]; do
  case "$1" in
    --lease) LEASE="$2"; shift 2;;
    --est-gen) EST_GEN="$2"; shift 2;;
    --est-ver) EST_VER="$2"; shift 2;;
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
RUNS=$WORK/runs/l6
L5RUNS=$WORK/runs/l5                                                      # B1t records (read only)
L4RUNS=$WORK/runs/l4                                                      # T2 references (read only)
L2RUNS=$WORK/runs/l2                                                      # V1 (read only)
L1RUNS=$WORK/runs/l1                                                      # B0 (read only)
LOG=runs/l6/chain_day21.log
# ---------------------------------------------------------------- pinned by the coordinator (after the tests and the smokes)
L6_VERIFIER_SHA=FILL_L6_VERIFIER_SHA                                       # scripts/l6_verifier.py
L6_GATE_SHA=FILL_L6_GATE_SHA                                               # scripts/l6_gate.py
L5_BELIEF_SHA=FILL_L5_BELIEF_SHA                                           # scripts/l5_belief.py
L2_MODEL_SHA=FILL_L2_MODEL_SHA                                             # scripts/l2_model.py
L1H_MODEL_SHA=FILL_L1H_MODEL_SHA                                           # scripts/l1h_model.py
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
# ---------------------------------------------------------------- the knobs (D8-96 sections 2 and 4)
CTX_SHARD=900
N_TRAIN=4000                                                              # train[:4000] (GEN); the VER jobs train one epoch over these rows
TRAIN_STOP_S=4500                                                         # the soft stop of the verifier training: lease start + 75 min
L6_FLAGS="--k-beams 4 --gen-batch 8 --score-batch 32 --batch 2 --lr-lora 1e-4 --warmup 100"
mkdir -p runs/l6
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day21_L6 started pid $$ host $(hostname) lease $LEASE sha256 $(sha256sum scripts/chain_day21_L6.sh | cut -c1-16) prereg $PREREG approval $APPROVAL est gen $EST_GEN ver $EST_VER min; cap $CTX_SHARD n_train $N_TRAIN"
if grep -q -e "FILL_" <<<"$L6_VERIFIER_SHA $L6_GATE_SHA $L5_BELIEF_SHA $L2_MODEL_SHA $L1H_MODEL_SHA $L1F_MODEL_SHA $L1E_MODEL_SHA $L1D_MODEL_SHA $L1C_MODEL_SHA $B0_CKPT_SHA_342101 $B0_CKPT_SHA_342202 $MODEL_SHA $TRAIN_SHA $FUSION_SHA $DATA_SHA $TRAINJ_SHA $VALJ_SHA $FRESHJ_SHA"; then log "ABORT: unfilled FILL_ placeholders (coordinator)"; exit 2; fi
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
check_shas(){
  [ "$(sha scripts/l6_verifier.py)" = "$L6_VERIFIER_SHA" ] || { log "ABORT: l6_verifier.py sha changed ($(sha scripts/l6_verifier.py))"; exit 2; }
  [ "$(sha scripts/l6_gate.py)" = "$L6_GATE_SHA" ] || { log "ABORT: l6_gate.py sha changed ($(sha scripts/l6_gate.py))"; exit 2; }
  [ "$(sha scripts/l5_belief.py)" = "$L5_BELIEF_SHA" ] || { log "ABORT: l5_belief.py sha changed ($(sha scripts/l5_belief.py))"; exit 2; }
  [ "$(sha scripts/l2_model.py)" = "$L2_MODEL_SHA" ] || { log "ABORT: l2_model.py sha changed ($(sha scripts/l2_model.py))"; exit 2; }
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
mkdir $RUNS.lock 2>/dev/null || { log "ABORT: a chain_day21_L6 lease is already running (lock $RUNS.lock exists)"; exit 2; }
trap 'rmdir $RUNS.lock 2>/dev/null' EXIT
for d in $RUNS $L5RUNS $L4RUNS $WORK/runs/l3 $L2RUNS $L1RUNS $WORK/runs/l1h $WORK/runs/l1b $WORK/runs/l1c $WORK/runs/l1d $WORK/runs/l1e $WORK/runs/l1f $WORK/runs/l1g; do [ -e $d/lease_$LEASE.json ] && { log "ABORT: $d/lease_$LEASE.json exists (lease numbers continue L1 .. L5; use the next number)"; exit 2; }; done
[ -e $L2RUNS.lock ] || [ -e $WORK/runs/l3.lock ] || [ -e $L4RUNS.lock ] || [ -e $L5RUNS.lock ] && { log "ABORT: runs/l2.lock, l3.lock, l4.lock or l5.lock exists (another chain is running)"; exit 2; }
FREE_MB=$(df -Pm /mnt/datasets | awk 'NR==2{print $4}')
if [ "$FREE_MB" -lt 5000 ]; then log "ABORT: only ${FREE_MB} MB free on /mnt/datasets (< 5,000)"; exit 2; fi
[ -d $HF_HUB_CACHE/models--Qwen--Qwen3-1.7B ] || { log "ABORT: Qwen3-1.7B is not in $HF_HUB_CACHE (the chain never downloads)"; exit 2; }
# ---------------------------------------------------------------- before the lease: import self-check (CPU)
SELF=$(CUDA_VISIBLE_DEVICES= timeout -k 30 300 $P -B -c "import sys; sys.path.insert(0, 'scripts'); import l6_verifier, l6_gate, l5_belief, l2_model, l1h_model, l1f_model, l1e_model, l1d_model, l1c_model, l1_model, l1_train, l1_fusion, l1_data, torch, transformers, peft; print('import self-check ok', torch.__version__, transformers.__version__, peft.__version__)" 2>&1 | tail -1)
case "$SELF" in "import self-check ok"*) log "L6 $SELF";; *) log "ABORT: import self-check failed: $SELF"; exit 2;; esac
# 5090 SOLO: no other chain, no project trainer / diag, no compute process on the GPU (wait, never preempt)
busy(){ pgrep -af "bash scripts/chain_" | grep -v "chain_day21_L6" | grep -q . || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "scripts/diag_" >/dev/null || pgrep -f "^$P scripts/g4_" >/dev/null || pgrep -f "scripts/l1_" >/dev/null || pgrep -f "scripts/l1b_" >/dev/null || pgrep -f "scripts/l1c_" >/dev/null || pgrep -f "scripts/l1d_" >/dev/null || pgrep -f "scripts/l1e_" >/dev/null || pgrep -f "scripts/l1f_" >/dev/null || pgrep -f "scripts/l1g_" >/dev/null || pgrep -f "scripts/l1h_" >/dev/null || pgrep -f "scripts/l2_train" >/dev/null || pgrep -f "scripts/l2_diag" >/dev/null || pgrep -f "scripts/l3_train" >/dev/null || pgrep -f "scripts/l5_belief" >/dev/null || pgrep -f "scripts/l6_verifier" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }   # never "scripts/l6_" (matches the queue -> waits on itself)
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
check_shas
# ---------------------------------------------------------------- the job of this lease (D8-96 section 4)
cond_done(){ [ -f $1/summary.json ]; }
l5_ok(){ [ -f $L5RUNS/B1t-same2-s$1/summary.json ] && [ -f $L5RUNS/B1t-same2-s$1/preds_val.json ] && [ -f $L5RUNS/B1t-same2-s$1/preds_fresh.json ]; }
NAME=""; OUT=""; EST=0; ARGS=""; WHY=""; KIND=""
case "$LEASE" in
  57) NAME=GEN-same2-s342101; KIND=gen; OUT=$RUNS/$NAME; EST=$EST_GEN; ARGS="--mode gen --pair same2 --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --v-from $L2RUNS/V1-same2-s342101 --limit-train $N_TRAIN"
      cond_done $L2RUNS/V1-same2-s342101 || WHY="needs runs/l2/V1-same2-s342101/summary.json";;
  58) NAME=GEN-same2-s342202; KIND=gen; OUT=$RUNS/$NAME; EST=$EST_GEN; ARGS="--mode gen --pair same2 --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202 --v-from $L2RUNS/V1-same2-s342202 --limit-train $N_TRAIN"
      cond_done $L2RUNS/V1-same2-s342202 || WHY="needs runs/l2/V1-same2-s342202/summary.json";;
  59) NAME=VER-same2-s342101; KIND=ver; OUT=$RUNS/$NAME; EST=$EST_VER; ARGS="--mode train --pair same2 --seed 342101 --receiver-from $L1RUNS/B0-hetero-s342101 --v-from $L2RUNS/V1-same2-s342101 --cands-from $RUNS/GEN-same2-s342101/cands_train.json --l5-from $L5RUNS/B1t-same2-s342101"
      cond_done $RUNS/GEN-same2-s342101 && [ -f $RUNS/GEN-same2-s342101/cands_train.json ] && l5_ok 342101 || WHY="needs runs/l6/GEN-same2-s342101 (summary + cands_train.json) and runs/l5/B1t-same2-s342101 (summary + preds)";;
  60) NAME=VER-same2-s342202; KIND=ver; OUT=$RUNS/$NAME; EST=$EST_VER; ARGS="--mode train --pair same2 --seed 342202 --receiver-from $L1RUNS/B0-hetero-s342202 --v-from $L2RUNS/V1-same2-s342202 --cands-from $RUNS/GEN-same2-s342202/cands_train.json --l5-from $L5RUNS/B1t-same2-s342202"
      cond_done $RUNS/GEN-same2-s342202 && [ -f $RUNS/GEN-same2-s342202/cands_train.json ] && l5_ok 342202 || WHY="needs runs/l6/GEN-same2-s342202 (summary + cands_train.json) and runs/l5/B1t-same2-s342202 (summary + preds)";;
  *) WHY="no job is assigned to lease $LEASE (D8-96 section 4: 61+ reserved)";;
esac
if [ -n "$WHY" ]; then
  log "L6 lease $LEASE: nothing started ($WHY)"
  echo "{\"lease\": $LEASE, \"job\": \"${NAME:-none}\", \"skipped\": true, \"why\": \"$WHY\", \"utc\": \"$(date -u +%FT%TZ)\"}" > $RUNS/lease_${LEASE}_skipped.json
  log "chain_day21_L6 lease $LEASE finished (skipped)"
  exit 0
fi
if [ -f $OUT/summary.json ]; then log "L6 lease $LEASE: $NAME summary.json exists -> nothing started"; exit 0; fi
if [ -e $OUT ]; then log "L6 lease $LEASE: $OUT exists without summary.json (partial / INCOMPLETE run; never overwrite) -> nothing started, the coordinator decides"; exit 0; fi
COMMON="--train $DATA/train.jsonl --val $DATA/val.jsonl --fresh $DATA/fresh.jsonl --hf-cache $HF_HUB_CACHE --keep-preds --max-ctx-tokens $CTX_SHARD"
RUNNER=scripts/l6_verifier.py
# ================================================================ the lease (7,200 s inclusive)
START=$(date +%s); DT=$((START+7150)); DL=$((DT-60))
EXTRA=""; [ "$KIND" = ver ] && EXTRA="--train-stop-epoch $((START+TRAIN_STOP_S))"
cat > $RUNS/lease_$LEASE.json <<JSON
{"lease": $LEASE, "lease_start_epoch": $START, "total_deadline_epoch": $DT, "job_deadline_epoch": $DL, "inclusive_seconds": 7200, "batch": "L6", "prereg": "$PREREG",
 "user_gpu_approval_utc": "$APPROVAL", "job": "$NAME", "kind": "$KIND", "out": "$OUT", "estimate_min": $EST, "ctx_shard": $CTX_SHARD, "n_train": $N_TRAIN, "train_stop_s": $TRAIN_STOP_S,
 "flags": "$L6_FLAGS", "args": "$ARGS $EXTRA", "data": "$DATA", "l1_runs": "$L1RUNS", "l2_runs": "$L2RUNS", "l4_runs": "$L4RUNS", "l5_runs": "$L5RUNS",
 "receiver_sha256": {"342101": "$B0_CKPT_SHA_342101", "342202": "$B0_CKPT_SHA_342202"},
 "data_sha256": {"train": "$TRAINJ_SHA", "val": "$VALJ_SHA", "fresh": "$FRESHJ_SHA"},
 "scripts_sha256": {"l6_verifier.py": "$L6_VERIFIER_SHA", "l6_gate.py": "$L6_GATE_SHA", "l5_belief.py": "$L5_BELIEF_SHA", "l2_model.py": "$L2_MODEL_SHA", "l1h_model.py": "$L1H_MODEL_SHA", "l1f_model.py": "$L1F_MODEL_SHA", "l1e_model.py": "$L1E_MODEL_SHA", "l1d_model.py": "$L1D_MODEL_SHA", "l1c_model.py": "$L1C_MODEL_SHA", "l1_model.py": "$MODEL_SHA", "l1_train.py": "$TRAIN_SHA", "l1_fusion.py": "$FUSION_SHA", "l1_data.py": "$DATA_SHA"},
 "chain_sha256": "$(sha scripts/chain_day21_L6.sh)", "hf_hub_cache": "$HF_HUB_CACHE", "offline": true, "renewal": "none", "resume": "none"}
JSON
log "L6 lease $LEASE start $START (job deadline +7090, kill at +7150): $NAME (estimate $EST min)"
STOP=0
now=$(date +%s); rem=$((DL-now))
if [ $((EST*60)) -gt $rem ]; then
  log "L6 lease $LEASE stops before $NAME: estimate $EST min > remaining $((rem/60)) min"; STOP=1
else
  n=0; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $n -lt 20 ]; do log "L6 $NAME: GPU busy before start; waiting"; sleep 30; n=$((n+1)); done
  check_shas
  log "L6 $NAME start: $RUNNER $ARGS $EXTRA $COMMON $L6_FLAGS"
  now=$(date +%s)
  timeout -k 30 $((DT-now)) $P -B $RUNNER $ARGS $EXTRA $COMMON $L6_FLAGS --out $OUT --deadline-epoch $DL > $RUNS/$NAME.log 2>&1
  x=$?
  log "L6 $NAME done (exit $x): $(tail -1 $RUNS/$NAME.log | cut -c1-300)"
  [ $x -eq 124 ] && log "L6 $NAME TIMED OUT (lease end); no summary.json -> the job is missing until re-run by the coordinator"
fi
END=$(date +%s)
echo "{\"lease\": $LEASE, \"job\": \"$NAME\", \"lease_end_epoch\": $END, \"lease_seconds\": $((END-START)), \"stopped_for_time\": $STOP}" > $RUNS/lease_end_$LEASE.json
log "L6 lease $LEASE GPU work over after $((END-START)) s; the GPU is not used again by this chain"
# ---------------------------------------------------------------- after the GPU work (CPU only): the gate of this job
check_shas
mkdir -p $RUNS/gates
CUDA_VISIBLE_DEVICES= timeout -k 30 600 nice -n 10 $P -B scripts/l6_gate.py --job $NAME --runs $RUNS --l5 $L5RUNS --l4 $L4RUNS --out $RUNS/gates > $RUNS/gates/$NAME.log 2>&1
log "L6 gate $NAME (exit $?): $(grep -a '^GATE' $RUNS/gates/$NAME.log | tr '\n' ' ')"
log "L6 disk: $RUNS $(du -sm $RUNS | cut -f1) MB; /mnt/datasets free $(df -Pm /mnt/datasets | awk 'NR==2{print $4}') MB"
log "chain_day21_L6 lease $LEASE finished"
