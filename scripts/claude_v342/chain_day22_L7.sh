#!/bin/bash
# Day-22 ladder L7 "agent 数 N 作为单变量：信念级共识对文字" (Claude). Pre-registration: OVERNIGHT_REPORT D8-97 (the per-job criteria are judged by
# l7_gate.py, which writes the marker files GATE_<job>_<verdict>). Runner: l7_nagents.py (one base model + the LoRA roles shared by the N agents) on
# top of the PINNED l1_model / l1_train / l1_fusion / l1_data files; inputs = the B0 checkpoint of the seed (runs/l1, the 'default' LoRA), the
# splits_l7 data (hotpot/splits_l7: train rows never seen by B0; VAL / fresh = the splits_l1 questions resharded for N = 4) and, per job, the
# earlier jobs of this chain (runs/l7). Structure and safety rules = chain_day21_L6.sh (uk-linux 5090: sha pins, FILL_ check, one lock dir, never
# overwrite, SOLO GPU wait, ONE 7,200 s lease per launch, timeout -k on our own child PIDs only, never pkill / killall, CPU gates after the GPU work).
#
# ONE LAUNCH = ONE LEASE = ONE JOB (--lease N selects the job from the FIXED table of D8-97 section 5; a lease whose condition is not met starts
# nothing and exits 0 so the queue moves on). N = 4 first (the main criterion), then N = 2:
#   63 TSEND-N4-s342101  64 TRECV-N4-s342101 [63]  65 RECV-N4-s342101  66 GEN-N4-s342101 [65]  67 VER-N4-s342101 [66 complete + 65]
#   68 TSEND-N4-s342202  69 TRECV-N4-s342202 [68]  70 RECV-N4-s342202  71 GEN-N4-s342202 [70]  72 VER-N4-s342202 [71 + 70]
#   73 TSEND-N2-s342101  74 TRECV-N2-s342101 [73]  75 RECV-N2-s342101  76 GEN-N2-s342101 [75]  77 VER-N2-s342101 [76 + 75]
#   78 TSEND-N2-s342202  79 TRECV-N2-s342202 [78]  80 RECV-N2-s342202  81 GEN-N2-s342202 [80]  82 VER-N2-s342202 [81 + 80]
#   83+ nothing (reserved for the coordinator)
#   GPU approval: the user's "批准，做" (2026-11-04, D8-97 leases 63-82), passed as $2 into every lease_<N>.json.
# Writes only under /mnt/datasets (runs/l7/..., runs/l7/chain_day22.log). Data, HF cache, runs/l1 are read, never written.
#
# The coordinator fills every FILL_ (after the tests and the smokes; the chain refuses FILL_):
#   L7_NAGENTS_SHA / L7_GATE_SHA / L7_DATA_SHA   B0_CKPT_SHA_342101 / 342202   TRAIN7_N2_SHA / TRAIN7_N4_SHA / VAL4_SHA / FRESH4_SHA
#   MODEL_SHA / TRAIN_SHA / FUSION_SHA / DATA_SHA / VALJ_SHA / FRESHJ_SHA = the L1 pins (already filled)
# Launch: one lease per launch, via scripts/l7_leases.sh, after the previous chain process has exited.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
PREREG="$1"; APPROVAL="$2"
[ "$PREREG" = "D8-97" ] && [ -n "$APPROVAL" ] || { echo "usage: chain_day22_L7.sh D8-97 <user GPU approval UTC> --lease N [--est-tsend 100 --est-trecv 90 --est-recv 90 --est-gen 90 --est-ver 80]"; exit 2; }
shift 2
LEASE=""; EST_TSEND=100; EST_TRECV=90; EST_RECV=90; EST_GEN=90; EST_VER=80
while [ $# -gt 0 ]; do
  case "$1" in
    --lease) LEASE="$2"; shift 2;;
    --est-tsend) EST_TSEND="$2"; shift 2;;
    --est-trecv) EST_TRECV="$2"; shift 2;;
    --est-recv) EST_RECV="$2"; shift 2;;
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
DATA=$WORK/hotpot/splits_l1                                               # VAL / fresh of N = 2 (read only)
DATA7=$WORK/hotpot/splits_l7                                              # train7_N{2,4}, val_N4, fresh_N4 (read only)
RUNS=$WORK/runs/l7
L1RUNS=$WORK/runs/l1                                                      # B0 (read only)
LOG=runs/l7/chain_day22.log
# ---------------------------------------------------------------- pinned by the coordinator (after the tests and the smokes)
L7_NAGENTS_SHA=FILL_L7_NAGENTS_SHA                                         # scripts/l7_nagents.py
L7_GATE_SHA=FILL_L7_GATE_SHA                                               # scripts/l7_gate.py
L7_DATA_SHA=FILL_L7_DATA_SHA                                               # scripts/l7_data.py
B0_CKPT_SHA_342101=FILL_B0_CKPT_SHA_342101                                 # runs/l1/B0-hetero-s342101/trainable.pt
B0_CKPT_SHA_342202=FILL_B0_CKPT_SHA_342202                                 # runs/l1/B0-hetero-s342202/trainable.pt
TRAIN7_N2_SHA=FILL_TRAIN7_N2_SHA                                           # $DATA7/train7_N2.jsonl
TRAIN7_N4_SHA=FILL_TRAIN7_N4_SHA                                           # $DATA7/train7_N4.jsonl
VAL4_SHA=FILL_VAL4_SHA                                                     # $DATA7/val_N4.jsonl
FRESH4_SHA=FILL_FRESH4_SHA                                                 # $DATA7/fresh_N4.jsonl
MODEL_SHA=aa3a4eec71ea38b11b034282cd550c671c6358d037c2a7179c80ca4d438f5b86                                                   # scripts/l1_model.py (L1 pin)
TRAIN_SHA=5f8eee7833fd53851650ff4aa2249269e3a7fbab76b8de92695c403f23b9f33a                                                   # scripts/l1_train.py (L1 pin)
FUSION_SHA=a0f56eb6de811a4247fa784f701c23ef43119626ff1e9e6821846d379fa6a474                                                 # scripts/l1_fusion.py (L1 pin)
DATA_SHA=41d0db811e710579ddcdab76c8c622358984423da54a8e792716fa76c75263dd                                                     # scripts/l1_data.py (L1 pin)
VALJ_SHA=b1e240bbe70873e1f2ac7586171caf6d3620f269c0e02807033a5fded5f68ecf                                                     # $DATA/val.jsonl (L1 pin)
FRESHJ_SHA=985610d8e092745c9edc37b7624ab7e502a9e94da9b0c7a280cd4f68d00acc60                                                 # $DATA/fresh.jsonl (L1 pin)
# ---------------------------------------------------------------- the knobs (D8-97 sections 2 and 5; appendix 1: equal agent-example budgets)
CTX=900
SEL_LAYER=18                                                              # the V1 window selection layer (runs/l2/V1-same2-s*/summary.json span.sel_layer)
K_EV=32
BUDGET=64
STEPS=2500                                                                # tsend / trecv stage 1 / recv: steps; x BATCH x N = 40,000 agent-examples at both N
STAGE2_STEPS=625
BATCH_N2=8;  BATCH_N4=4                                                   # examples per step: equal agent-examples per step (16) at both N
SUB_N2=5000; SUB_N4=2500                                                  # stage-2 subset rows: 10,000 agent-examples at both N
VER_BATCH=16                                                              # verifier: rows per step (2 sequences each); --steps 0 = one pass over the 20,000 rows
STOP_TSEND=3700; STOP_TRECV=4800; STOP_RECV=5100; STOP_VER=3600           # soft stops of the training loops (seconds after the lease start); the rest of the job follows
L7_FLAGS="--k-ev $K_EV --budget $BUDGET --sel-layer $SEL_LAYER --eval-batch 16 --gen-batch 8 --score-batch 32 --warmup 100 --lr-lora 2e-4 --lr-recv 1e-4 --lr-verify 1e-4 --stage2-steps $STAGE2_STEPS"
mkdir -p runs/l7
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day22_L7 started pid $$ host $(hostname) lease $LEASE sha256 $(sha256sum scripts/chain_day22_L7.sh | cut -c1-16) prereg $PREREG approval $APPROVAL est tsend $EST_TSEND trecv $EST_TRECV recv $EST_RECV gen $EST_GEN ver $EST_VER min; ctx $CTX steps $STEPS"
if grep -q -e "FILL_" <<<"$L7_NAGENTS_SHA $L7_GATE_SHA $L7_DATA_SHA $B0_CKPT_SHA_342101 $B0_CKPT_SHA_342202 $TRAIN7_N2_SHA $TRAIN7_N4_SHA $VAL4_SHA $FRESH4_SHA $MODEL_SHA $TRAIN_SHA $FUSION_SHA $DATA_SHA $VALJ_SHA $FRESHJ_SHA"; then log "ABORT: unfilled FILL_ placeholders (coordinator)"; exit 2; fi
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
check_shas(){
  [ "$(sha scripts/l7_nagents.py)" = "$L7_NAGENTS_SHA" ] || { log "ABORT: l7_nagents.py sha changed ($(sha scripts/l7_nagents.py))"; exit 2; }
  [ "$(sha scripts/l7_gate.py)" = "$L7_GATE_SHA" ] || { log "ABORT: l7_gate.py sha changed ($(sha scripts/l7_gate.py))"; exit 2; }
  [ "$(sha scripts/l7_data.py)" = "$L7_DATA_SHA" ] || { log "ABORT: l7_data.py sha changed ($(sha scripts/l7_data.py))"; exit 2; }
  [ "$(sha scripts/l1_model.py)" = "$MODEL_SHA" ] || { log "ABORT: l1_model.py sha changed ($(sha scripts/l1_model.py))"; exit 2; }
  [ "$(sha scripts/l1_train.py)" = "$TRAIN_SHA" ] || { log "ABORT: l1_train.py sha changed ($(sha scripts/l1_train.py))"; exit 2; }
  [ "$(sha scripts/l1_fusion.py)" = "$FUSION_SHA" ] || { log "ABORT: l1_fusion.py sha changed ($(sha scripts/l1_fusion.py))"; exit 2; }
  [ "$(sha scripts/l1_data.py)" = "$DATA_SHA" ] || { log "ABORT: l1_data.py sha changed ($(sha scripts/l1_data.py))"; exit 2; }
  [ "$(sha $DATA/val.jsonl)" = "$VALJ_SHA" ] || { log "ABORT: $DATA/val.jsonl sha changed or missing"; exit 2; }
  [ "$(sha $DATA/fresh.jsonl)" = "$FRESHJ_SHA" ] || { log "ABORT: $DATA/fresh.jsonl sha changed or missing"; exit 2; }
  [ "$(sha $DATA7/train7_N2.jsonl)" = "$TRAIN7_N2_SHA" ] || { log "ABORT: $DATA7/train7_N2.jsonl sha changed or missing"; exit 2; }
  [ "$(sha $DATA7/train7_N4.jsonl)" = "$TRAIN7_N4_SHA" ] || { log "ABORT: $DATA7/train7_N4.jsonl sha changed or missing"; exit 2; }
  [ "$(sha $DATA7/val_N4.jsonl)" = "$VAL4_SHA" ] || { log "ABORT: $DATA7/val_N4.jsonl sha changed or missing"; exit 2; }
  [ "$(sha $DATA7/fresh_N4.jsonl)" = "$FRESH4_SHA" ] || { log "ABORT: $DATA7/fresh_N4.jsonl sha changed or missing"; exit 2; }
  [ "$(sha $L1RUNS/B0-hetero-s342101/trainable.pt)" = "$B0_CKPT_SHA_342101" ] || { log "ABORT: B0-hetero-s342101/trainable.pt sha changed or missing"; exit 2; }
  [ "$(sha $L1RUNS/B0-hetero-s342202/trainable.pt)" = "$B0_CKPT_SHA_342202" ] || { log "ABORT: B0-hetero-s342202/trainable.pt sha changed or missing"; exit 2; }
}
check_shas
for s in 342101 342202; do [ -f $L1RUNS/B0-hetero-s$s/summary.json ] || { log "ABORT: $L1RUNS/B0-hetero-s$s/summary.json missing"; exit 2; }; done
mkdir $RUNS.lock 2>/dev/null || { log "ABORT: a chain_day22_L7 lease is already running (lock $RUNS.lock exists)"; exit 2; }
trap 'rmdir $RUNS.lock 2>/dev/null' EXIT
for d in $RUNS $WORK/runs/l6 $WORK/runs/l5 $WORK/runs/l4 $WORK/runs/l3 $WORK/runs/l2 $L1RUNS $WORK/runs/l1h $WORK/runs/l1b $WORK/runs/l1c $WORK/runs/l1d $WORK/runs/l1e $WORK/runs/l1f $WORK/runs/l1g; do [ -e $d/lease_$LEASE.json ] && { log "ABORT: $d/lease_$LEASE.json exists (lease numbers continue L1 .. L6; use the next number)"; exit 2; }; done
for l in l2 l3 l4 l5 l6; do [ -e $WORK/runs/$l.lock ] && { log "ABORT: runs/$l.lock exists (another chain is running)"; exit 2; }; done
FREE_MB=$(df -Pm /mnt/datasets | awk 'NR==2{print $4}')
if [ "$FREE_MB" -lt 5000 ]; then log "ABORT: only ${FREE_MB} MB free on /mnt/datasets (< 5,000)"; exit 2; fi
[ -d $HF_HUB_CACHE/models--Qwen--Qwen3-1.7B ] || { log "ABORT: Qwen3-1.7B is not in $HF_HUB_CACHE (the chain never downloads)"; exit 2; }
# ---------------------------------------------------------------- before the lease: import self-check (CPU)
SELF=$(CUDA_VISIBLE_DEVICES= timeout -k 30 300 $P -B -c "import sys; sys.path.insert(0, 'scripts'); import l7_nagents, l7_gate, l7_data, l1_model, l1_train, l1_fusion, l1_data, torch, transformers, peft; print('import self-check ok', torch.__version__, transformers.__version__, peft.__version__)" 2>&1 | tail -1)
case "$SELF" in "import self-check ok"*) log "L7 $SELF";; *) log "ABORT: import self-check failed: $SELF"; exit 2;; esac
# 5090 SOLO: no other chain, no project trainer / diag, no compute process on the GPU (wait, never preempt)
busy(){ pgrep -af "bash scripts/chain_" | grep -v "chain_day22_L7" | grep -q . || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "scripts/diag_" >/dev/null || pgrep -f "^$P scripts/g4_" >/dev/null || pgrep -f "scripts/l1_" >/dev/null || pgrep -f "scripts/l1b_" >/dev/null || pgrep -f "scripts/l1c_" >/dev/null || pgrep -f "scripts/l1d_" >/dev/null || pgrep -f "scripts/l1e_" >/dev/null || pgrep -f "scripts/l1f_" >/dev/null || pgrep -f "scripts/l1g_" >/dev/null || pgrep -f "scripts/l1h_" >/dev/null || pgrep -f "scripts/l2_train" >/dev/null || pgrep -f "scripts/l2_diag" >/dev/null || pgrep -f "scripts/l3_train" >/dev/null || pgrep -f "scripts/l5_belief" >/dev/null || pgrep -f "scripts/l6_verifier" >/dev/null || pgrep -f "scripts/l7_nagents" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }   # never "scripts/l7_" (matches the queue -> waits on itself)
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
check_shas
# ---------------------------------------------------------------- the job of this lease (D8-97 section 5)
cond_done(){ [ -f $1/summary.json ]; }
gen_complete(){ [ -f $1/summary.json ] && [ -f $1/cands_train.json ] && grep -q '"complete": true' $1/summary.json; }
NAME=""; OUT=""; EST=0; ARGS=""; WHY=""; KIND=""; NAG=""; SEED=""
case "$LEASE" in
  63) KIND=tsend; NAG=4; SEED=342101;;  64) KIND=trecv; NAG=4; SEED=342101;;  65) KIND=recv; NAG=4; SEED=342101;;  66) KIND=gen; NAG=4; SEED=342101;;  67) KIND=ver; NAG=4; SEED=342101;;
  68) KIND=tsend; NAG=4; SEED=342202;;  69) KIND=trecv; NAG=4; SEED=342202;;  70) KIND=recv; NAG=4; SEED=342202;;  71) KIND=gen; NAG=4; SEED=342202;;  72) KIND=ver; NAG=4; SEED=342202;;
  73) KIND=tsend; NAG=2; SEED=342101;;  74) KIND=trecv; NAG=2; SEED=342101;;  75) KIND=recv; NAG=2; SEED=342101;;  76) KIND=gen; NAG=2; SEED=342101;;  77) KIND=ver; NAG=2; SEED=342101;;
  78) KIND=tsend; NAG=2; SEED=342202;;  79) KIND=trecv; NAG=2; SEED=342202;;  80) KIND=recv; NAG=2; SEED=342202;;  81) KIND=gen; NAG=2; SEED=342202;;  82) KIND=ver; NAG=2; SEED=342202;;
  *) WHY="no job is assigned to lease $LEASE (D8-97 section 5: 83+ reserved)";;
esac
if [ -z "$WHY" ]; then
  UP=$(echo $KIND | tr a-z A-Z); NAME=$UP-N$NAG-s$SEED; OUT=$RUNS/$NAME
  if [ "$NAG" = 2 ]; then BATCH=$BATCH_N2; SUB=$SUB_N2; TRAIN=$DATA7/train7_N2.jsonl; VAL=$DATA/val.jsonl; FRESH=$DATA/fresh.jsonl
  else BATCH=$BATCH_N4; SUB=$SUB_N4; TRAIN=$DATA7/train7_N4.jsonl; VAL=$DATA7/val_N$NAG.jsonl; FRESH=$DATA7/fresh_N$NAG.jsonl; fi
  BASE="--job $KIND --n-agents $NAG --seed $SEED --b0-from $L1RUNS/B0-hetero-s$SEED --train $TRAIN --val $VAL --fresh $FRESH"
  TS=$RUNS/TSEND-N$NAG-s$SEED; RC=$RUNS/RECV-N$NAG-s$SEED; GN=$RUNS/GEN-N$NAG-s$SEED
  case "$KIND" in
    tsend) EST=$EST_TSEND; STOP=$STOP_TSEND; ARGS="$BASE --steps $STEPS --batch $BATCH --stage2-examples $SUB";;
    trecv) EST=$EST_TRECV; STOP=$STOP_TRECV; ARGS="$BASE --tsend-from $TS --steps $STEPS --batch $BATCH --stage2-examples $SUB"
           cond_done $TS && [ -f $TS/messages.json ] && [ -f $TS/trainable.pt ] || WHY="needs $TS (summary + messages.json + trainable.pt)";;
    recv)  EST=$EST_RECV; STOP=$STOP_RECV; ARGS="$BASE --steps $STEPS --batch $BATCH";;
    gen)   EST=$EST_GEN; STOP=0; ARGS="$BASE --recv-from $RC"
           cond_done $RC && [ -f $RC/trainable.pt ] || WHY="needs $RC (summary + trainable.pt)";;
    ver)   EST=$EST_VER; STOP=$STOP_VER; ARGS="$BASE --recv-from $RC --cands-from $GN/cands_train.json --steps 0 --batch $VER_BATCH"
           gen_complete $GN && cond_done $RC && [ -f $RC/trainable.pt ] || WHY="needs $GN (summary complete + cands_train.json) and $RC (summary + trainable.pt)";;
  esac
fi
if [ -n "$WHY" ]; then
  log "L7 lease $LEASE: nothing started ($WHY)"
  echo "{\"lease\": $LEASE, \"job\": \"${NAME:-none}\", \"skipped\": true, \"why\": \"$WHY\", \"utc\": \"$(date -u +%FT%TZ)\"}" > $RUNS/lease_${LEASE}_skipped.json
  log "chain_day22_L7 lease $LEASE finished (skipped)"
  exit 0
fi
if [ -f $OUT/summary.json ]; then log "L7 lease $LEASE: $NAME summary.json exists -> nothing started"; exit 0; fi
if [ -e $OUT ]; then log "L7 lease $LEASE: $OUT exists without summary.json (partial / INCOMPLETE run; never overwrite) -> nothing started, the coordinator decides"; exit 0; fi
COMMON="--hf-cache $HF_HUB_CACHE --keep-preds --max-ctx-tokens $CTX"
RUNNER=scripts/l7_nagents.py
# ================================================================ the lease (7,200 s inclusive)
START=$(date +%s); DT=$((START+7150)); DL=$((DT-60))
EXTRA=""; [ "$STOP" -gt 0 ] && EXTRA="--train-stop-epoch $((START+STOP))"
cat > $RUNS/lease_$LEASE.json <<JSON
{"lease": $LEASE, "lease_start_epoch": $START, "total_deadline_epoch": $DT, "job_deadline_epoch": $DL, "inclusive_seconds": 7200, "batch": "L7", "prereg": "$PREREG",
 "user_gpu_approval_utc": "$APPROVAL", "job": "$NAME", "kind": "$KIND", "n_agents": $NAG, "seed": $SEED, "out": "$OUT", "estimate_min": $EST, "ctx": $CTX, "steps": $STEPS, "batch": $BATCH, "stage2_examples": $SUB,
 "train_stop_s": $STOP, "sel_layer": $SEL_LAYER, "k_ev": $K_EV, "budget": $BUDGET, "flags": "$L7_FLAGS", "args": "$ARGS $EXTRA", "data": "$DATA7", "l1_runs": "$L1RUNS",
 "receiver_sha256": {"342101": "$B0_CKPT_SHA_342101", "342202": "$B0_CKPT_SHA_342202"},
 "data_sha256": {"train7_N2": "$TRAIN7_N2_SHA", "train7_N4": "$TRAIN7_N4_SHA", "val_N4": "$VAL4_SHA", "fresh_N4": "$FRESH4_SHA", "val": "$VALJ_SHA", "fresh": "$FRESHJ_SHA"},
 "scripts_sha256": {"l7_nagents.py": "$L7_NAGENTS_SHA", "l7_gate.py": "$L7_GATE_SHA", "l7_data.py": "$L7_DATA_SHA", "l1_model.py": "$MODEL_SHA", "l1_train.py": "$TRAIN_SHA", "l1_fusion.py": "$FUSION_SHA", "l1_data.py": "$DATA_SHA"},
 "chain_sha256": "$(sha scripts/chain_day22_L7.sh)", "hf_hub_cache": "$HF_HUB_CACHE", "offline": true, "renewal": "none", "resume": "none"}
JSON
log "L7 lease $LEASE start $START (job deadline +7090, kill at +7150): $NAME (estimate $EST min)"
STOPPED=0
now=$(date +%s); rem=$((DL-now))
if [ $((EST*60)) -gt $rem ]; then
  log "L7 lease $LEASE stops before $NAME: estimate $EST min > remaining $((rem/60)) min"; STOPPED=1
else
  n=0; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $n -lt 20 ]; do log "L7 $NAME: GPU busy before start; waiting"; sleep 30; n=$((n+1)); done
  check_shas
  log "L7 $NAME start: $RUNNER $ARGS $EXTRA $COMMON $L7_FLAGS"
  now=$(date +%s)
  timeout -k 30 $((DT-now)) $P -B $RUNNER $ARGS $EXTRA $COMMON $L7_FLAGS --out $OUT --deadline-epoch $DL > $RUNS/$NAME.log 2>&1
  x=$?
  log "L7 $NAME done (exit $x): $(tail -1 $RUNS/$NAME.log | cut -c1-300)"
  [ $x -eq 124 ] && log "L7 $NAME TIMED OUT (lease end); no summary.json -> the job is missing until re-run by the coordinator"
fi
END=$(date +%s)
echo "{\"lease\": $LEASE, \"job\": \"$NAME\", \"lease_end_epoch\": $END, \"lease_seconds\": $((END-START)), \"stopped_for_time\": $STOPPED}" > $RUNS/lease_end_$LEASE.json
log "L7 lease $LEASE GPU work over after $((END-START)) s; the GPU is not used again by this chain"
# ---------------------------------------------------------------- after the GPU work (CPU only): the gate of this job
check_shas
mkdir -p $RUNS/gates
CUDA_VISIBLE_DEVICES= timeout -k 30 600 nice -n 10 $P -B scripts/l7_gate.py --job $NAME --runs $RUNS --out $RUNS/gates > $RUNS/gates/$NAME.log 2>&1
log "L7 gate $NAME (exit $?): $(grep -a '^GATE' $RUNS/gates/$NAME.log | tr '\n' ' ')"
log "L7 disk: $RUNS $(du -sm $RUNS | cut -f1) MB; /mnt/datasets free $(df -Pm /mnt/datasets | awk 'NR==2{print $4}') MB"
log "chain_day22_L7 lease $LEASE finished"
