#!/bin/bash
# Day-4 batch T (Claude, written 2026-10-23 ~15:10 UTC after batch S, before any batch-T run). HANDOFF_20261023 section 6 step 2.
# Question: is DECODER CAPACITY a lever for HouseExpo accuracy within the 64-dimensional constraint? Batch S showed the training-length
# lever halving (+2.4 -> +1.0/+1.3 pp per cycle); the decoder (630k of 866k parameters) has never been varied.
# ONE intervention: train_v386.py --decoder-width 2.0 (every decoder channel count x2 => decoder 2,491,330 params, 3.95x; encoder, head,
# code, wire, fusion, loss, data, schedule unchanged; W=1.0 is bit-identical to v378 -- test_v386_width.py passed locally and on uk-linux).
# Protocol = batch Q (from scratch, houseexpo_fit, 14,400-house cache, 4 epochs = 57,600 updates, lr 3e-4, max fusion, posterior_only).
# CONTROL = batch Q itself (runs/v382-long/d4800e4-s<seed>): same seed => identical encoder/head init and identical data order; the only
# difference in initial weights is the decoder (analyze_v386.py checks this bit-exactly on initial.pt).
# Pre-registered reading (final VAL R8 wall F1, 6-stratum equal weight, final weights, both seeds 342101 / 342202):
#   WIDE >= CONTROL + 1 pp on BOTH seeds  => decoder capacity is a lever at this budget (then: train the wide decoder on the 3-cycle schedule);
#   |WIDE - CONTROL| < 1 pp on BOTH seeds => not a lever at this budget (remaining levers lie outside the 64-dim/decoder pair: user decision);
#   otherwise                             => ambiguous (keep the narrow decoder; do not chain further width steps on this evidence).
#   Protection: VAL R8 unknown-known FPR may not rise by > 0.3 pp. Secondary (reported): K0, fresh, 8+-room stratum.
#   Caveat written in advance: equal update budget conflates learning speed and capacity; a wide-decoder win is "capacity helps at
#   57,600 updates", not a ceiling statement.
# Budget / speed (reviews: batch N M1; code_review_v386_decoder M1-M3): a 200-update CUDA preflight for both seeds IN PARALLEL is
#   validated and projected by scripts/preflight_check_v386.py (rate = float epoch_seconds/200 incl. warm-up; need = load + 57,600 x rate
#   + 4 x epoch-end validations; valid only if both preflights completed 200 finite-loss updates with the W=2 decoder and control-equal
#   encoder, and the parallel rate is >= 0.094 s/update = batch Q's two-job W=1 rate, otherwise the two jobs did not overlap).
#   VALID FIT (1.05 x need <= time to train deadline, 2 x val <= final window) => both seeds in parallel under this lease.
#   VALID NOFIT => a SOLO preflight (seed 342101 alone) decides the pre-registered fallback: seeds run SEQUENTIALLY, each under its own
#   inclusive 7,200 s lease (lease.json, lease2.json), if 1.05 x need_solo <= 6,500 s; else the batch aborts INCONCLUSIVE, nothing launched.
#   Any INVALID preflight => ABORT, nothing launched. Incomplete runs (updates != 57,600) => INCONCLUSIVE.
# Writes only under /mnt/datasets (runs/v386-decoder). GPU must be free of our jobs (anchored pgrep guard).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
C=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_T started pid $$ sha256 $(sha256sum scripts/chain_day4_T.sh | cut -c1-16)"
TSHA=$(sha256sum scripts/train_v386.py | cut -d' ' -f1)
[ "$TSHA" = "77e8bc73895bd9765765a9afebc6fe701603e826b99a83c30da3b2b7dd44a068" ] || { log "ABORT: train_v386.py sha changed ($TSHA)"; exit 2; }
for seed in 342101 342202; do
  for f in final.json run.json checkpoints/initial.pt validation/final-val.json; do [ -f runs/v382-long/d4800e4-s$seed/$f ] || { log "ABORT: control runs/v382-long/d4800e4-s$seed/$f missing"; exit 2; }; done
done
ls -d runs/v386-decoder/wide-s* >/dev/null 2>&1 && { log "ABORT: runs/v386-decoder/wide-s* already exists (never overwrite history)"; exit 2; }
pgrep -f "^$P scripts/train_v3" >/dev/null && { log "ABORT: a trainer process is already running"; exit 2; }
mkdir -p runs/v386-decoder
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
COMMON="--source $S --world houseexpo_fit --train-cache $C/houseexpo_fit-k4800 --arm owned --size 128 --device cuda --epochs 4 --lr 3e-4 --val-every 1 --refine-steps 0 --readout posterior_only --fusion max --expect-updates 57600 --decoder-width 2.0"
# ---- preflight (review M1/M2): 200 updates per seed, both IN PARALLEL (same contention as a parallel launch); validity + projection by
# scripts/preflight_check_v386.py (rate = float epoch_seconds/200 incl. warm-up; need = load + 57,600 x rate + 4 x val; invalid if any
# preflight did not complete 200 finite-loss updates with the W=2 decoder / control-equal encoder, or if the parallel rate is below
# batch Q's two-job W=1 rate 0.094 s/update, i.e. the two jobs did not overlap). Preflight dirs are timestamped so a re-run never collides.
TS=$(date -u +%H%M%S)
log "batch T preflight start (200 updates x 2 seeds in parallel, decoder width 2.0)"
for seed in 342101 342202; do
  $P scripts/train_v386.py $COMMON --out runs/v386-decoder/pf-par-$TS-s$seed --seed $seed --max-updates 200 --deadline-epoch $DE --deadline-total $START > runs/v386-decoder/pf-par-$TS-s$seed.log 2>&1 &
done
wait
NOW=$(date +%s)
CHK=$($P scripts/preflight_check_v386.py --preflight runs/v386-decoder/pf-par-$TS-s342101 runs/v386-decoder/pf-par-$TS-s342202 --control-run runs/v382-long/d4800e4-s342101/run.json --avail $((DE-NOW)) --final-window $((DT-DE)) --mode parallel 2>&1 | tail -1)
log "batch T preflight (parallel): $CHK"
case "$CHK" in
  "VALID FIT"*) MODE=PARALLEL ;;
  "VALID NOFIT"*)
    # review M3: the sequential fallback is decided on a SOLO preflight (one job on the GPU), each seed then gets its own 7,200 s lease
    TS2=$(date -u +%H%M%S)
    log "batch T preflight (solo) start: seed 342101 alone, 200 updates"
    $P scripts/train_v386.py $COMMON --out runs/v386-decoder/pf-solo-$TS2-s342101 --seed 342101 --max-updates 200 --deadline-epoch $DE --deadline-total $START > runs/v386-decoder/pf-solo-$TS2-s342101.log 2>&1
    CHK2=$($P scripts/preflight_check_v386.py --preflight runs/v386-decoder/pf-solo-$TS2-s342101 --control-run runs/v382-long/d4800e4-s342101/run.json --avail 6500 --final-window 550 --mode solo 2>&1 | tail -1)
    log "batch T preflight (solo): $CHK2"
    case "$CHK2" in
      "VALID FIT"*) MODE=SEQUENTIAL ;;
      "VALID NOFIT"*) log "ABORT INCONCLUSIVE: even one job at a time cannot finish 57,600 updates within a lease ($CHK2); nothing launched"; exit 3 ;;
      *) log "ABORT: solo preflight invalid ($CHK2); nothing launched"; exit 2 ;;
    esac ;;
  *) log "ABORT: parallel preflight invalid ($CHK); nothing launched"; exit 2 ;;
esac
log "batch T plan: $MODE"
case "$MODE" in
  PARALLEL)
    cat > runs/v386-decoder/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "houseexpo_fit", "updates_each": 57600,
 "mode": "parallel", "preflight": "$CHK", "trainer": "scripts/train_v386.py", "trainer_sha256": "$TSHA", "decoder_width": 2.0,
 "control": "runs/v382-long/d4800e4-s<seed> (batch Q, decoder width 1.0, same seed/recipe)",
 "expectation": "WIDE >= Q + 1pp final VAL R8 on both seeds => decoder capacity is a lever; |gain| < 1pp on both => not a lever at this budget; else ambiguous; incomplete => INCONCLUSIVE; FPR may not rise > 0.3pp"}
JSON
    log "batch T start (parallel, decoder width 2.0, from scratch, batch-Q recipe)"
    for seed in 342101 342202; do
      nohup $P scripts/train_v386.py $COMMON --out runs/v386-decoder/wide-s$seed --seed $seed --deadline-epoch $DE --deadline-total $DT > runs/v386-decoder/wide-s$seed.log 2>&1 &
      log "wide seed $seed pid $!"
    done
    sleep 60
    while pgrep -f "^$P scripts/train_v386.py .*runs/v386-decoder/wide" >/dev/null; do sleep 60; done
    ;;
  SEQUENTIAL)
    for seed in 342101 342202; do
      LS=$(date +%s); LDE=$((LS+6600)); LDT=$((LS+7150)); LF=runs/v386-decoder/lease$([ $seed = 342101 ] && echo "" || echo 2).json
      cat > $LF <<JSON
{"lease_start_epoch": $LS, "train_deadline_epoch": $LDE, "total_deadline_epoch": $LDT, "inclusive_seconds": 7200, "world": "houseexpo_fit", "updates_each": 57600,
 "mode": "sequential", "seed": $seed, "preflight_parallel": "$CHK", "preflight_solo": "$CHK2", "trainer": "scripts/train_v386.py", "trainer_sha256": "$TSHA", "decoder_width": 2.0,
 "control": "runs/v382-long/d4800e4-s$seed (batch Q, decoder width 1.0, same seed/recipe)",
 "expectation": "WIDE >= Q + 1pp final VAL R8 on both seeds => decoder capacity is a lever; |gain| < 1pp on both => not a lever at this budget; else ambiguous; incomplete => INCONCLUSIVE; FPR may not rise > 0.3pp"}
JSON
      log "batch T start seed $seed (sequential lease $LF, decoder width 2.0, from scratch, batch-Q recipe)"
      nohup $P scripts/train_v386.py $COMMON --out runs/v386-decoder/wide-s$seed --seed $seed --deadline-epoch $LDE --deadline-total $LDT > runs/v386-decoder/wide-s$seed.log 2>&1 &
      log "wide seed $seed pid $!"
      sleep 60
      while pgrep -f "^$P scripts/train_v386.py .*runs/v386-decoder/wide-s$seed" >/dev/null; do sleep 60; done
      log "wide seed $seed done"
    done
    ;;
esac
log "batch T done"
RES=$($P scripts/analyze_v386.py --runs runs --out runs/v386-decoder/analysis.json 2>&1 | tail -1)
log "batch T analysis: $RES"
