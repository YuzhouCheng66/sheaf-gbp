#!/bin/bash
# Day-5 batch V (Claude, written 2026-10-24 ~02:00 UTC after batches T/U/W, before any batch-V run). fp_day5_decoder.md candidate 2.
# Question: is the decoder's SCALE / BLOCK-PHASE BLINDNESS a lever? gate_v388 (D5-3) measured that 53 % of the HouseExpo residual is 1-px
# wall jitter; the decoder gets only (x, y, 1/size) planes, so the physical pixel scale (cell = L/128, 2.5x range across houses) and the
# phase of the 8x8 public block grid must be inferred. v392 adds three PUBLIC planes at every decoder stage: block phase u, v and L/20.
# ONE intervention: train_v392.py --frame-cond (on top of the wide decoder, --decoder-width 2.0, which is the current numerically best
# architecture: D5-8). Identity-initialised (new conv columns zero) => the arm starts as exactly the batch-T function; test_v392_frame.py.
# Protocol = batch Q / T (from scratch, houseexpo_fit, 14,400-house cache, 4 epochs = 57,600 updates, lr 3e-4, max fusion, posterior_only).
# CONTROL = batch T itself (runs/v386-decoder/wide-s<seed>, W = 2.0, no conditioning): same seed => identical encoder/head init, identical
# decoder init except the zero columns, identical data order (analyze_v392.py checks encoder/head bit-exactly on initial.pt).
# Pre-registered reading (final VAL R8 wall F1, 6-stratum equal weight, final weights, both seeds 342101 / 342202):
#   FRAME >= T + 1 pp on BOTH seeds   => the public frame (scale + phase) is a lever (then: carry it through the 3-cycle schedule);
#   |FRAME - T| < 1 pp on BOTH seeds  => not a lever: the decoder already infers scale and phase from content;
#   otherwise                         => ambiguous (keep the unconditioned decoder).
#   Mechanism check (reported, not decisive): the gain should be largest on the largest charts (8+-room stratum) where cell size departs
#   most from the 20 m chart; a gain that is flat across strata would point to the block-phase planes rather than scale.
#   Protection: VAL R8 unknown-known FPR may not rise by > 0.3 pp. Secondary: K0, fresh, 1-px-tolerance F1 (post hoc, gate script).
# Budget: wide decoder => solo runs (batch T: 0.1435 s/update shared vs 0.0828 solo). The same validated preflight decides PARALLEL vs
#   SEQUENTIAL (expected SEQUENTIAL: two leases, ~3 h); --width 2.0 --decoder-params 2511202 --frame-cond 1. Incomplete => INCONCLUSIVE.
# Writes only under /mnt/datasets (runs/v392-frame). GPU must be free of our jobs (anchored pgrep guard).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
C=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_V started pid $$ sha256 $(sha256sum scripts/chain_day4_V.sh | cut -c1-16)"
TSHA=$(sha256sum scripts/train_v392.py | cut -d' ' -f1)
[ "$TSHA" = "ef4dc60082ef9f5f7076cf6bec4b7173b987ee730dd5f63f26a8321d40db00d2" ] || { log "ABORT: train_v392.py sha changed ($TSHA)"; exit 2; }
for seed in 342101 342202; do
  for f in final.json run.json checkpoints/initial.pt validation/final-val.json; do [ -f runs/v386-decoder/wide-s$seed/$f ] || { log "ABORT: control (batch T) runs/v386-decoder/wide-s$seed/$f missing"; exit 2; }; done
done
ls -d runs/v392-frame/frame-s* >/dev/null 2>&1 && { log "ABORT: runs/v392-frame/frame-s* already exists (never overwrite history)"; exit 2; }
pgrep -f "^$P scripts/train_v3" >/dev/null && { log "ABORT: a trainer process is already running"; exit 2; }
mkdir -p runs/v392-frame
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
COMMON="--source $S --world houseexpo_fit --train-cache $C/houseexpo_fit-k4800 --arm owned --size 128 --device cuda --epochs 4 --lr 3e-4 --val-every 1 --refine-steps 0 --readout posterior_only --fusion max --expect-updates 57600 --decoder-width 2.0 --frame-cond"
# ---- preflight (review M1/M2): 200 updates per seed, both IN PARALLEL (same contention as a parallel launch); validity + projection by
# scripts/preflight_check_v386.py (rate = float epoch_seconds/200 incl. warm-up; need = load + 57,600 x rate + 4 x val; invalid if any
# preflight did not complete 200 finite-loss updates with the W=2 decoder / control-equal encoder, or if the parallel rate is below
# batch Q's two-job W=1 rate 0.094 s/update, i.e. the two jobs did not overlap). Preflight dirs are timestamped so a re-run never collides.
TS=$(date -u +%H%M%S)
log "batch V preflight start (200 updates x 2 seeds in parallel, decoder width 2.0)"
for seed in 342101 342202; do
  $P scripts/train_v392.py $COMMON --out runs/v392-frame/pf-par-$TS-s$seed --seed $seed --max-updates 200 --deadline-epoch $DE --deadline-total $START > runs/v392-frame/pf-par-$TS-s$seed.log 2>&1 &
done
wait
NOW=$(date +%s)
CHK=$($P scripts/preflight_check_v386.py --preflight runs/v392-frame/pf-par-$TS-s342101 runs/v392-frame/pf-par-$TS-s342202 --control-run runs/v386-decoder/wide-s342101/run.json --avail $((DE-NOW)) --final-window $((DT-DE)) --mode parallel --width 2.0 --decoder-params 2511202 --frame-cond 1 2>&1 | tail -1)
log "batch V preflight (parallel): $CHK"
case "$CHK" in
  "VALID FIT"*) MODE=PARALLEL ;;
  "VALID NOFIT"*)
    # review M3: the sequential fallback is decided on a SOLO preflight (one job on the GPU), each seed then gets its own 7,200 s lease
    TS2=$(date -u +%H%M%S)
    log "batch V preflight (solo) start: seed 342101 alone, 200 updates"
    $P scripts/train_v392.py $COMMON --out runs/v392-frame/pf-solo-$TS2-s342101 --seed 342101 --max-updates 200 --deadline-epoch $DE --deadline-total $START > runs/v392-frame/pf-solo-$TS2-s342101.log 2>&1
    CHK2=$($P scripts/preflight_check_v386.py --preflight runs/v392-frame/pf-solo-$TS2-s342101 --control-run runs/v386-decoder/wide-s342101/run.json --avail 6500 --final-window 550 --mode solo --width 2.0 --decoder-params 2511202 --frame-cond 1 2>&1 | tail -1)
    log "batch V preflight (solo): $CHK2"
    case "$CHK2" in
      "VALID FIT"*) MODE=SEQUENTIAL ;;
      "VALID NOFIT"*) log "ABORT INCONCLUSIVE: even one job at a time cannot finish 57,600 updates within a lease ($CHK2); nothing launched"; exit 3 ;;
      *) log "ABORT: solo preflight invalid ($CHK2); nothing launched"; exit 2 ;;
    esac ;;
  *) log "ABORT: parallel preflight invalid ($CHK); nothing launched"; exit 2 ;;
esac
log "batch V plan: $MODE"
case "$MODE" in
  PARALLEL)
    cat > runs/v392-frame/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "houseexpo_fit", "updates_each": 57600,
 "mode": "parallel", "preflight": "$CHK", "trainer": "scripts/train_v392.py", "trainer_sha256": "$TSHA", "decoder_width": 2.0, "frame_cond": true,
 "control": "runs/v386-decoder/wide-s<seed> (batch T, decoder width 2.0, no frame conditioning, same seed/recipe)",
 "expectation": "FRAME >= T + 1pp final VAL R8 on both seeds => public-frame conditioning is a lever; |gain| < 1pp on both => not a lever; else ambiguous; incomplete => INCONCLUSIVE; FPR may not rise > 0.3pp"}
JSON
    log "batch V start (parallel, wide decoder + frame conditioning, from scratch, batch-Q recipe)"
    for seed in 342101 342202; do
      nohup $P scripts/train_v392.py $COMMON --out runs/v392-frame/frame-s$seed --seed $seed --deadline-epoch $DE --deadline-total $DT > runs/v392-frame/frame-s$seed.log 2>&1 &
      log "frame seed $seed pid $!"
    done
    sleep 60
    while pgrep -f "^$P scripts/train_v392.py .*runs/v392-frame/frame" >/dev/null; do sleep 60; done
    ;;
  SEQUENTIAL)
    for seed in 342101 342202; do
      LS=$(date +%s); LDE=$((LS+6600)); LDT=$((LS+7150)); LF=runs/v392-frame/lease$([ $seed = 342101 ] && echo "" || echo 2).json
      cat > $LF <<JSON
{"lease_start_epoch": $LS, "train_deadline_epoch": $LDE, "total_deadline_epoch": $LDT, "inclusive_seconds": 7200, "world": "houseexpo_fit", "updates_each": 57600,
 "mode": "sequential", "seed": $seed, "preflight_parallel": "$CHK", "preflight_solo": "$CHK2", "trainer": "scripts/train_v392.py", "trainer_sha256": "$TSHA", "decoder_width": 2.0, "frame_cond": true,
 "control": "runs/v386-decoder/wide-s$seed (batch T, decoder width 2.0, no frame conditioning, same seed/recipe)",
 "expectation": "FRAME >= T + 1pp final VAL R8 on both seeds => public-frame conditioning is a lever; |gain| < 1pp on both => not a lever; else ambiguous; incomplete => INCONCLUSIVE; FPR may not rise > 0.3pp"}
JSON
      log "batch V start seed $seed (sequential lease $LF, wide decoder + frame conditioning, from scratch, batch-Q recipe)"
      nohup $P scripts/train_v392.py $COMMON --out runs/v392-frame/frame-s$seed --seed $seed --deadline-epoch $LDE --deadline-total $LDT > runs/v392-frame/frame-s$seed.log 2>&1 &
      log "frame seed $seed pid $!"
      sleep 60
      while pgrep -f "^$P scripts/train_v392.py .*runs/v392-frame/frame-s$seed" >/dev/null; do sleep 60; done
      log "frame seed $seed done"
    done
    ;;
esac
log "batch V done"
RES=$($P scripts/analyze_v392.py --runs runs --out runs/v392-frame/analysis.json 2>&1 | tail -1)
log "batch V analysis: $RES"
