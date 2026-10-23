#!/bin/bash
# Day-4 batch U (Claude, written 2026-10-23 ~16:10 UTC after the Day-5 first-principles round, before any batch-U run).
# Question: does the decoder under-use the 8-bit code axis? Three independent analysts (fp_day5_fusion/decoder/representation.md)
# converged on the mechanism "the decoder reads each block's 8-bit scalar as a smooth 1-D quantity, so 255 levels index only a 1-D
# family of patterns; max-join never interpolates, so the code may be read as a symbol". The CPU gate on v384 (gate_v388_codeaxis.py,
# runs/v388-gate/) must have said LAUNCH before this chain is started (its verdict is copied into the lease).
# ONE intervention: train_v388.py --code-embed 8 (a learned 255 x 8 code-level table at the decoder input, identity-initialised so the
# initial decoder FUNCTION equals v378/v386 exactly; encoder, head, wire, fusion, loss, data, schedule unchanged; test_v388_embed.py).
# Protocol = batch Q (from scratch, houseexpo_fit, 14,400-house cache, 4 epochs = 57,600 updates, lr 3e-4, max fusion, posterior_only).
# CONTROL = batch Q itself (runs/v382-long/d4800e4-s<seed>): same seed => identical encoder/head init and data order; identical decoder
# init except the new table and the zero-initialised new conv columns (analyze_v388.py checks encoder/head bit-exactly on initial.pt).
# Pre-registered reading (final VAL R8 wall F1, 6-stratum equal weight, final weights, both seeds 342101 / 342202):
#   EMBED >= CONTROL + 1 pp on BOTH seeds   => the code axis was under-used: symbol reading is a lever (then: combine with batch U's
#                                              winner and the 3-cycle schedule);
#   |EMBED - CONTROL| < 1 pp on BOTH seeds  => the smooth reading is not the limit; the residual is the hard one-scalar-per-block limit;
#   otherwise                               => ambiguous (keep v378 reading).
#   "Structural" claim only if both seeds >= +2 pp AND seed-mean >= +3 pp on VAL and on fresh (adversary rule, fp_day5_adversary.md 2.0).
#   Protection: VAL R8 unknown-known FPR may not rise by > 0.3 pp. Secondary (reported): K0, fresh, 8+-room stratum (mechanism predicts
#   the gain concentrates there), and a post-hoc +-1/+-3-level perturbation test on the trained arm (wire-noise robustness).
# Scheduling: this chain WAITS until no train_v386.py (batch T) process is running (anchored pgrep), then runs the same validated
# preflight -> PARALLEL / SEQUENTIAL fallback as batch T (scripts/preflight_check_v386.py, --width 1.0 --decoder-params 636570
# --code-embed 8). Compute is ~ that of the current model, so PARALLEL is expected. Each run has its own inclusive 7,200 s lease.
# Writes only under /mnt/datasets (runs/v388-embed).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
C=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_U started pid $$ sha256 $(sha256sum scripts/chain_day4_U.sh | cut -c1-16)"
TSHA=$(sha256sum scripts/train_v388.py | cut -d' ' -f1)
[ "$TSHA" = "1e34ce3d60d47ae4a57576508ac6ae603cbf71ca126eb2d5098769cb08f44dcb" ] || { log "ABORT: train_v388.py sha changed ($TSHA)"; exit 2; }
GATE=$(grep -o '"verdict": "[^"]*"' runs/v388-gate/gate-s342202.json | head -1)
case "$GATE" in *LAUNCH*) ;; *) log "ABORT: code-axis gate did not say LAUNCH ($GATE)"; exit 2 ;; esac
log "batch U gate: $GATE"
# wait for batch T (train_v386.py) to finish; never co-run with it
while pgrep -f "^$P scripts/train_v386.py " >/dev/null; do sleep 120; done
for seed in 342101 342202; do
  for f in final.json run.json checkpoints/initial.pt validation/final-val.json; do [ -f runs/v382-long/d4800e4-s$seed/$f ] || { log "ABORT: control runs/v382-long/d4800e4-s$seed/$f missing"; exit 2; }; done
done
ls -d runs/v388-embed/embed-s* >/dev/null 2>&1 && { log "ABORT: runs/v388-embed/embed-s* already exists (never overwrite history)"; exit 2; }
pgrep -f "^$P scripts/train_v3" >/dev/null && { log "ABORT: a trainer process is already running"; exit 2; }
mkdir -p runs/v388-embed
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
COMMON="--source $S --world houseexpo_fit --train-cache $C/houseexpo_fit-k4800 --arm owned --size 128 --device cuda --epochs 4 --lr 3e-4 --val-every 1 --refine-steps 0 --readout posterior_only --fusion max --expect-updates 57600 --decoder-width 1.0 --code-embed 8"
# ---- preflight (review M1/M2): 200 updates per seed, both IN PARALLEL (same contention as a parallel launch); validity + projection by
# scripts/preflight_check_v386.py (rate = float epoch_seconds/200 incl. warm-up; need = load + 57,600 x rate + 4 x val; invalid if any
# preflight did not complete 200 finite-loss updates with the W=2 decoder / control-equal encoder, or if the parallel rate is below
# batch Q's two-job W=1 rate 0.094 s/update, i.e. the two jobs did not overlap). Preflight dirs are timestamped so a re-run never collides.
TS=$(date -u +%H%M%S)
log "batch U preflight start (200 updates x 2 seeds in parallel, code embed 8)"
for seed in 342101 342202; do
  $P scripts/train_v388.py $COMMON --out runs/v388-embed/pf-par-$TS-s$seed --seed $seed --max-updates 200 --deadline-epoch $DE --deadline-total $START > runs/v388-embed/pf-par-$TS-s$seed.log 2>&1 &
done
wait
NOW=$(date +%s)
CHK=$($P scripts/preflight_check_v386.py --preflight runs/v388-embed/pf-par-$TS-s342101 runs/v388-embed/pf-par-$TS-s342202 --control-run runs/v382-long/d4800e4-s342101/run.json --avail $((DE-NOW)) --final-window $((DT-DE)) --mode parallel --width 1.0 --decoder-params 636570 --code-embed 8 2>&1 | tail -1)
log "batch U preflight (parallel): $CHK"
case "$CHK" in
  "VALID FIT"*) MODE=PARALLEL ;;
  "VALID NOFIT"*)
    # review M3: the sequential fallback is decided on a SOLO preflight (one job on the GPU), each seed then gets its own 7,200 s lease
    TS2=$(date -u +%H%M%S)
    log "batch U preflight (solo) start: seed 342101 alone, 200 updates"
    $P scripts/train_v388.py $COMMON --out runs/v388-embed/pf-solo-$TS2-s342101 --seed 342101 --max-updates 200 --deadline-epoch $DE --deadline-total $START > runs/v388-embed/pf-solo-$TS2-s342101.log 2>&1
    CHK2=$($P scripts/preflight_check_v386.py --preflight runs/v388-embed/pf-solo-$TS2-s342101 --control-run runs/v382-long/d4800e4-s342101/run.json --avail 6500 --final-window 550 --mode solo --width 1.0 --decoder-params 636570 --code-embed 8 2>&1 | tail -1)
    log "batch U preflight (solo): $CHK2"
    case "$CHK2" in
      "VALID FIT"*) MODE=SEQUENTIAL ;;
      "VALID NOFIT"*) log "ABORT INCONCLUSIVE: even one job at a time cannot finish 57,600 updates within a lease ($CHK2); nothing launched"; exit 3 ;;
      *) log "ABORT: solo preflight invalid ($CHK2); nothing launched"; exit 2 ;;
    esac ;;
  *) log "ABORT: parallel preflight invalid ($CHK); nothing launched"; exit 2 ;;
esac
log "batch U plan: $MODE"
case "$MODE" in
  PARALLEL)
    cat > runs/v388-embed/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "houseexpo_fit", "updates_each": 57600,
 "mode": "parallel", "preflight": "$CHK", "trainer": "scripts/train_v388.py", "trainer_sha256": "$TSHA", "decoder_width": 1.0, "code_embed": 8, "gate": "$GATE",
 "control": "runs/v382-long/d4800e4-s<seed> (batch Q, code_embed 0, same seed/recipe)",
 "expectation": "EMBED >= Q + 1pp final VAL R8 on both seeds => symbol reading is a lever; |gain| < 1pp on both => not a lever; else ambiguous; structural only if both >= +2pp and mean >= +3pp on VAL and fresh; incomplete => INCONCLUSIVE; FPR may not rise > 0.3pp"}
JSON
    log "batch U start (parallel, code embed 8, from scratch, batch-Q recipe)"
    for seed in 342101 342202; do
      nohup $P scripts/train_v388.py $COMMON --out runs/v388-embed/embed-s$seed --seed $seed --deadline-epoch $DE --deadline-total $DT > runs/v388-embed/embed-s$seed.log 2>&1 &
      log "embed seed $seed pid $!"
    done
    sleep 60
    while pgrep -f "^$P scripts/train_v388.py .*runs/v388-embed/embed" >/dev/null; do sleep 60; done
    ;;
  SEQUENTIAL)
    for seed in 342101 342202; do
      LS=$(date +%s); LDE=$((LS+6600)); LDT=$((LS+7150)); LF=runs/v388-embed/lease$([ $seed = 342101 ] && echo "" || echo 2).json
      cat > $LF <<JSON
{"lease_start_epoch": $LS, "train_deadline_epoch": $LDE, "total_deadline_epoch": $LDT, "inclusive_seconds": 7200, "world": "houseexpo_fit", "updates_each": 57600,
 "mode": "sequential", "seed": $seed, "preflight_parallel": "$CHK", "preflight_solo": "$CHK2", "trainer": "scripts/train_v388.py", "trainer_sha256": "$TSHA", "decoder_width": 1.0, "code_embed": 8, "gate": "$GATE",
 "control": "runs/v382-long/d4800e4-s$seed (batch Q, code_embed 0, same seed/recipe)",
 "expectation": "EMBED >= Q + 1pp final VAL R8 on both seeds => symbol reading is a lever; |gain| < 1pp on both => not a lever; else ambiguous; structural only if both >= +2pp and mean >= +3pp on VAL and fresh; incomplete => INCONCLUSIVE; FPR may not rise > 0.3pp"}
JSON
      log "batch U start seed $seed (sequential lease $LF, code embed 8, from scratch, batch-Q recipe)"
      nohup $P scripts/train_v388.py $COMMON --out runs/v388-embed/embed-s$seed --seed $seed --deadline-epoch $LDE --deadline-total $LDT > runs/v388-embed/embed-s$seed.log 2>&1 &
      log "embed seed $seed pid $!"
      sleep 60
      while pgrep -f "^$P scripts/train_v388.py .*runs/v388-embed/embed-s$seed" >/dev/null; do sleep 60; done
      log "embed seed $seed done"
    done
    ;;
esac
log "batch U done"
RES=$($P scripts/analyze_v388.py --runs runs --out runs/v388-embed/analysis.json 2>&1 | tail -1)
log "batch U analysis: $RES"
