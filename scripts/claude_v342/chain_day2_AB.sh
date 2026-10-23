#!/bin/bash
# Day-2 queue, phases A and B (Claude, written 2026-10-22 ~14:35 UTC, BEFORE any of these runs started and before the v348 k-bit
# results were known). Runs after the v348 k-bit batch (all four final.json present, or its total deadline + 180 s).
#
# Phase A (one 7200 s inclusive lease, 3 trainers in parallel + light eval-only jobs):
#   v352  anchored-readout-under-max pair: v347 max fusion, --readout anchored (decoder reads own (q,h) AND posterior (q,h) planes;
#         both are bottleneck quantities the README allows), seeds 342101/342202, 36 epochs, matched to runs/v347-fusion-pair/max-s*.
#         Rationale: under max selection the node's own partial view of a block is DISCARDED whenever a peer's factor is more
#         precise there; the own planes give it back at zero bytes. Under sum fusion (N2-1) anchored vs posterior-only made no
#         difference, which is why posterior-only was adopted; the question is re-opened only because the fusion rule changed.
#         GATE (pre-registered): adopt anchored iff on BOTH seeds own@R8 F1 >= max-posterior-only + 1.5 pp AND R8 F1 >= +1.0 pp
#         AND unknown FPR <= +0.3 pp. Otherwise posterior-only stays.
#   v350  dynamic fine-tune (item 3): train_v350_dyn_edge.py --dynamic (T=6 exploration slices of the TRAIN worlds), warm start from
#         runs/v347-fusion-pair/max-s342202/checkpoints/final.pt, 6 dynamic epochs (= 21,312 updates, the same count as 36 static
#         epochs), lr 1e-4. Launched only if runs/v350-dynamic-ft/REVIEW_OK exists (independent code review verdict).
#         GATE (pre-registered, eval_dynamic_v349 on VALIDATION episodes idx 0-7): own K0 F1 at t=5 >= zero-shot + 3 pp AND crdt F1
#         at t=5 >= zero-shot + 3 pp, with static VALIDATION K0 and R8 F1 not worse than -1 pp vs max-s342202.
#   v353  edge-selection restriction maps (item 2), zero-shot on the v347 max weights: --epochs 0 --edge-select R, R in 15/12/10/8/6 m,
#         both seeds, sequential (light). Reported as accuracy vs fraction of coordinates sent (edge_select_bytes_val128.json).
# Phase B (one 7200 s inclusive lease): endpoint-long pair, 72 epochs (2x updates), seeds 342101/342202, max fusion, with
#   wire   = kbit-8 (train_v348_kbit.py --wire kbit --mu-bits 8 --q-bits 8) iff runs/v348-kbit/b8 R8 F1 >= max-s342101 R8 - 0.5 pp AND
#            K0 >= -0.5 pp (same seed; the k-bit wire is the simpler exact-union wire), else the sealed profile-4 wire (train_v347_maxsel.py);
#   readout = anchored iff the v352 gate above passed, else posterior_only.
#   Purpose: the 36-epoch curves were still rising at the end of the cosine schedule (ep30->36: +0.7/+1.5 pp R8); this is a budget
#   change, not a design change, and is reported as such (VAL + fresh panel, no selection on the fresh panel).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
LOG=runs/chain_day2.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day2_AB started pid $$ sha256 $(sha256sum scripts/chain_day2_AB.sh | cut -c1-16)"

# ---------------------------------------------------------------- wait for the k-bit batch
while true; do
  ok=1; for b in 8 6 5 4; do [ -f runs/v348-kbit/b$b/final.json ] || ok=0; done
  [ $ok = 1 ] && break
  if [ -f runs/v348-kbit/lease.json ]; then
    DTK=$(python3 -c "import json;print(int(json.load(open('runs/v348-kbit/lease.json'))['total_deadline_epoch']))")
    if [ $(date +%s) -gt $((DTK+180)) ]; then log "k-bit total deadline passed; continuing without all four final.json"; break; fi
  fi
  sleep 60
done
log "k-bit batch finished; phase A start"

# ---------------------------------------------------------------- Phase A
mkdir -p runs/v352-anchored-max runs/v350-dynamic-ft runs/v353-edge-select
START=$(date +%s); DE=$((START+5400)); DT=$((START+7000))
cat > runs/v352-anchored-max/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "epochs": 36, "size": 128,
 "seeds": [342101, 342202], "design": "v347 max fusion + anchored readout (own planes + posterior planes), matched to runs/v347-fusion-pair/max-s*",
 "gate": "adopt anchored iff on BOTH seeds own@R8 F1 >= max-posterior-only + 1.5pp AND R8 F1 >= +1.0pp AND unknown FPR <= +0.3pp",
 "shared_lease_with": ["runs/v350-dynamic-ft", "runs/v353-edge-select"], "trainer_sha256": "$(sha256sum scripts/train_v347_maxsel.py | cut -d' ' -f1)"}
JSON
for seed in 342101 342202; do
  nohup $P scripts/train_v347_maxsel.py --source $S --out runs/v352-anchored-max/anch-s$seed --arm owned --size 128 --device cuda --epochs 36 --val-every 6 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout anchored --fusion max > runs/v352-anchored-max/anch-s$seed.log 2>&1 &
  log "anch-s$seed pid $!"
done
if [ -f runs/v350-dynamic-ft/REVIEW_OK ]; then
  cat > runs/v350-dynamic-ft/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "dynamic_epochs": 6, "updates_planned": 21312, "size": 128,
 "warm_start": "runs/v347-fusion-pair/max-s342202/checkpoints/final.pt", "lr": 1e-4, "fusion": "max", "readout": "posterior_only",
 "data": "dyn_data_v349 T=6 exploration slices of the TRAIN worlds (same v293 geometry/sensor/radio); VALIDATION episodes idx 0-7 for the gate; TEST never generated",
 "gate": "eval_dynamic_v349: own K0 F1 at t=5 >= zero-shot + 3pp AND crdt F1 at t=5 >= zero-shot + 3pp; static VALIDATION K0 and R8 F1 >= max-s342202 - 1pp",
 "review": "$(cat runs/v350-dynamic-ft/REVIEW_OK)", "trainer_sha256": "$(sha256sum scripts/train_v350_dyn_edge.py | cut -d' ' -f1)", "data_sha256": "$(sha256sum scripts/dyn_data_v349.py | cut -d' ' -f1)"}
JSON
  nohup $P scripts/train_v350_dyn_edge.py --source $S --out runs/v350-dynamic-ft/dynft-s342202 --arm owned --size 128 --device cuda --dynamic --fusion max --readout posterior_only --init-from runs/v347-fusion-pair/max-s342202/checkpoints/final.pt --epochs 6 --lr 1e-4 --val-every 2 --seed 342202 --deadline-epoch $DE --deadline-total $DT --refine-steps 0 > runs/v350-dynamic-ft/dynft-s342202.log 2>&1 &
  log "dynft-s342202 pid $!"
  cat > runs/v353-edge-select/lease.json <<JSON
{"lease_start_epoch": $START, "total_deadline_epoch": $DT, "design": "zero-shot evaluation of the v347 max weights with the edge-stalk selection restriction map (blocks within R m of the edge midpoint), R in 15/12/10/8/6 m, both seeds; --epochs 0",
 "report": "wall F1 / acc / FPR vs mean fraction of coordinates sent per directed message (edge_select_bytes_val128.json); the union target is unchanged (dropped information counts as error)"}
JSON
  ( for seed in 342101 342202; do for R in 15 12 10 8 6; do
      [ $(date +%s) -gt $DT ] && break
      $P scripts/train_v350_dyn_edge.py --source $S --out runs/v353-edge-select/zs-s$seed-R$R --arm owned --size 128 --device cuda --fusion max --readout posterior_only --init-from runs/v347-fusion-pair/max-s$seed/checkpoints/final.pt --epochs 0 --edge-select $R --seed $seed --deadline-total $DT --refine-steps 0 > runs/v353-edge-select/zs-s$seed-R$R.log 2>&1
    done; done; log "edge-select zero-shot evals done" ) &
else
  log "runs/v350-dynamic-ft/REVIEW_OK missing: dynamic fine-tune and edge-select evals NOT launched"
fi
sleep 120
while pgrep -f "runs/v352-anchored-max/anch-s" >/dev/null || pgrep -f "runs/v350-dynamic-ft/dynft" >/dev/null; do sleep 60; done
log "phase A trainers finished"
if [ -f runs/v350-dynamic-ft/dynft-s342202/checkpoints/final.pt ]; then
  $P scripts/eval_dynamic_v349.py --source $S --checkpoint runs/v350-dynamic-ft/dynft-s342202/checkpoints/final.pt --trainer scripts/train_v350_dyn_edge.py --out runs/v349-dynamic/dynft-s342202.json --device cuda > runs/v349-dynamic/dynft-s342202.log 2>&1
  log "dynamic eval of dynft-s342202 done"
fi

# ---------------------------------------------------------------- Phase B
CHOICE=$(python3 - <<PY
import json, os
def g(path):
    f = json.load(open(path))["val"]; return f["0"]["all"]["wall_f1"], f["8"]["all"]["wall_f1"], f["8"]["all"]["unknown_known_fpr"], f["8"]["all"]["own_wall_f1"]
wire = "sealed"
try:
    k0b, r8b, _, _ = g("runs/v348-kbit/b8/final.json"); k0m, r8m, _, _ = g("runs/v347-fusion-pair/max-s342101/final.json")
    if r8b >= r8m - 0.005 and k0b >= k0m - 0.005: wire = "kbit"
except Exception as e: pass
readout = "posterior_only"
try:
    ok = True
    for s in ("342101", "342202"):
        k0a, r8a, fa, oa = g(f"runs/v352-anchored-max/anch-s{s}/final.json"); k0m, r8m, fm, om = g(f"runs/v347-fusion-pair/max-s{s}/final.json")
        if not (oa >= om + 0.015 and r8a >= r8m + 0.01 and fa <= fm + 0.003): ok = False
    if ok: readout = "anchored"
except Exception as e: pass
print(wire, readout)
PY
)
WIRE=${CHOICE% *}; READOUT=${CHOICE#* }
log "phase B choice: wire=$WIRE readout=$READOUT"
mkdir -p runs/v354-endpoint-long
START=$(date +%s); DE=$((START+5400)); DT=$((START+7000))
cat > runs/v354-endpoint-long/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "epochs": 72, "size": 128, "seeds": [342101, 342202],
 "fusion": "max", "wire": "$WIRE", "readout": "$READOUT", "selection_rules": "wire=kbit-8 iff b8 R8>=max-s342101 R8-0.5pp and K0>=-0.5pp; readout=anchored iff v352 gate passed on both seeds",
 "purpose": "endpoint with a 2x longer cosine schedule (the 36-epoch curves were still rising); reported on VAL and the fresh panel; not a design change"}
JSON
for seed in 342101 342202; do
  if [ "$WIRE" = kbit ]; then
    nohup $P scripts/train_v348_kbit.py --source $S --out runs/v354-endpoint-long/long-s$seed --arm owned --size 128 --device cuda --epochs 72 --val-every 6 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout $READOUT --fusion max --wire kbit --mu-bits 8 --q-bits 8 > runs/v354-endpoint-long/long-s$seed.log 2>&1 &
  else
    nohup $P scripts/train_v347_maxsel.py --source $S --out runs/v354-endpoint-long/long-s$seed --arm owned --size 128 --device cuda --epochs 72 --val-every 6 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout $READOUT --fusion max > runs/v354-endpoint-long/long-s$seed.log 2>&1 &
  fi
  log "long-s$seed pid $!"
done
log "phase B launched"
