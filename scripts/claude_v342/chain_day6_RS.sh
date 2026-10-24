#!/bin/bash
# Day-6 batch RS-A (Claude, written 2026-10-24 ~22:30 UTC, before any real-scan training run). Pre-registered in D7-0.
# Question (first principles: hold the system fixed, change only the world): the HouseExpo working baseline (wide decoder + public-frame
# conditioning, runs/v393-frame-cycles/c3-s<seed>, VAL R8 .915 / .913 on HouseExpo) meets REAL scans: floor slices of Replica meshes under
# the v371 fixture contract (mapping_realscan2d_data_v410: same chart rule, sensor, poses, radio, N in {8,16}; only the geometry differs).
# Arms per seed (paired, same starting weights): ZS = the baseline evaluated as is (train_v410 --epochs 0); FT = the same weights trained
# on the Replica TRAIN cache (15 floors of 14 scenes, random yaw per fixture, 4,800 fixtures x 3 epochs = 14,400 updates, peak lr 1e-4,
# cosine, the cycle settings of batches W / X). Secondary: FT evaluated back on HouseExpo VAL (forgetting).
# Disclosure: a CPU smoke run of ZS on 2 VAL fixtures (scene 1 = apartment_2, N8, placements 0 and 8) gave R8 wall F1 .495 / .496 (K0
# .568 / .580): the HouseExpo prior draws straight thin walls; the real free space is bounded by ragged furniture edges. The thresholds
# below were chosen before that smoke run and are not changed by it.
# Pre-registered reading (final VAL, 4 scenes x N{8,16} x 8 placements, R8 wall F1 equal weight per (scene, N), BOTH seeds):
#   G = FT - ZS >= +3 pp on both seeds      => real-scan data is the lever (training on HM3D is the next priority);
#   |G| < 1 pp on both seeds                => the HouseExpo training already transfers at this scale;
#   otherwise                               => partial. Protection: unknown-known FPR may not rise by > 0.3 pp (FT vs ZS).
#   Reported, not judged: ZS and FT vs HouseExpo VAL (.915 / .913, the transfer gap), K0, the fresh panel (placements 8-15 of the same
#   VAL scenes), per-scene rows, and FT's HouseExpo VAL R8 vs .915 / .913 (forgetting).
# Caveats recorded in advance: 15 training floors is tiny (HouseExpo: 14,400 houses), so FT measures how fast real data helps, not its
# ceiling; VALIDATION is new rooms (office_3/4 share a building with office_0-2) more than new buildings; the fresh panel re-uses the VAL
# scenes; Replica has no TEST split (claims at scale come from HM3D).
# Scheduling: solo, seeds strictly sequential, each seed (ZS + FT + forgetting eval) under ONE inclusive 7,200 s lease, every run under a
# hard `timeout` at the lease's total deadline. Waits for batch 3D-B's seed-342202 re-run. Writes only under /mnt/datasets (runs/v410-realscan).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
export REALSCAN_BASE_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/realscan-v410/replica-base
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
C=/mnt/datasets/yuzhou-sheaf-gbp-research/realscan-v410/replica-train-k4800
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day6_RS started pid $$ sha256 $(sha256sum scripts/chain_day6_RS.sh | cut -c1-16)"
TSHA=$(sha256sum scripts/train_v410.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_realscan2d_data_v410.py | cut -d' ' -f1)
$P -c "import json,sys; m=json.load(open('$C/manifest.json')); sys.exit(0 if m['scenes']==4800 and m['world']=='realscan2d' and m['size']==128 and m['data_module_sha256']=='$DSHA' else 1)" || { log "ABORT: real-scan cache does not match the data module / count"; exit 2; }
for s in 342101 342202; do
  [ -f runs/v393-frame-cycles/c3-s$s/checkpoints/final.pt ] || { log "ABORT: baseline c3-s$s missing"; exit 2; }
  $P -c "import json,sys; f=json.load(open('runs/v393-frame-cycles/c3-s$s/final.json')); sys.exit(0 if f.get('updates')==57600 else 1)" || { log "ABORT: baseline c3-s$s incomplete"; exit 2; }
done
ls -d runs/v410-realscan/zs-s* runs/v410-realscan/ft-s* >/dev/null 2>&1 && { log "ABORT: runs/v410-realscan/{zs,ft}-s* already exist (never overwrite history)"; exit 2; }
while pgrep -f "chain_day5_3DB_s2[.]sh" >/dev/null || pgrep -f "^$P scripts/train_3d_v401.py " >/dev/null; do sleep 120; done
pgrep -f "^$P scripts/train_" >/dev/null && { log "ABORT: a trainer process is already running"; exit 2; }
log "batch RS-A: 3D-B is gone; starting (solo, one lease per seed)"
mkdir -p runs/v410-realscan
MODEL="--source $S --arm owned --size 128 --device cuda --readout posterior_only --fusion max --decoder-width 2.0 --frame-cond --refine-steps 0"
for seed in 342101 342202; do
  INIT=runs/v393-frame-cycles/c3-s$seed/checkpoints/final.pt
  START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
  cat > runs/v410-realscan/lease_s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v410 real-scan floor slices (Replica)", "seed": $seed,
 "runs": ["zs-s$seed (evaluation only)", "ft-s$seed (4800 x 3 = 14400 updates)", "ft-s$seed-hx (HouseExpo VAL, evaluation only)"], "init": "$INIT",
 "trainer": "scripts/train_v410.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "cache": "$C",
 "expectation": "G = FT - ZS R8 wall F1 (VAL, equal weight per scene x N): >= +3pp on both seeds => real-scan data is the lever; |G| < 1pp on both => HouseExpo training transfers; else partial; FPR may not rise > 0.3pp; incomplete => INCONCLUSIVE"}
JSON
  log "batch RS-A seed $seed: zero-shot evaluation"
  timeout -k 60 $((DT - $(date +%s))) $P scripts/train_v410.py $MODEL --world realscan2d --epochs 0 --seed $seed --init-from $INIT --deadline-total $DT --out runs/v410-realscan/zs-s$seed > runs/v410-realscan/zs-s$seed.log 2>&1
  log "batch RS-A seed $seed: fine-tune (14400 updates)"
  timeout -k 60 $((DT - $(date +%s))) $P scripts/train_v410.py $MODEL --world realscan2d --train-cache $C --epochs 3 --lr 1e-4 --val-every 1 --expect-updates 14400 --seed $seed --init-from $INIT --deadline-epoch $DE --deadline-total $DT --out runs/v410-realscan/ft-s$seed > runs/v410-realscan/ft-s$seed.log 2>&1
  if [ -f runs/v410-realscan/ft-s$seed/checkpoints/final.pt ]; then
    log "batch RS-A seed $seed: HouseExpo VAL of the fine-tuned weights"
    timeout -k 60 $((DT - $(date +%s))) $P scripts/train_v410.py $MODEL --world houseexpo_fit --epochs 0 --seed $seed --init-from runs/v410-realscan/ft-s$seed/checkpoints/final.pt --deadline-total $DT --out runs/v410-realscan/ft-s$seed-hx > runs/v410-realscan/ft-s$seed-hx.log 2>&1
  fi
  log "batch RS-A seed $seed done"
done
log "batch RS-A done"
RES=$($P - <<'PY'
import json
S = None
def eq(f, panel, hop="8", key="wall_f1"):
    st = f[panel][hop]["strata"]; return sum(v[key] for v in st.values()) / len(st)
rows = {}
for s in ("342101", "342202"):
    try:
        zs = json.load(open(f"runs/v410-realscan/zs-s{s}/final.json")); ft = json.load(open(f"runs/v410-realscan/ft-s{s}/final.json"))
        if ft["updates"] != 14400: rows[s] = f"INCONCLUSIVE (fine-tune incomplete: {ft['updates']} updates)"; continue
        base = json.load(open(f"runs/v393-frame-cycles/c3-s{s}/final.json"))
        try:
            hx = json.load(open(f"runs/v410-realscan/ft-s{s}-hx/final.json")); hx_r8 = round(eq(hx, "val"), 4)
        except Exception:
            hx_r8 = None
        rows[s] = dict(ZS_R8=round(eq(zs, "val"), 4), FT_R8=round(eq(ft, "val"), 4), gain_pp=round(100 * (eq(ft, "val") - eq(zs, "val")), 2),
                       ZS_K0=round(eq(zs, "val", "0"), 4), FT_K0=round(eq(ft, "val", "0"), 4),
                       fpr_delta_pp=round(100 * (eq(ft, "val", key="unknown_known_fpr") - eq(zs, "val", key="unknown_known_fpr")), 3),
                       ZS_fresh_R8=round(eq(zs, "fresh"), 4), FT_fresh_R8=round(eq(ft, "fresh"), 4),
                       houseexpo_R8_baseline=round(eq(base, "val"), 4), houseexpo_R8_after_FT=hx_r8,
                       FT_per_scene_R8={k: round(v["wall_f1"], 4) for k, v in ft["val"]["8"]["strata"].items()},
                       ZS_per_scene_R8={k: round(v["wall_f1"], 4) for k, v in zs["val"]["8"]["strata"].items()})
    except Exception as e:
        rows[s] = f"INCONCLUSIVE ({e})"
ok = all(isinstance(r, dict) for r in rows.values())
if ok:
    g = [r["gain_pp"] for r in rows.values()]
    verdict = "real-scan data is the lever" if min(g) >= 3 else ("HouseExpo training transfers" if max(abs(x) for x in g) < 1 else "partial")
    fpr_ok = all(r["fpr_delta_pp"] <= 0.3 for r in rows.values())
else:
    verdict = "INCONCLUSIVE"; fpr_ok = None
json.dump(dict(rows=rows, verdict=verdict, fpr_ok=fpr_ok), open("runs/v410-realscan/analysis.json", "w"), indent=1)
print(verdict, "| fpr_ok", fpr_ok, "|", json.dumps(rows))
PY
)
log "batch RS-A analysis: $RES"
