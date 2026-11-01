#!/bin/bash
# L2 pre-launch on uk-linux (5090), detached: unit tests (CPU) -> smokes (R1 70 steps, V1 70 + 70, Hu-hetero 70, R1-hetero 30) -> E0 diagnostics.
# Writes runs/l2_smoke/* and runs/l2_diag/*; never touches runs/l2 or runs/l1h. Each step is skipped when its output already exists.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HF_HUB_CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/hf-cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
D=hotpot/splits_l1
L=runs/l2_smoke
mkdir -p $L runs/l2_diag $TMPDIR
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $L/prelaunch.log; }
log "prelaunch start pid $$"
if [ ! -f $L/pytest.ok ]; then
  (cd scripts && CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P -B -m pytest test_l2_model.py -q -p no:cacheprovider > ../$L/pytest.log 2>&1); x=$?
  log "pytest exit $x: $(tail -n 1 $L/pytest.log | cut -c1-200)"
  [ $x -eq 0 ] || { log "ABORT: tests failed"; exit 2; }
  date -u > $L/pytest.ok
fi
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { log "ABORT: GPU busy"; exit 3; }
pgrep -af 'bash scripts/chai[n]_' | grep -q . && { log "ABORT: a chain is running"; exit 3; }
COMMON="--train $D/train.jsonl --val $D/val.jsonl --fresh $D/fresh.jsonl --hf-cache $HF_HUB_CACHE --keep-preds --max-ctx-tokens 900 --eval-batch 32 --tau 1.0 --n-blocks 2 --limit-eval 64"
smoke(){                                                                          # $1 = name, $2.. = trainer args
  local name=$1; shift
  [ -e $L/$name ] && { log "smoke $name exists -> skipped"; return 0; }
  log "smoke $name start"
  timeout -k 30 2400 $P -B scripts/l2_train.py "$@" $COMMON --out $L/$name > $L/$name.log 2>&1
  local x=$?
  log "smoke $name exit $x: $(tail -n 1 $L/$name.log | cut -c1-250)"
  grep -a "stage [12] done\|step time\|VAL p=0.0\|slot accuracy\|oracle spans\|peak" $L/$name.log | cut -c1-260 | tee -a $L/prelaunch.log
  return $x
}
smoke R1-s342101-70steps --arm R1 --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --init-from runs/l1h/Hu-same2-s342101 --stage2-steps 70
smoke V1-s342101-70steps --arm V1 --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --sel-layer 18 --stage1-steps 70 --stage2-steps 70
smoke Hu-hetero-s342101-70steps --arm Hu --pair hetero --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --stage1-steps 70
smoke R1-hetero-s342101-30steps --arm R1 --pair hetero --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --init-from $L/Hu-hetero-s342101-70steps --stage2-steps 30
if [ ! -f runs/l2_diag/E0.json ]; then
  log "E0 start"
  timeout -k 30 5400 $P -B scripts/l2_diag.py --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --init-from runs/l1h/Hu-same2-s342101 \
    --train $D/train.jsonl --val $D/val.jsonl --hf-cache $HF_HUB_CACHE --out runs/l2_diag --layers 6,9,12,15,18,21 --n-train 300 --n-val 500 > runs/l2_diag/E0.out 2>&1
  log "E0 exit $?: $(grep -a E0C_BEST_LAYER runs/l2_diag/E0.out)"
  grep -a "E0a\|E0b\|none (B0)\|E0c\|select_quality layer" runs/l2_diag/log.txt | cut -c1-300 | tee -a $L/prelaunch.log
fi
log "prelaunch done"
