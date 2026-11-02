#!/bin/bash
# L3 pre-launch on uk-linux (5090), detached: unit tests (CPU) -> smokes: V2 (70 + 70), S2a (30 steps, from the V2 smoke), S2b (30), T2 (30).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HF_HUB_CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/hf-cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
D=hotpot/splits_l1
L=runs/l3_smoke
mkdir -p $L $TMPDIR
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $L/prelaunch.log; }
log "L3 prelaunch start pid $$"
(cd scripts && CUDA_VISIBLE_DEVICES= timeout -k 30 2400 $P -B -m pytest test_l2_model.py test_l3_model.py -q -p no:cacheprovider > ../$L/pytest.log 2>&1); x=$?
log "pytest exit $x: $(tail -n 1 $L/pytest.log | cut -c1-200)"
[ $x -eq 0 ] || { log "ABORT: tests failed"; exit 2; }
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { log "ABORT: GPU busy"; exit 3; }
pgrep -af 'bash scripts/chai[n]_' | grep -q . && { log "ABORT: a chain is running"; exit 3; }
COMMON="--train $D/train.jsonl --val $D/val.jsonl --fresh $D/fresh.jsonl --hf-cache $HF_HUB_CACHE --keep-preds --max-ctx-tokens 900 --eval-batch 32 --limit-eval 64"
run(){                                                                            # $1 = name, $2 = trainer, $3.. = args
  local name=$1 trainer=$2; shift 2
  [ -e $L/$name ] && { log "smoke $name exists -> skipped"; return 0; }
  log "smoke $name start"
  timeout -k 30 3000 $P -B scripts/$trainer "$@" $COMMON --out $L/$name > $L/$name.log 2>&1
  local x=$?
  log "smoke $name exit $x: $(tail -n 1 $L/$name.log | cut -c1-250)"
  grep -a "stage [0-9] done\|step time\|VAL p=0.0\|VAL round-2\|oracle spans\|selector quality\|round-1 cache\|C assertion\|peak\|Traceback" $L/$name.log | cut -c1-300 | tee -a $L/prelaunch.log
  return $x
}
run V2-s342101-70steps l2_train.py --arm V2 --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --stage1-steps 70 --stage2-steps 70 --tau 1.0 --n-blocks 2
run S2a-s342101-30steps l3_train.py --arm S2a --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --v2-from $L/V2-s342101-70steps --stage3-steps 30
run S2b-s342101-30steps l3_train.py --arm S2b --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --v2-from $L/V2-s342101-70steps --init-from $L/S2a-s342101-30steps --stage4-steps 30
run T2-s342101-30steps l3_train.py --arm T2 --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --v2-from $L/V2-s342101-70steps --init-from $L/S2a-s342101-30steps --stage4-steps 30
log "L3 prelaunch done"
