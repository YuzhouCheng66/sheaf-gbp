#!/bin/bash
# L6 pre-launch on uk-linux (5090), detached: unit tests (CPU) -> smokes: GEN s1 on train[:48] and VER s1 (24 steps x 2 examples on those rows,
# evaluation on 48 VAL / fresh examples with the L5 B1t records): timing, memory, readouts.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HF_HUB_CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/hf-cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
D=hotpot/splits_l1
L=runs/l6_smoke
mkdir -p $L $TMPDIR
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $L/prelaunch.log; }
log "L6 prelaunch start pid $$"
(cd scripts && CUDA_VISIBLE_DEVICES= timeout -k 30 2400 $P -B -m pytest test_l6_verifier.py -q -p no:cacheprovider > ../$L/pytest.log 2>&1); x=$?
log "pytest exit $x: $(tail -n 1 $L/pytest.log | cut -c1-200)"
[ $x -eq 0 ] || { log "ABORT: tests failed"; exit 2; }
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { log "ABORT: GPU busy"; exit 3; }
pgrep -af 'bash scripts/chai[n]_' | grep -q . && { log "ABORT: a chain is running"; exit 3; }
COMMON="--train $D/train.jsonl --val $D/val.jsonl --fresh $D/fresh.jsonl --hf-cache $HF_HUB_CACHE --keep-preds --max-ctx-tokens 900 --k-beams 4 --gen-batch 8 --score-batch 32"
run(){                                                                            # $1 = name, $2.. = args
  local name=$1; shift 1
  [ -e $L/$name ] && { log "smoke $name exists -> skipped"; return 0; }
  log "smoke $name start"
  timeout -k 30 3000 $P -B scripts/l6_verifier.py "$@" $COMMON --out $L/$name > $L/$name.log 2>&1
  local x=$?
  log "smoke $name exit $x: $(tail -n 1 $L/$name.log | cut -c1-300)"
  grep -a "VAL:\|fresh:\|gen:\|train step\|train done\|evidence\|verify adapters\|Traceback\|Error" $L/$name.log | cut -c1-400 | tee -a $L/prelaunch.log
  return $x
}
run GEN-s342101-48 --mode gen --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --v-from runs/l2/V1-same2-s342101 --limit-train 48
run VER-s342101-48 --mode train --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --v-from runs/l2/V1-same2-s342101 --cands-from $L/GEN-s342101-48/cands_train.json \
    --l5-from runs/l5/B1t-same2-s342101 --batch 2 --steps 24 --lr-lora 1e-4 --warmup 4 --log-every 4 --limit-eval 48
log "L6 prelaunch done"
