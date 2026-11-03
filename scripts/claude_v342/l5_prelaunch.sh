#!/bin/bash
# L5 pre-launch on uk-linux (5090), detached: unit tests (CPU) -> smokes: B1 s1 and B1t s1 on 48 VAL / fresh examples each (timing, memory, readouts).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HF_HUB_CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/hf-cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
D=hotpot/splits_l1
L=runs/l5_smoke
mkdir -p $L $TMPDIR
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $L/prelaunch.log; }
log "L5 prelaunch start pid $$"
(cd scripts && CUDA_VISIBLE_DEVICES= timeout -k 30 2400 $P -B -m pytest test_l5_belief.py -q -p no:cacheprovider > ../$L/pytest.log 2>&1); x=$?
log "pytest exit $x: $(tail -n 1 $L/pytest.log | cut -c1-200)"
[ $x -eq 0 ] || { log "ABORT: tests failed"; exit 2; }
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { log "ABORT: GPU busy"; exit 3; }
pgrep -af 'bash scripts/chai[n]_' | grep -q . && { log "ABORT: a chain is running"; exit 3; }
COMMON="--val $D/val.jsonl --fresh $D/fresh.jsonl --hf-cache $HF_HUB_CACHE --keep-preds --max-ctx-tokens 900 --k-beams 4 --gen-batch 8 --score-batch 32 --limit-eval 48"
run(){                                                                            # $1 = name, $2.. = args
  local name=$1; shift 1
  [ -e $L/$name ] && { log "smoke $name exists -> skipped"; return 0; }
  log "smoke $name start"
  timeout -k 30 3000 $P -B scripts/l5_belief.py "$@" $COMMON --out $L/$name > $L/$name.log 2>&1
  local x=$?
  log "smoke $name exit $x: $(tail -n 1 $L/$name.log | cut -c1-300)"
  grep -a "VAL:\|fresh:\|l5 val: \|peak\|Traceback\|Error" $L/$name.log | cut -c1-400 | tee -a $L/prelaunch.log
  return $x
}
run B1-s342101-48 --arm B1 --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --v-from runs/l2/V3-same2-s342101 --round1-cache runs/l4/S3a-same2-s342101/round1_cache.json
run B1t-s342101-48 --arm B1t --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --v-from runs/l2/V1-same2-s342101
log "L5 prelaunch done"
