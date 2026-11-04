#!/bin/bash
# L7 pre-launch on uk-linux (5090), detached: unit tests (CPU) -> the data check -> smokes of the five jobs at N = 4 (seed 342101, 48 training
# rows, 24 VAL / fresh examples, a few steps) and of RECV / GEN / VER at N = 2: step times, s/example, peak memory, readouts.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HF_HUB_CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/hf-cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
D=hotpot/splits_l1
D7=hotpot/splits_l7
L=runs/l7_smoke
mkdir -p $L $TMPDIR
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $L/prelaunch.log; }
log "L7 prelaunch start pid $$"
(cd scripts && CUDA_VISIBLE_DEVICES= timeout -k 30 2400 $P -B -m pytest test_l7_nagents.py -q -p no:cacheprovider > ../$L/pytest.log 2>&1); x=$?
log "pytest exit $x: $(tail -n 1 $L/pytest.log | cut -c1-200)"
[ $x -eq 0 ] || { log "ABORT: tests failed"; exit 2; }
for f in $D7/train7_N2.jsonl $D7/train7_N4.jsonl $D7/val_N4.jsonl $D7/fresh_N4.jsonl $D7/splits7_meta.json; do [ -f $f ] || { log "ABORT: $f missing"; exit 2; }; done
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { log "ABORT: GPU busy"; exit 3; }
pgrep -af 'bash scripts/chai[n]_' | grep -q . && { log "ABORT: a chain is running"; exit 3; }
COMMON="--hf-cache $HF_HUB_CACHE --keep-preds --max-ctx-tokens 900 --k-ev 32 --budget 64 --sel-layer 18 --eval-batch 16 --gen-batch 8 --score-batch 32 --warmup 4 --log-every 4 --lr-lora 2e-4 --lr-recv 1e-4 --lr-verify 1e-4 --limit-eval 24"
run(){                                                                            # $1 = name, $2.. = args
  local name=$1; shift 1
  [ -e $L/$name ] && { log "smoke $name exists -> skipped"; return 0; }
  log "smoke $name start"
  timeout -k 30 3000 $P -B scripts/l7_nagents.py "$@" $COMMON --out $L/$name > $L/$name.log 2>&1
  local x=$?
  log "smoke $name exit $x: $(tail -n 1 $L/$name.log | cut -c1-300)"
  grep -a "step 12/\|step 6/\|VAL\|fresh\|messages:\|gen:\|train done\|round-1\|adapter\|Traceback\|Error" $L/$name.log | cut -c1-500 | tee -a $L/prelaunch.log
  return $x
}
B0=runs/l1/B0-hetero-s342101
N4="--n-agents 4 --seed 342101 --b0-from $B0 --train $D7/train7_N4.jsonl --val $D7/val_N4.jsonl --fresh $D7/fresh_N4.jsonl"
N2="--n-agents 2 --seed 342101 --b0-from $B0 --train $D7/train7_N2.jsonl --val $D/val.jsonl --fresh $D/fresh.jsonl"
run TSEND-N4-48 --job tsend $N4 --limit-train 48 --steps 12 --batch 4 --stage2-examples 8
run TRECV-N4-48 --job trecv $N4 --limit-train 48 --tsend-from $L/TSEND-N4-48 --steps 12 --stage2-steps 4 --batch 4 --stage2-examples 8
run RECV-N4-48 --job recv $N4 --limit-train 48 --steps 12 --batch 4
run GEN-N4-48 --job gen $N4 --limit-train 48 --recv-from $L/RECV-N4-48
run VER-N4-48 --job ver $N4 --recv-from $L/RECV-N4-48 --cands-from $L/GEN-N4-48/cands_train.json --steps 6 --batch 16
run RECV-N2-48 --job recv $N2 --limit-train 48 --steps 12 --batch 8
run GEN-N2-48 --job gen $N2 --limit-train 48 --recv-from $L/RECV-N2-48
run VER-N2-48 --job ver $N2 --recv-from $L/RECV-N2-48 --cands-from $L/GEN-N2-48/cands_train.json --steps 6 --batch 16
log "L7 prelaunch done"
