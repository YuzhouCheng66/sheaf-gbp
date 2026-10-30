#!/bin/bash
# toy_run.sh ARM SEED STEPS TAG [extra args]  -- CPU only, 3 threads, nice 10; out runs/<ARM>-s<SEED>-<TAG>
ARM=$1; SEED=$2; STEPS=$3; TAG=$4; shift 4
T=/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/toy_l1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
OUT=$T/runs/$ARM-s$SEED-$TAG
cd $T
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false nohup nice -n 10 $P -B toy_train.py --arm $ARM --pair hetero --seed $SEED \
  --train data/train.jsonl --val data/val.jsonl --fresh data/fresh.jsonl --out $OUT --tiny --steps $STEPS --batch 8 --warmup 20 \
  --eval-batch 50 --max-new-tokens 2 --no-grad-ckpt --log-every 50 "$@" > $OUT.stdout 2>&1 &
echo $! > $OUT.pid
echo "started $ARM s$SEED $TAG pid $(cat $OUT.pid)"
