#!/bin/bash
# sequential, one process at a time (shared machine near its commit limit); each script retried up to 6 times
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=D:/anaconda3/envs/jax-gpu/python.exe
for s in p1_visibility p4b_occ_given_known p3b_delta_static p3c_protocol_bytes; do
  for i in 1 2 3 4 5 6; do
    timeout 3000 $PY -I $s.py > $s.out 2>&1 && { echo "$s ok (attempt $i)"; break; }
    echo "$s failed attempt $i: $(tail -1 $s.out)"
  done
done
