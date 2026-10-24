#!/bin/bash
# D6-2 pre-registered post-hoc items that need no GPU and no lease (Claude, 2026-10-24): (1) re-evaluate the batch 3D-A finals with the v401
# metrics (adds tolerance-1 surface F1); (2) compression accounting of the private maps at 64^3 and 128^3 (2-bit packed raw vs gzip -9),
# same 16 TRAIN scenes. Waits for the 128^3 cache generation to finish. CPU only, nice 15. Writes under runs/v401-3d128/posthoc/.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
mkdir -p runs/v401-3d128/posthoc
while pgrep -f "gen_3d_cache_v401[.]py" >/dev/null; do sleep 30; done
for s in 342101 342202; do
  nice -n 15 $P scripts/eval_3d_v401_posthoc.py --run runs/v400-3d/a64-s$s --out runs/v401-3d128/posthoc/a64-s$s-tol1.json --threads 6 --gen-workers 6 > runs/v401-3d128/posthoc/a64-s$s-tol1.log 2>&1
done
nice -n 15 $P - > runs/v401-3d128/posthoc/compression_64_vs_128.json 2> runs/v401-3d128/posthoc/compression.err <<'PY'
import gzip, json, numpy as np
from pathlib import Path
out = {}
for size, cache in ((64, Path('/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v400/train64-k1600')), (128, Path('/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v401/train128-k1600'))):
    gz, known = [], []
    for i in range(16):
        priv = np.load(cache / f'i{i:06d}.npz')['private']
        for a in range(priv.shape[0]):
            m = priv[a].reshape(-1).astype(np.int16) + 1                       # -1/0/1 -> 0/1/2, two bits per voxel
            packed = (m[0::4] | (m[1::4] << 2) | (m[2::4] << 4) | (m[3::4] << 6)).astype(np.uint8)
            gz.append(len(gzip.compress(packed.tobytes(), 9))); known.append(float((priv[a] >= 0).mean()))
    out[str(size)] = dict(voxels=size ** 3, raw_2bit_bytes=size ** 3 // 4, gzip_bytes_median=int(np.median(gz)), gzip_bytes_min=int(min(gz)), gzip_bytes_max=int(max(gz)),
                          known_fraction_median=round(float(np.median(known)), 4), code_bytes=512, ratio_vs_raw=round(size ** 3 // 4 / 512, 1), ratio_vs_gzip_median=round(float(np.median(gz)) / 512, 2), maps=len(gz))
print(json.dumps(out, indent=1))
PY
echo "$(date -u +%FT%TZ) posthoc_3DB_prep done" >> runs/v401-3d128/posthoc/done.txt
