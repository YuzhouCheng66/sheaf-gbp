"""A09 probe 3 (Claude, 2026-10-27, CPU, analytic): MACs per step of the current trainer and of candidate next-generation recipes, turned
into RTX 5090 seconds with the measured time model of analysis_20261025/analyst_C_architecture.md (t = 0.085 s fixed + 0.0041 s per GMAC of
ONE decode, 40 decodes per update; fits 3S-A .12 / 3S-W .21 / D2 .20 s; 3S-Z measured .206 s). The decoder count is read off train_3d_v431
(Decoder3D: pre at the folded code grid, lift at 16x16x8, upsampling blocks to the 64x64x16 compute grid, 1x1 output conv).
Also: ctx code of the whole-house GT occupancy (what a prior-first 'teacher' latent would have to carry) on the 6 demo houses.
Output: probes/A09/p3_compute.json"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def conv(vox, cin, cout, k=27):
    return vox * k * cin * cout


def decoder3d_gmac(W=2.0, code_in=8, out=16):
    """Decoder3D (train_3d_v431) at R1, compute grid 64x64x16; code_in = channels after the fold (2 * fold^2 for 1-channel stalks)"""
    c0 = int(round(64 * W)); chans = [int(round(ch * W)) for ch in (48, 36)]
    mac = conv(16 * 16 * 1, code_in, c0) + conv(16 * 16 * 1, c0, c0)                     # pre (at the folded code grid, BZ = 1)
    mac += conv(16 * 16 * 8, c0 + 4, c0)                                                 # lift at the 16x16x8 start grid
    mac += conv(32 * 32 * 16, c0 + 4, chans[0]) + conv(32 * 32 * 16, chans[0], chans[0])
    mac += conv(64 * 64 * 16, chans[0] + 4, chans[1]) + conv(64 * 64 * 16, chans[1], chans[1])
    mac += 64 * 64 * 16 * chans[1] * out
    return mac / 1e9


def trunk_gmac(bx, by, bz):
    """PatchTrunk3D per agent: blocks of (64/bx, 64/by, 16/bz) compute cells, convs 28-32-64-96-128 (strides 1,2,2,2)"""
    s = np.array([64 // bx, 64 // by, 16 // bz]); D = bx * by * bz; mac = 0; cin = 28
    for cout, stride in ((32, 1), (64, 2), (96, 2), (128, 2)):
        if stride == 2:
            s = np.maximum(1, (s + 1) // 2)
        mac += conv(int(np.prod(s)), cin, cout); cin = cout
    return D * mac / 1e9


def t_step(gmac_decode, decodes=40, fixed=0.085, amp=1.0):
    return fixed + 0.0041 * gmac_decode * (decodes / 40) / amp


rows = {}
base = decoder3d_gmac(2.0, 8, 16)
rows['3S-X 32x32x1 W2 (fold 2)'] = dict(gmac_decode=base, trunk_gmac_per_agent=trunk_gmac(32, 32, 1), s_update=t_step(base))
z = decoder3d_gmac(2.0, 32, 16)
rows['3S-Z 64x64x1 W2 (fold 4)'] = dict(gmac_decode=z, trunk_gmac_per_agent=trunk_gmac(64, 64, 1), s_update=t_step(z), measured=0.206)
d1 = decoder3d_gmac(2.0, 16 * 9, 8)                                                      # 64x64 x (C = 8 latents + 1 precision), occupancy head only
rows['D1 64x64xC8, occ head only'] = dict(gmac_decode=d1, s_update=t_step(d1, fixed=0.080))
rows['D1 + hop sampling (K0 + 1 random hop: 16 decodes)'] = dict(s_update=t_step(d1, decodes=16, fixed=0.065))
rows['D1 + hop sampling + bf16 (assume 2x on conv)'] = dict(s_update=t_step(d1, decodes=16, fixed=0.060, amp=2.0))
rows['K0-only screening run (8 decodes)'] = dict(s_update=t_step(z, decodes=8, fixed=0.055))
# D2 stage A: teacher AE on GT worlds, batch 8 maps: 8 trunk passes + 8 decodes, no rollout, 8 x 524k voxel loss
rows['D2 stage A teacher AE, batch 8 GT worlds'] = dict(s_update=0.030 + 0.0041 * d1 * (8 / 40))
# D3: BEV 2D U-Net detector on 128x128 (z folded: 96 ch), widths 64-128-256-128-64, batch 8 own maps
bev = (conv(128 * 128, 96, 64, 9) + conv(128 * 128, 64, 64, 9) + conv(64 * 64, 64, 128, 9) + conv(64 * 64, 128, 128, 9) + conv(32 * 32, 128, 256, 9)
       + conv(32 * 32, 256, 256, 9) + conv(64 * 64, 256 + 128, 128, 9) + conv(64 * 64, 128, 128, 9) + conv(128 * 128, 128 + 64, 64, 9) + conv(128 * 128, 64, 64, 9)) / 1e9
rows['D3 BEV detector (per map)'] = dict(gmac_per_map=bev, s_update_batch8=0.020 + 0.0041 * bev * 8 / 40 * 3 / 3)
for k, v in rows.items():
    v['updates_per_lease_6600s'] = int(6600 / v['s_update']) if 's_update' in v else int(6600 / v['s_update_batch8'])

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
from a09lib import load, ctx_bytes_binary   # noqa: E402
world = []
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path)
    world.append(dict(house=m['house'], ctx_gt_occ=ctx_bytes_binary(gt['occ']), ctx_gt_obj=ctx_bytes_binary(gt['obj']),
                      ctx_gt_struct=ctx_bytes_binary(gt['occ'] & ~gt['obj'] & ~gt['slab']), n_objects=m['n_objects']))
out = dict(steps=rows, world=world, world_median=dict(ctx_gt_occ=float(np.median([w['ctx_gt_occ'] for w in world])),
                                                     ctx_gt_obj=float(np.median([w['ctx_gt_obj'] for w in world])),
                                                     ctx_gt_struct=float(np.median([w['ctx_gt_struct'] for w in world]))))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, 'p3_compute.json'), 'w'), indent=1)
