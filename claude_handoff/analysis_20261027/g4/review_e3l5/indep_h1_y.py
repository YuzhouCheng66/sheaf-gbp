"""Reviewer's independent recomputation (E3-L5 code review, 2026-10-28) of the H1 Y-zone readouts on TRAIN H1 smoke maps only.
Does NOT import g4_e3l5 or g4_ood_panel / g4_h1_e3l3 scoring. Uses frozen infrastructure only for the model path (encode / GaBP / decode),
the B6 relay sums and G.Target / G.House / G.forest. Zone, flip, F1, phantom: written here from the D8-53 / D8-76 text.
Usage: python -I indep_h1_y.py <root> <npz> [<npz> ...]   (TRAIN files only; asserted)"""
import os, sys, json, hashlib
root = sys.argv[1]
sys.path.insert(0, os.path.join(root, 'scripts', 'claude_v342'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from scipy import ndimage
import torch
torch.set_num_threads(2)
import g4_e3l3 as L3
L = L3.L; R = L3.R; G = L3.G

AN = os.path.join(root, 'claude_handoff', 'analysis_20261027')
FIT = os.path.join(AN, 'g4', 'e1c', 'fit.json')
CRIT = json.load(open(os.path.join(AN, 'g4', 'e7', 'criteria_w.json')))
Y_FAM = {'ellipsoid', 'hanging', 'clutter', 'pole'}
PAIR = {'342101': 'A', '342202': 'B'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dil_face(m):                       # one-cell dilation, 6-neighbourhood (my own)
    return ndimage.binary_dilation(m, structure=ndimage.generate_binary_structure(3, 1))


def counts(p, T, H, Z):                # scope = target-known & furniture region; truth = target occupied in region
    pr = (p == 1) & T.known & H.region
    tr = (T.t == 1) & H.region
    return np.array([(pr & tr & Z).sum(), (pr & ~tr & Z).sum(), (tr & ~pr & Z).sum()], np.int64)


def flips(p, pb, S, T, H, Z):          # arm occupied, B6 free, zero-learning evidence S <= 0, in scope; split true / false inside Z
    sel = (p == 1) & (pb == 0) & (T.t >= 0) & H.region & (S <= 0)
    tr = (T.t == 1) & H.region
    return np.array([(sel & tr & Z).sum(), (sel & ~tr & Z).sum(), (sel & tr).sum(), (sel & ~tr).sum()], np.int64)


def phantoms(p, kc, H, near):          # 26-connected predicted blocks (k > 0, region) with no generator occupancy within 1 cell (Chebyshev)
    m = (p == 1) & (kc > 0) & H.region
    lab, nb = ndimage.label(m, structure=np.ones((3, 3, 3), bool))
    if nb == 0:
        return 0
    hit = set(np.unique(lab[m & near]).tolist())
    return sum(1 for b in range(1, nb + 1) if b not in hit)


def one(path, ts='342101'):
    name = os.path.basename(path)
    assert '_train_' in name, 'TRAIN smoke maps only'
    z = np.load(path)
    occ = z['occ'] == 1
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell'])
    priv = z['private']; n = priv.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    maps = list(z['L_k0_cam_nf'])
    # Y zone: my own union of Y-family object voxels, dilated by one cell
    idx, off = z['objvox_idx'], z['objvox_off']; fam = [str(x) for x in z['obj_family']]
    Ym = np.zeros(occ.size, bool)
    for i, f in enumerate(fam):
        if f in Y_FAM:
            Ym[idx[off[i]:off[i + 1]]] = True
    Yz = dil_face(Ym.reshape(occ.shape))
    near = ndimage.binary_dilation(occ, structure=np.ones((3, 3, 3), bool))
    # B6 (own pairing) and counts
    ev = R.Evidence.from_spec(f'c:{FIT}:{PAIR[ts]}:cam_nf')
    sums, _, _ = R.relay_sums(maps, F, ev.table)
    # E3-L3 frozen checkpoint at w = 4 with G4-E7's w = 4 threshold
    ckp = os.path.join(AN, 'review_E3R', 'inputs', f'e3l3_s{ts}', 'final.pt')
    ck, model = L3.load_model(ckp, 'cpu')
    jo = int(CRIT['thresholds'][ts]['w4']['R8']['j_occ'])
    tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
    assert np.array_equal(tb.numpy().astype(np.int64), np.asarray(ev.table, np.int64))
    with torch.no_grad():
        zp, lp = L.encode_maps(model, maps, tb, 'cpu')
        h, lq, zq = L3.source(zp, lp, 'round')
    arm_c = np.zeros(3, np.int64); arm_f = np.zeros(4, np.int64); arm_ph = 0.0; b6_c = np.zeros(3, np.int64); b6_ph = 0.0; km = 0
    for ci, cm in enumerate(F['comps']):
        S, kc = sums[ci]
        T = G.Target(priv[cm].max(0), H, mult=(priv[cm] >= 0).sum(0))
        pb = R.linear_readout(S, kc, ev.theta)
        b6_c += counts(pb, T, H, Yz) * len(cm); b6_ph += phantoms(pb, kc, H, near) * len(cm) / n
        Fl = L.local_forest(F, cm)
        with torch.no_grad():
            hb, lb, _ = L.gabp(h[cm], lq[cm], Fl, 4.0, 'round')
            lg = L.decode_np(model, hb, lb)
        for i in range(len(cm)):
            km += int(((lb.numpy()[i] > 0) != (kc > 0)).sum())
            p = np.where(kc > 0, (R.logit_bin(lg[i]) >= jo).astype(np.int8), np.int8(-1))
            arm_c += counts(p, T, H, Yz); arm_f += flips(p, pb, S * R.UNIT, T, H, Yz); arm_ph += phantoms(p, kc, H, near) / n
    return dict(file=name, sha12=sha(path)[:12], ts=ts, j_occ=jo, y_zone_voxels=int(Yz.sum()), known_mismatch=km,
                arm_Y_tp_fp_fn=arm_c.tolist(), arm_flips_Ytrue_Yfalse_alltrue_allfalse=arm_f.tolist(), arm_phantoms_per_house=arm_ph,
                b6_Y_tp_fp_fn=b6_c.tolist(), b6_phantoms_per_house=b6_ph)


if __name__ == '__main__':
    out = [one(p, ts) for p in sys.argv[2:] for ts in ('342101', '342202')]
    for o in out:
        print(json.dumps(o))
