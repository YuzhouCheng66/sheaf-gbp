"""Apply the pre-launch review patches A and B (code_review_v350_dynamic_edge.md, 2026-10-22) to train_v350_dyn_edge.py.
A: strict warm start (meta assertions; partial load fatal unless --allow-partial-init); no TRAIN generation for --epochs 0.
B: no float32 DT device cache for dynamic training slices (computed per update; identical values, same function)."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else 'train_v350_dyn_edge.py'
s = open(p).read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)
rep("    ap.add_argument('--init-from', default='', help='warm start: load all same-name same-shape tensors from this checkpoint (e.g. H128 final.pt for an H256 continuation); mismatched/new tensors keep fresh init')\n",
    "    ap.add_argument('--init-from', default='', help='warm start: load all same-name same-shape tensors from this checkpoint (e.g. H128 final.pt for an H256 continuation); mismatched/new tensors keep fresh init')\n"
    "    ap.add_argument('--allow-partial-init', action='store_true', help='permit a warm start that leaves some tensors at fresh init (H128->H256 continuation); otherwise any fresh/skipped tensor is fatal (review M1)')\n")
rep("    train_specs = [s for s in base.scene_plan('TRAIN') if s['size'] == size]\n",
    "    train_specs = [s for s in base.scene_plan('TRAIN') if s['size'] == size] if args.epochs > 0 else []   # --epochs 0: evaluation only, no training cache (review M8)\n")
rep("        assert payload.get('meta', {}).get('arm') == args.arm, 'warm start must come from the same arm'\n",
    "        assert payload.get('meta', {}).get('arm') == args.arm, 'warm start must come from the same arm'\n"
    "        for key, val in (('readout_mode', args.readout), ('precision_mode', args.precision), ('mu_bits', args.mu_bits), ('q_bits', args.q_bits), ('wire', args.wire)):\n"
    "            assert payload.get('meta', {}).get(key, val) == val, f'warm start {key} mismatch: checkpoint {payload.get(\"meta\", {}).get(key)} vs CLI {val}'\n"
    "        if not args.allow_partial_init:\n"
    "            assert not missing and not skipped, f'partial warm start (fresh {missing}, skipped {skipped}); pass --allow-partial-init if intended'\n")
rep("        sc['dt'] = {k: truncated_dt(sc['tgt'][k].reshape(n, size, size) == 1, T).reshape(n, -1) for k in HOPS}\n",
    "        if not (args.dynamic and 't' in sc['spec']):                      # dynamic training slices (3552 items, 13 GiB of float32 DT): computed per update instead (review M7)\n"
    "            sc['dt'] = {k: truncated_dt(sc['tgt'][k].reshape(n, size, size) == 1, T).reshape(n, -1) for k in HOPS}\n")
rep("                l, kn, wl = stage_loss(model.decoder, torch.cat((feats[0], feats[k]), -1), sc['tgt'][k], sc['dt'][k])\n",
    "                dt_k = sc['dt'][k] if 'dt' in sc else truncated_dt(sc['tgt'][k].reshape(-1, size, size) == 1, T).reshape(-1, size * size)\n"
    "                l, kn, wl = stage_loss(model.decoder, torch.cat((feats[0], feats[k]), -1), sc['tgt'][k], dt_k)\n")
open(p, 'w').write(s)
print('patched', p)
