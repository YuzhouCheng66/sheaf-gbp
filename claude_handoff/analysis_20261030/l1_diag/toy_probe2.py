"""Does the trained LLM read the soft tokens at all? Answer loss of a finished run under swapped soft tokens (CPU, no grad)."""
import json, os, sys, torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T)
import l1_model as M, l1_fusion as FU, toy_train as TT
run, n = sys.argv[1], int(sys.argv[2])
ck = torch.load(os.path.join(run, 'trainable.pt'), map_location='cpu', weights_only=False)
cfg = ck['config']; agent_kw = {k: cfg[k] for k in TT.AGENT_KEYS}
rows = M.DATA.read_jsonl(cfg['val'])
tiny = None
if cfg.get('tiny'):
    tiny = M.tiny_models(M.DATA.read_jsonl(cfg['train']) + rows + M.DATA.read_jsonl(cfg['fresh']), cfg['seed'])
pair = M.build_pair(cfg['arm'], cfg['pair'], torch.device('cpu'), model=cfg.get('model'), hf_cache=cfg.get('hf_cache'), tiny=tiny, grad_ckpt=False,
                    max_ctx=cfg['max_ctx_tokens'], **agent_kw)
if cfg.get('full_ft'):
    for a in pair.agents:
        for p in a.lm.parameters(): p.requires_grad_(True)
M.load_trainable_state(pair, ck['state']); pair.eval()
val = [r for r in rows if r['answer_owner'] is not None][:n]
own_of = torch.tensor([r['answer_owner'] for r in val])
print('run', run, 'n', len(val), 'owners', own_of.tolist(), flush=True)
torch.manual_seed(0)
with torch.no_grad():
    z, lam = pair.encode_both(val)
    x = FU.fuse_direct(z.float(), lam.float(), M.EDGES, pair.w)
    for i in range(2):
        ag = pair.agents[i]; pieces = pair.pieces(val, i, 'latent')
        ref = ag.reader(x[i], ag.dtype)
        variants = [('fused', ref), ('own_only_p1', ag.reader(z[i], ag.dtype)), ('zeros', torch.zeros_like(ref)),
                    ('random_LN', ag.reader(torch.randn_like(z[i]), ag.dtype)), ('random_x0.03', (torch.randn_like(ref) * 0.03).to(ag.dtype))]
        for name, soft in variants:
            loss, info = ag.loss(pieces, soft)
            per = info['per_example']
            own = per[own_of == i]; oth = per[own_of != i]
            print(f'agent {i} {ag.name}: {name:14s} loss all {float(per.mean()):.4f}  own-owner {float(own.mean()):.4f} (n={len(own)})  other-owner {float(oth.mean()):.4f} (n={len(oth)})', flush=True)
