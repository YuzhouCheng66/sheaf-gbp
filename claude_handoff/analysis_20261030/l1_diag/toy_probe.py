"""lambda / z probe of a finished run (toy or real): rebuild the pair, load trainable.pt, encode a VAL batch, print lam statistics."""
import json, os, sys, torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T)
import l1_model as M
import toy_train as TT
run, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50
ck = torch.load(os.path.join(run, 'trainable.pt'), map_location='cpu', weights_only=False)
cfg = ck['config']
agent_kw = {k: cfg[k] for k in TT.AGENT_KEYS}
rows = {s: M.DATA.read_jsonl(cfg[s]) if hasattr(M.DATA, 'read_jsonl') else [json.loads(l) for l in open(cfg[s])] for s in ('train', 'val', 'fresh')}
if cfg.get('limit_train'): rows['train'] = rows['train'][:cfg['limit_train']]
tiny = M.tiny_models(rows['train'] + rows['val'] + rows['fresh'], cfg['seed']) if cfg.get('tiny') else None
dev = torch.device('cpu')
pair = M.build_pair(cfg['arm'], cfg['pair'], dev, model=cfg.get('model'), hf_cache=cfg.get('hf_cache'), tiny=tiny, grad_ckpt=False,
                    max_ctx=cfg['max_ctx_tokens'], **agent_kw)
if cfg.get('full_ft'):
    for a in pair.agents:
        for p in a.lm.parameters(): p.requires_grad_(True)
missing = M.load_trainable_state(pair, ck['state'])
print('run', run, 'arm', cfg['arm'], 'missing', len(missing), 'w', pair.w_value)
pair.eval()
val = rows['val'][:n]
with torch.no_grad():
    z, lam = pair.encode_both(val)
    for i in range(2):
        l = lam[i].flatten().float(); ll = l.log()
        q = torch.quantile(ll, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99]))
        print(f'agent {i} {pair.agents[i].name}: log lam quantiles 1/10/50/90/99 % = {[round(float(v), 2) for v in q]}; '
              f'frac at low clamp {(ll <= M.LOG_LAM_MIN + 1e-3).float().mean():.3f} at high clamp {(ll >= M.LOG_LAM_MAX - 1e-3).float().mean():.3f}; '
              f'z abs mean {z[i].abs().mean():.3f} max {z[i].abs().max():.3f}')
    import l1_fusion as FU
    x = FU.fuse_direct(z.float(), lam.float(), M.EDGES, pair.w)
    for i in range(2):
        j = 1 - i
        # share of the own z in the fused state, per coordinate: a = (x - z_j) / (z_i - z_j)
        d = (z[i] - z[j]); ok = d.abs() > 1e-3
        a = ((x[i] - z[j]) / d)[ok]
        print(f'agent {i}: own-share a quantiles 10/50/90 % = {[round(float(v), 3) for v in torch.quantile(a, torch.tensor([0.1, 0.5, 0.9]))]} (1 = pure own z, 0 = pure other)')
        s = pair.agents[i].reader(x[i])
        e = pair.agents[i].embed.weight
        print(f'agent {i}: soft-token L2 norm mean {s.norm(dim=-1).mean():.2f}; vocab embedding L2 norm mean {e.norm(dim=-1).mean():.2f} (ratio {float(s.norm(dim=-1).mean() / e.norm(dim=-1).mean()):.1f})')
