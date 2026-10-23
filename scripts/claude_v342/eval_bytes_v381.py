"""v381 = v380 + ENTROPY-CODED presence / change masks (Claude, 2026-10-23). The 64-bit mask of every packet is coded in the same
arithmetic stream with a causal 2-D context model on the 8x8 block grid (left, up, up-left, up-right bits -> 16 contexts; separate
tables for evidence masks and DIFF change masks), integer tables fitted on the TRAIN split (v293 TRAIN plan, same flood) and shared
offline like the symbol tables. VAL bytes are then coded and DECODED back (mask and symbols, bit-exact). The v380 raw-mask figures are
reported alongside for the same messages.

v380: Exact per-message bytes of the max-fusion protocol with ENTROPY-CODED k-bit codes (Claude, 2026-10-23; review code_review_v368_rate.md M2).

What it measures (v293 VALIDATION panel, H128, the trainer's 48 scenes; evaluation only):
  * every source code (observed block k of agent i): mean index mi in [-127,127] and precision index ci in [0, nq-2] exactly as the
    trainer's hard codes / k-bit wire (floor log codebook over [1e-8,128] of p = q + prior);
  * the synchronous forest flood of rollout_max (v368: lattice join, own factor as wire-encoded) simulated EXACTLY at the index level:
    the message i->j in round r carries, per coordinate, the lexicographic max of (ci, mi) over i's own code and the round-(r-1)
    messages into i except the one from j (max precision, ties by quantised mean == the v368 lattice join); prior-only coordinates
    are implicit (public prior) and not sent;
  * each message is ARITHMETIC-CODED for real (32-bit Witten-Neal-Cleary coder, integer frequency tables of total 2^16 derived from the
    checkpoint's learned factorised entropy model; tables are model parameters shared offline) and DECODED back; round trip is
    asserted bit-exact for every message.
Billing per directed message: FULL = 64-bit presence mask + coded symbols of all evidence coordinates; DIFF = silent if nothing changed
since the last packet on that directed edge, else 64-bit change mask + coded symbols of the changed coordinates. Bytes = ceil(bits/8);
an 8-byte header per non-silent packet is reported separately. Comparator on the SAME messages: the fixed k-bit 8/8 wire (2 B per
coordinate). Also reported: the ideal model code length (-log2 P, what the trainer's 'model_bits_per_observed_coordinate' measures).
python eval_bytes_v381.py --source <sealed> --checkpoint <final.pt> --out <json> [--device cuda] [--fit-scenes 200]
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

TOTAL = 1 << 16


# ----------------------------------------------------------------------------- arithmetic coder (Witten, Neal & Cleary 1987, 32-bit)
class Encoder:
    def __init__(self):
        self.low, self.high, self.pending, self.bits = 0, (1 << 32) - 1, 0, []

    def _emit(self, b):
        self.bits.append(b); self.bits.extend([1 - b] * self.pending); self.pending = 0

    def encode(self, cum_lo, cum_hi, total=TOTAL):
        r = self.high - self.low + 1
        self.high = self.low + r * cum_hi // total - 1
        self.low = self.low + r * cum_lo // total
        while True:
            if self.high < (1 << 31):
                self._emit(0)
            elif self.low >= (1 << 31):
                self._emit(1); self.low -= 1 << 31; self.high -= 1 << 31
            elif self.low >= (1 << 30) and self.high < (3 << 30):
                self.pending += 1; self.low -= 1 << 30; self.high -= 1 << 30
            else:
                break
            self.low <<= 1; self.high = (self.high << 1) | 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < (1 << 30) else 1)
        return self.bits


class Decoder:
    def __init__(self, bits):
        self.bits, self.pos = bits, 0
        self.low, self.high, self.value = 0, (1 << 32) - 1, 0
        for _ in range(32):
            self.value = (self.value << 1) | self._next()

    def _next(self):
        b = self.bits[self.pos] if self.pos < len(self.bits) else 0
        self.pos += 1; return b

    def decode(self, cum, total=TOTAL):
        r = self.high - self.low + 1
        target = ((self.value - self.low + 1) * total - 1) // r
        s = int(np.searchsorted(cum, target, side='right') - 1)
        self.high = self.low + r * int(cum[s + 1]) // total - 1
        self.low = self.low + r * int(cum[s]) // total
        while True:
            if self.high < (1 << 31):
                pass
            elif self.low >= (1 << 31):
                self.low -= 1 << 31; self.high -= 1 << 31; self.value -= 1 << 31
            elif self.low >= (1 << 30) and self.high < (3 << 30):
                self.low -= 1 << 30; self.high -= 1 << 30; self.value -= 1 << 30
            else:
                break
            self.low <<= 1; self.high = (self.high << 1) | 1; self.value = (self.value << 1) | self._next()
        return s


def int_table(p):
    """probabilities -> integer frequencies (each >= 1) summing to TOTAL; cumulative array of len K+1"""
    p = np.asarray(p, np.float64); p = p / p.sum()
    f = np.maximum(1, np.floor(p * (TOTAL - len(p))).astype(np.int64))
    f[np.argmax(f)] += TOTAL - f.sum()
    assert f.min() >= 1 and f.sum() == TOTAL
    return np.concatenate([[0], np.cumsum(f)])


def mask_contexts(bits64):
    """causal 2-D contexts of an 8x8 raster-order mask: (left, up, up-left, up-right) -> 0..15"""
    g = np.zeros((9, 10), np.int64); g[1:, 1:9] = np.asarray(bits64, np.int64).reshape(8, 8)
    return (g[1:, 0:8] | (g[0:8, 1:9] << 1) | (g[0:8, 0:8] << 2) | (g[0:8, 2:10] << 3)).reshape(64)


def mask_tables(counts):
    """counts [16, 2] -> per-context cumulative binary tables (KT-smoothed)"""
    return [int_table(counts[c] + .5) for c in range(16)]


def code_message_masked(maskbits, symbols, cum_mu, cum_q, levels, mtab):
    enc = Encoder(); ctx = mask_contexts(maskbits)
    for k in range(64):
        b = int(maskbits[k]); cm = mtab[ctx[k]]; enc.encode(int(cm[b]), int(cm[b + 1]))
    for mi, ci in symbols:
        a = mi + levels
        enc.encode(int(cum_mu[a]), int(cum_mu[a + 1])); enc.encode(int(cum_q[ci]), int(cum_q[ci + 1]))
    bits = enc.finish()
    dec = Decoder(bits); got = []
    for k in range(64):
        g = np.zeros(64, np.int64); g[:len(got)] = got
        c = mask_contexts(g)[k]; got.append(dec.decode(mtab[c]))
    assert got == [int(b) for b in maskbits], 'mask round trip failed'
    for mi, ci in symbols:
        assert dec.decode(cum_mu) == mi + levels and dec.decode(cum_q) == ci, 'symbol round trip failed'
    return bits


def code_message(symbols, cum_mu, cum_q, levels):
    """symbols: list of (mi, ci). Returns the coded bit list; asserts exact decode."""
    enc = Encoder()
    for mi, ci in symbols:
        a = mi + levels
        enc.encode(int(cum_mu[a]), int(cum_mu[a + 1])); enc.encode(int(cum_q[ci]), int(cum_q[ci + 1]))
    bits = enc.finish()
    dec = Decoder(bits)
    for mi, ci in symbols:
        assert dec.decode(cum_mu) == mi + levels and dec.decode(cum_q) == ci, 'arithmetic round trip failed'
    return bits


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--rounds', type=int, default=8); ap.add_argument('--fit-scenes', type=int, default=200)
    args = ap.parse_args()
    src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid')); sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib.util
    spec_ = importlib.util.spec_from_file_location('t368', Path(__file__).resolve().parent / 'train_v368.py'); t = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(t)
    from scripts import mapping_multiroom_data_v293 as data
    from scripts import mapping_native_bank_v306 as bank
    from scripts import train_mapping_support_native_v306 as base
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False); meta = payload['meta']
    assert meta.get('wire') == 'kbit' and payload.get('entropy_model'), 'k-bit checkpoint with an entropy model required'
    t.MU_BITS, t.Q_BITS, t.WIRE = meta['mu_bits'], meta['q_bits'], 'kbit'
    t.READOUT_MODE = meta.get('readout_mode', 'posterior_only'); t.PRECISION_MODE = meta.get('precision_mode', 'learned')
    dev = torch.device(args.device)
    model = t.SheafModel(meta['arm'], meta['size'], meta['seed']); model.load_state_dict(payload['state_dict']); model.to(dev).eval()
    em = t.EntropyModel(); em.load_state_dict(payload['entropy_model']); em.to(dev).eval()
    levels, nq, lo, hi, s = em.geometry()
    with torch.no_grad():
        xm = torch.arange(-levels, levels + 1, device=dev, dtype=torch.float64) / levels
        xq = lo + (torch.arange(0, nq - 1, device=dev, dtype=torch.float64) + .5) * s
        p_mu = (2. ** -em.mu.bits(xm, 1. / levels, -1., 1.)).cpu().numpy(); p_q = (2. ** -em.lp.bits(xq, s, lo, hi)).cpu().numpy()
    assert abs(p_mu.sum() - 1) < 1e-6 and abs(p_q.sum() - 1) < 1e-6, (p_mu.sum(), p_q.sum())
    cum_mu, cum_q = int_table(p_mu), int_table(p_q)
    ideal_mu, ideal_q = -np.log2(p_mu), -np.log2(p_q)

    def scene_messages(sp):
        """yields (kind, maskbits, symbols) for one scene: kind in own / full / diff / silent"""
        fx = data.make_fixture(**sp); inp = fx['inputs']; n = inp['private'].shape[0]
        fsel = bank.select_forest(inp['positions_m'], inp['adjacency'], inp['public_frame']['communication_radius_m'])
        own_t = torch.from_numpy(inp['private']).to(dev)
        with torch.no_grad():
            mu_c, q, mask = model.head(model.trunk(own_t), own_t)
            logp = (q.double() + t.LATENT_PRIOR / (n * 64)).log()
            mi, ci = em.hard_codes(mu_c, logp)
        mi = mi.long().cpu().numpy(); ci = ci.long().cpu().numpy(); ob = mask.bool().cpu().numpy()
        out = []
        for i in range(n):
            out.append(('own', ob[i].astype(np.int64), [(int(mi[i, k]), int(ci[i, k])) for k in range(64) if ob[i, k]]))
        own_key = np.where(ob, ci * 1000 + (mi + levels), -1)
        edges = [(int(a), int(b)) for a, b, _ in fsel['edges']]
        directed = [(a, b) for a, b in edges] + [(b, a) for a, b in edges]
        incoming = {j: [d for d, (a2, b2) in enumerate(directed) if b2 == j] for j in range(n)}
        msg = [np.full(64, -1, np.int64) for _ in directed]; last = {}
        dec = lambda m, ks: [(int(m[k] % 1000) - levels, int(m[k] // 1000)) for k in ks]
        for r in range(1, args.rounds + 1):
            msg = [np.max(np.stack([own_key[i]] + [msg[b] for b in incoming[i] if directed[b][0] != j]), 0) for d, (i, j) in enumerate(directed)]
            for d in range(len(directed)):
                m = msg[d]; ev = np.flatnonzero(m >= 0)
                out.append(('full', (m >= 0).astype(np.int64), dec(m, ev)))
                prev = last.get(d)
                chg = (m >= 0) if prev is None else (m != prev)
                out.append(('diff', chg.astype(np.int64), dec(m, np.flatnonzero(chg))) if chg.any() else ('silent', None, None))
                last[d] = m.copy()
        return out

    # ---- pass 1: mask context counts on the TRAIN split (tables are model side-information, fitted once offline)
    train_specs = [sp for sp in base.scene_plan('TRAIN') if sp['size'] == meta['size']][:args.fit_scenes]
    counts = {'ev': np.zeros((16, 2)), 'chg': np.zeros((16, 2))}
    t0 = time.time()
    for sp in train_specs:
        for kind, mb, _ in scene_messages(sp):
            if kind == 'silent':
                continue
            c = mask_contexts(mb); key = 'chg' if kind == 'diff' else 'ev'
            np.add.at(counts[key], (c, mb), 1)
    mtab = {k: mask_tables(v) for k, v in counts.items()}
    print(f'mask tables fitted on {len(train_specs)} TRAIN scenes ({time.time() - t0:.0f}s)', flush=True)

    # ---- pass 2: VALIDATION, raw-mask (v380) and coded-mask (v381) bytes for the same messages, all round trips asserted
    specs = [sp for sp in base.scene_plan('VALIDATION') if sp['size'] == meta['size']]
    tot = {k: 0 for k in ('own_raw', 'own_coded', 'own_fixed', 'own_msgs', 'own_symbols', 'full_raw', 'full_coded', 'full_fixed', 'diff_raw', 'diff_coded', 'diff_fixed',
                          'messages', 'silent', 'own_mask_bits_coded')}
    for sp in specs:
        for kind, mb, sym in scene_messages(sp):
            if kind == 'silent':
                tot['silent'] += 1; continue
            raw = math.ceil((64 + len(code_message(sym, cum_mu, cum_q, levels))) / 8)
            coded_bits = code_message_masked(mb, sym, cum_mu, cum_q, levels, mtab['chg' if kind == 'diff' else 'ev'])
            coded = math.ceil(len(coded_bits) / 8); fixed = 8 + 2 * len(sym)
            if kind == 'own':
                tot['own_raw'] += raw; tot['own_coded'] += coded; tot['own_fixed'] += fixed; tot['own_msgs'] += 1; tot['own_symbols'] += len(sym)
                tot['own_mask_bits_coded'] += len(code_message_masked(mb, [], cum_mu, cum_q, levels, mtab['ev']))
            else:
                tot['messages'] += (kind == 'full')
                tot[f'{kind}_raw'] += raw; tot[f'{kind}_coded'] += coded; tot[f'{kind}_fixed'] += fixed
        print(f"{sp['rooms']} {sp['agents']} {sp['index']} | {time.time() - t0:.0f}s", flush=True)
    nonsilent = tot['messages'] - tot['silent']
    res = dict(script='eval_bytes_v381.py', checkpoint=args.checkpoint, rate_lambda=meta.get('args', {}).get('rate_lambda'), fit_scenes=len(train_specs), val_scenes=len(specs),
               mask_context_counts={k: v.tolist() for k, v in counts.items()}, totals=tot,
               own_bytes_per_message=dict(fixed_kbit=tot['own_fixed'] / tot['own_msgs'], entropy_raw_mask=tot['own_raw'] / tot['own_msgs'], entropy_coded_mask=tot['own_coded'] / tot['own_msgs'],
                                          coded_mask_bits_alone=tot['own_mask_bits_coded'] / tot['own_msgs']),
               flood_bytes=dict(full_fixed=tot['full_fixed'], full_entropy_raw_mask=tot['full_raw'], full_entropy_coded_mask=tot['full_coded'],
                                diff_fixed=tot['diff_fixed'], diff_entropy_raw_mask=tot['diff_raw'], diff_entropy_coded_mask=tot['diff_coded'], header_bytes_if_8B=8 * nonsilent),
               ratio_vs_fixed=dict(own=tot['own_coded'] / tot['own_fixed'], full=tot['full_coded'] / tot['full_fixed'], diff=tot['diff_coded'] / max(tot['diff_fixed'], 1)))
    json.dump(res, open(args.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('totals', 'mask_context_counts')}, indent=1))


if __name__ == '__main__':
    main()
