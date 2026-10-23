"""Exact per-message bytes of the max-fusion protocol with ENTROPY-CODED k-bit codes (Claude, 2026-10-23; review code_review_v368_rate.md M2).

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
python eval_bytes_v380.py --source <sealed> --checkpoint <final.pt> --out <json> [--device cuda]
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
    ap.add_argument('--device', default='cpu'); ap.add_argument('--rounds', type=int, default=8)
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

    specs = [sp for sp in base.scene_plan('VALIDATION') if sp['size'] == meta['size']]
    tot = dict(messages=0, silent=0, full_ac_bytes=0, diff_ac_bytes=0, full_fixed_bytes=0, diff_fixed_bytes=0, own_ac_bytes=0, own_fixed_bytes=0,
               own_ideal_bits=0., own_symbols=0, own_messages=0, full_symbols=0, diff_symbols=0, headers_nonsilent=0)
    t0 = time.time()
    for sp in specs:
        fx = data.make_fixture(**sp); inp = fx['inputs']; n = inp['private'].shape[0]
        fsel = bank.select_forest(inp['positions_m'], inp['adjacency'], inp['public_frame']['communication_radius_m'])
        own_t = torch.from_numpy(inp['private']).to(dev)
        with torch.no_grad():
            mu_c, q, mask = model.head(model.trunk(own_t), own_t)
            logp = (q.double() + t.LATENT_PRIOR / (n * 64)).log()
            mi, ci = em.hard_codes(mu_c, logp)
        mi = mi.long().cpu().numpy(); ci = ci.long().cpu().numpy(); ob = mask.bool().cpu().numpy()
        # own source messages
        for i in range(n):
            sym = [(int(mi[i, k]), int(ci[i, k])) for k in range(64) if ob[i, k]]
            bits = code_message(sym, cum_mu, cum_q, levels)
            tot['own_ac_bytes'] += math.ceil((64 + len(bits)) / 8); tot['own_fixed_bytes'] += 8 + 2 * len(sym)
            tot['own_ideal_bits'] += float(sum(ideal_mu[m + levels] + ideal_q[c] for m, c in sym)); tot['own_symbols'] += len(sym); tot['own_messages'] += 1
        # exact index-level flood (lexicographic max of (ci, mi); -1 = prior-only)
        own_key = np.where(ob, ci * 1000 + (mi + levels), -1)            # (ci, mi) lexicographic as one integer
        edges = [(int(a), int(b)) for a, b, _ in fsel['edges']]
        directed = [(a, b) for a, b in edges] + [(b, a) for a, b in edges]
        incoming = {j: [d for d, (a2, b2) in enumerate(directed) if b2 == j] for j in range(n)}
        msg = [np.full(64, -1, np.int64) for _ in directed]; last = {}
        for r in range(1, args.rounds + 1):
            new = []
            for d, (i, j) in enumerate(directed):
                cand = [own_key[i]] + [msg[b] for b in incoming[i] if directed[b][0] != j]
                new.append(np.max(np.stack(cand), 0))
            msg = new
            for d, (i, j) in enumerate(directed):
                m = msg[d]; ev = np.flatnonzero(m >= 0)
                sym = [(int(m[k] % 1000) - levels, int(m[k] // 1000)) for k in ev]
                bits = code_message(sym, cum_mu, cum_q, levels)
                tot['messages'] += 1; tot['full_ac_bytes'] += math.ceil((64 + len(bits)) / 8); tot['full_fixed_bytes'] += 8 + 2 * len(sym); tot['full_symbols'] += len(sym)
                prev = last.get(d)
                changed = ev if prev is None else np.flatnonzero(m != prev)
                if len(changed) == 0:
                    tot['silent'] += 1
                else:
                    symc = [(int(m[k] % 1000) - levels, int(m[k] // 1000)) for k in changed]
                    bc = code_message(symc, cum_mu, cum_q, levels)
                    tot['diff_ac_bytes'] += math.ceil((64 + len(bc)) / 8); tot['diff_fixed_bytes'] += 8 + 2 * len(symc); tot['diff_symbols'] += len(symc); tot['headers_nonsilent'] += 1
                last[d] = m.copy()
        print(f"{sp['rooms']} {sp['agents']} {sp['index']} | msgs {tot['messages']} | {time.time() - t0:.0f}s", flush=True)
    res = dict(script='eval_bytes_v380.py', checkpoint=args.checkpoint, rate_lambda=meta.get('args', {}).get('rate_lambda'), scenes=len(specs), rounds=args.rounds, totals=tot,
               own_ideal_bits_per_symbol=tot['own_ideal_bits'] / max(tot['own_symbols'], 1),
               own_ac_payload_bits_per_symbol=(8 * tot['own_ac_bytes'] - 64 * tot['own_messages']) / max(tot['own_symbols'], 1),
               own_bytes_per_message=dict(entropy_coded=tot['own_ac_bytes'] / tot['own_messages'], fixed_kbit=tot['own_fixed_bytes'] / tot['own_messages']),
               flood_bytes=dict(full_entropy=tot['full_ac_bytes'], full_fixed=tot['full_fixed_bytes'], diff_entropy=tot['diff_ac_bytes'], diff_fixed=tot['diff_fixed_bytes'],
                                header_bytes_if_8B=8 * tot['headers_nonsilent']),
               ratio_entropy_over_fixed=dict(own=tot['own_ac_bytes'] / tot['own_fixed_bytes'], full=tot['full_ac_bytes'] / tot['full_fixed_bytes'], diff=tot['diff_ac_bytes'] / max(tot['diff_fixed_bytes'], 1)))
    json.dump(res, open(args.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'totals'}, indent=1))


if __name__ == '__main__':
    main()
