"""l7_data.py (Claude, 2026-11-04): the data of ladder L7 (pre-registration D8-97): HotpotQA resharded over N agents.
Pure python (json, random, hashlib); no torch.

  * splits_l7: n_train examples drawn (random.Random(seed)) from the usable examples of the official HotpotQA train file whose ids are NOT in
    splits_l1 (train / val / fresh), i.e. examples that B0 and every earlier adapter never saw. VAL / fresh stay the splits_l1 questions.
  * N-sharding of one example (reshard_row): the 10 paragraphs go to N shards whose sizes are 10 // N (+1 for 10 % N of them, which ones is
    random); the two gold paragraphs go to two DISTINCT random shards, the 8 distractors are shuffled and dealt in shard order, the order
    inside every shard is shuffled. sup[i] = the supporting sentences of the gold paragraph in shard i ([] when shard i has none: the
    text sender target is then ' (none)', as in L1), gold_shards = [shard of gold 0, shard of gold 1], answer_owner = the index of the
    shard whose gold paragraph contains the answer string when exactly one does, else None (the splits_l1 semantics).
    N = 2 gives the splits_l1 structure (one gold paragraph per shard); the N = 2 VAL / fresh files of splits_l1 are reused unchanged.
  * Files: DIR/train7_N{N}.jsonl for every N, DIR/val_N{N}.jsonl / fresh_N{N}.jsonl for N != 2, DIR/splits7_meta.json.

  python l7_data.py --train-json hotpot/hotpot_train_v1.1.json --exclude-dir hotpot/splits_l1 --val hotpot/splits_l1/val.jsonl \
      --fresh hotpot/splits_l1/fresh.jsonl --out hotpot/splits_l7 [--seed 342101 --n-train 20000 --n-agents 2,4]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_data as D  # noqa: E402

BATCH = 'L7'
PREREG = 'D8-97'
SEED = 342101
N_TRAIN = 20000
N_LIST = (2, 4)
N_PARAGRAPHS = D.N_PARAGRAPHS


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def shard_sizes(n_par, n_agents, rng):
    """the shard sizes (sum n_par): n_par // n_agents each, +1 for n_par % n_agents random shards"""
    base, extra = divmod(int(n_par), int(n_agents))
    sizes = [base + (1 if j < extra else 0) for j in range(n_agents)]
    rng.shuffle(sizes)
    return sizes


def reshard_row(row, n_agents, rng):
    """a splits_l1-format row (all_paragraphs, gold_title [2], sup [2] aligned with gold_title, question, answer, ...) -> a new row with N shards.
    rng calls in a fixed order: the sizes shuffle, the two gold shards (rng.sample), the distractor shuffle, one shuffle per shard."""
    n_agents = int(n_agents)
    if n_agents < 2 or n_agents > N_PARAGRAPHS - 1:
        raise ValueError(f'n_agents {n_agents} must be in [2, {N_PARAGRAPHS - 1}]')
    paras = [[t, list(s)] for t, s in row['all_paragraphs']]
    if len(paras) != N_PARAGRAPHS:
        raise ValueError(f'{row.get("id")}: {len(paras)} paragraphs')
    gold_titles = list(row['gold_title'])
    sup_by_title = {t: list(s) for t, s in zip(gold_titles, row['sup'])}
    if len(gold_titles) != 2 or len(sup_by_title) != 2:
        raise ValueError(f'{row.get("id")}: gold_title / sup malformed')
    gold = [p for p in paras if p[0] in sup_by_title]                       # context order
    distractors = [p for p in paras if p[0] not in sup_by_title]
    if len(gold) != 2 or len(distractors) != N_PARAGRAPHS - 2:
        raise ValueError(f'{row.get("id")}: gold paragraphs {len(gold)}')
    sizes = shard_sizes(N_PARAGRAPHS, n_agents, rng)
    gs = rng.sample(range(n_agents), 2)                                    # distinct shards for the two gold paragraphs
    rng.shuffle(distractors)
    shards = [[] for _ in range(n_agents)]
    shards[gs[0]].append(gold[0])
    shards[gs[1]].append(gold[1])
    it = iter(distractors)
    for j in range(n_agents):
        while len(shards[j]) < sizes[j]:
            shards[j].append(next(it))
    rest = list(it)
    if rest:
        raise RuntimeError('distractors left over')
    for sh in shards:
        rng.shuffle(sh)
    sup = [[] for _ in range(n_agents)]
    sup[gs[0]] = sup_by_title[gold[0][0]]
    sup[gs[1]] = sup_by_title[gold[1][0]]
    ans = str(row['answer']).strip().lower()
    hit = [ans in ' '.join(g[1]).lower() for g in gold]
    owner = gs[0] if hit[0] and not hit[1] else gs[1] if hit[1] and not hit[0] else None
    out = dict(row)
    out.update(shards=shards, sup=sup, gold_title=[gold[0][0], gold[1][0]], gold_shards=[gs[0], gs[1]], answer_owner=owner, n_agents=n_agents,
               shard_sizes=[len(sh) for sh in shards])
    return out


def split_stats(rows):
    n_ag = rows[0]['n_agents'] if rows else 0
    owner = Counter('none' if r['answer_owner'] is None else str(r['answer_owner']) for r in rows)
    gs = Counter(tuple(sorted(r['gold_shards'])) for r in rows)
    return dict(n=len(rows), n_agents=n_ag, type=dict(sorted(Counter(str(r['type']) for r in rows).items())),
                answer_owner={k: owner.get(k, 0) for k in [str(i) for i in range(n_ag)] + ['none']},
                gold_shard_pairs={'-'.join(map(str, k)): v for k, v in sorted(gs.items())},
                shard_sizes_mean=[float(sum(r['shard_sizes'][j] for r in rows) / max(1, len(rows))) for j in range(n_ag)])


def read_ids(path):
    return {r['id'] for r in D.read_jsonl(path)} if os.path.isfile(path) else set()


def make_splits7(train_json, exclude_dir, val_path, fresh_path, out_dir, seed=SEED, n_train=N_TRAIN, n_list=N_LIST, limit_raw=0):
    rng = random.Random(seed)
    raw = D.load_hotpot_json(train_json)
    if limit_raw:
        raw = raw[:limit_raw]
    excluded = set()
    for name in ('train', 'val', 'fresh'):
        excluded |= read_ids(os.path.join(exclude_dir, name + '.jsonl'))
    rows, reasons = [], Counter()
    for ex in raw:
        sh = D.shard_example(ex, rng)                                        # the canonical row (filter + the 2-shard sup / gold_title fields)
        if sh is None:
            reasons[D.filter_reason(ex)] += 1
        else:
            rows.append(sh)
    pool = [r for r in rows if r['id'] not in excluded]
    if n_train > len(pool):
        raise ValueError(f'n_train={n_train} > {len(pool)} usable unseen examples')
    train = rng.sample(pool, n_train)
    if len({r['id'] for r in train}) != len(train):
        raise ValueError('duplicate ids')
    val, fresh = D.read_jsonl(val_path), D.read_jsonl(fresh_path)
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(batch=BATCH, prereg=PREREG, seed=int(seed), n_train=int(n_train), n_list=[int(n) for n in n_list],
                sources=dict(train_json=dict(path=os.path.abspath(train_json), sha256=file_sha256(train_json)), exclude_dir=os.path.abspath(exclude_dir),
                             val=dict(path=os.path.abspath(val_path), sha256=file_sha256(val_path)), fresh=dict(path=os.path.abspath(fresh_path), sha256=file_sha256(fresh_path))),
                counts=dict(loaded=len(raw), filtered_out={k: reasons.get(k, 0) for k in D.FILTER_REASONS}, usable=len(rows), excluded_ids=len(excluded),
                            unseen_pool=len(pool), written_train=len(train)),
                files={}, per_file={}, script_sha256=file_sha256(os.path.abspath(__file__)), l1_data_sha256=file_sha256(os.path.abspath(D.__file__)))
    for n in n_list:
        n = int(n)
        todo = [('train7', train)] + ([('val', val), ('fresh', fresh)] if n != 2 else [])
        for k, (name, src) in enumerate(todo):
            r2 = random.Random(seed * 1000 + n * 10 + k)
            out = [reshard_row(r, n, r2) for r in src]
            fn = f'{name}_N{n}.jsonl'
            path = os.path.join(out_dir, fn)
            D.write_jsonl(path, out)
            meta['files'][fn] = dict(path=fn, n=len(out), sha256=file_sha256(path), source=name)
            meta['per_file'][fn] = split_stats(out)
    with open(os.path.join(out_dir, 'splits7_meta.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(meta, fh, indent=1)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--train-json', required=True)
    ap.add_argument('--exclude-dir', required=True, help='splits_l1: its train / val / fresh ids are never drawn')
    ap.add_argument('--val', required=True); ap.add_argument('--fresh', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--n-train', type=int, default=N_TRAIN)
    ap.add_argument('--n-agents', default=','.join(str(n) for n in N_LIST))
    ap.add_argument('--limit-raw', type=int, default=0, help='tests: only the first n raw examples')
    a = ap.parse_args(argv)
    n_list = [int(x) for x in a.n_agents.split(',') if x.strip()]
    meta = make_splits7(a.train_json, a.exclude_dir, a.val, a.fresh, a.out, a.seed, a.n_train, n_list, a.limit_raw)
    print(json.dumps(dict(counts=meta['counts'], files={k: dict(n=v['n'], sha256=v['sha256'][:16]) for k, v in meta['files'].items()},
                          per_file={k: dict(answer_owner=v['answer_owner'], gold_shard_pairs=v['gold_shard_pairs'], shard_sizes_mean=v['shard_sizes_mean']) for k, v in meta['per_file'].items()}),
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
