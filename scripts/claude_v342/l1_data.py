"""l1_data.py (Claude, 2026-10-29): data side of batch L1 "LLM latent collaboration with sheaf consensus"
(claude_handoff/DESIGN_LLM_LATENT_SHEAF_GBP.md: section 2 task / data / splits, section 4 the arms' prompts,
section 5 metrics; pre-registration OVERNIGHT_REPORT D8-82). Pure python (json, random, hashlib, re, string,
collections); no torch, no transformers.

What this file fixes (design section 2, "写死" column):
  * HotpotQA distractor setting. An example is kept iff it has exactly 10 context paragraphs with distinct titles
    and exactly 2 distinct supporting-fact titles, both present in the context (so "the gold paragraph" is well
    defined). The 10 paragraphs are split 5 + 5 so that every shard holds exactly one gold paragraph: which gold
    paragraph goes to shard 0 is a coin flip, the 8 distractors are shuffled and split 4 / 4, and the order inside
    each shard is shuffled. One random.Random(seed) does all of this in file order (train file, then dev file) and
    then draws the splits, so the whole data set is a function of (files, seed).
  * Splits: TRAIN n_train from the official train file; VAL n_val (the verdict set) and fresh n_fresh (degrade
    check) from the official dev-distractor file, disjoint by id; every other dev example is the sealed TEST and
    is never written, only counted. Seed 342101, sizes 20,000 / 1,000 / 1,000.
  * Prompts of every arm (section 4): latent arms get (prefix, suffix) and the model side inserts the K soft tokens
    between them; B0 / B4 / T / T0 get plain strings.
  * Official HotpotQA EM / F1 normalisation (hotpot_evaluate_v1.py; section 5).

Raw HotpotQA and the jsonl products never enter the public repo (*.jsonl is gitignored; data lives on uk-linux
/mnt/datasets, see run_l1_data.sh).

Usage:
  python l1_data.py --train-json hotpot_train_v1.1.json --dev-json hotpot_dev_distractor_v1.json --out DIR
                    [--seed 342101 --n-train 20000 --n-val 1000 --n-fresh 1000]
  -> DIR/train.jsonl, DIR/val.jsonl, DIR/fresh.jsonl, DIR/splits_meta.json; prints the meta as JSON.
"""
import argparse
import hashlib
import json
import os
import random
import re
import string
from collections import Counter

SEED = 342101                       # design section 2: 划分种子 342101 (sharding and split draws)
N_TRAIN, N_VAL, N_FRESH = 20000, 1000, 1000
N_PARAGRAPHS, N_GOLD, SHARD_SIZE = 10, 2, 5
SPLITS = ('train', 'val', 'fresh')
FILTER_REASONS = ('n_paragraphs', 'duplicate_titles', 'n_sup_titles', 'sup_title_missing')


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- loading and sharding (design section 2)
def load_hotpot_json(path):
    """Official HotpotQA JSON (train v1.1 or dev distractor v1): a list of dicts with keys _id, question, answer, type,
    level, supporting_facts [[title, sent_idx], ...], context [[title, [sentence, ...]], ...]."""
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f'{path}: expected a list of HotpotQA examples')
    return data


def filter_reason(ex):
    """None if the example can be sharded, else one of FILTER_REASONS (counted in splits_meta.json)."""
    ctx = ex.get('context') or []
    if len(ctx) != N_PARAGRAPHS:
        return 'n_paragraphs'
    titles = [p[0] for p in ctx]
    if len(set(titles)) != N_PARAGRAPHS:
        return 'duplicate_titles'
    sup_titles = list(dict.fromkeys(t for t, _ in ex.get('supporting_facts') or []))
    if len(sup_titles) != N_GOLD:
        return 'n_sup_titles'
    if any(t not in titles for t in sup_titles):
        return 'sup_title_missing'
    return None


def shard_example(ex, rng):
    """Split one HotpotQA example into 2 shards of SHARD_SIZE paragraphs, exactly one gold paragraph each.

    rng calls, in this order (so the same ex and the same rng state give the same output): coin flip = which gold
    paragraph (context order) goes to shard 0; shuffle of the 8 distractors, then split 4 / 4; shuffle inside shard 0;
    shuffle inside shard 1. Returns None for examples that filter_reason() rejects. The input is not modified.
    """
    if filter_reason(ex) is not None:
        return None
    ctx = [[t, list(s)] for t, s in ex['context']]
    sup_titles = {t for t, _ in ex['supporting_facts']}
    gold = [p for p in ctx if p[0] in sup_titles]            # context order
    distractors = [p for p in ctx if p[0] not in sup_titles]
    if rng.randrange(2):                                      # coin flip: gold paragraph for shard 0
        gold.reverse()
    rng.shuffle(distractors)
    k = SHARD_SIZE - 1
    shards = [[gold[0]] + distractors[:k], [gold[1]] + distractors[k:]]
    for sh in shards:
        rng.shuffle(sh)
    sup = []
    for g in gold:
        idx = sorted({int(i) for t, i in ex['supporting_facts'] if t == g[0] and 0 <= int(i) < len(g[1])})
        sup.append([g[1][i] for i in idx])
    ans = str(ex['answer']).strip().lower()
    hit = [ans in ' '.join(g[1]).lower() for g in gold]
    owner = 0 if hit[0] and not hit[1] else 1 if hit[1] and not hit[0] else None
    return dict(id=ex['_id'], question=ex['question'], answer=ex['answer'], type=ex.get('type'), level=ex.get('level'),
                all_paragraphs=[[t, list(s)] for t, s in ex['context']], shards=shards,
                gold_title=[gold[0][0], gold[1][0]], answer_owner=owner, sup=sup)


# ---------------------------------------------------------------- splits (design section 2)
def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:    # LF on every OS: the sha256 is part of the record
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _split_stats(rows):
    owner = Counter('none' if r['answer_owner'] is None else str(r['answer_owner']) for r in rows)
    return dict(n=len(rows), type=dict(sorted(Counter(str(r['type']) for r in rows).items())),
                answer_owner={k: owner.get(k, 0) for k in ('0', '1', 'none')})


def make_splits(train_json, dev_json, out_dir, seed=SEED, n_train=N_TRAIN, n_val=N_VAL, n_fresh=N_FRESH):
    """Shard both files with one random.Random(seed) (train file then dev file, file order), draw TRAIN from the
    train file and VAL + fresh from the dev file (rng.sample; first n_val -> VAL, next n_fresh -> fresh), write
    out_dir/{train,val,fresh}.jsonl and out_dir/splits_meta.json, and return the meta dict. The remaining dev
    examples are the sealed TEST: counted, never written. Deterministic across runs and platforms."""
    rng = random.Random(seed)
    sources = {'train': train_json, 'dev': dev_json}
    loaded, filtered, kept = {}, {}, {}
    for name, path in sources.items():
        data = load_hotpot_json(path)
        ok, reasons = [], Counter()
        for ex in data:
            sh = shard_example(ex, rng)
            if sh is None:
                reasons[filter_reason(ex)] += 1
            else:
                ok.append(sh)
        loaded[name], kept[name] = len(data), ok
        filtered[name] = {k: reasons.get(k, 0) for k in FILTER_REASONS}
    if n_train > len(kept['train']):
        raise ValueError(f'n_train={n_train} > {len(kept["train"])} usable train examples')
    if n_val + n_fresh > len(kept['dev']):
        raise ValueError(f'n_val + n_fresh={n_val + n_fresh} > {len(kept["dev"])} usable dev examples')
    train = rng.sample(kept['train'], n_train)
    picked = rng.sample(kept['dev'], n_val + n_fresh)
    if len({r['id'] for r in picked}) != len(picked) or len({r['id'] for r in train}) != len(train):
        raise ValueError('duplicate ids inside a split')
    splits = {'train': train, 'val': picked[:n_val], 'fresh': picked[n_val:]}
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(batch='L1', prereg='D8-82', seed=int(seed), n_train=int(n_train), n_val=int(n_val), n_fresh=int(n_fresh),
                sources={k: dict(path=os.path.abspath(p), sha256=file_sha256(p)) for k, p in sources.items()},
                counts=dict(loaded=loaded, filtered_out=filtered, usable={k: len(v) for k, v in kept.items()},
                            written={k: len(v) for k, v in splits.items()},
                            sealed_test_count=len(kept['dev']) - n_val - n_fresh),
                per_split={}, files={}, script_sha256=file_sha256(os.path.abspath(__file__)))
    for name in SPLITS:
        path = os.path.join(out_dir, name + '.jsonl')
        write_jsonl(path, splits[name])
        meta['files'][name] = dict(path=name + '.jsonl', n=len(splits[name]), sha256=file_sha256(path))
        meta['per_split'][name] = _split_stats(splits[name])
    with open(os.path.join(out_dir, 'splits_meta.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(meta, fh, indent=1)
    return meta


# ---------------------------------------------------------------- prompts (design section 4)
def paragraphs_text(paragraphs):
    return ''.join(f"Title: {t}\n{' '.join(s)}\n" for t, s in paragraphs)


def latent_prompt(ex, agent):
    """(prefix, suffix) for the latent arms (U, S1, S1L1): the K soft tokens go between them (section 3, 读头)."""
    return 'Documents:\n' + paragraphs_text(ex['shards'][agent]), f"Question: {ex['question']}\nAnswer:"


def full_prompt(ex):
    """B4: one model, all 10 paragraphs."""
    return 'Documents:\n' + paragraphs_text(ex['all_paragraphs']) + f"Question: {ex['question']}\nAnswer:"


def no_comm_prompt(ex, agent):
    """B0: own shard only (= latent prompt with nothing in between)."""
    prefix, suffix = latent_prompt(ex, agent)
    return prefix + suffix


def text_sender_prompt(ex, agent):
    """T / T0 sender: write the supporting sentences from the own shard."""
    return ('Documents:\n' + paragraphs_text(ex['shards'][agent])
            + f"Question: {ex['question']}\nWrite the sentences from the documents that help answer the question:")


def text_sender_target(ex, agent):
    """T sender's gold message: the supporting-fact sentences of the own gold paragraph."""
    sup = ex['sup'][agent]
    return ' ' + ' '.join(sup) if sup else ' (none)'


def text_receiver_prompt(ex, agent, message):
    return ('Documents:\n' + paragraphs_text(ex['shards'][agent])
            + f"Message from the other agent: {message}\nQuestion: {ex['question']}\nAnswer:")


def answer_target(ex):
    return ' ' + ex['answer']


# ---------------------------------------------------------------- metrics (design section 5; hotpot_evaluate_v1.py)
_ARTICLES = re.compile(r'\b(a|an|the)\b')
_PUNCT = set(string.punctuation)
_SPECIAL = ('yes', 'no', 'noanswer')


def normalize_answer(s):
    """lower, strip punctuation, strip articles a / an / the, collapse whitespace (SQuAD / HotpotQA)."""
    s = ''.join(ch for ch in str(s).lower() if ch not in _PUNCT)
    return ' '.join(_ARTICLES.sub(' ', s).split())


def exact_match(pred, gold):
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def f1_score(pred, gold):
    """Token F1 after normalisation; HotpotQA's rule: a yes / no / noanswer pred or gold that does not match the other
    side exactly scores 0."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if (p in _SPECIAL or g in _SPECIAL) and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    common = sum((Counter(pt) & Counter(gt)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pt), common / len(gt)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description='L1 (D8-82) HotpotQA sharding and splits; see module docstring.')
    ap.add_argument('--train-json', required=True)
    ap.add_argument('--dev-json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--n-train', type=int, default=N_TRAIN)
    ap.add_argument('--n-val', type=int, default=N_VAL)
    ap.add_argument('--n-fresh', type=int, default=N_FRESH)
    a = ap.parse_args(argv)
    meta = make_splits(a.train_json, a.dev_json, a.out, seed=a.seed, n_train=a.n_train, n_val=a.n_val, n_fresh=a.n_fresh)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
