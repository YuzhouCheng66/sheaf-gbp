"""Toy key-passing data in the l1_data schema (CPU diagnostic, not part of batch L1).
Each example: question 'what is the code ?'; the answer word sits in ONE shard ('the code is <A> .'); the other shard's gold
paragraph holds a distractor fact ('the box is <B> .'); each shard also has one filler paragraph. answer_owner = the shard with <A>.
B0 (own shard only) can answer only when it owns the code (50 %), otherwise it must guess among 40 names."""
import json, random, sys
out, n_train, n_val, n_fresh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
NAMES = ('apple banana cherry grape lemon mango melon olive peach pear plum kiwi fig lime date coral amber jade onyx opal '
         'ruby topaz pearl ivory ebony cedar maple birch pine oak elm ash fern moss reed rose lily iris tulip daisy').split()
assert len(NAMES) == 40 and len(set(NAMES)) == 40
FILL = ['the sky is blue .', 'birds sing at dawn .', 'rivers flow to the sea .', 'the cat sleeps on the mat .', 'snow falls in winter .',
        'bread needs flour .', 'the moon is bright .', 'trees grow slowly .', 'the road is long .', 'rain falls on the hill .']
TITLES = 'alpha beta gamma delta zeta theta kappa sigma'.split()

def make(rng, i):
    a = rng.choice(NAMES)
    b = rng.choice([n for n in NAMES if n != a])
    owner = rng.randrange(2)
    t = rng.sample(TITLES, 4)
    g_ans = [t[0], [f'the code is {a} .']]
    g_oth = [t[1], [f'the box is {b} .']]
    d = [[t[2], [rng.choice(FILL)]], [t[3], [rng.choice(FILL)]]]
    gold = [g_ans, g_oth] if owner == 0 else [g_oth, g_ans]
    shards = [[gold[0], d[0]], [gold[1], d[1]]]
    for sh in shards:
        rng.shuffle(sh)
    return dict(id=f'toy{i}', question='what is the code ?', answer=a, type='bridge', level='easy',
                all_paragraphs=shards[0] + shards[1], shards=shards, gold_title=[gold[0][0], gold[1][0]], answer_owner=owner,
                sup=[list(gold[0][1]), list(gold[1][1])])

for name, n, seed in (('train', n_train, 11), ('val', n_val, 22), ('fresh', n_fresh, 33)):
    rng = random.Random(seed)
    rows = [make(rng, f'{name}{i}') for i in range(n)]
    with open(f'{out}/{name}.jsonl', 'w', encoding='utf-8', newline='\n') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    own = sum(r['answer_owner'] == 0 for r in rows)
    print(name, n, 'owner0', own, 'owner1', n - own)
print(json.dumps(rows[0]))
