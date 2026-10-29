"""l1_aggregate.py (Claude, 2026-10-29): the verdict logic of batch L1 and nothing else
(claude_handoff/DESIGN_LLM_LATENT_SHEAF_GBP.md section 6; pre-registration OVERNIGHT_REPORT D8-82).
Reads runs/<arm>-<pair>-s<seed>/summary.json written by the model side (l1_train.py) and writes
runs/aggregate.json + runs/aggregate.md. Pure python.

summary.json schema (summary_schema() returns it as a dict the model side can import; validate_summary() checks it):
{
  "arm":   "B0" | "B4" | "T" | "T0" | "U" | "S1" | "S1L1",
  "pair":  "hetero" | "same",
  "seed":  342101 | 342202,
  "val":   {"em": float, "f1": float, "em_agent": [a0, a1] (B4: [a]),
            "em_model": {"Qwen/Qwen3-1.7B": em, "meta-llama/Llama-3.2-1B-Instruct": em}},
  "fresh": {same keys},
  "c_assert": true | false,          # latent arms; required for S1 and S1L1 (GBP fixed point == direct solve, arm C)
  "scripts_sha": {"<script>": "<sha256>", ...}
}
Every em / f1 is in PERCENTAGE POINTS (0-100): the thresholds below are in points. "em" is the mean over the
answering agents (design section 5, main metric); "em_model" is what P0 reads (per model of the hetero pair).

Criteria (VAL, both seeds must pass; design section 6, D8-82):
  P0  stage-0 gate, per model of the hetero pair, each seed: B4 - B0 >= 15.0 and >= 5 * |B0(seed 1) - B0(seed 2)|.
  P1  hetero S1 - T    >= 2.0       P2  hetero S1 - U >= 1.0       P3  hetero S1 - S1L1 >= 1.0
Verdict: P1-P3 all pass -> LEVER; P1 fails -> NO; otherwise AMBIGUOUS.
INCOMPLETE: a required summary is missing (hetero B0, B4, T, U, S1, S1L1 x both seeds), any c_assert is false (or
missing for S1 / S1L1), scripts_sha differs between runs of the same script, a summary's arm / pair / seed
disagrees with its directory name, or P0 fails (stage 1 was not supposed to run; design: 不过就停).
Degrade: (S1 - T) on fresh lower than on VAL by more than 1.0 for either seed caps LEVER at AMBIGUOUS.
T0 and the same-model pair are report-only: same-pair S1 - T is shown against the P1 threshold as the diagnostic
"alignment (P_i) is the bottleneck" and never changes the verdict.

Usage:
  python l1_aggregate.py --runs DIR            -> DIR/aggregate.json, DIR/aggregate.md; prints "VERDICT: ..."
  python l1_aggregate.py --runs DIR --stage0   -> same files, P0 only; prints "P0: PASS" / "P0: FAIL" / "P0: INCOMPLETE"
"""
import argparse
import hashlib
import json
import os
import time

ARMS = ('B0', 'B4', 'T', 'T0', 'U', 'S1', 'S1L1')
PAIRS = ('hetero', 'same')
SEEDS = (342101, 342202)
MODELS = ('Qwen/Qwen3-1.7B', 'meta-llama/Llama-3.2-1B-Instruct')
LATENT_ARMS = ('U', 'S1', 'S1L1')
GBP_ARMS = ('S1', 'S1L1')               # arms whose evaluation runs real GBP messages: c_assert is mandatory
REQUIRED_ARMS = ('B0', 'B4', 'T', 'U', 'S1', 'S1L1')
STAGE0_ARMS = ('B0', 'B4')
P0_MIN_GAIN, P0_SEED_FACTOR = 15.0, 5.0
CRITERIA = {'P1': dict(claim='latent consensus beats text', arms=('S1', 'T'), threshold=2.0),
            'P2': dict(claim='symmetric consensus not below one-way projection', arms=('S1', 'U'), threshold=1.0),
            'P3': dict(claim='precision is useful', arms=('S1', 'S1L1'), threshold=1.0)}
DEGRADE_PT = 1.0
VERDICTS = ('LEVER', 'AMBIGUOUS', 'NO', 'INCOMPLETE')
ROUND = 6                               # margins are rounded to 6 decimals before the >= test (45.3 - 43.3 = 1.9999999999999964)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run_dir_name(arm, pair, seed):
    return f'{arm}-{pair}-s{int(seed)}'


def summary_schema():
    """Required keys of runs/<arm>-<pair>-s<seed>/summary.json, as {key: description}; importable by the model side."""
    split = {'em': 'float, exact match in percentage points, mean over the answering agents',
             'f1': 'float, F1 in percentage points, mean over the answering agents',
             'em_agent': 'list of floats, EM per agent in percentage points: [agent0, agent1]; B4: [model]',
             'em_model': 'dict model id -> EM in percentage points; hetero pair keys: ' + ', '.join(MODELS)}
    return {'arm': 'str, one of ' + ', '.join(ARMS),
            'pair': 'str, one of ' + ', '.join(PAIRS),
            'seed': 'int, one of ' + ', '.join(map(str, SEEDS)) + ' (= the s<seed> of the directory name)',
            'val': split, 'fresh': dict(split),
            'c_assert': 'bool; required for ' + ', '.join(GBP_ARMS) + ' (GBP fixed point == direct solve); optional for U; '
                        'false anywhere -> INCOMPLETE',
            'scripts_sha': 'dict script name -> sha256 of the script that produced the run; must agree across runs'}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_summary(s, arm, pair, seed):
    """List of problems (empty = valid) of one summary dict against the directory it was read from."""
    if not isinstance(s, dict):
        return ['summary is not a dict']
    problems = []
    for k in ('arm', 'pair', 'seed', 'val', 'fresh', 'scripts_sha'):
        if k not in s:
            problems.append(f'missing key {k}')
    if 'arm' in s and 'pair' in s and 'seed' in s:
        try:
            same = s['arm'] == arm and s['pair'] == pair and int(s['seed']) == int(seed)
        except (TypeError, ValueError):
            same = False
        if not same:
            problems.append(f'arm / pair / seed {s.get("arm")}, {s.get("pair")}, {s.get("seed")} disagree with the directory')
    for split in ('val', 'fresh'):
        blk = s.get(split)
        if not isinstance(blk, dict):
            if split in s:
                problems.append(f'{split} is not a dict')
            continue
        for k in ('em', 'f1', 'em_agent', 'em_model'):
            if k not in blk:
                problems.append(f'{split}.{k} missing')
        for k in ('em', 'f1'):
            if k in blk and (not _num(blk[k]) or not 0.0 <= float(blk[k]) <= 100.0):
                problems.append(f'{split}.{k} is not a number in [0, 100] (percentage points)')
        if 'em_model' in blk and not isinstance(blk['em_model'], dict):
            problems.append(f'{split}.em_model is not a dict')
        if 'em_agent' in blk and not isinstance(blk['em_agent'], list):
            problems.append(f'{split}.em_agent is not a list')
    if arm in GBP_ARMS and 'c_assert' not in s:
        problems.append('c_assert missing')
    if 'c_assert' in s and s['c_assert'] is not True:
        problems.append('c_assert false')
    if 'scripts_sha' in s and not isinstance(s['scripts_sha'], dict):
        problems.append('scripts_sha is not a dict')
    return problems


def load_runs(runs_dir, arms=ARMS, pairs=PAIRS):
    """{run name: {path, summary, problems}} for every summary.json found, plus the list of absent run names."""
    runs, absent = {}, []
    for pair in pairs:
        for arm in arms:
            for seed in SEEDS:
                name = run_dir_name(arm, pair, seed)
                path = os.path.join(runs_dir, name, 'summary.json')
                if not os.path.isfile(path):
                    absent.append(name)
                    continue
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        s = json.load(fh)
                    problems = validate_summary(s, arm, pair, seed)
                except (OSError, ValueError) as e:
                    s, problems = None, [f'unreadable: {e}']
                runs[name] = dict(path=os.path.abspath(path), summary=s, problems=problems)
    return runs, absent


def _get(runs, arm, pair, seed, split, key='em', model=None):
    r = runs.get(run_dir_name(arm, pair, seed))
    if r is None or r['summary'] is None:
        return None
    try:
        v = r['summary'][split][key]
        if model is not None:
            v = v[model]
        return float(v)
    except (KeyError, TypeError, ValueError):
        return None


def sha_conflicts(runs):
    """{script: {sha: [run names]}} for scripts whose sha differs between runs."""
    seen = {}
    for name, r in runs.items():
        shas = (r['summary'] or {}).get('scripts_sha')
        if isinstance(shas, dict):
            for script, sha in shas.items():
                seen.setdefault(script, {}).setdefault(str(sha), []).append(name)
    return {k: v for k, v in seen.items() if len(v) > 1}


def scripts_sha_union(runs):
    out = {}
    for r in runs.values():
        shas = (r['summary'] or {}).get('scripts_sha')
        if isinstance(shas, dict):
            for k, v in shas.items():
                out.setdefault(k, str(v))
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- criteria
def p0_block(runs):
    """P0 per model of the hetero pair: B4 - B0 >= 15 pt and >= 5 x |B0 seed 1 - B0 seed 2|, both seeds."""
    out = dict(threshold_pt=P0_MIN_GAIN, seed_factor=P0_SEED_FACTOR, pair='hetero', per_model={}, missing=[], **{'pass': None})
    for m in MODELS:
        b0 = {str(s): _get(runs, 'B0', 'hetero', s, 'val', 'em_model', m) for s in SEEDS}
        b4 = {str(s): _get(runs, 'B4', 'hetero', s, 'val', 'em_model', m) for s in SEEDS}
        if any(v is None for v in list(b0.values()) + list(b4.values())):
            out['missing'].append(m)
            out['per_model'][m] = dict(b0=b0, b4=b4, gain=None, delta_b0_seeds=None, required_gain=None, pass_seed=None, **{'pass': None})
            continue
        d = round(abs(b0[str(SEEDS[0])] - b0[str(SEEDS[1])]), ROUND)
        req = max(P0_MIN_GAIN, P0_SEED_FACTOR * d)
        gain = {s: round(b4[s] - b0[s], ROUND) for s in b0}
        ok = {s: bool(gain[s] >= P0_MIN_GAIN and gain[s] >= P0_SEED_FACTOR * d) for s in gain}
        out['per_model'][m] = dict(b0=b0, b4=b4, gain=gain, delta_b0_seeds=d, required_gain=req, pass_seed=ok, **{'pass': all(ok.values())})
    if not out['missing']:
        out['pass'] = all(v['pass'] for v in out['per_model'].values())
    return out


def criterion_block(runs, name, pair='hetero', split='val'):
    c = CRITERIA[name]
    a, b = c['arms']
    margin, ok = {}, {}
    for s in SEEDS:
        va, vb = _get(runs, a, pair, s, split), _get(runs, b, pair, s, split)
        margin[str(s)] = None if va is None or vb is None else round(va - vb, ROUND)
        ok[str(s)] = None if margin[str(s)] is None else bool(margin[str(s)] >= c['threshold'])
    complete = all(v is not None for v in ok.values())
    return dict(claim=c['claim'], arms=list(c['arms']), pair=pair, split=split, threshold=c['threshold'],
                margin=margin, pass_seed=ok, **{'pass': all(ok.values()) if complete else None})


def degrade_block(runs):
    """(S1 - T) on fresh vs VAL (hetero): a drop of more than DEGRADE_PT on either seed caps LEVER at AMBIGUOUS."""
    val = criterion_block(runs, 'P1', 'hetero', 'val')['margin']
    fresh = criterion_block(runs, 'P1', 'hetero', 'fresh')['margin']
    drop = {s: None if val[s] is None or fresh[s] is None else round(val[s] - fresh[s], ROUND) for s in val}
    complete = all(v is not None for v in drop.values())
    return dict(arms=['S1', 'T'], pair='hetero', max_drop_pt=DEGRADE_PT, val=val, fresh=fresh, drop=drop,
                triggered=any(v > DEGRADE_PT for v in drop.values()) if complete else None)


def same_pair_block(runs):
    """Diagnostic only: same-model S1 - T against the P1 threshold (never changes the verdict)."""
    c = criterion_block(runs, 'P1', 'same', 'val')
    c['note'] = 'diagnostic only; same-pair pass with hetero fail = alignment (P_i) is the bottleneck'
    return c


def decide(criteria, degrade, incomplete_reasons):
    if incomplete_reasons:
        return 'INCOMPLETE'
    if all(criteria[k]['pass'] for k in ('P1', 'P2', 'P3')):
        v = 'LEVER'
    elif not criteria['P1']['pass']:
        v = 'NO'
    else:
        v = 'AMBIGUOUS'
    if v == 'LEVER' and degrade['triggered']:
        v = 'AMBIGUOUS'
    return v


# ---------------------------------------------------------------- aggregate
def _runs_block(runs):
    out = {}
    for name, r in sorted(runs.items()):
        s = r['summary'] or {}
        out[name] = dict(path=r['path'], problems=r['problems'],
                         **{k: s.get(k) for k in ('arm', 'pair', 'seed', 'val', 'fresh', 'c_assert', 'scripts_sha')})
    return out


def aggregate(runs_dir, stage0=False):
    runs, absent = load_runs(runs_dir, arms=STAGE0_ARMS if stage0 else ARMS, pairs=('hetero',) if stage0 else PAIRS)
    required = [run_dir_name(a, 'hetero', s) for a in (STAGE0_ARMS if stage0 else REQUIRED_ARMS) for s in SEEDS]
    reasons = [f'missing required run {n}' for n in required if n not in runs]
    reasons += [f'{n}: {p}' for n, r in sorted(runs.items()) for p in r['problems']]
    conflicts = sha_conflicts(runs)
    reasons += [f'scripts_sha of {k} differs between runs: ' + '; '.join(f'{sha[:12]} <- {sorted(v)}' for sha, v in sorted(vv.items()))
                for k, vv in sorted(conflicts.items())]
    ems = [v for a in ARMS for p in PAIRS for s in SEEDS for sp in ('val', 'fresh') for v in [_get(runs, a, p, s, sp)] if v is not None]
    if ems and max(ems) <= 1.0:        # unit guard: EM must be percentage points (0-100), not fractions
        reasons.append('every EM is <= 1.0: summary.json must report percentage points (0-100), not fractions')
    p0 = p0_block(runs)
    out = dict(batch='L1', prereg='D8-82', design='claude_handoff/DESIGN_LLM_LATENT_SHEAF_GBP.md section 6',
               stage=0 if stage0 else 1, runs_dir=os.path.abspath(runs_dir), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               runs=_runs_block(runs), absent=absent, required=required, scripts_sha=scripts_sha_union(runs),
               sha_conflicts=conflicts, p0=p0, script_sha256=file_sha256(os.path.abspath(__file__)))
    p0_line = 'P0: ' + ('INCOMPLETE' if reasons or p0['pass'] is None else 'PASS' if p0['pass'] else 'FAIL')
    if stage0:
        out.update(incomplete_reasons=reasons, p0_line=p0_line, verdict=None)
        return out
    if p0['pass'] is False:
        reasons.append('P0 gate failed (stage 1 was not supposed to run)')
    criteria = {k: criterion_block(runs, k) for k in CRITERIA}
    degrade = degrade_block(runs)
    out.update(criteria=criteria, degrade=degrade, same_pair=same_pair_block(runs), incomplete_reasons=reasons,
               p0_line=p0_line, verdict=decide(criteria, degrade, reasons))
    return out


def _f(x, nd=1):
    return '-' if x is None else f'{x:.{nd}f}'


def _pf(x):
    return '-' if x is None else 'PASS' if x else 'FAIL'


def to_markdown(agg):
    L = [f"# L1 aggregate (D8-82, design section 6) - stage {agg['stage']}", '',
         f"runs: `{agg['runs_dir']}`  utc: {agg['utc']}", '',
         f"**{agg['p0_line']}**" if agg['stage'] == 0 else f"**VERDICT: {agg['verdict']}**  ({agg['p0_line']})", '',
         '## EM / F1 (percentage points; mean over agents) by arm x pair x seed', '',
         '| run | arm | pair | seed | VAL EM | VAL F1 | fresh EM | fresh F1 | c_assert | problems |', '|---|---|---|---|---|---|---|---|---|---|']
    for name, r in agg['runs'].items():
        v, fr = r.get('val') or {}, r.get('fresh') or {}
        L.append(f"| {name} | {r.get('arm')} | {r.get('pair')} | {r.get('seed')} | {_f(v.get('em'))} | {_f(v.get('f1'))} | "
                 f"{_f(fr.get('em'))} | {_f(fr.get('f1'))} | {r.get('c_assert')} | {'; '.join(r['problems']) or '-'} |")
    if agg['absent']:
        L += ['', 'absent runs: ' + ', '.join(agg['absent'])]
    p0 = agg['p0']
    L += ['', f"## P0 (stage-0 gate, hetero, VAL): B4 - B0 >= {p0['threshold_pt']} pt and >= {p0['seed_factor']} x |B0 seed1 - B0 seed2|", '',
          '| model | seed | B0 | B4 | B4 - B0 | required | pass |', '|---|---|---|---|---|---|---|']
    for m, b in p0['per_model'].items():
        for s in map(str, SEEDS):
            g = None if b['gain'] is None else b['gain'][s]
            ok = None if b['pass_seed'] is None else b['pass_seed'][s]
            L.append(f"| {m} | {s} | {_f(b['b0'][s])} | {_f(b['b4'][s])} | {_f(g, 2)} | {_f(b['required_gain'], 2)} | {_pf(ok)} |")
    L.append(f"| **all** | | | | | | **{_pf(p0['pass'])}** |")
    if agg['stage'] == 1:
        L += ['', '## P1-P3 (hetero, VAL, both seeds)', '', '| crit | claim | arms | threshold | margin s1 | margin s2 | pass |', '|---|---|---|---|---|---|---|']
        for k, c in agg['criteria'].items():
            m = c['margin']
            L.append(f"| {k} | {c['claim']} | {c['arms'][0]} - {c['arms'][1]} | >= {c['threshold']} | {_f(m[str(SEEDS[0])], 2)} | "
                     f"{_f(m[str(SEEDS[1])], 2)} | {_pf(c['pass'])} |")
        d = agg['degrade']
        L += ['', f"## Degrade (S1 - T on fresh vs VAL; drop > {d['max_drop_pt']} pt on either seed caps LEVER at AMBIGUOUS)", '',
              '| seed | VAL margin | fresh margin | drop | triggered |', '|---|---|---|---|---|']
        for s in map(str, SEEDS):
            L.append(f"| {s} | {_f(d['val'][s], 2)} | {_f(d['fresh'][s], 2)} | {_f(d['drop'][s], 2)} | {d['triggered']} |")
        sp = agg['same_pair']
        L += ['', '## Same-model pair (diagnostic only; never changes the verdict)', '',
              f"S1 - T (same, VAL): seed {SEEDS[0]} {_f(sp['margin'][str(SEEDS[0])], 2)}, seed {SEEDS[1]} {_f(sp['margin'][str(SEEDS[1])], 2)}; "
              f"P1 threshold >= {sp['threshold']}: {_pf(sp['pass'])}", '', sp['note']]
    L += ['', '## INCOMPLETE reasons', ''] + ([f'- {r}' for r in agg['incomplete_reasons']] or ['- none'])
    L += ['', f"scripts_sha: " + (', '.join(f'{k} {v[:12]}' for k, v in agg['scripts_sha'].items()) or '-'),
          f"l1_aggregate.py sha256 {agg['script_sha256'][:12]}", '']
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description='L1 (D8-82) verdict from runs/<arm>-<pair>-s<seed>/summary.json; see module docstring.')
    ap.add_argument('--runs', required=True, help='directory holding <arm>-<pair>-s<seed>/summary.json')
    ap.add_argument('--stage0', action='store_true', help='evaluate only P0 (B0 and B4 of the hetero pair)')
    a = ap.parse_args(argv)
    agg = aggregate(a.runs, stage0=a.stage0)
    with open(os.path.join(a.runs, 'aggregate.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(agg, fh, indent=1)
    with open(os.path.join(a.runs, 'aggregate.md'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(to_markdown(agg))
    for r in agg['incomplete_reasons']:
        print('INCOMPLETE:', r)
    print(agg['p0_line'])
    if not a.stage0:
        print('VERDICT:', agg['verdict'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
