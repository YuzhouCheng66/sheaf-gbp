"""c2_votes.py (moderator C2, 2026-10-27): merge the 10 debate vote tables into one table.
Every debate table has the columns: design | proposer | furniture | compression | feasibility | compliance | falsifiability | total.
Rows are mapped to a canonical design id from the proposer id (A01..A10) plus name keywords; syntheses are kept apart (each was rated
only by its own author).  Output: probes/C2/c2_votes.json and a printed table (mean over all voters, mean without self-votes,
min / max / std of totals, compliance flags)."""
import glob, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEB = os.path.abspath(os.path.join(HERE, '..', '..', 'debate'))

NAMES = {
    'A01-I': 'LL-Sheaf 光线序无损', 'A01-II': 'Hyper-Sheaf 学习超先验', 'A01-III': 'Prog-Sheaf 生成器反演',
    'A02-G1': 'GenFact 事实格+生成先验', 'A02-G2': '验证式生成解码', 'A02-G3': '场景 token 潜扩散',
    'A03-A': 'RAY-SHEAF 平面+盒+投射', 'A03-B': 'RK-Δ', 'A03-C': '学习的结构语言',
    'A04-1': 'ΔV-Sheaf', 'A04-2': 'PV-Sheaf', 'A04-3': 'PR-Sheaf 程序+残差',
    'A05-1': 'RAYFIELD 稀疏家具隐式场', 'A05-2': '盒子词元生成 sheaf', 'A05-3': '码让位给家具',
    'A06-1': 'ProgSheaf 一致集最短程序', 'A06-2': 'RaySheaf ctx(O)+位姿', 'A06-3': 'RayField 保 join 神经码',
    'A07-D1': 'GELS 精确格层+学习熵模型', 'A07-D2': '噪声世界+真值评分', 'A07-D3': '图元超先验+位姿配准',
    'A08-D1': 'RC-Sheaf 几何层+投射', 'A08-D2': 'Vocab-Sheaf 家具是词', 'A08-D3': '融合=解码 潜空间 GBP',
    'A09-A': 'W-Sheaf 世界码+重演', 'A09-B': '教师潜空间+高斯 GBP', 'A09-C': '程序+残差',
    'A10-α': 'EVS-Sheaf 可见面原子+新息', 'A10-β': '现码+新息线路+家具残差', 'A10-γ': '物体区间 sheaf',
}


def canon(name, prop):
    m = re.search(r'A(0[1-9]|10)', prop)
    if not m:
        return None
    p = 'A' + m.group(1)
    syn = ('合成' in name) or ('综合' in name) or ('合成' in prop) or ('综合' in prop)
    if p == 'A01':
        if syn or 'RAY-PROG' in name: return 'A01-syn'
        if 'Hyper' in name or 'Ⅱ' in prop: return 'A01-II'
        if 'Prog-Sheaf' in name or 'Ⅲ' in prop: return 'A01-III'
        if 'LL-Sheaf' in name or '光线序' in name: return 'A01-I'
    if p == 'A02':
        if syn or 'S\\*' in name or 'S*' in name: return 'A02-syn'
        if 'G1' in name or 'GenFact' in name: return 'A02-G1'
        if 'G2' in name or '验证' in name or '潜码' in name: return 'A02-G2'
        if 'G3' in name or '场景' in name: return 'A02-G3'
    if p == 'A03':
        if syn or '⁺' in name: return 'A03-syn'
        if 'RK' in name: return 'A03-B'
        if '结构语言' in name or '骨架' in name: return 'A03-C'
        if 'RAY-SHEAF' in name: return 'A03-A'
    if p == 'A04':
        if syn or 'EX-Sheaf' in name: return 'A04-syn'
        if 'ΔV' in name: return 'A04-1'
        if 'PV' in name: return 'A04-2'
        if 'PR' in name: return 'A04-3'
    if p == 'A05':
        if syn or '双层' in name: return 'A05-syn'
        if '码让位' in name: return 'A05-3'
        if '盒' in name or 'BoxTok' in name: return 'A05-2'
        if 'RAYFIELD' in name: return 'A05-1'
    if p == 'A06':
        if syn or 'EPR' in name: return 'A06-syn'
        if 'ProgSheaf' in name: return 'A06-1'
        if 'RaySheaf' in name: return 'A06-2'
        if 'RayField' in name: return 'A06-3'
    if p == 'A07':
        if syn or 'NX' in name: return 'A07-syn'
        if 'GELS' in name: return 'A07-D1'
        if '配准' in name or '图元' in name or 'D3' in name or 'D3' in prop: return 'A07-D3'
        if '噪声' in name or 'D2' in name or 'D2' in prop: return 'A07-D2'
        if 'D1' in name: return 'A07-D1'
    if p == 'A08':
        if syn or 'TRS' in name: return 'A08-syn'
        if 'RC-Sheaf' in name or 'D1' in name: return 'A08-D1'
        if 'Vocab' in name or 'D2' in name: return 'A08-D2'
        if 'D3' in name or '融合' in name or '潜空间' in name: return 'A08-D3'
    if p == 'A09':
        if syn or 'LRS' in name: return 'A09-syn'
        if 'W-Sheaf' in name: return 'A09-A'
        if '教师' in name: return 'A09-B'
        if '程序' in name: return 'A09-C'
    if p == 'A10':
        if syn or 'FACT' in name: return 'A10-syn'
        if 'α' in name or 'EVS' in name: return 'A10-α'
        if 'β' in name or '新息' in name: return 'A10-β'
        if 'γ' in name or '物体区间' in name: return 'A10-γ'
    return None


def num(s):
    d = re.sub(r'[^0-9]', '', s)
    return int(d) if d else None


votes = {}          # canon -> list of dict(voter, furn, comp, feas, fals, total, compliance)
unmapped = []
for path in sorted(glob.glob(os.path.join(DEB, 'A*_debate.md'))):
    voter = os.path.basename(path)[:3]
    text = open(path, encoding='utf-8').read()
    sec = text[text.rfind('投票'):] if '投票' in text else text
    # the vote table: the first markdown table after the last '投票' heading whose header has 8 columns incl. 总
    lines = sec.splitlines()
    in_tab = False
    for ln in lines:
        if ln.startswith('|') and ('总分' in ln or '| 总 |' in ln) and '设计' in ln:
            in_tab = True; continue
        if in_tab:
            if not ln.startswith('|'):
                if ln.strip() == '':
                    continue
                break
            cells = [c.strip() for c in ln.strip().strip('|').split('|')]
            if set(cells[0]) <= set('-: '):
                continue
            if len(cells) < 8:
                continue
            name = cells[0].replace('**', ''); prop = cells[1].replace('**', '')
            key = canon(name, prop)
            rec = dict(voter=voter, name=name, prop=prop, furn=num(cells[2]), comp=num(cells[3]), feas=num(cells[4]), compliance=cells[5],
                       fals=num(cells[6]), total=num(cells[7]))
            if key is None:
                unmapped.append(rec); continue
            votes.setdefault(key, []).append(rec)

rows = []
for key, v in votes.items():
    base = key.split('-')[0]
    tot = np.array([r['total'] for r in v], float)
    nonself = np.array([r['total'] for r in v if r['voter'] != base], float)
    comp_flags = sorted({r['compliance'] for r in v})
    rows.append(dict(key=key, name=NAMES.get(key, key), n=len(v), mean=float(tot.mean()), mean_nonself=float(nonself.mean()) if len(nonself) else None,
                     n_nonself=int(len(nonself)), min=float(tot.min()), max=float(tot.max()), std=float(tot.std()),
                     furn=float(np.mean([r['furn'] for r in v])), comp=float(np.mean([r['comp'] for r in v])), feas=float(np.mean([r['feas'] for r in v])),
                     fals=float(np.mean([r['fals'] for r in v])), cond=sum(('有条件' in r['compliance']) for r in v), voters=[r['voter'] for r in v]))
main = sorted([r for r in rows if not r['key'].endswith('syn')], key=lambda r: -(r['mean_nonself'] if r['mean_nonself'] is not None else r['mean']))
syn = sorted([r for r in rows if r['key'].endswith('syn')], key=lambda r: -r['mean'])
for i, r in enumerate(main, 1):
    r['rank'] = i
json.dump(dict(main=main, syntheses=syn, unmapped=unmapped), open(os.path.join(HERE, 'c2_votes.json'), 'w'), indent=1, ensure_ascii=False)
print(f"{'rk':>2} {'key':8s} {'n':>2} {'mean':>5} {'nonself':>7} {'min':>4} {'max':>4} {'std':>4}  furn comp feas fals  cond  name")
for r in main:
    print(f"{r['rank']:>2} {r['key']:8s} {r['n']:>2} {r['mean']:5.2f} {r['mean_nonself']:7.2f} {r['min']:4.0f} {r['max']:4.0f} {r['std']:4.2f}  {r['furn']:.1f}  {r['comp']:.1f}  {r['feas']:.1f}  {r['fals']:.1f}  {r['cond']:>3}  {r['name']}")
print('syntheses:')
for r in syn:
    print(f"   {r['key']:8s} {r['n']:>2} {r['mean']:5.2f}  voters {r['voters']}")
print('unmapped:', [(u['voter'], u['name'], u['prop']) for u in unmapped])
