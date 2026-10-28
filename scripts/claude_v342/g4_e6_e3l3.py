"""g4_e6_e3l3.py (Claude, 2026-10-28): 批次 G4-E6L "主档 cam 上，E3-L3 接在按岛组合位姿之后"（纯 CPU，0 个租约，冻结检查点，不重训）。
预注册 claude_handoff/OVERNIGHT_REPORT_20261021.md D8-70（逐条照做）；原文草案 review_E3L3/RI_accuracy.md §5.2，原型 review_E3L3/RI/ri_probe.py（job_e6）。
只 import 冻结代码，从不修改：g4_island_compose.py（G4-E6，sha b7438b98：e6_corrections / believed_positions / load_readouts / readouts /
pinned_e4_manifest / expected_jobs / 常数）、g4_registration.py（E4，e9574c7e：register_pair / evidence / planes_tern / _integrate / params）、
g4_noise_panels.py（场景、噪声、V3 积分、fuse、score）、g4_e3r.py（scored、net_readout）、g4_e3l3.py 及 g4_e3l2 / g4_e3l（E3-L3 编码、GaBP、
解码）；E3-L3 的加载与解码取自本批的 g4_h1_e3l3.py（load_e3l3 / e3l3_preds，与 E3-L3 正式评估逐行相同，sha 记入每个任务）。

面板（D8-70）：VAL + fresh × 噪声种子 1、2 × cam = G4-E6 的 96 个任务（OI.expected_jobs）。
  地图重建、E4 边估计（读 E4 冻结 job，不重跑匹配器）、按岛组合（OI.e6_corrections，只读白名单字段 + 相信位置）、整体对齐行（每分量 B4 融合信念
  对真值 register_pair 得 G_c，每机用 G_c · K_a 重新 V3 积分）全部照 g4_island_compose / E4 eval_arm。
  在同一组对齐后的地图上：B4 / B6_A / B6_B / NET_342101 / NET_342202（OI.readouts，冻结）与 E3L3_342101 / E3L3_342202（每台机器解码自己的
  GaBP 信念，学到的 w，逐跳取整；已知 = Lam_b > 0，逐体素断言 = k > 0）。
一致性闸门（D8-70）：NET / B6 的对齐行必须与 g4/e6/jobs 逐行相等（JSON 往返后逐值比较），否则 INCOMPLETE。
判据（D8-70，写死在 CRIT；判决行 = 整体对齐行，分量展开均值；VAL 4 格 = 训练种子 342101 / 342202 × 噪声种子 1 / 2）：
  非劣：VAL 4 格都满足 E3L3(e6) >= NET(e6) − 1.0 pp；
  组合有效：E3L3(e6) − E3L3(noreg) >= +3 pp，4 格都满足；
  保护：空闲召回 E3L3(e6) >= NET(e6) − 0.5 pp；每格伤害房（该房 E3L3(e6) 比 NET(e6) 低超过 2 pp）<= 3 / 24；
        fresh 4 格均值 E3L3(e6) − NET(e6) >= −1.0 pp，不满足只降级；
  结论：全过 → NONINFERIOR（"E3-L3 在主档非劣"）；任一 VAL 格 < −2 pp → NOT_NONINFERIOR（"不非劣"）；其余 AMBIGUOUS。
只报告（D8-70）：岛间损失（cam_nf 真位姿 − e6，对 E3-L3 与 NET 各自）；按岛数分组；配对池化 F1；有限 w 读法（--w，见 O9）。

操作化（D8-70 没写死、在这里写定；写于看任何正式读数之前）
  O1 每任务只做判决与只报告需要的行：cam 档 noreg、e6 两个臂的整体对齐行；cam_nf 档 noreg 臂的整体对齐行（= G4-E6 的 truepose 行，O13）。
     原样行、真值规范行、e4rule / ceiling 臂、harsh 档都不做（G4-E6 已报）。
  O2 地图重建逐行照 OI.run_task（公共随机数、to_public、V3、k0 缓存），断言 pub_sha256 / k0_sha256 与 E4 job 相同（不符 → 任务失败）。
     E4 job 文件 sha 必须等于 OI 钉住的清单条目（OI.pinned_e4_manifest，96 个）；G4-E6 job 文件 sha 必须等于本脚本钉住的 G4-E6 清单
     （E6_MANIFEST_SHA256，96 个文件 "名 sha256" 行的 sha256），且由 G4-E6 正式脚本（b7438b98）写成、未失败、未跳过。
  O3 G4-E6 的 K：OI.e6_corrections(n, E4 cam 边记录, OI.believed_positions(pub 扫描))；断言接受边岛与 G4-E6 job 的 islands 相同、每机 K 的
     R.params 与 job 记录逐值相同（O7 闸门的一部分）。两个臂 K 逐元素差 <= 1e-12 时复用读数（= OI.same_K，同 G4-E6 O18）。
  O4 对齐：逐分量 R.register_pair(真值平面, B4 融合信念平面)，与 OI.eval_arm6 的对齐段逐行相同；断言每分量 G 的 R.params 与 G4-E6 job 的
     align[ci].G 逐值相同（O7 闸门的一部分）。
  O5 E3-L3 读出 = g4_h1_e3l3.e3l3_preds（= g4_e3l3.eval_arm 的 R8 段：encode_maps → source 'round' → 每分量 gabp(w, 'round') → decode_np(chunk 2)
     → R.net_readout(logit, k, j_occ)），k = 该分量对齐后地图的观测计数（同 E3-L3 正式评估）；行 = [R.scored(p_i, R8T, H, k), 1]，每台机器一行。
     GaBP 核对（L.gbp_checks）默认开启（--no-gbp-check 关闭），失败 → INCOMPLETE。
  O6 计分与汇总：判决数 = 分量展开（每台机器一行）后 furn_f1 / free_recall 的均值（G.summarise_rows），与 G4-E6 / E3-L3 同口径。
     伤害房：该房 E3L3(e6) 的房均值（该房各行展开后、F1 有定义的行的均值）− NET(e6) 的房均值 < −2 pp；分母 = 该格任务数（24）。
     保护项（空闲召回、伤害房）只在 VAL 4 格判；fresh 只判"4 格均值 >= −1.0 pp"（不满足只降级），fresh 的保护读数只报告（同 G4-E6 O8）。
  O7 一致性闸门：每个任务、每个计算的行（cam noreg / cam e6 / cam_nf noreg 的整体对齐行），B4 / B6_A / B6_B / NET_342101 / NET_342202 的行
     与 G4-E6 job 的同一行 JSON 往返后逐值相等，且 O3 的岛与 K、O4 的 G 相等；任一不等 → 该任务记 e6_mismatch（附诊断），汇总判 INCOMPLETE。
  O8 完整性：预期 = VAL / fresh 每房 × 种子 1、2 中 E4 job 存在且未跳过者（OI.expected_jobs，正式模式断言 = 96）；缺失、失败、闸门不等、
     已知不一致 > 0、GaBP 核对失败、脚本 / 冻结模块 / 检查点 / 阈值 sha 混用 → INCOMPLETE。
  O9 --w（默认 learned）：数值 w 必须配 --w-thr（G4-E7 第 0 步为该 w 在 TRAIN 上重选的阈值，JSON 同 g4_h1_e3l3 O9）；有限 w 的运行只报告，
     verdict 固定为 "REPORT ONLY (finite w)"，判据数字照算、照列（would_be）。NET / B6 与闸门不受 w 影响。
  O10 只报告的行：
     岛间损失 = truepose（cam_nf noreg 对齐行）− e6（cam 对齐行），E3L3 / NET / B6 各自（pp）；
     按岛数分组 = 按该房 cam 接受边岛数（e6_corrections 的 islands 个数）分组的房均值差 E3L3(e6) − NET(e6)，以及组内房数；
     配对池化 F1 = 两臂在相同计分体素上合计 TP / FP / FN 的 F1（E3L3 每机一行，NET 分量行按智能体数展开，逐行配对）及其差；
       配对逐行均值 = 只在两臂 F1 都有定义的行上取均值；
     另报 noreg 对齐行上的 E3L3 − NET、cam_nf 对齐行上的 E3L3 − NET。

资源（冒烟实测见回复）：每任务单线程约 5–7 分钟（含 2 次 NET 与 E3-L3 各 3 行）；峰值约 3 GB / 进程。
用法（总是 python -B；CPU；本机 <= 10 进程）：
  python -B g4_e6_e3l3.py --out DIR --jobs val_r1_i0_s1,val_r2_i3_s1 --workers 2       # 冒烟（只用 R-I 看过的 VAL 种子 1 任务）
  python -B g4_e6_e3l3.py --out DIR --formal --workers 8                                 # 正式（96 个任务）
  python -B g4_e6_e3l3.py --out DIR --aggregate-only [--formal]
  可选：--w 4 --w-thr THR.json（只报告）、--no-gbp-check、--threads 1
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True

import argparse, glob, hashlib, json, math, multiprocessing, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_island_compose as OI  # noqa: E402   (frozen G4-E6; imports g4_e3r, g4_registration)
import g4_h1_e3l3 as HL  # noqa: E402          (this batch: E3-L3 loading / decoding identical to the formal E3-L3 eval)

E = OI.E            # g4_e3r
REG = OI.R          # g4_registration
G = OI.G
W = OI.W
L = HL.L
assert HL.R is E and HL.G is G, 'g4_island_compose and g4_e3l3 must share g4_e3r / g4_noise_panels'
torch = E.torch
UNK = E.UNK
I3 = OI.I3

A27 = OI.A27
E6_JOBS = os.path.join(A27, 'g4', 'e6', 'jobs')
E6_SCRIPT_SHA256 = 'b7438b987c15ff4a4cc4d9c204763a11dd49d74cb18cb9aae793936b4917c8d8'
E6_MANIFEST_SHA256 = '4c98930c7ff8c3a790bfdcc3b82d6f6487640c952496f2dd82ee9c9bf6d1b03e'   # sha256 of the sorted "name sha256" lines of g4/e6/jobs
FROZEN = dict(HL.FROZEN, g4_island_compose=E6_SCRIPT_SHA256, g4_registration=OI.FROZEN['g4_registration'])
OUT_DIR = os.path.join(A27, 'g4', 'e6l')
SEEDS = HL.SEEDS
PAIR = HL.PAIR
NETS = tuple(f'NET_{ts}' for ts in SEEDS)
E3S = tuple(f'E3L3_{ts}' for ts in SEEDS)
CONTROLS = OI.READOUTS                              # B4, B6_A, B6_B, NET_342101, NET_342202
GATED = ('B4', 'B6_A', 'B6_B', 'NET_342101', 'NET_342202')
PLAN = (('cam_nf', 'noreg'), ('cam', 'noreg'), ('cam', 'e6'))
JUDGED_COND, ROW = 'cam', 'aligned'
PANELS, NOISE_SEEDS = OI.PANELS, OI.SEEDS
N_EXPECTED = OI.N_EXPECTED
# D8-70 criteria (pp), written before the run
CRIT = dict(noninf_pp=-1.0, comp_eff_pp=3.0, free_guard_pp=-0.5, harm_pp=-2.0, harm_max=3, fresh_mean_pp=-1.0, not_noninf_pp=-2.0)
JOB_FORMAT = 1


def sha(p):
    return G.file_sha256(p)


def _jload(p):
    with open(p) as fh:
        return json.load(fh)


def _jr(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


def _check(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def frozen_files():
    d = HL.frozen_files()
    d.update(g4_island_compose=OI.__file__, g4_registration=REG.__file__)
    return d


def check_frozen():
    for nm, p in frozen_files().items():
        s = sha(p)
        _check(s == FROZEN[nm], f'{nm}.py sha {s[:12]} != frozen {FROZEN[nm][:12]}')


def e6_manifest(jobs_dir=E6_JOBS):
    m = {os.path.basename(p)[:-5]: sha(p) for p in sorted(glob.glob(os.path.join(jobs_dir, '*.json')))}
    txt = '\n'.join(f'{k} {m[k]}' for k in sorted(m))
    return m, hashlib.sha256(txt.encode()).hexdigest()


def pinned_e6_manifest():
    m, s = e6_manifest()
    _check(s == E6_MANIFEST_SHA256, f'G4-E6 job directory manifest sha {s} != pinned {E6_MANIFEST_SHA256} ({len(m)} files)')
    return m


# ================================================================ models (per worker)
_M = None


def load_models(threads, w, w_thr_path):
    global _M
    if _M is None:
        RO = OI.load_readouts()                                         # B6_A / B6_B / NET (frozen, sha-pinned inside OI)
        w_thr = HL.parse_w_thr(w_thr_path) if w_thr_path else None
        e3 = HL.load_e3l3(threads, None if w == 'learned' else float(w), w_thr)
        for ts, m in e3.items():
            _check(np.array_equal(m['table'], RO['b6'][f'B6_{PAIR[ts]}'].table), f'E3L3_{ts} evidence table != B6_{PAIR[ts]}')
        _M = dict(RO=RO, e3=e3)
    return _M


# ================================================================ one task
def e3l3_rows(m, maps, F, R8T, H, gbp_check=True):
    """O5: k = per-component observation count of these maps (as the formal E3-L3 eval); every member decodes its own belief;
    rows [R.scored(p_i, R8T, H, k), 1], one per agent in component order"""
    kcs = [sum((maps[a] != UNK).astype(np.int32) for a in cm) for cm in F['comps']]
    ep, _, _, info = HL.e3l3_preds(m, maps, F, kcs, gbp_check)
    rws = [[E.scored(p, R8T[ci], H, kcs[ci]), 1] for ci in range(len(F['comps'])) for p in ep[ci]]
    return rws, dict(known_mismatch=info['known_mismatch'], gbp=info['gbp'], w=info['w'], w_label=info['w_label'], j_occ=info['j_occ'])


def _gate_rows(mine, theirs):
    """O7: {readout: equal?} of my aligned rows vs the G4-E6 job's (JSON round trip), plus S2-style diagnostics when unequal"""
    eq, diag = {}, {}
    for ro in GATED:
        t = (theirs or {}).get(ro)
        eq[ro] = t is not None and _jr(mine[ro]) == t
        if not eq[ro] and t is not None:
            diag[ro] = OI.rows_diff(mine[ro], t)
    return eq, diag


def run_task(job):
    panel, rooms, index, seed, opts = job
    _check(__debug__, 'python -O refused')
    _check(sha(os.path.abspath(__file__)) == opts['script_sha256'], 'g4_e6_e3l3.py changed since the run started')
    _check(sha(HL.__file__) == opts['hl_sha256'], 'g4_h1_e3l3.py changed since the run started')
    check_frozen()
    _check(index < 16, 'only VAL / fresh')
    torch.set_num_threads(max(1, int(opts['threads'])))
    name = OI.job_name(panel, rooms, index, seed); t_start = time.time(); timing = {}
    e4p = os.path.join(OI.E4_JOBS, name + '.json'); e6p = os.path.join(E6_JOBS, name + '.json')
    e4_sha, e6_sha = sha(e4p), sha(e6p)
    _check(e4_sha == opts['e4_sha256'][name], f'E4 job {name} sha != the pinned E4 manifest entry')
    _check(e6_sha == opts['e6_sha256'][name], f'G4-E6 job {name} sha != the pinned G4-E6 manifest entry')
    e4 = _jload(e4p); e6j = _jload(e6p)
    out = dict(panel=panel, house=f'r{rooms}_i{index}', rooms=rooms, index=index, seed=seed, job=name, job_format=JOB_FORMAT,
               script_sha256=opts['script_sha256'], hl_sha256=opts['hl_sha256'], frozen=FROZEN, w=opts['w'], w_thr_sha256=opts.get('w_thr_sha256'),
               gbp_check=opts['gbp_check'], e4_job_sha256=e4_sha, e6_job_sha256=e6_sha, conds={}, gate=dict(ok=True, problems=[]))
    if e4.get('skipped'):
        out['skipped'] = 'E4 job skipped'; return out
    _check(not e4.get('failed'), 'E4 job is a failed record')
    _check(e6j.get('script_sha256') == E6_SCRIPT_SHA256 and not e6j.get('failed') and not e6j.get('skipped'), 'G4-E6 job not a completed b7438b98 record')
    prm = e4['fingerprint']['match']
    _check(prm == json.loads(json.dumps(REG.MATCH)), 'E4 job matcher settings != frozen MATCH')
    M = load_models(opts['threads'], opts['w'], opts.get('w_thr'))
    RO = M['RO']; out['readouts'] = RO['info']
    out['e3l3'] = {ts: dict(ckpt_sha256=m['ckpt_sha256'], thr_sha256=m['thr_sha256'], w=m['w'], w_label=m['w_label'], j_occ=m['jo']) for ts, m in M['e3'].items()}
    fx = W.make_fixture('VALIDATION', index, rooms); timing['fixture'] = time.time() - t_start
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    H = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n)
    _check(W.array_hash(private) == e4['private_sha256'] and fx['world_id'] == e4['world_id'], 'fixture != E4 job')
    _check([list(c) for c in F['comps']] == e4['components'] == e6j['components'], 'components != E4 / G4-E6 job')
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool)
    T8 = G.stream_targets(private, reached8)
    R8T = [G.Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    rays = W.fibonacci_sphere(W.BEAMS)
    t0 = time.time()
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    timing['trace'] = time.time() - t0
    rstats = G.RelayStats(); k0cache = {}
    out.update(agents=n, components=F['comps'])
    for cname in ('cam_nf', 'cam'):
        c = G.CONDITIONS[cname]; e4c = e4['conds'][cname]; e6c = e6j['conds'][cname]; tc = time.time()
        has_inter = c['inter_t'] > 0 or c['inter_yaw'] > 0; ck = G._ck(c)
        scans_pub, L_pub = [], []
        for a in range(n):                                              # O2: = OI.run_task
            frame = None
            if has_inter:
                z = G.frame_draws(seed, rooms, index, a)
                frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
            scans = [G.measure(c, *traces[a][j], traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)[0] for j in range(traj.shape[0])]
            pub = [G.to_public(s, frame) for s in scans]
            if (ck, a) not in k0cache:
                k0cache[(ck, a)] = G.integrate_agent(shape, cell, ds, scans, ('V3',))['V3']
            L_pub.append(G.integrate_agent(shape, cell, ds, pub, ('V3',))['V3'] if has_inter else k0cache[(ck, a)])
            scans_pub.append(pub)
        rec = dict(pub_sha256=W.array_hash(np.stack(L_pub)), k0_sha256=W.array_hash(np.stack([k0cache[(ck, a)] for a in range(n)])))
        _check(rec['pub_sha256'] == e4c['pub_sha256'] and rec['pub_sha256'] == e6c['pub_sha256'], f'{cname}: rebuilt public-frame maps != E4 / G4-E6')
        _check(rec['k0_sha256'] == e4c['k0_sha256'], f'{cname}: rebuilt K0 maps != E4 k0_sha256')
        timing[f'{cname}_integrate'] = time.time() - tc
        Ks = dict(noreg=[I3.copy() for _ in range(n)])
        if cname == JUDGED_COND:                                        # O3
            bpos = OI.believed_positions(scans_pub)
            Ks['e6'], _, isl = OI.e6_corrections(n, e4c['registration']['edges'], bpos)
            rec['islands'] = isl; rec['n_islands'] = len(isl)
            if _jr(isl) != e6c['e6']['islands']:
                out['gate']['problems'].append(f'{cname}: islands != G4-E6 job')
            if _jr([REG.params(x) for x in Ks['e6']]) != e6c['arms']['e6']['K']:
                out['gate']['problems'].append(f'{cname}: e6 K != G4-E6 job')
        arms, done = {}, []
        for arm in [a for cn, a in PLAN if cn == cname]:
            K = Ks[arm]; ta = time.time()
            prev = next((nm for nm, Kp in done if OI.same_K(K, Kp)), None)
            if prev is not None:                                        # O3 reuse; the gate still compares this arm's G4-E6 rows
                arms[arm] = dict(arms[prev], reused_from=prev); done.append((arm, K))
                eq, _ = _gate_rows(arms[prev]['aligned'], e6c['arms'][arm].get(ROW))
                if not all(eq.values()) or _jr(arms[prev]['G']) != _jr([x['G'] for x in e6c['arms'][arm]['align']]):
                    out['gate']['problems'].append(f'{cname}/{arm} (reused from {prev}): rows / G != G4-E6 job')
                continue
            maps0 = [L_pub[a] if np.array_equal(K[a], I3) else REG._integrate(shape, cell, ds, scans_pub[a], K[a]) for a in range(n)]
            _, preds = G.fuse(maps0, F, 'sum', rstats)                  # O4: = OI.eval_arm6's alignment
            al = [None] * n; Gs = []
            for ci, cm in enumerate(F['comps']):
                ev_t = REG.evidence(REG.planes_tern(T8[cm[0]], cell, prm), cell, prm)
                ev_b = REG.evidence(REG.planes_tern(preds[cm[0]], cell, prm), cell, prm)
                res = REG.register_pair(ev_t, ev_b, cell, prm); Gc = res['M']; Gs.append(REG.params(Gc))
                for a in cm:
                    al[a] = maps0[a] if np.array_equal(Gc, I3) else REG._integrate(shape, cell, ds, scans_pub[a], Gc @ K[a])
            del preds, maps0
            e6arm = e6c['arms'][arm]
            if _jr(Gs) != _jr([x['G'] for x in e6arm['align']]):
                out['gate']['problems'].append(f'{cname}/{arm}: alignment G != G4-E6 job')
            ta2 = time.time()
            rows = OI.readouts(al, F, R8T, H, RO, rstats)               # frozen B4 / B6 / NET on the aligned maps
            t_ctrl = time.time() - ta2
            eq, diag = _gate_rows(rows, e6arm.get(ROW))
            if not all(eq.values()):
                out['gate']['problems'].append(f'{cname}/{arm}: aligned rows != G4-E6 job for {[k for k, v in eq.items() if not v]}')
            e3info = {}
            ta3 = time.time()
            for ts in SEEDS:
                rows[f'E3L3_{ts}'], e3info[f'E3L3_{ts}'] = e3l3_rows(M['e3'][ts], al, F, R8T, H, opts['gbp_check'])
            del al
            arms[arm] = dict(G=Gs, aligned=rows, e3=e3info, gate_rows_equal=eq, gate_diag=diag,
                             seconds=dict(total=time.time() - ta, controls=t_ctrl, e3l3=time.time() - ta3))
            done.append((arm, K))
            timing[f'{cname}_{arm}'] = time.time() - ta
        rec['arms'] = arms
        out['conds'][cname] = rec
        timing[f'{cname}_total'] = time.time() - tc
        del scans_pub, L_pub
        if cname == 'cam_nf':
            k0cache = {k: v for k, v in k0cache.items() if k[0] == G._ck(G.CONDITIONS['cam'])}
    out['gate']['ok'] = not out['gate']['problems']
    out['relay'] = dict(combines=rstats.combines, belief_checks=rstats.belief_checks)
    timing['total'] = time.time() - t_start
    out['timing'] = timing; out['peak_mb'] = G._peak_mb()
    return out


def _run_and_save(job):
    panel, rooms, index, seed, opts = job
    name = OI.job_name(panel, rooms, index, seed)
    try:
        res = run_task(job)
    except Exception:  # noqa: BLE001 (recorded; aggregation -> INCOMPLETE)
        res = dict(panel=panel, house=f'r{rooms}_i{index}', rooms=rooms, index=index, seed=seed, job=name, failed=traceback.format_exc()[-3000:],
                   script_sha256=opts['script_sha256'], w=opts['w'])
    path = os.path.join(opts['out'], 'jobs', name + '.json'); tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(res, fh, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(tmp, path)
    return name, (res.get('timing') or {}).get('total'), res.get('peak_mb'), res.get('skipped'), bool(res.get('failed')), (res.get('gate') or {}).get('ok')


def _worker_init(threads):
    sys.dont_write_bytecode = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    G.setup_env()
    torch.set_num_threads(max(1, int(threads)))


# ================================================================ aggregation and verdict (D8-70)
def _rows(job, cond, arm, ro):
    a = job['conds'][cond]['arms'][arm]
    return a['aligned'][ro]


def _mean(rows, key):
    return G.summarise_rows(rows, keys=(key,))[key]


def _pp(x, y):
    return None if (x is None or y is None) else 100. * (x - y)


def _house_mean(comp_rows, key='furn_f1'):
    xs = [r[key] for r in G._expand(comp_rows) if r.get(key) is not None]
    return float(np.mean(xs)) if xs else None


def paired(rows_a, rows_b):
    """O10: paired pooled F1 (sum TP / FP / FN over identical scored voxels, row by row) and the paired row mean (rows where both F1 defined)"""
    _check(len(rows_a) == len(rows_b), 'paired rows: different row counts')
    out = {}
    for nm, rows in (('a', rows_a), ('b', rows_b)):
        tp = sum(r['furn_tp'] for r in rows); fp = sum(r['furn_fp'] for r in rows); fn = sum(r['furn_fn'] for r in rows)
        out[nm] = dict(tp=tp, fp=fp, fn=fn, f1=(2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else None)
    both = [(x['furn_f1'], y['furn_f1']) for x, y in zip(rows_a, rows_b) if x.get('furn_f1') is not None and y.get('furn_f1') is not None]
    out['pooled_d_pp'] = _pp(out['a']['f1'], out['b']['f1'])
    out['paired_row_mean_a'] = float(np.mean([x for x, _ in both])) if both else None
    out['paired_row_mean_b'] = float(np.mean([y for _, y in both])) if both else None
    out['paired_row_d_pp'] = _pp(out['paired_row_mean_a'], out['paired_row_mean_b'])
    out['n_rows'] = len(rows_a); out['n_rows_both_defined'] = len(both)
    return out


def cell_stats(J, ts):
    """one (panel, noise seed, checkpoint) cell on the judged row (cam, aligned), plus the report-only rows"""
    e3, net, b6 = f'E3L3_{ts}', f'NET_{ts}', f'B6_{PAIR[ts]}'
    ex = lambda j, cond, arm, ro: G._expand(_rows(j, cond, arm, ro))       # noqa: E731
    R = {}
    for cond, arm in PLAN:
        for ro in (e3, net, b6):
            rows = [r for j in J for r in ex(j, cond, arm, ro)]
            R[(cond, arm, ro)] = dict(furn=_mean(rows, 'furn_f1'), free=_mean(rows, 'free_recall'), rows=rows)
    e3e6, nete6 = R[('cam', 'e6', e3)], R[('cam', 'e6', net)]
    per = {}
    for j in J:
        per[j['job']] = _pp(_house_mean(_rows(j, 'cam', 'e6', e3)), _house_mean(_rows(j, 'cam', 'e6', net)))
    harm = sorted(k for k, v in per.items() if v is not None and v < CRIT['harm_pp'])
    c = dict(e3_e6=e3e6['furn'], net_e6=nete6['furn'], d_pp=_pp(e3e6['furn'], nete6['furn']),
             e3_noreg=R[('cam', 'noreg', e3)]['furn'], comp_eff_pp=_pp(e3e6['furn'], R[('cam', 'noreg', e3)]['furn']),
             free_e3=e3e6['free'], free_net=nete6['free'], free_d_pp=_pp(e3e6['free'], nete6['free']),
             n_harm=len(harm), harm_houses=harm, per_house_pp=per, n_houses=len(J), n_rows=len(e3e6['rows']))
    # report only (O10)
    tp = {ro: R[('cam_nf', 'noreg', ro)]['furn'] for ro in (e3, net, b6)}
    c['island_loss_pp'] = {ro: _pp(tp[ro], R[('cam', 'e6', ro)]['furn']) for ro in (e3, net, b6)}
    c['truepose_furn'] = tp
    c['net_comp_eff_pp'] = _pp(nete6['furn'], R[('cam', 'noreg', net)]['furn'])
    c['b6_e6'] = R[('cam', 'e6', b6)]['furn']
    c['d_noreg_pp'] = _pp(R[('cam', 'noreg', e3)]['furn'], R[('cam', 'noreg', net)]['furn'])
    c['d_cam_nf_pp'] = _pp(R[('cam_nf', 'noreg', e3)]['furn'], R[('cam_nf', 'noreg', net)]['furn'])
    c['paired_e6'] = paired(e3e6['rows'], nete6['rows'])
    c['paired_noreg'] = paired(R[('cam', 'noreg', e3)]['rows'], R[('cam', 'noreg', net)]['rows'])
    groups = {}
    for j in J:
        k = j['conds']['cam']['n_islands']; d = per[j['job']]
        g = groups.setdefault(str(k), dict(houses=0, d_pp=[]))
        g['houses'] += 1
        if d is not None:
            g['d_pp'].append(d)
    c['by_islands'] = {k: dict(houses=g['houses'], mean_d_pp=(float(np.mean(g['d_pp'])) if g['d_pp'] else None)) for k, g in sorted(groups.items(), key=lambda x: int(x[0]))}
    return c


def verdict(cells, reasons, finite_w=False):
    """D8-70, written before the run"""
    if reasons:
        return dict(verdict='INCOMPLETE', reasons=reasons)
    val = {f'{ts}|{s}': cells['val'][str(s)][ts] for ts in SEEDS for s in NOISE_SEEDS}
    fr = {f'{ts}|{s}': cells['fresh'][str(s)][ts] for ts in SEEDS for s in NOISE_SEEDS}
    if any(c[k] is None for c in val.values() for k in ('d_pp', 'comp_eff_pp', 'free_d_pp')) or any(c['d_pp'] is None for c in fr.values()):
        return dict(verdict='INCOMPLETE', reasons=['undefined cell'])
    fmean = float(np.mean([c['d_pp'] for c in fr.values()]))
    checks = dict(noninf=all(c['d_pp'] >= CRIT['noninf_pp'] - 1e-9 for c in val.values()),
                  comp_eff=all(c['comp_eff_pp'] >= CRIT['comp_eff_pp'] - 1e-9 for c in val.values()),
                  free_guard=all(c['free_d_pp'] >= CRIT['free_guard_pp'] - 1e-9 for c in val.values()),
                  harm_guard=all(c['n_harm'] <= CRIT['harm_max'] for c in val.values()))
    fresh_ok = fmean >= CRIT['fresh_mean_pp'] - 1e-9
    not_noninf = any(c['d_pp'] < CRIT['not_noninf_pp'] for c in val.values())
    notes = []
    if all(checks.values()):
        v = 'NONINFERIOR' if fresh_ok else 'AMBIGUOUS'
        if not fresh_ok:
            notes.append(f'NONINFERIOR demoted: fresh 4-cell mean {fmean:+.2f} pp < {CRIT["fresh_mean_pp"]} pp')
    elif not_noninf:
        v = 'NOT_NONINFERIOR'
    else:
        v = 'AMBIGUOUS'
    out = dict(verdict=v, checks=checks, fresh_ok=fresh_ok, fresh_mean_pp=fmean, notes=notes,
               val_d_pp={k: c['d_pp'] for k, c in val.items()}, val_comp_eff_pp={k: c['comp_eff_pp'] for k, c in val.items()},
               val_free_d_pp={k: c['free_d_pp'] for k, c in val.items()}, val_n_harm={k: c['n_harm'] for k, c in val.items()},
               val_harm_houses={k: c['harm_houses'] for k, c in val.items()}, fresh_d_pp={k: c['d_pp'] for k, c in fr.items()},
               fresh_guards_report_only=dict(note='只报告 / report only (O6)', free_d_pp={k: c['free_d_pp'] for k, c in fr.items()},
                                             n_harm={k: c['n_harm'] for k, c in fr.items()}, comp_eff_pp={k: c['comp_eff_pp'] for k, c in fr.items()}))
    if finite_w:
        out['would_be'] = out['verdict']; out['verdict'] = 'REPORT ONLY (finite w)'
    return out


def aggregate(out_dir, panels=PANELS, seeds=NOISE_SEEDS, smoke=False, only=None):
    recs = {}
    for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json'))):
        r = _jload(p); recs[r.get('job') or os.path.basename(p)[:-5]] = r
    if only:
        exp = list(only)
    else:
        exp, _, missing_e4 = OI.expected_jobs(panels, seeds)
    S = dict(expected=len(exp), failed=[], missing=[], skipped=[], gate_failed={}, known_mismatch={}, gbp_failed=[], versions={}, panels={})
    reasons = []
    for k in ('script_sha256', 'hl_sha256', 'frozen', 'w', 'w_thr_sha256', 'gbp_check', 'readouts', 'e3l3'):
        S['versions'][k] = sorted({json.dumps(r.get(k), sort_keys=True) for r in recs.values() if not r.get('failed')})
        if len(S['versions'][k]) > 1:
            reasons.append(f'mixed {k} across task files')
    good = {}
    for nm in exp:
        r = recs.get(nm)
        if r is None:
            S['missing'].append(nm)
        elif r.get('failed'):
            S['failed'].append(nm)
        elif r.get('skipped'):
            S['skipped'].append(nm)
        else:
            if not r['gate']['ok']:
                S['gate_failed'][nm] = r['gate']['problems']
            for cond, rec in r['conds'].items():
                for arm, a in rec['arms'].items():
                    for e3, info in a['e3'].items():
                        if info['known_mismatch']:
                            S['known_mismatch'][f'{nm}:{cond}/{arm}/{e3}'] = info['known_mismatch']
                        if not info['gbp']['pass']:
                            S['gbp_failed'].append(f'{nm}:{cond}/{arm}/{e3}')
            good[nm] = r
    if smoke:
        reasons.append('SMOKE: not a verdict (subset of tasks)')
    if not only and len(exp) != N_EXPECTED:
        reasons.append(f'expected {N_EXPECTED} tasks, the E4 directory gives {len(exp)}')
    for k in ('missing', 'failed', 'skipped', 'gate_failed', 'known_mismatch', 'gbp_failed'):
        if S.get(k):
            reasons.append(f'{len(S[k])} {k}: {list(S[k])[:6]}')
    try:
        OI.pinned_e4_manifest(); pinned_e6_manifest()
    except RuntimeError as e:
        reasons.append(str(e)[:300])
    cells = {}
    for panel in panels:
        cells[panel] = {}
        for seed in seeds:
            J = [good[nm] for nm in exp if nm in good and OI.parse_name(nm)[0] == panel and OI.parse_name(nm)[3] == seed]
            if not J:
                continue
            cells[panel][str(seed)] = {ts: cell_stats(J, ts) for ts in SEEDS}
            S['panels'].setdefault(panel, {})[str(seed)] = dict(houses=[j['job'] for j in J],
                                                                seconds_median=float(np.median([j['timing']['total'] for j in J])),
                                                                n_islands_mean=float(np.mean([j['conds']['cam']['n_islands'] for j in J])))
    S['cells'] = cells
    complete = all(str(s) in cells.get(p, {}) for p in PANELS for s in NOISE_SEEDS)
    if not complete and not reasons:
        reasons.append('a judged panel / seed has no tasks')
    finite = any(json.loads(w) != 'learned' for w in S['versions'].get('w', []))
    S['verdict'] = verdict(cells, reasons, finite) if complete else dict(verdict='INCOMPLETE', reasons=reasons)
    S['criteria'] = dict(CRIT, judged_row='cam aligned (component-expanded mean)', prereg='D8-70')
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(S, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return S


def _f(x, fmt):
    return 'None' if x is None else format(x, fmt)


def report(S):
    L_ = [f"expected {S['expected']}, failed {len(S['failed'])}, missing {len(S['missing'])}, gate failed {len(S['gate_failed'])}, "
          f"known mismatch {len(S['known_mismatch'])}, GaBP failed {len(S['gbp_failed'])}; versions w {S['versions'].get('w')}"]
    for panel, P in S['cells'].items():
        for seed, cs in P.items():
            for ts, c in cs.items():
                L_.append(f"{panel} s{seed} {ts}: E3L3(e6) {_f(c['e3_e6'], '.4f')} vs NET(e6) {_f(c['net_e6'], '.4f')} d {_f(c['d_pp'], '+.2f')} pp | "
                          f"E3L3 e6 - noreg {_f(c['comp_eff_pp'], '+.2f')} pp (NET {_f(c['net_comp_eff_pp'], '+.2f')}) | free {_f(c['free_d_pp'], '+.2f')} pp | "
                          f"harm {c['n_harm']}/{c['n_houses']} {c['harm_houses']}")
                L_.append(f"    report only: island loss (truepose - e6) E3L3 {_f(c['island_loss_pp'][f'E3L3_{ts}'], '.2f')} NET {_f(c['island_loss_pp'][f'NET_{ts}'], '.2f')} "
                          f"B6 {_f(c['island_loss_pp'][f'B6_{PAIR[ts]}'], '.2f')} pp; paired pooled d {_f(c['paired_e6']['pooled_d_pp'], '+.2f')} pp "
                          f"(E3L3 {_f(c['paired_e6']['a']['f1'], '.4f')} NET {_f(c['paired_e6']['b']['f1'], '.4f')}), paired row d {_f(c['paired_e6']['paired_row_d_pp'], '+.2f')}; "
                          f"noreg d {_f(c['d_noreg_pp'], '+.2f')}, cam_nf d {_f(c['d_cam_nf_pp'], '+.2f')}")
                L_.append('    by islands: ' + ', '.join(f"{k}: {g['houses']} houses {_f(g['mean_d_pp'], '+.2f')} pp" for k, g in c['by_islands'].items()))
    L_.append(f"VERDICT: {json.dumps(S['verdict'], default=str)[:2500]}")
    return '\n'.join(L_)


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=''); ap.add_argument('--jobs', default='', help='explicit task names (smoke), e.g. val_r1_i0_s1,val_r2_i3_s1')
    ap.add_argument('--formal', action='store_true'); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--aggregate-only', action='store_true'); ap.add_argument('--w', default='learned'); ap.add_argument('--w-thr', default='')
    ap.add_argument('--no-gbp-check', action='store_true')
    a = ap.parse_args(argv)
    _check(__debug__, 'python -O refused')
    _check(a.workers <= 10, 'at most 10 processes on this machine')
    only = [x for x in a.jobs.split(',') if x] or None
    _check(bool(only) != bool(a.formal), 'exactly one of --formal (96 tasks) / --jobs (smoke)')
    if a.w != 'learned':
        float(a.w); _check(bool(a.w_thr) and os.path.exists(a.w_thr), '--w needs --w-thr (O9)')
    else:
        _check(not a.w_thr, '--w-thr only with a finite --w')
    out = a.out or (OUT_DIR + ('' if a.w == 'learned' else f'_w{a.w}'))
    os.makedirs(os.path.join(out, 'jobs'), exist_ok=True)
    if not a.aggregate_only:
        check_frozen()
        opts = dict(out=out, threads=a.threads, script_sha256=sha(os.path.abspath(__file__)), hl_sha256=sha(HL.__file__), w=a.w,
                    w_thr=(os.path.abspath(a.w_thr) if a.w_thr else None), w_thr_sha256=(sha(a.w_thr) if a.w_thr else None),
                    gbp_check=not a.no_gbp_check, e4_sha256=OI.pinned_e4_manifest(), e6_sha256=pinned_e6_manifest())
        names = only if only else OI.expected_jobs(PANELS, NOISE_SEEDS)[0]
        if a.formal:
            _check(len(names) == N_EXPECTED, f'expected {N_EXPECTED} tasks, found {len(names)}')
        for nm in names:
            _check(nm.split('_')[0] in PANELS and nm in opts['e6_sha256'], f'{nm}: not a G4-E6 VAL / fresh task')
        todo = []
        for nm in names:
            p = os.path.join(out, 'jobs', nm + '.json')
            if os.path.exists(p):
                r = _jload(p)
                if not r.get('failed') and r.get('script_sha256') == opts['script_sha256'] and r.get('w') == a.w:
                    continue
            panel, rooms, index, seed = OI.parse_name(nm)
            todo.append((panel, rooms, index, seed, opts))
        run = dict(script='g4_e6_e3l3.py', script_sha256=opts['script_sha256'], hl_sha256=opts['hl_sha256'], frozen=FROZEN, args=vars(a),
                   n_todo=len(todo), start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        with open(os.path.join(out, 'run.json'), 'w') as fh:
            json.dump(run, fh, indent=1)
        print(f'{len(todo)} tasks ({len(names) - len(todo)} done), {a.workers} workers x {a.threads} threads, w {a.w} -> {out}', flush=True)
        t0 = time.time()
        if a.workers <= 1:
            _worker_init(a.threads)
            for jb in todo:
                print(*_run_and_save(jb), f'{time.time() - t0:.0f}s', flush=True)
        else:
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(a.workers, initializer=_worker_init, initargs=(a.threads,), maxtasksperchild=4) as pool:
                for res in pool.imap_unordered(_run_and_save, todo):
                    print(*res, f'{time.time() - t0:.0f}s', flush=True)
        run.update(end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), wall_seconds=time.time() - t0)
        with open(os.path.join(out, 'run.json'), 'w') as fh:
            json.dump(run, fh, indent=1)
    S = aggregate(out, smoke=bool(only), only=only)
    txt = report(S)
    open(os.path.join(out, 'report.txt'), 'w', encoding='utf-8').write(txt); print(txt)
    return S


if __name__ == '__main__':
    main()
