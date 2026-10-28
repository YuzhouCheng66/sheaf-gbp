"""g4_e3l4.py（Claude，2026-10-28）：批次 E3-L4"在固定的有限 w* 下训练学习隐码 sheaf GBP"（预注册 OVERNIGHT_REPORT D8-68；
uk-linux RTX 5090，一个 7,200 s 租约，两个种子顺序跑，每个 <= 3,000 s）。
薄包装：只 import 冻结的 g4_e3l3.py（E3-L3，sha c7523cc8...；经它再 import g4_e3l2 / g4_e3l / g4_e3r / g4_noise_panels），不改任何文件。

相对 E3-L3 的唯一变量（D8-68）：w 固定为 w*，omega = log w* 冻结（requires_grad = False，不进优化），训练与评估都用。
其余逐项复用 g4_e3l3：编码器 / 解码器（R.UNet3，宽 16，skip=False）、三级字母表 z = round(tanh u)（直通）、Lam = 足迹、
已知 = Lam_b > 0、逐跳步长 1 取整（训练 'ste'，评估 'round'）、O7 线路字节、数据（cache-e3r-v1，186 / 19 房，cam_nf 200-202，
clean 0.1）、连通子树采样、BCE（scope & Lam_b > 0）、AdamW / 预热 300 + 余弦 / 裁剪 1.0 / batch 8、探针（每 1,000 次）、
第 6,000 次快照、阈值规则（P10）、评估（P11）、判决数值（P12 = D8-63）。

操作化（D8-68 没写死、由本实现写死的部分；回复里逐条列出）
  Q1 模型：E3L4 = g4_e3l3.E3L3 的子类，构造后 log_w = log w*（float32），requires_grad_(False)。configure(w*) 在本进程里把
     L3.E3L3 / L3.W0 / L3.ARM 换成 E3L4 / w* / 'E3L4'，于是 L3.train 构造的模型、recipe['w0']、检查点、L3.load_model 都是 E3-L4 的。
     构造顺序（enc、dec、log_w）与 torch.manual_seed(seed) 不变；log_w 的构造不消耗随机数，所以同种子下编解码器初值与 E3-L3 逐位
     相同（单测）。优化器参数组照 L3 原样：log_w 仍在无权重衰减组里，但 grad 恒为 None，AdamW 跳过它，clip_grad_norm_ 也忽略它。
     CPU float32 下 exp(float32(log w)) 对网格值 3 / 4 / 6 / 8 恰好等于 w（单测）；训练后、select 前、eval 前都断言检查点的
     exp(log_w) == w*（非网格的 --smoke 值只要求相对差 <= 1e-6）。
  Q2 w* 由 --w-star 给出，必须在 G4-E7 第 0 步的网格 {3, 4, 6, 8} 里（--smoke 例外，但仍须 > 1：w <= 1 时 Lam 消息 w / (w + 1)
     取整为 0，已知掩码被破坏，E3-L3 审查 §2.3）。链里由协调者钉死（W_STAR=FILL_W_STAR）。
  Q3 更新数：--prereg-updates 9562（D8-68 = E3-L3 计划数，两种子相同）。正式训练（--max-updates 0）要求 --updates <= 9562；
     --updates < 9562 即"第二个差异"，必须给 --second-note（链在预跑后按 E3-L3 规则算出能放下的数时写入），写进 <run>/e3l4.json、
     final.json['e3l4'] 与日志。预跑（--max-updates > 0）不受此限。< 2,000 次 -> INCOMPLETE（链与判决）。"未截断" = 完成计划的
     更新数（L3 的 complete）。
  Q4 train / select 由 L3.main 原样执行（参数原样转发，argparse allow_abbrev=False，免得 --updates 被当成本包装参数的缩写）；
     本包装只在前后加记录与断言：训练前写 <run>/e3l4.json（已存在则拒绝），训练后在 final.json 加 'e3l4' 块。
  Q5 eval：复刻 L3.main eval（同一 L3.eval_job / eval_arm / 计分 / 字节 / GaBP 断言），只是工作进程先 configure，使模型类为 E3L4、
     臂名为 E3L4_<seed>；任务文件的 script_sha256 = 本文件 sha（另记 g4_e3l3 sha）。导出只读、不覆盖 jobs/；--expect-seeds 必须
     等于导出记录里的噪声种子集合；所有记录 split = VALIDATION、panel in {val, fresh}（不生成也不读 TEST；--smoke 例外）。
     判决面板 = G4-E5 导出（VAL / fresh x 噪声种子 3、4，tiers clean / cam_nf / cam）；只报告面板 = E3-R 导出（噪声种子 1、2）。
  Q6 对照 NET：E3-R 冻结检查点 + threshold.json，在同一批 G4-E5 地图上直接跑 `g4_e3r.py eval`（E3-R eval 口径，字节开），
     再 `ctrlagg` = R.aggregate（无判决），得到与 E3-R summary 同格式的对照（NET_<seed> 的 R8 / K0 / K0_sep / rnet_relay_median）。
     判决读对照时断言：没有失败 / 跳过的任务；g4_e3r sha 与本进程 import 的相同；检查点 sha 包含两个训练种子 run.json 里配对的
     E3-R 检查点 sha。对照字节 = rnet_relay_median 的 sum + count（同 E3-L3）。
  Q7 E1 复现：G4-E5 导出的噪声种子 3、4 没有 E1 任务（记录里 e1_match 三档都是 None）。aggregate 对种子 3、4 只接受"三档全 None"，
     列入 e1_not_applicable；其他种子仍要求三档全 True（同 L3）。
  Q8 判决：L3.verdict 在 D8_68（= D8-63 的数，noise_seeds = 3、4）下运行（临时替换 L3.D8_63 与 L3.ARM，调用后复原）；外加
     INCOMPLETE：缺 e3l4.json 或其中 w* / 预注册不符；final.json 没有 e3l4 块或最终 w != w*；格内 w != w*；计划更新数 != 9,562
     且没有写明第二个差异；对照不合格（Q6）。格的键 e3l3_* 改名为 arm_*（值不变）。
  Q9 只报告：配对池化 F1（furn_f1_pooled 之差）、机间信念差、字节上界比；z 三级分布与熵；训练探针摘要（每 1,000 次的留出房家具、
     字节比、z、w）；E5 summary 的 Q-inf 行与本对照 NET 的 R8 家具差（同图一致性）及 QPOST4；种子 1、2 的连续性格（report12：
     对 E3-R summary 的 NET，并列 E3-L3 summary 的同格读数）。E3-L3 在种子 3、4 上的同格读数属于 G4-E7（互不读读数），不在本批算。

用法（总是 python -B；CPU 阶段 CUDA_VISIBLE_DEVICES= ）：
  train     --w-star W --prereg-updates 9562 [--second-note TEXT] <g4_e3l3 train 的全部参数>
  select    --w-star W <g4_e3l3 select 的全部参数>
  eval      --w-star W --export X --out O --ckpt R1/checkpoints/final.pt,R2/... --expect-seeds 3,4 [--workers 9]
  ctrlagg   --out CTRL_EVAL_DIR --expect-seeds 3,4                    （对照 = g4_e3r.py eval 的输出目录）
  aggregate --w-star W --out O --runs R1,R2 --control-summary CTRL/summary.json [--e5-summary E5.json]
  report12  --w-star W --out O12 --control-summary E3R/eval/summary.json [--e3l3-summary E3L3.json]
  selfcheck --pyroot P --w-star W
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, contextlib, glob, json, math, time  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_e3l3 as L3  # noqa: E402   (frozen; read only)

L2 = L3.L2
L = L3.L
R = L3.R
G = L3.G
import torch  # noqa: E402

ARM = 'E3L4'
PREREG = 'D8-68'
BATCH = 'E3-L4'
UPDATES_PREREG = 9562                  # D8-68: E3-L3's planned count, both seeds
BUDGET_S = 3000                        # per seed on the 5090, load + compile included (D8-68)
W_GRID = (3.0, 4.0, 6.0, 8.0)          # G4-E7 step 0 range (D8-67)
W_RTOL_SMOKE = 1e-6
JUDGED_SEEDS = ('3', '4')              # G4-E5 export (judged VAL, fresh demotion)
REPORT_SEEDS = ('1', '2')              # E3-R export (report only)
NO_E1_SEEDS = frozenset({3, 4})        # Q7
RUN_FORMAT = 1
_ORIG = dict(E3L3=L3.E3L3, W0=L3.W0, ARM=L3.ARM, D8_63=L3.D8_63)
_W = dict(w=None)


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


def _jdump(obj, path):
    with open(path + '.tmp', 'w') as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + '.tmp', path)


def file_sha256(p):
    return G.file_sha256(p)


def _utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# ================================================================ Q1 / Q2: the model with omega frozen at log w*
class E3L4(_ORIG['E3L3']):
    """E3-L3's model; omega = log w* is frozen (requires_grad False): w = exp(omega) = w* in training and evaluation"""

    def __init__(self, width=16, w0=None):
        w0 = _W['w'] if w0 is None else w0
        assert w0 is not None, 'E3L4 needs w* (configure(w*) or an explicit w0)'
        super().__init__(width, w0)
        self.log_w.requires_grad_(False)


def check_w_star(w, smoke=False):
    w = float(w)
    assert math.isfinite(w) and w > 1.0, f'w* = {w}: must be > 1 (w <= 1 breaks the known mask after per-hop rounding)'
    assert smoke or w in W_GRID, f'w* = {w} is not in the G4-E7 step-0 grid {W_GRID}'
    return w


def configure(w_star):
    """Q1: route g4_e3l3's model class, initial w and arm name to E3-L4 in this process"""
    _W['w'] = float(w_star)
    L3.E3L3 = E3L4
    L3.W0 = float(w_star)
    L3.ARM = ARM


def D8_68(w_star=None):
    C = dict(_ORIG['D8_63'], prereg=PREREG, noise_seeds=list(JUDGED_SEEDS), budget_seconds=BUDGET_S, updates_prereg=UPDATES_PREREG,
             control_arm='NET_{seed}: E3-R checkpoints + thresholds re-evaluated with g4_e3r.py eval on the same G4-E5 maps (noise seeds 3 / 4)',
             single_variable='w fixed at w* (omega = log w* frozen), training and evaluation',
             supplement='D8-68 = the D8-63 criteria on VAL cam_nf noise seeds 3 / 4 (G4-E5 export): lever = 4 cells d >= -1 pp vs NET, bytes <= .15 x, '
                        'free / surface >= -0.5 pp, FPR identical, clean >= .99, K0 >= -3 pp, not truncated, GaBP checks at finite w; cam 4-cell mean '
                        '< -3 pp or fresh (noise 3 / 4) 4-cell mean < -1 pp demote LEVER only; not enough = any cell < -3 pp or bytes > .5 x; '
                        'updates fixed at 9,562 unless they do not fit (second difference, written down)')
    if w_star is not None:
        C['w_star'] = float(w_star)
    return C


@contextlib.contextmanager
def _as_e3l4(w_star):
    saved = {k: getattr(L3, k) for k in ('E3L3', 'W0', 'ARM', 'D8_63')}
    saved_w = _W['w']
    configure(w_star); L3.D8_63 = D8_68(w_star)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(L3, k, v)
        _W['w'] = saved_w


def ckpt_w(path):
    """exp(log_w) of a checkpoint, evaluated in float32 on the CPU (as eval / select do)"""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return float(torch.exp(ck['state_dict']['log_w'].float())), ck


def _w_ok(w, w_star, smoke):
    return (w == w_star) if (not smoke and w_star in W_GRID) else abs(w - w_star) <= W_RTOL_SMOKE * w_star


def _sub_args(rest, *names):
    """read a few forwarded options without consuming them (Q4)"""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for n in names:
        if n == '--smoke':
            p.add_argument(n, action='store_true')
        else:
            p.add_argument(n, default=None)
    a, _ = p.parse_known_args(rest)
    return a


# ================================================================ train (Q3 / Q4)
def train(args, rest):
    w_star = check_w_star(args.w_star, args.smoke)
    a = _sub_args(rest, '--out', '--updates', '--max-updates', '--seed', '--smoke')
    smoke = bool(args.smoke or a.smoke)
    assert a.out and a.updates is not None and a.seed is not None, 'train needs --out, --updates, --seed'
    upd = int(a.updates); mx = int(a.max_updates or 0); preflight = mx > 0
    assert args.prereg_updates == UPDATES_PREREG or smoke, f'--prereg-updates {args.prereg_updates} != D8-68 {UPDATES_PREREG}'
    second = (not preflight) and upd != args.prereg_updates
    if not preflight and not smoke:
        assert upd <= args.prereg_updates, f'--updates {upd} > the pre-registered {args.prereg_updates} (D8-68 fixes the count)'
    if second and not args.second_note.strip():
        raise SystemExit(f'--updates {upd} != {args.prereg_updates}: the second difference must be written down (--second-note)')
    os.makedirs(a.out, exist_ok=True)
    mp = os.path.join(a.out, 'e3l4.json')
    if os.path.exists(mp):
        raise SystemExit(f'{mp} exists (never overwrite)')
    meta = dict(format=RUN_FORMAT, batch=BATCH, prereg=PREREG, arm=ARM, seed=int(a.seed), w_star=w_star, log_w_star=math.log(w_star), omega_frozen=True,
                updates_prereg=int(args.prereg_updates), updates=upd, max_updates=mx, preflight=preflight, second_difference=bool(second),
                second_difference_note=(args.second_note.strip() or None), wrapper_sha256=file_sha256(os.path.abspath(__file__)),
                g4_e3l3_sha256=file_sha256(L3.__file__), g4_e3l2_sha256=file_sha256(L2.__file__), g4_e3l_sha256=file_sha256(L.__file__),
                g4_e3r_sha256=file_sha256(R.__file__), smoke=smoke, utc_start=_utc(), pid=os.getpid())
    _jdump(meta, mp)
    if second:
        print(f'E3-L4 SECOND DIFFERENCE (D8-68): {upd} updates instead of {args.prereg_updates}: {args.second_note.strip()}', flush=True)
    print(f'E3-L4 train seed {a.seed}: w* = {w_star} (omega = log w* frozen), {upd} updates{" (preflight, max " + str(mx) + ")" if preflight else ""}', flush=True)
    configure(w_star)
    fin = L3.main(['train'] + list(rest) + (['--smoke'] if args.smoke and not a.smoke else []))
    ckp = os.path.join(a.out, 'checkpoints', 'final.pt')
    wf, ck = ckpt_w(ckp)
    ok_w = _w_ok(wf, w_star, smoke) and abs(float(ck['recipe']['w0']) - w_star) <= W_RTOL_SMOKE * w_star
    assert ok_w, f'final checkpoint w {wf!r} (recipe w0 {ck["recipe"]["w0"]}) != w* {w_star}'
    fp = os.path.join(a.out, 'final.json')
    fj = _jload(fp)
    fj['e3l4'] = dict(meta, w_final=wf, w_final_equals_w_star=bool(ok_w), utc_end=_utc(),
                      complete_means='updates_done == planned (the planned count is the chain\'s UPD; with a second difference it is below 9,562)')
    _jdump(fj, fp)
    print(f'E3-L4 train seed {a.seed} done: {fj["updates_done"]} / {fj["updates_planned"]} updates ({"complete" if fj["complete"] else "NOT complete"}), '
          f'w {wf} (= w* {w_star}), second difference: {("yes, " + args.second_note.strip()) if second else "no"}', flush=True)
    return fj


# ================================================================ select (Q4): pre-checks, then g4_e3l3's select
def _check_run(run, w_star, smoke):
    mp = os.path.join(run, 'e3l4.json')
    assert os.path.exists(mp), f'{run}: no e3l4.json'
    m = _jload(mp)
    assert m.get('prereg') == PREREG and float(m.get('w_star')) == w_star, f'{run}: e3l4.json prereg / w* {m.get("prereg")} / {m.get("w_star")} != {PREREG} / {w_star}'
    wf, ck = ckpt_w(os.path.join(run, 'checkpoints', 'final.pt'))
    assert _w_ok(wf, w_star, smoke), f'{run}: checkpoint w {wf!r} != w* {w_star}'
    return m


def select(args, rest):
    w_star = check_w_star(args.w_star, args.smoke)
    a = _sub_args(rest, '--run', '--smoke')
    assert a.run, 'select needs --run'
    _check_run(a.run, w_star, bool(args.smoke or a.smoke))
    configure(w_star)
    return L3.main(['select'] + list(rest) + (['--smoke'] if args.smoke and not a.smoke else []))


# ================================================================ eval (Q5)
def _eval_and_save(job):
    """worker entry (spawn): route g4_e3l3 to E3-L4 in this process, then g4_e3l3's own eval job"""
    configure(job[-1]['w_star'])
    return L3._eval_and_save(job)


def eval_cmd(args, json_dir, index_file):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    w_star = check_w_star(args.w_star, args.smoke)
    ckpts = [p for p in args.ckpt.split(',') if p]
    assert ckpts, 'no checkpoints'
    thr = {}
    for p in ckpts:
        run = os.path.dirname(os.path.dirname(p))
        thr[p] = os.path.join(run, 'threshold.json')
        assert os.path.exists(thr[p]), f'{p}: no {thr[p]}'
        _check_run(run, w_star, args.smoke)
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(args.export, 'records', '*.json')))]
    assert recs, f'no export records in {args.export}/records'
    seeds = sorted({int(r['seed']) for r in recs}); want = sorted(int(s) for s in args.expect_seeds.split(',') if s)
    if not args.smoke:
        assert seeds == want, f'export noise seeds {seeds} != expected {want}'
        bad = [r for r in recs if r.get('split') != 'VALIDATION' or r.get('panel') not in ('val', 'fresh')]
        assert not bad, f'{len(bad)} records are not VALIDATION val / fresh (TEST is never read)'
    if args.only:
        assert args.smoke, '--only is for the smoke'
        keep = set(args.only.split(','))
        recs = [r for r in recs if f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}" in keep]
    jd = os.path.join(args.out, 'jobs')
    if os.path.isdir(jd) and os.listdir(jd):
        raise SystemExit(f'{jd} is not empty (never overwrite)')
    sha = file_sha256(os.path.abspath(__file__))
    opts = dict(export=args.export, out=args.out, ckpts=ckpts, thr=thr, ckpt_shas={p: file_sha256(p) for p in ckpts}, tiers=list(R.EVAL_TIERS),
                threads=args.threads, chunk=args.chunk, script_sha256=sha, smoke=bool(args.smoke), w_star=w_star, g4_e3l3_sha256=file_sha256(L3.__file__))
    os.makedirs(jd, exist_ok=True)
    _jdump(dict(stage='eval', batch=BATCH, prereg=PREREG, w_star=w_star, export=os.path.abspath(args.export), expect_seeds=want, seeds=seeds,
                ckpts=opts['ckpt_shas'], script_sha256=sha, g4_e3l3_sha256=opts['g4_e3l3_sha256'], utc=_utc()), os.path.join(args.out, 'eval_run.json'))
    for r in recs:
        if r.get('failed') or r.get('skipped'):
            _jdump(dict(stage='eval', panel=r['panel'], rooms=r['rooms'], index=r['index'], seed=r['seed'], failed=bool(r.get('failed')),
                        skipped=r.get('skipped'), error='export ' + str(r.get('error') or r.get('skipped'))),
                   os.path.join(jd, f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}.json"))
    jobs = [(r, opts) for r in recs if not r.get('failed') and not r.get('skipped')]
    t0 = time.time()
    print(f'E3-L4 eval: {len(jobs)} jobs, w* {w_star}, checkpoints {ckpts}, export {args.export} (seeds {seeds}), {args.workers} workers x {args.threads} threads', flush=True)
    for i, rec in R._pool(args.workers, _eval_and_save, jobs, json_dir, index_file):
        print(f"[{i}/{len(jobs)}] {rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']} {'FAILED ' + rec.get('error', '') if rec.get('failed') else ''}"
              f"{rec.get('seconds', 0):.0f}s peak {rec.get('peak_mb') or 0:.0f} MB  {time.time() - t0:.0f}s", flush=True)
    return None


# ================================================================ aggregation (Q7) and the control summary (Q6)
def aggregate(out_dir):
    S = L3.aggregate(out_dir)
    na = []
    for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json'))):
        r = _jload(p)
        tag = f"{r.get('panel')}_r{r.get('rooms')}_i{r.get('index')}_s{r.get('seed')}"
        e1 = r.get('e1_match')
        if tag in S['e1_mismatch'] and int(r.get('seed', -1)) in NO_E1_SEEDS and isinstance(e1, dict) and e1 and all(v is None for v in e1.values()):
            na.append(tag)
    S['e1_mismatch'] = [t for t in S['e1_mismatch'] if t not in na]
    S['e1_not_applicable'] = na
    S['e1_note'] = 'noise seeds 3 / 4 (G4-E5 export) have no E1 job: e1_match None on all tiers is accepted for them only (Q7)'
    return S


def ctrlagg(args):
    """Q6: E3-R's own aggregation of the NET control re-evaluated on the G4-E5 maps (no verdict)"""
    p = os.path.join(args.out, 'summary.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} exists (never overwrite)')
    S = R.aggregate(args.out, smoke=args.smoke)
    want = sorted(str(s) for s in args.expect_seeds.split(',') if s)
    S['e3l4_control'] = dict(batch=BATCH, prereg=PREREG, expect_seeds=want, seeds={pn: sorted(P['seeds']) for pn, P in S['panels'].items()},
                             g4_e3r_sha256=file_sha256(R.__file__), wrapper_sha256=file_sha256(os.path.abspath(__file__)), utc=_utc(),
                             note='NET_<seed> = frozen E3-R checkpoint + threshold, g4_e3r.py eval (bytes on) on the G4-E5 export; e1_mismatch '
                                  'lists seeds 3 / 4 only because E1 never ran them')
    _jdump(S, p)
    print(f"E3-L4 control summary {p}: panels {S['e3l4_control']['seeds']}, failed {S['failed']}, skipped {S['skipped']}, sha256 {file_sha256(p)}", flush=True)
    return S


def control_problems(Sc, runs_info, expect_seeds=JUDGED_SEEDS):
    out = []
    if Sc.get('failed') or Sc.get('skipped'):
        out.append(f"control: failed / skipped jobs {Sc.get('failed')} / {Sc.get('skipped')}")
    if Sc.get('script_sha256') != [file_sha256(R.__file__)]:
        out.append(f"control: g4_e3r sha {Sc.get('script_sha256')} != the frozen {file_sha256(R.__file__)}")
    shas = set()
    for s in Sc.get('ckpts') or []:
        try:
            shas |= set(json.loads(s).values())
        except (TypeError, ValueError):
            out.append('control: unreadable checkpoint list')
    for sd, pair in runs_info.items():
        if not pair or pair.get('ckpt_sha256') not in shas:
            out.append(f'control: the E3-R checkpoint paired with training seed {sd} is not among the control checkpoints')
    for panel in ('val', 'fresh'):
        got = sorted((Sc.get('panels', {}).get(panel) or {}).get('seeds', {}))
        if got != sorted(expect_seeds):
            out.append(f'control: panel {panel} noise seeds {got} != {sorted(expect_seeds)}')
    return out


# ================================================================ verdict (Q8) and report-only readouts (Q9)
def _rename(c):
    if not isinstance(c, dict):
        return c
    return {(k.replace('e3l3_', 'arm_', 1) if k.startswith('e3l3_') else k): v for k, v in c.items()}


def _probe_digest(run):
    try:
        H = _jload(os.path.join(run, 'history.json'))
    except Exception:
        return None
    out = []
    for h in H:
        p = h.get('probe')
        if p:
            out.append(dict(updates=h['updates'], furn_f1_at0=p.get('furn_f1_at0'), furn_f1_oracle=p.get('furn_f1_oracle'), bytes_ratio_median=p.get('bytes_ratio_median'),
                            z_minus=p.get('z_minus'), z_zero=p.get('z_zero'), z_plus=p.get('z_plus'), z_entropy_bits=p.get('z_entropy_bits'),
                            tanh_saturation=p.get('tanh_saturation'), w=p.get('w'), known_mismatch=p.get('known_mismatch'), holdout_bce=h.get('holdout_bce')))
        elif h.get('probe_skipped'):
            out.append(dict(updates=h['updates'], skipped=h['probe_skipped']))
    return out


def _arm_ctrl(S, Sc, panel, ns, ts, tier='cam_nf'):
    try:
        a = S['panels'][panel]['seeds'][ns]['tiers'][tier][f'{ARM}_{ts}']
        c = Sc['panels'][panel]['seeds'][ns]['tiers'][tier][f'NET_{ts}']
        return a, c
    except (KeyError, TypeError):
        return None, None


def _extras(S, Sc, panel, key):
    ts, ns = key.split('|')
    a, c = _arm_ctrl(S, Sc, panel, ns, ts)
    if a is None or c is None:
        return {}
    x = dict(belief_disagreement_rms=a.get('belief_disagreement_rms'), report_bytes_upper_median=a.get('report_bytes_upper_median'))
    pa, pc = a['R8'].get('furn_f1_pooled'), c['R8'].get('furn_f1_pooled')
    if pa is not None and pc is not None:
        x.update(arm_R8_furn_pooled=pa, control_R8_furn_pooled=pc, d_furn_pooled=pa - pc)
    return x


def _e5_crosscheck(cells, Sc, e5_path):
    """report only: the re-evaluated NET (control) vs G4-E5's Q-inf row on the same maps, and QPOST4"""
    if not e5_path:
        return None
    try:
        E = _jload(e5_path)
    except Exception as e:
        return dict(error=repr(e)[:200])
    out = {}
    for k in cells:
        ts, ns = k.split('|')
        try:
            q = E['panels']['val'][ns]['cells'][ts]['cam_nf']
            c = Sc['panels']['val']['seeds'][ns]['tiers']['cam_nf'][f'NET_{ts}']['R8']
            out[k] = dict(e5_qinf_furn=q['Qinf_abs']['furn'], control_net_furn=c['furn_f1'], diff=c['furn_f1'] - q['Qinf_abs']['furn'],
                          e5_qinf_free=q['Qinf_abs'].get('free'), control_net_free=c.get('free_recall'),
                          qpost4=dict(furn=q['QPOST4'].get('furn', q['QPOST4'].get('arm_furn')), d_furn=q['QPOST4'].get('d_furn'), ratio=q['QPOST4'].get('ratio'))
                          if 'QPOST4' in q else None)
        except (KeyError, TypeError) as e:
            out[k] = dict(error=f'missing {e}')
    return out


def verdict(S, runs=None, control_summary=None, w_star=None, smoke=False, e5_summary=None):
    """D8-68 (Q8): g4_e3l3's D8-63 verdict with the noise seeds 3 / 4 and the E3-L4 arm, plus the E3-L4 INCOMPLETE conditions"""
    extra = []
    if w_star is None:
        extra.append('w* not given')
        w_star = float('nan')
    with _as_e3l4(w_star if math.isfinite(w_star) else 4.0):
        V = L3.verdict(S, runs, control_summary, smoke=smoke, refs=None)
    run_meta, pairs = {}, {}
    planned = set()
    for r in runs or []:
        try:
            m = _jload(os.path.join(r, 'e3l4.json')); sd = str(m['seed'])
        except Exception as e:
            extra.append(f'run {r}: e3l4.json unreadable ({e!r})'); continue
        run_meta[sd] = m
        if m.get('prereg') != PREREG or m.get('w_star') != w_star or not m.get('omega_frozen'):
            extra.append(f'training seed {sd}: e3l4.json prereg / w* / omega_frozen = {m.get("prereg")} / {m.get("w_star")} / {m.get("omega_frozen")} '
                         f'(expected {PREREG} / {w_star} / True)')
        try:
            fj = _jload(os.path.join(r, 'final.json')); e = fj.get('e3l4') or {}
            if not e.get('w_final_equals_w_star') or not _w_ok(float(e.get('w_final', float('nan'))), w_star, smoke):
                extra.append(f'training seed {sd}: final w {e.get("w_final")} != w* {w_star} (or no e3l4 block in final.json)')
            planned.add(int(fj.get('updates_planned', -1)))
            if int(fj.get('updates_planned', -1)) != UPDATES_PREREG and not (m.get('second_difference') and m.get('second_difference_note')):
                extra.append(f'training seed {sd}: {fj.get("updates_planned")} planned updates != {UPDATES_PREREG} and no second difference written down')
        except Exception as e:
            extra.append(f'training seed {sd}: final.json unreadable ({e!r})')
        try:
            pairs[sd] = _jload(os.path.join(r, 'run.json')).get('pair')
        except Exception:
            pairs[sd] = None
    for c in V['cells'].values():
        if c.get('w') is None or not _w_ok(float(c['w']), w_star, smoke):
            extra.append(f'cell w {c.get("w")} != w* {w_star}'); break
    Sc = None
    if control_summary:
        Sc = _jload(control_summary)
        extra += control_problems(Sc, pairs)
    v = V['verdict']
    if extra:
        v = (v + '; ' if v.startswith('INCOMPLETE') else 'INCOMPLETE: ') + '; '.join(extra)
    second = sorted({sd for sd, m in run_meta.items() if m.get('second_difference')})
    cells = {k: dict(_rename(c), **(_extras(S, Sc, 'val', k) if Sc else {})) for k, c in V['cells'].items()}
    fresh = {k: dict(_rename(c), **(_extras(S, Sc, 'fresh', k) if Sc else {})) for k, c in V['fresh_cells'].items()}
    out = dict(V, verdict=v, cells=cells, fresh_cells=fresh, criteria=D8_68(w_star), w_star=w_star, e3l4_incomplete_reasons=extra,
               second_difference=dict(seeds=second, notes={sd: run_meta[sd].get('second_difference_note') for sd in second},
                                      planned_updates=sorted(planned), prereg_updates=UPDATES_PREREG),
               training_probes={r: _probe_digest(r) for r in (runs or [])},
               report_e5=_e5_crosscheck(cells, Sc, e5_summary) if Sc else None,
               report_note='E3-L3 on noise seeds 3 / 4 belongs to G4-E7 (the two batches do not read each other); seeds 1 / 2 in report12')
    out.pop('report_refs', None)
    return out


def report12(args):
    """Q9 report only: continuity on noise seeds 1 / 2 (E3-R export) vs the E3-R summary NET and the E3-L3 summary; never a verdict"""
    p = os.path.join(args.out, 'summary.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} exists (never overwrite)')
    w_star = check_w_star(args.w_star, args.smoke)
    S = aggregate(args.out)
    Sc = _jload(args.control_summary)
    E3 = _jload(args.e3l3_summary) if args.e3l3_summary else None
    C = dict(D8_68(w_star), noise_seeds=list(REPORT_SEEDS))
    cells, fresh, reasons = {}, {}, []
    with _as_e3l4(w_star):
        for panel, store, judged in (('val', cells, True), ('fresh', fresh, False)):
            for ts in C['train_seeds']:
                for ns in REPORT_SEEDS:
                    c = L3._cell(S, Sc, panel, ts, ns, C, reasons, f'{panel} cell (train {ts}, noise {ns})', judged)
                    if c is None:
                        continue
                    c = dict(_rename(c), **_extras(S, Sc, panel, f'{ts}|{ns}'))
                    if E3 is not None:
                        try:
                            e = E3['panels'][panel]['seeds'][ns]['tiers']['cam_nf'][f'E3L3_{ts}']
                            c.update(e3l3_R8_furn=e['R8']['furn_f1'], e3l3_bytes_median=e.get('bytes_median'),
                                     d_furn_e3l4_minus_e3l3=c['arm_R8_furn'] - e['R8']['furn_f1'],
                                     bytes_ratio_e3l4_over_e3l3=(c['bytes_median'] / e['bytes_median']) if e.get('bytes_median') else None)
                            cam = E3['panels'][panel]['seeds'][ns]['tiers']['cam'][f'E3L3_{ts}']
                            if 'cam_R8_furn' in c:
                                c['d_cam_e3l4_minus_e3l3'] = c['cam_R8_furn'] - cam['R8']['furn_f1']
                        except (KeyError, TypeError):
                            c['e3l3'] = None
                    store[f'{ts}|{ns}'] = c
    S['report12'] = dict(report_only=True, w_star=w_star, cells=cells, fresh_cells=fresh, problems=reasons, e1_mismatch=S['e1_mismatch'],
                         control_summary=dict(path=os.path.abspath(args.control_summary), sha256=file_sha256(args.control_summary)),
                         e3l3_summary=(dict(path=os.path.abspath(args.e3l3_summary), sha256=file_sha256(args.e3l3_summary)) if args.e3l3_summary else None),
                         gbp=S.get('gbp'), known_mismatch=S.get('known_mismatch'))
    _jdump(S, p)
    print('E3-L4 report12 (noise seeds 1 / 2, report only):', flush=True)
    for k, c in cells.items():
        print(f"  cell {k}: E3-L4 R8 furn {c['arm_R8_furn']:.4f} vs NET {c['control_R8_furn']:.4f} (d {c['d_furn']:+.4f}); bytes {c['bytes_ratio']:.3f} x; "
              f"E3-L3 {c.get('e3l3_R8_furn')}; problems {len(reasons)}", flush=True)
    return S


# ================================================================ main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('train', 'select', 'eval', 'ctrlagg', 'aggregate', 'report12', 'selfcheck'):
        p = sub.add_parser(name, allow_abbrev=False)
        p.add_argument('--houseexpo-json-dir', default=None); p.add_argument('--houseexpo-index', default=None)
        p.add_argument('--smoke', action='store_true')
        if name != 'ctrlagg':
            p.add_argument('--w-star', type=float, required=True)
        if name == 'train':
            p.add_argument('--prereg-updates', type=int, required=True); p.add_argument('--second-note', default='')
        if name == 'eval':
            p.add_argument('--export', required=True); p.add_argument('--out', required=True); p.add_argument('--ckpt', required=True)
            p.add_argument('--expect-seeds', required=True); p.add_argument('--workers', type=int, default=6); p.add_argument('--threads', type=int, default=1)
            p.add_argument('--chunk', type=int, default=2); p.add_argument('--only', default='')
        if name == 'ctrlagg':
            p.add_argument('--out', required=True); p.add_argument('--expect-seeds', required=True)
        if name == 'aggregate':
            p.add_argument('--out', required=True); p.add_argument('--runs', default=''); p.add_argument('--control-summary', default='')
            p.add_argument('--e5-summary', default='')
        if name == 'report12':
            p.add_argument('--out', required=True); p.add_argument('--control-summary', required=True); p.add_argument('--e3l3-summary', default='')
        if name == 'selfcheck':
            p.add_argument('--pyroot', required=True)
    return ap


def main(argv=None):
    ap = build_parser()
    args, rest = ap.parse_known_args(argv)
    if rest and args.cmd not in ('train', 'select'):
        ap.error(f'unrecognised arguments: {rest}')
    json_dir = args.houseexpo_json_dir or R._HX[0] or G.HOUSEEXPO_JSON_DIR
    index_file = args.houseexpo_index or R._HX[1] or G.HOUSEEXPO_INDEX
    for opt, val in (('--houseexpo-json-dir', args.houseexpo_json_dir), ('--houseexpo-index', args.houseexpo_index)):
        if val and args.cmd in ('train', 'select'):
            rest = list(rest) + [opt, val]
    G.setup_env(json_dir, index_file)
    if args.cmd == 'train':
        return train(args, rest)
    if args.cmd == 'select':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return select(args, rest)
    if args.cmd == 'eval':
        return eval_cmd(args, json_dir, index_file)
    if args.cmd == 'ctrlagg':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return ctrlagg(args)
    if args.cmd == 'selfcheck':
        check_w_star(args.w_star, args.smoke)
        for m in (L3, L2, L, R):
            assert os.path.dirname(os.path.abspath(m.__file__)) == HERE, f'{m.__name__} imported from {m.__file__}, not {HERE}'
        print(f'E3-L4 w* {args.w_star} ok; g4_e3l3 {file_sha256(L3.__file__)[:16]}', flush=True)
        return R.main(['selfcheck', '--pyroot', args.pyroot, '--houseexpo-json-dir', json_dir, '--houseexpo-index', index_file])
    if args.cmd == 'aggregate':
        p = os.path.join(args.out, 'summary.json')
        if os.path.exists(p):
            raise SystemExit(f'{p} exists (never overwrite)')
        w_star = check_w_star(args.w_star, args.smoke)
        S = aggregate(args.out)
        runs = [r for r in args.runs.split(',') if r] or None
        S['decision'] = verdict(S, runs, args.control_summary or None, w_star=w_star, smoke=args.smoke, e5_summary=args.e5_summary or None)
        S['decision']['control_summary_file'] = (dict(path=os.path.abspath(args.control_summary), sha256=file_sha256(args.control_summary))
                                                 if args.control_summary else None)
        S['decision']['script_sha256'] = file_sha256(os.path.abspath(__file__))
        _jdump(S, p)
        D = S['decision']
        print('E3-L4 verdict:', D['verdict'], flush=True)
        print(f"  w* {w_star}; second difference: {D['second_difference']}", flush=True)
        for k, c in D['cells'].items():
            print(f"  cell {k}: E3-L4 R8 furn {c['arm_R8_furn']:.4f} vs NET {c['control_R8_furn']:.4f} (d {c['d_furn']:+.4f}); bytes {c['bytes_median']:.0f} = "
                  f"{c['bytes_ratio']:.3f} x control; cam d {c.get('cam_d_furn', float('nan')):+.4f}; guards {c.get('guards')}", flush=True)
        print('  fresh:', json.dumps(D.get('fresh')), flush=True)
        print('  gbp:', json.dumps(D.get('gbp')), '| known mismatch:', json.dumps(D.get('known_mismatch')), flush=True)
        return S
    if args.cmd == 'report12':
        return report12(args)


if __name__ == '__main__':
    main()
