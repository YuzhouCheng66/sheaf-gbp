"""mapping_3d_storey_data_v420div.py (Claude, 2026-10-28): batch E3-L5 的 TRAIN 生成器"带形状多样性的训练分布"。
预注册 claude_handoff/OVERNIGHT_REPORT_20261021.md D8-76（背景 D8-53 / D8-55 / D8-71）。只 import 冻结的 mapping_3d_storey_data_v420.py
（v420）与 mapping_3d_storey_data_v420ood.py（v420ood，G4-H1 派生生成器，sha 9e1c7c2a...），从不修改它们；平面、传感、智能体、
无线电全部调用 v420 的函数，新图元的栅格化、旋转件与挂墙层板的造型调用 v420ood 的函数。

一、D8-76 写死的部分
  训练分布 = 原生成器 v420 50 % + 派生变体 'train_div' 50 %（按房子分配）。train_div 里可见的家族 X：
    旋转家具（rot_bed / rot_sofa / rot_table，v420ood 的造型），尺度 x0.6 / x1.5（v420ood 'scale' 的造型），挂墙层板 wall_shelf
    （v420ood 的造型，底面离地 1.0-1.9 m）。
  留出家族 Y（训练里绝不出现，断言）：椭球 / 球（ellipsoid、ball）、悬挂物（hanging）、杂物（clutter）、细立杆（coat_rack，家族 pole）。
  户型与留出房同 E3-L3（= E3-R 的 TRAIN 缓存 cache-e3r-v1：scene_plan('TRAIN', 70) 去掉它跳过的 5 房 = 205 房，留出 = index % 10 == 9，
  186 训练 / 19 留出）。只生成 TRAIN（VALIDATION / TEST 拒绝）。

二、操作化（D8-76 没写死、这里定下的部分；回复里逐条列出）
  V1 按房分配（确定性、与 index 奇偶无关）：205 房按 (房间数分层 r1 / r2 / r3, 是否留出) 分 6 组，每组按
     sha256('v420div:assign:TRAIN:r{rooms}:i{index}') 排序，前 floor(n / 2) 房用 train_div，其余用 v420
     -> 训练 92 train_div / 94 v420（r1 30/30、r2 31/32、r3 31/32），留出 8 / 11（r1 2/3、r2 3/4、r3 3/4）。
     （index 奇偶不能用：留出房全是奇数 index；按分层是为了让两半的房子大小分布相同。）
  V2 v420 那一半就是 v420 本身：world_id = 'v420:TRAIN:{house_id}'，make_fixture = v420 原函数，因此这一半与 cache-e3r-v1 的同名房
     逐数组相同（npz 的 zip 时间戳会让文件 sha 不同；生成后在 uk-linux 上逐数组核对）。
  V3 train_div 的世界：world_id = 'v420div:train_div:TRAIN:{house_id}'；平面逐位不变（层高、门高取 v420 自己 world_id 的前两次几何抽样，
     同 v420ood O1），家具几何随机数 = sha(world_id + ':geometry')，先烧掉同样的两次抽样；放置循环逐行照 v420ood（= v420 的规则：每
     1.2 m^2 一次尝试、房间类别 + 30 % 通用、每物体 20 次尝试、靠墙推入、背面贴墙、离开口 0.6 m、剩余地面连通 >= 95 %）。
  V4 每个槽位（原类别 src，与 v420 的槽位表相同）在放置前从"可用模式"里等概率抽一个（几何随机数）：
       可用模式 = {rotate 若 src in (bed, sofa, dining_table, coffee_table, desk)} + {scale（全部类别）} + {shelf 若 src 靠墙}；
       rotate -> v420ood 的 rot_bed / rot_sofa / rot_table（绕 z 转 U(15, 75) 度，自由放置），**桌面杂物去掉**（v420ood 以 0.5 概率在
                 rot_table 上放 clutter，属于 Y；这里照常抽随机数但丢弃这些图元，断言没有附带物体）；
       scale  -> f in {0.6, 1.5} 等概率（逐物体一次），v420ood 'scale' 的造型（含 v420 的椅子规则）；
       shelf  -> v420ood 的 wall_shelf（0.5 概率 1-3 块 3 cm 层板，否则吊柜；底面离地 U(1.0, 1.9)）。
     所以 train_div 房里每个物体都属于 X（没有原样家具；原样家具来自另一半房子）。组合件 l_shape / step_shape 既不在 X 也不在 Y，不出现。
     家族名：rotated、scaled_0.6、scaled_1.5、wall_shelf（与 v420ood 的 FAMILY / family_of 一致）。
  V5 Y 为 0 的断言（world_spec 出口）：没有 kind in Y_KINDS、没有 family in Y_FAMILIES、没有 'ell' 图元、family 全在 X_FAMILIES、没有
     table_top 附带物体。v420 那一半另断言 kind 全在 v420 的类别表里、图元只有 box / cyl。
  V6 智能体放不下（v420 agent_positions 的三种 ValueError）时，train_div 房按 world_id 后缀 ':a1'..':a3' 重抽家具（最多 4 次），保证
     房子集合与 cache-e3r-v1 相同；尝试序号写进 fixture['div_attempt'] 与统计。4 次都失败 -> ValueError（gen 记 skipped，manifest 的
     房数会不等于 205，g4_e3l5 gen 会报错）。
  V7 make_fixture 的返回键与 v420 相同（另加 variant / div_attempt）；module_mixed() 返回一个模块对象（v420 的全部属性 + 按 V1 分派的
     world_spec / make_fixture，rasterize = v420ood.rasterize，对只有 box / cyl 的 v420 世界逐位等于 v420.rasterize），给 g4_e3r 的
     house_ctx（它通过模块全局 W 调 make_fixture）用，冻结文件不改。
  V8 统计（family_stats）：每房按家族 / 类别的物体数与体素数（v420ood.object_masks：每物体图元并集、扣墙柱；并集逐位等于 obj ∪ 楼板部分），
     Y 家族物体数与体素数必须为 0。

HouseExpo 原始数据只从 HOUSEEXPO_JSON_DIR 读取，不复制、不导出。
"""
import functools
import hashlib
import json
import math
import os
import types

import numpy as np
from scipy import ndimage

import mapping_3d_storey_data_v420 as W               # frozen; import only
import mapping_3d_storey_data_v420ood as WO           # frozen; import only

VERSION = '420div'
VARIANT = 'train_div'
ORIG = 'v420'
X_FAMILIES = ('rotated', 'scaled_0.6', 'scaled_1.5', 'wall_shelf')
Y_KINDS = ('ball', 'ellipsoid', 'hanging', 'clutter', 'coat_rack')
Y_FAMILIES = ('ellipsoid', 'hanging', 'clutter', 'pole')
Y_PRIMS = ('ell',)
ROT_OF = dict(bed='rot_bed', sofa='rot_sofa', dining_table='rot_table', coffee_table='rot_table', desk='rot_table')
MODES = ('rotate', 'scale', 'shelf')
E3R_SKIPPED = ('TRAIN_r1_i26', 'TRAIN_r1_i29', 'TRAIN_r1_i46', 'TRAIN_r1_i49', 'TRAIN_r1_i66')   # cache-e3r-v1 manifest 'skipped'
E3R_MANIFEST_SHA256 = '14e90b409a04ff5bf64484b3d8035dd964a787494c16cdda80b849a9e44b7e57'          # cache-e3r-v1/manifest.json
TRAIN_PER_STRATUM = 70
HOLDOUT_MOD, HOLDOUT_REM = 10, 9
ATTEMPTS = 4
AGENT_SKIPS = ('no free floor', 'not enough free floor with clearance', 'could not place agents')   # v420 agent_positions' ValueErrors

assert {WO.FAMILY[k] for k in Y_KINDS} == set(Y_FAMILIES)
assert all(WO.NOVEL_OF[k] == v for k, v in ROT_OF.items()) and set(ROT_OF.values()) == set(WO.ROTATED)
assert {WO.FAMILY[k] for k in ROT_OF.values()} == {'rotated'} and WO.FAMILY['wall_shelf'] == 'wall_shelf'
assert {f'scaled_{f:g}' for f in WO.SCALE_FACTORS} == {'scaled_0.6', 'scaled_1.5'}
assert not set(X_FAMILIES) & set(Y_FAMILIES)


# ----------------------------------------------------------------------------- V1: the plan and the per-house assignment
def house_name(rooms, index):
    return f'TRAIN_r{int(rooms)}_i{int(index)}'


def is_holdout(index):
    return int(index) % HOLDOUT_MOD == HOLDOUT_REM


def train_plan():
    """the 205 TRAIN houses of cache-e3r-v1: scene_plan('TRAIN', 70) without the 5 houses it skipped"""
    plan = [(int(s['rooms']), int(s['index'])) for s in W.scene_plan('TRAIN', TRAIN_PER_STRATUM)]
    return [h for h in plan if house_name(*h) not in E3R_SKIPPED]


def _rank_key(rooms, index):
    return hashlib.sha256(f'v420div:assign:TRAIN:r{int(rooms)}:i{int(index)}'.encode()).hexdigest()


@functools.lru_cache(maxsize=1)
def _assignment():
    out = {}
    plan = train_plan()
    for rooms in (1, 2, 3):
        for ho in (False, True):
            grp = sorted([h for h in plan if h[0] == rooms and is_holdout(h[1]) == ho], key=lambda h: _rank_key(*h))
            nd = len(grp) // 2
            for i, h in enumerate(grp):
                out[h] = VARIANT if i < nd else ORIG
    assert len(out) == len(plan) == 205
    return out


def assignment():
    """{(rooms, index): 'train_div' | 'v420'} over the 205 TRAIN houses (V1)"""
    return dict(_assignment())


def variant_of(split, index, rooms):
    assert str(split).upper() == 'TRAIN', 'v420div: TRAIN only (VALIDATION / TEST are refused)'
    key = (int(rooms), int(index))
    a = _assignment()
    if key not in a:
        raise KeyError(f'{house_name(*key)} is not one of the 205 cache-e3r-v1 TRAIN houses')
    return a[key]


def slot_modes(src):
    """V4: the X modes available for a slot of v420 kind `src` (fixed order rotate, scale, shelf)"""
    m = []
    if src in ROT_OF:
        m.append('rotate')
    m.append('scale')
    if W.SIZES_KIND[src][2]:
        m.append('shelf')
    return tuple(m)


# ----------------------------------------------------------------------------- V5: the Y = 0 assertions
def check_div_objects(objects):
    bad = [o['kind'] for o in objects if o['kind'] in Y_KINDS or o.get('family') in Y_FAMILIES or o.get('placement') != 'floor'
           or any(p[0] in Y_PRIMS for p in o['prims'])]
    assert not bad, f'train_div world holds held-out (Y) objects / attachments: {bad}'
    fams = {o.get('family') for o in objects}
    assert fams <= set(X_FAMILIES), f'train_div families {sorted(map(str, fams))} not within X {X_FAMILIES}'


def check_orig_objects(objects):
    bad = [o['kind'] for o in objects if o['kind'] not in W.SIZES_KIND or any(p[0] not in ('box', 'cyl') for p in o['prims'])]
    assert not bad, f'v420 world holds non-v420 objects: {bad}'


# ----------------------------------------------------------------------------- V3 / V4: the train_div world (v420ood's placement loop)
def world_spec(split, index, rooms=1, attempt=0, identity=False):
    """train_div world of a TRAIN house (identity=True: v420's own world_id and library -- used only by the tests to prove that the
    placement loop is v420's, exactly as v420ood's internal 'v420' variant)"""
    assert str(split).upper() == 'TRAIN', 'v420div: TRAIN only (VALIDATION / TEST are refused)'
    base = W.v371.world_spec(split, index, rooms)
    L = base['chart_m']
    d = json.load(open(os.path.join(os.environ['HOUSEEXPO_JSON_DIR'], base['house_id'] + '.json')))
    v = np.asarray(d['verts'], np.float64); centre = (v.min(0) + v.max(0)) / 2
    shift_ = -centre + L / 2
    room_rects = [(cat, (b[0] + shift_[0], b[1] + shift_[1], b[2] + shift_[0], b[3] + shift_[1])) for cat, bl in sorted(d.get('room_category', {}).items()) for b in bl]
    v420_id = f"v420:{base['split']}:{base['house_id']}"
    world_id = v420_id if identity else f"v420div:{VARIANT}:{base['split']}:{base['house_id']}" + (f':a{int(attempt)}' if attempt else '')
    prng = np.random.default_rng(W._seed(v420_id + ':geometry'))     # the plan (ceiling, door height) is v420's
    H_w = float(prng.uniform(2.4, 3.0)); h_d = float(prng.uniform(2.0, 2.15))
    rng = np.random.default_rng(W._seed(world_id + ':geometry'))
    rng.uniform(2.4, 3.0); rng.uniform(2.0, 2.15)                    # burnt (identity: the furniture rng state is exactly v420's)
    g512 = W.v371.rasterize(base, 512); free_all = g512['walls'] == 0; cell = L / 512
    door = g512['door_mask'] | np.kron(W.v371.rasterize(base, 128)['door_mask'], np.ones((4, 4), bool))
    near_door = ndimage.distance_transform_edt(~door) * cell < 0.6
    lab, nlab = ndimage.label(free_all)
    free = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:]))) if nlab else free_all
    taken = np.zeros_like(free)
    objects = []
    if not room_rects:
        rr, cc = np.nonzero(free)
        if len(rr):
            room_rects = [('Generic', (cc.min() * cell, rr.min() * cell, (cc.max() + 1) * cell, (rr.max() + 1) * cell))]
    for cat, rect_box in room_rects:
        c0b, r0b = max(0, int(math.floor(rect_box[0] / cell))), max(0, int(math.floor(rect_box[1] / cell)))
        c1b, r1b = min(512, int(math.ceil(rect_box[2] / cell))), min(512, int(math.ceil(rect_box[3] / cell)))
        sub = free[r0b:r1b, c0b:c1b]
        if sub.size == 0 or not sub.any():
            continue
        rows = np.nonzero(sub.any(1))[0]; cols = np.nonzero(sub.any(0))[0]
        rect = ((c0b + cols[0]) * cell, (r0b + rows[0]) * cell, (c0b + cols[-1] + 1) * cell, (r0b + rows[-1] + 1) * cell)
        kinds = W.ROOM_KINDS.get(cat, W.GENERIC_KINDS if cat == 'Generic' else ['box', 'plant'])
        area = max(0., (rect[2] - rect[0]) * (rect[3] - rect[1]))
        n = int(min(len(kinds) + 14, max(1, rng.poisson(area / W.DENSITY_M2))))
        extra = [str(rng.choice(W.GENERIC_KINDS)) if rng.random() < 0.3 else str(rng.choice(kinds)) for _ in range(max(0, n - len(kinds)))]
        order = list(kinds) + extra
        for src in order[:n]:
            if identity:
                mode, kind, f, pv = 'orig', src, None, 'ctrl'
            else:                                                       # V4: one X mode per slot
                mode = str(rng.choice(slot_modes(src)))
                if mode == 'rotate':
                    kind, f, pv = ROT_OF[src], None, 'novel'
                elif mode == 'shelf':
                    kind, f, pv = 'wall_shelf', None, 'novel'
                else:
                    kind, f, pv = src, float(rng.choice(WO.SCALE_FACTORS)), 'scale'
            for _ in range(20):                                         # placement attempts
                depth, width, against, payload = WO._plan(pv, kind, f, rng, H_w)
                if payload is not None:
                    payload = dict(payload, top=[])                     # V4: the table-top clutter (Y) is drawn but dropped
                inner = (rect[0] + 0.02, rect[1] + 0.02, rect[2] - 0.02, rect[3] - 0.02)
                if against:
                    side = str(rng.choice(['x0', 'x1', 'y0', 'y1']))
                    fp = W._against(side, inner, depth, width, rng)
                else:
                    side = None
                    if inner[2] - inner[0] < width + 0.6 or inner[3] - inner[1] < depth + 0.6: fp = None
                    else:
                        x = float(rng.uniform(inner[0] + 0.3, inner[2] - 0.3 - width)); y = float(rng.uniform(inner[1] + 0.3, inner[3] - 0.3 - depth))
                        fp = (x, y, x + width, y + depth)
                if fp is None:
                    continue
                prims, attached = WO._build(pv, kind, fp, side, against, payload, f, rng, H_w)
                assert not attached, 'train_div: an attached (table-top) object survived'
                if not prims:
                    continue
                push = {'x0': (1, 0), 'x1': (-1, 0), 'y0': (0, 1), 'y1': (0, -1)}.get(side, (0, 0)) if against else (0, 0)
                ok = False
                for k_push in range(0, 25 if against else 1):
                    dx, dy = push[0] * 0.05 * k_push, push[1] * 0.05 * k_push
                    fx0, fy0, fx1, fy1 = WO.footprint(prims); fx0 += dx; fx1 += dx; fy0 += dy; fy1 += dy
                    c0, r0 = int(math.floor(fx0 / cell)), int(math.floor(fy0 / cell)); c1, r1 = int(math.ceil(fx1 / cell)), int(math.ceil(fy1 / cell))
                    if c0 < 0 or r0 < 0 or c1 > 512 or r1 > 512:
                        continue
                    win = np.s_[r0:r1, c0:c1]
                    if free[win].all():
                        ok = True; break
                if not ok or taken[win].any() or near_door[win].any():
                    continue
                if against:
                    back = {'x0': np.s_[r0:r1, max(0, c0 - 2):c0], 'x1': np.s_[r0:r1, c1:min(512, c1 + 2)],
                            'y0': np.s_[max(0, r0 - 2):r0, c0:c1], 'y1': np.s_[r1:min(512, r1 + 2), c0:c1]}[side]
                    strip = free_all[back]
                    if strip.size == 0 or (~strip).mean() < 0.5:
                        continue
                if dx or dy:
                    prims = [WO.shift(p, dx, dy) for p in prims]
                trial = taken.copy(); m = max(1, int(round(0.3 / cell))); trial[max(0, r0 - m):r1 + m, max(0, c0 - m):c1 + m] = True
                remaining = free & ~trial
                lab, _ = ndimage.label(remaining)
                if lab.max() == 0 or np.bincount(lab.ravel())[1:].max() < 0.95 * remaining.sum():
                    continue
                taken[win] = True
                o = dict(kind=kind, room=cat, prims=[list(p) for p in prims])
                if not identity:
                    fam = 'rotated' if mode == 'rotate' else ('wall_shelf' if mode == 'shelf' else f'scaled_{f:g}')
                    o.update(src_kind=src, mode=mode, family=fam, scale=f, placement='floor',
                             angle_deg=(math.degrees(payload['theta']) if payload else None))
                objects.append(o)
                break
    out = dict(version=W.VERSION, split=base['split'], index=int(index), rooms=int(rooms), house_id=base['house_id'], world_id=world_id,
               chart_m=L, height_m=W.HEIGHT_M, floor_t=W.FLOOR_T, wall_h=H_w, door_h=h_d, verts_m=base['verts_m'],
               room_rects=[(c, list(r)) for c, r in room_rects], objects=objects, base_spec=base)
    if not identity:
        check_div_objects(objects)
        out.update(div_version=VERSION, variant=VARIANT, plan_world_id=v420_id, attempt=int(attempt))
    return out


rasterize = WO.rasterize                    # v420 rasterize for box / cyl + v420ood's obox (no 'ell' ever occurs here)
object_masks = WO.object_masks


def _agent_skip(e):
    return isinstance(e, ValueError) and str(e) in AGENT_SKIPS


def make_fixture(split, index, rooms=1, dims=W.DIMS['R1'], agents=W.AGENTS):
    """the train_div fixture: = v420 make_fixture on the train_div world (same keys, same calls); V6 re-draws the furniture when the
    agents cannot be placed"""
    last = None
    for attempt in range(ATTEMPTS):
        spec = world_spec(split, index, rooms, attempt)
        r = rasterize(spec, tuple(dims)); occ = r['occ']; cell = r['cell']
        rng = np.random.default_rng(W._seed(spec['world_id'] + f':poses:{agents}'))
        ref = r if tuple(dims) == W.DIMS['R1'] else rasterize(spec, W.DIMS['R1'])
        try:
            centres = W.agent_positions(spec, ref['occ'].astype(bool), ref['cell'], agents, rng)
        except ValueError as e:
            if not _agent_skip(e):
                raise
            last = e
            continue
        jitter = rng.uniform(-.1, .1, (W.POSES, agents, 3)); jitter[:, :, 2] = 0.
        trajectory = centres[None] + jitter
        rays = W.fibonacci_sphere(W.BEAMS)
        private = np.stack([W.observe(occ, cell, trajectory[:, a], rays) for a in range(agents)])
        adj, edges = W.radio_forest(centres)
        reached = np.stack([W.forest_reached(edges, agents, h) for h in W.HOPS])
        return dict(spec=dict(split=split, index=int(index), rooms=int(rooms), dims=list(dims), agents=agents, version=W.VERSION, div_version=VERSION,
                              variant=VARIANT, attempt=attempt), world_id=spec['world_id'], chart_m=spec['chart_m'], occ=occ, slab=r['slab'], obj=r['obj'],
                    cell=cell, private=private, positions_m=centres, trajectory_m=trajectory, adjacency=adj, edges=edges, reached=reached, hops=W.HOPS,
                    n_objects=len(spec['objects']), wall_h=spec['wall_h'], variant=VARIANT, div_attempt=attempt)
    raise ValueError(str(last))


# ----------------------------------------------------------------------------- V7: the mixed TRAIN module (dispatch by V1)
def world_spec_mixed(split, index, rooms=1):
    if variant_of(split, index, rooms) == ORIG:
        spec = W.world_spec(split, index, rooms)
        check_orig_objects(spec['objects'])
        return spec
    return world_spec(split, index, rooms)


def make_fixture_mixed(split, index, rooms=1, dims=W.DIMS['R1'], agents=W.AGENTS):
    if variant_of(split, index, rooms) == ORIG:
        fx = W.make_fixture(split, index, rooms, dims, agents)
        fx['variant'] = ORIG; fx['div_attempt'] = None
        return fx
    return make_fixture(split, index, rooms, dims, agents)


def module_mixed():
    """a module object with every v420 attribute, world_spec / make_fixture dispatched by V1 (for g4_e3r's house_ctx via E.W)"""
    m = types.ModuleType('mapping_3d_storey_data_v420div_mixed')
    m.__dict__.update({k: v for k, v in vars(W).items() if not (k.startswith('__') and k.endswith('__'))})
    m.__file__ = os.path.abspath(__file__)
    m.DIV_VERSION = VERSION; m.VARIANT = 'mixed'
    m.world_spec = world_spec_mixed
    m.rasterize = rasterize
    m.make_fixture = make_fixture_mixed
    return m


# ----------------------------------------------------------------------------- V8: per-house statistics (no ray casting)
def family_stats(split, index, rooms=1, dims=W.DIMS['R1'], attempt=None):
    """objects and voxels per family / kind of the house's assigned world (attempt: the train_div attempt actually used, default 0);
    asserts that the per-object masks reproduce the object raster and that the held-out families are absent"""
    var = variant_of(split, index, rooms); dims = tuple(dims)
    if var == ORIG:
        spec = W.world_spec(split, index, rooms); check_orig_objects(spec['objects'])
        for o in spec['objects']:
            o.setdefault('family', 'orig')
    else:
        spec = world_spec(split, index, rooms, int(attempt or 0))
    r = rasterize(spec, dims)
    masks = object_masks(spec, dims)
    union = np.zeros(dims, bool)
    for mk in masks:
        union |= mk
    assert np.array_equal(union & ~r['slab'], r['obj']), 'per-object masks do not reproduce the object raster'
    fam, kin = {}, {}
    for o, mk in zip(spec['objects'], masks):
        nv = int((mk & ~r['slab']).sum())
        a = fam.setdefault(o['family'], [0, 0]); a[0] += 1; a[1] += nv
        b = kin.setdefault(o['kind'], [0, 0]); b[0] += 1; b[1] += nv
    y_obj = sum(v[0] for k, v in fam.items() if k in Y_FAMILIES) + sum(v[0] for k, v in kin.items() if k in Y_KINDS)
    y_vox = sum(v[1] for k, v in fam.items() if k in Y_FAMILIES) + sum(v[1] for k, v in kin.items() if k in Y_KINDS)
    y_prims = sum(1 for o in spec['objects'] for p in o['prims'] if p[0] in Y_PRIMS)
    assert y_obj == 0 and y_vox == 0 and y_prims == 0, f'held-out (Y) objects in {house_name(rooms, index)}: {y_obj} objects, {y_vox} voxels'
    return dict(house=house_name(rooms, index), variant=var, world_id=spec['world_id'], attempt=(None if var == ORIG else int(attempt or 0)),
                n_objects=len(spec['objects']), obj_voxels=int(r['obj'].sum()), families=fam, kinds=kin, y_objects=int(y_obj), y_voxels=int(y_vox),
                y_prims=int(y_prims), wall_h=spec['wall_h'])
