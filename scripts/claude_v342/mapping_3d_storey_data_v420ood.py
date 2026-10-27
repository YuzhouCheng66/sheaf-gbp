"""mapping_3d_storey_data_v420ood.py (Claude, 2026-10-27): batch G4-H1 留出家具形状面板的派生生成器。
预注册 claude_handoff/OVERNIGHT_REPORT_20261021.md D8-53（背景 D8-48、D8-50；设计 review_E3R/RA_accuracy.md §3、RC_sheaf_plan.md §6 草案 A）。
v420（mapping_3d_storey_data_v420.py）冻结，这里只 import 它，从不修改；平面、传感、智能体、无线电全部调用 v420 的函数。

一、变体（单变量 = 测试世界的家具形状分布；D8-53 原文）
  ctrl     原家具库（v420 ROOM_KINDS / GENERIC_KINDS / SIZES_KIND / _make 及其椅子规则），新的世界种子；
  novel    只放新物体（见三）；
  scale    已有类别，每个物体整体乘 f ∈ {0.6, 1.5}（等概率，逐物体抽一次）；
  floating 只放离地的挂墙层板与上悬物。
  world_id = v420ood:{variant}:{split}:{house_id}，几何随机数 rng = default_rng(sha(world_id + ':geometry'))，随 world_id 独立；
  智能体位姿随机数照 v420 用 world_id + ':poses:8'，所以也随变体独立。

二、操作化（D8-53 没写死、这里定下的部分）
  O1 平面逐位不变：墙、门洞、楼板由 v371 光栅决定；层高 H_w 与门高 h_d 在 v420 里是几何 rng 的前两次抽样。这里一律取 v420 自己的
     world_id（v420:{split}:{house_id}）种子的前两次抽样，所以四个变体与 v420 的墙、楼板、天花板、门楣体素逐位相同（单测断言）。
     家具 rng 先烧掉同样的两次抽样，再进入放置循环；因此内部变体 'v420'（world_id 取 v420 原值、原家具库）逐物体复现 v420.world_spec
     （单测断言 objects 完全相等，rasterize 逐位相等）——这证明放置循环是 v420 的忠实拷贝，只有家具库换了。
  O2 放置：放置循环逐行照抄 v420（每 1.2 m² 一次尝试、房间自身类别 + 30 % 通用类别、每物体 20 次尝试、靠墙物体向内推、背面 ≥ 50 %
     贴墙、离开口 0.6 m、0.3 m 步行余量下剩余地面连通 ≥ 95 %）；碰撞与连通都用物体全部图元的外包轴对齐矩形（旋转件即旋转后的外包
     矩形）。"占地比例不变"操作化为"尝试密度与放置规则不变"，实际占地份额随物体形状变化，在冒烟 / 几何检查里报告。
  O3 每个房间的物体槽位与 v420 相同（类别表、数量、30 % 通用抽样），novel / floating 把每个槽位的原类别映射成新类别（NOVEL_OF /
     FLOATING_OF），这样房间语义与每房物体数的分布保持不变。
  O4 新物体（novel；尺寸为米，z 为离地板顶的高度）：
     rot_bed / rot_sofa / rot_table：v420 的床、沙发、桌子（_make 原样，局部坐标靠背 / 床头在 y0 侧）整体绕 z 转 θ ~ U(15°, 75°)，
       自由放置（转过的家具不能贴墙）；桌高 U(0.40, 0.78)；桌面上以 0.5 概率放一组小件杂物（单独记为一个 clutter 物体）；
     ball：球，直径 U(0.4, 0.9)，落地；ellipsoid：轴对齐椭球（豆袋），rx, ry 由占地给出，rz ~ U(0.2, 0.45)，落地；
     wall_shelf：靠墙，无腿，底面离地 U(1.0, 1.9)；0.5 概率为 1–3 块 3 cm 层板（间距 0.3 m），否则为高 U(0.3, 0.6) 的吊柜；
       顶面不高于天花板下 5 cm；
     clutter：0.10–0.25 m 的小盒（每边独立 U(0.10, 0.25)，绕 z 任意转），地上一堆 1–3 个，每个以 0.5 概率在上面再叠一个；
     l_shape：靠墙，主臂沿墙（进深 U(0.45, 0.65)），一端有垂直副臂伸出到总进深，两臂高度各 U(0.45, 0.95)；
     step_shape：靠墙，2–4 级台阶沿墙排列，首级高 U(0.3, 0.6)，每级加 U(0.25, 0.45)；
     coat_rack：细立杆，杆半径 U(0.010, 0.015)（杆宽 2–3 cm），高 U(1.6, 1.9)，底座 3 cm 厚圆盘，顶端两根 2 cm 交叉挂钩。
  O5 floating：靠墙类槽位 -> wall_shelf，自由类槽位 -> hanging（0.5 概率为吊灯：椭球灯罩中心离地 U(1.6, 2.1)、
     半高 U(0.10, 0.25)，1 cm 吊线到天花板；否则为吊箱：底面离地 U(1.5, 2.0)、高 U(0.2, 0.5)，两根 1 cm 吊杆到天花板）。
  O6 scale：先按 f 缩放后的尺寸取占地（放置规则看到真实大小），再在未缩放的占地上用 v420 _make（含椅子）造物体，然后以锚点
     （靠墙物体为背面中点，自由物体为占地中心）为中心在 x, y 上乘 f、z 乘 f、圆柱半径乘 f；z 超过天花板的部分截掉。
  O7 新图元（与 'box'、'cyl' 并列，单位米）：
     ('obox', cx, cy, hx, hy, theta, z0, z1)：绕 z 旋转 theta（弧度）的盒子，半宽 hx（局部 u 轴）、hy（局部 v 轴）；
     ('ell', cx, cy, cz, rx, ry, rz)：轴对齐椭球，cz 为球心离地板顶的高度。
     栅格化沿用"格心包含"，并把"比一格薄的部件保留含其中心的一格"推广为：
       obox 的某个局部轴全宽 2h < 一格（水平格宽）时，该轴只保留中线（u = 0 或 v = 0）穿过的格子（格子在该轴上的投影半宽
       w = c/2 (|cos θ| + |sin θ|)，条件 -w < u ≤ w）；两轴都薄时保留含中心的那一格；z 轴规则与 v420 'box' 相同。
       θ = 0 时这条规则逐格退化为 v420 'box' 的逐轴规则（单测断言）。若结果为空，保留含中心的格。
       ell 的某轴直径 2r < 该轴格宽时，该轴只保留含中心的那一层（该轴项取 0）；结果为空时保留含中心的格。
     与 v420 相同：物体体素扣掉墙柱（obj &= ~walls），occ |= obj，obj 掩码再扣掉楼板。
  O8 计分分区掩码：每个物体带 ood 标志（novel：全部新物体；scale：全部被缩放的物体；floating：全部离地物体；ctrl：无）。
     make_fixture 额外输出 ood_mask（带标志物体体素之并，已扣墙柱）、每物体体素的平铺下标（objvox_idx / objvox_off）、类别与家族，
     评估脚本把 ood_mask 膨胀 1 格（G.dil6）作为"新物体区 / 被缩放或离地物体区"。
  O9 G / g4_e3r 的接口：make_fixture(split, index, rooms=1, dims, agents) 的返回键与 v420 一致（另加上述键）。G.W 与 E.W 在 import
     时绑定 v420 模块对象；module_for(variant) 返回一个模块对象（v420 的全部属性 + 绑定了变体的 world_spec / rasterize /
     make_fixture），评估脚本只做一处替换 E.W = module_for(variant)（E.house_ctx 通过模块全局 W 调 make_fixture），冻结文件不改。

不生成 TEST（v365 / v371 的 world_spec 在 TEST 上抛错）。HouseExpo 原始数据只从 HOUSEEXPO_JSON_DIR 读取，不复制、不导出。
"""
import json
import math
import os
import types

import numpy as np
from scipy import ndimage

import mapping_3d_storey_data_v420 as W            # frozen; import only

VERSION = '420ood'
VARIANTS = ('ctrl', 'novel', 'scale', 'floating')
IDENTITY = 'v420'                                   # internal: v420's own world_id + original library (reproduces v420 exactly)
ROT_DEG = (15.0, 75.0)
SCALE_FACTORS = (0.6, 1.5)
SHELF_Z = (1.0, 1.9)
CLUTTER_EDGE = (0.10, 0.25)
POLE_R = (0.010, 0.015)
TOP_CLUTTER_P = 0.5
CEIL_GAP = 0.05

# ----------------------------------------------------------------------------- kinds
NOVEL_OF = dict(bed='rot_bed', sofa='rot_sofa', dining_table='rot_table', coffee_table='rot_table', desk='rot_table', round_table='ellipsoid',
                wardrobe='step_shape', cabinet_tall='step_shape', fridge='step_shape', car='step_shape',
                cabinet='l_shape', tv_cabinet='l_shape', shoe_cabinet='l_shape', sink='l_shape', counter='l_shape', bathtub='l_shape',
                toilet='l_shape', nightstand='clutter', bookshelf='wall_shelf', lamp='coat_rack', plant='ball', box='clutter')
FLOATING_OF = {k: ('wall_shelf' if v[2] else 'hanging') for k, v in W.SIZES_KIND.items()}
# kind -> (depth range, width range, against); rotated kinds: the LOCAL (unrotated) depth / width; ball: depth = diameter
NEW_SIZES = dict(rot_bed=((1.9, 2.1), (1.2, 1.8), False), rot_sofa=((0.85, 0.95), (1.6, 2.4), False), rot_table=((0.6, 1.0), (0.9, 1.6), False),
                 ball=((0.4, 0.9), (0.4, 0.9), False), ellipsoid=((0.5, 1.0), (0.6, 1.2), False), clutter=((0.3, 0.6), (0.3, 0.6), False),
                 coat_rack=((0.4, 0.5), (0.4, 0.5), False), hanging=((0.4, 0.9), (0.4, 0.9), False),
                 l_shape=((1.0, 1.6), (1.2, 2.2), True), step_shape=((0.4, 0.8), (0.9, 1.8), True), wall_shelf=((0.2, 0.35), (0.6, 1.6), True))
ROTATED = {'rot_bed': 'bed', 'rot_sofa': 'sofa', 'rot_table': 'table'}
FAMILY = dict(rot_bed='rotated', rot_sofa='rotated', rot_table='rotated', ball='ellipsoid', ellipsoid='ellipsoid', clutter='clutter',
              coat_rack='pole', l_shape='composite', step_shape='composite', wall_shelf='wall_shelf', hanging='hanging')
assert set(NOVEL_OF) == set(W.SIZES_KIND) and set(FLOATING_OF) == set(W.SIZES_KIND)


def variant_kind(variant, src):
    if variant == 'novel':
        return NOVEL_OF[src]
    if variant == 'floating':
        return FLOATING_OF[src]
    return src


def family_of(variant, kind, f=None):
    if variant in ('ctrl', IDENTITY):
        return 'orig'
    if variant == 'scale':
        return f'scaled_{f:g}'
    return FAMILY[kind]


# ----------------------------------------------------------------------------- primitive helpers
def footprint(prims):
    """axis-aligned bounding rectangle (x0, y0, x1, y1) of all primitives (= v420 _footprint on box / cyl)"""
    if all(p[0] in ('box', 'cyl') for p in prims):
        return W._footprint(prims)
    a = []
    for p in prims:
        if p[0] == 'box':
            a.append((p[1], p[2], p[4], p[5]))
        elif p[0] == 'cyl':
            a.append((p[1] - p[3], p[2] - p[3], p[1] + p[3], p[2] + p[3]))
        elif p[0] == 'obox':
            _, cx, cy, hx, hy, th, _, _ = p; c, s = abs(math.cos(th)), abs(math.sin(th))
            ex, ey = c * hx + s * hy, s * hx + c * hy
            a.append((cx - ex, cy - ey, cx + ex, cy + ey))
        elif p[0] == 'ell':
            a.append((p[1] - p[4], p[2] - p[5], p[1] + p[4], p[2] + p[5]))
        else:
            raise ValueError(p[0])
    a = np.asarray(a, np.float64)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max())


def shift(p, dx, dy):
    if p[0] in ('box', 'cyl'):
        return W._shift(p, dx, dy)
    if p[0] in ('obox', 'ell'):
        return (p[0], p[1] + dx, p[2] + dy) + tuple(p[3:])
    raise ValueError(p[0])


def rotate_local(prims, theta):
    """rotate primitives given in local coordinates (about the origin) by theta about z: box -> obox, cyl -> cyl, obox -> obox"""
    c, s = math.cos(theta), math.sin(theta); out = []
    for p in prims:
        if p[0] == 'box':
            _, x0, y0, z0, x1, y1, z1 = p; mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            out.append(('obox', c * mx - s * my, s * mx + c * my, (x1 - x0) / 2, (y1 - y0) / 2, theta, z0, z1))
        elif p[0] == 'cyl':
            _, mx, my, r, z0, z1 = p
            out.append(('cyl', c * mx - s * my, s * mx + c * my, r, z0, z1))
        elif p[0] == 'obox':
            _, mx, my, hx, hy, th, z0, z1 = p
            out.append(('obox', c * mx - s * my, s * mx + c * my, hx, hy, th + theta, z0, z1))
        else:
            raise ValueError(p[0])
    return out


def _wall_box(fp, side, a0, a1, o0, o1, z0, z1):
    """a box at along-wall interval [a0, a1] and out-of-wall interval [o0, o1] (metres from the wall) of a footprint backed on `side`"""
    x0, y0, x1, y1 = fp
    if side == 'y0':
        return ('box', x0 + a0, y0 + o0, z0, x0 + a1, y0 + o1, z1)
    if side == 'y1':
        return ('box', x0 + a0, y1 - o1, z0, x0 + a1, y1 - o0, z1)
    if side == 'x0':
        return ('box', x0 + o0, y0 + a0, z0, x0 + o1, y0 + a1, z1)
    return ('box', x1 - o1, y0 + a0, z0, x1 - o0, y0 + a1, z1)


def _along_out(fp, side):
    x0, y0, x1, y1 = fp
    return ((x1 - x0), (y1 - y0)) if side in ('y0', 'y1') else ((y1 - y0), (x1 - x0))


def _clutter_items(rng, cx, cy, half_x, half_y, z_base):
    """1-3 small boxes (edges U(0.10, 0.25), any yaw) in [cx +- half_x] x [cy +- half_y] on z_base, each with p = .5 one stacked on top"""
    out = []
    for _ in range(int(rng.integers(1, 4))):
        ex, ey, ez = (float(rng.uniform(*CLUTTER_EDGE)) for _ in range(3))
        r = 0.5 * math.hypot(ex, ey)
        px = float(rng.uniform(cx - max(0., half_x - r), cx + max(0., half_x - r)))
        py = float(rng.uniform(cy - max(0., half_y - r), cy + max(0., half_y - r)))
        out.append(('obox', px, py, ex / 2, ey / 2, float(rng.uniform(0, math.pi / 2)), z_base, z_base + ez))
        if rng.random() < 0.5:
            fx, fy, fz = (float(rng.uniform(*CLUTTER_EDGE)) for _ in range(3))
            out.append(('obox', px, py, fx / 2, fy / 2, float(rng.uniform(0, math.pi / 2)), z_base + ez, z_base + ez + fz))
    return out


def _v420_chairs(kind, fp, side, against, rng):
    """the chairs v420's world_spec adds around dining tables and wall-backed desks (copied verbatim, same rng calls)"""
    prims = []
    if kind == 'dining_table':
        x0, y0, x1, y1 = fp; cw = 0.45
        for k in range(int(rng.integers(2, 4))):
            cx = x0 + (k + 0.5) * (x1 - x0) / 3 - cw / 2
            prims += W._chair(cx, y0 - 0.5, cx + cw, y0 - 0.05, 'y0', rng) + W._chair(cx, y1 + 0.05, cx + cw, y1 + 0.5, 'y1', rng)
    if kind == 'desk' and against:
        x0, y0, x1, y1 = fp; cw = 0.45
        if side in ('y0', 'y1'):
            cx = (x0 + x1) / 2 - cw / 2
            prims += W._chair(cx, y1 + 0.05, cx + cw, y1 + 0.5, 'y1', rng) if side == 'y0' else W._chair(cx, y0 - 0.5, cx + cw, y0 - 0.05, 'y0', rng)
        else:
            cy = (y0 + y1) / 2 - cw / 2
            prims += W._chair(x1 + 0.05, cy, x1 + 0.5, cy + cw, 'x1', rng) if side == 'x0' else W._chair(x0 - 0.5, cy, x0 - 0.05, cy + cw, 'x0', rng)
    return prims


def _scale_prim(p, ax, ay, f, wall_h):
    if p[0] == 'box':
        _, x0, y0, z0, x1, y1, z1 = p
        z0, z1 = f * z0, min(f * z1, wall_h)
        if z0 >= wall_h:
            return None
        return ('box', ax + f * (x0 - ax), ay + f * (y0 - ay), z0, ax + f * (x1 - ax), ay + f * (y1 - ay), z1)
    if p[0] == 'cyl':
        _, cx, cy, r, z0, z1 = p
        z0, z1 = f * z0, min(f * z1, wall_h)
        if z0 >= wall_h:
            return None
        return ('cyl', ax + f * (cx - ax), ay + f * (cy - ay), f * r, z0, z1)
    raise ValueError(p[0])


def _anchor(fp, side, against):
    x0, y0, x1, y1 = fp
    if not against:
        return (x0 + x1) / 2, (y0 + y1) / 2
    return {'y0': ((x0 + x1) / 2, y0), 'y1': ((x0 + x1) / 2, y1), 'x0': (x0, (y0 + y1) / 2), 'x1': (x1, (y0 + y1) / 2)}[side]


# ----------------------------------------------------------------------------- per-kind planning and building
def _plan(variant, kind, f, rng, wall_h):
    """one placement attempt's footprint size: (depth = y extent, width = x extent for free-standing objects; out / along for
    wall-backed ones), against, payload. Original kinds draw exactly v420's (depth, width) calls."""
    if kind in W.SIZES_KIND and variant in ('ctrl', 'scale', IDENTITY):
        (dlo, dhi), (wlo, whi), against = W.SIZES_KIND[kind]
        depth, width = float(rng.uniform(dlo, dhi)), float(rng.uniform(wlo, whi))
        if variant == 'scale':
            depth, width = depth * f, width * f
        return depth, width, against, None
    (dlo, dhi), (wlo, whi), against = NEW_SIZES[kind]
    depth, width = float(rng.uniform(dlo, dhi)), float(rng.uniform(wlo, whi))
    if kind == 'ball':
        width = depth
    if kind not in ROTATED:
        return depth, width, against, None
    theta = math.radians(float(rng.uniform(*ROT_DEG)))
    lfp = (-width / 2, -depth / 2, width / 2, depth / 2)
    if ROTATED[kind] == 'table':
        h = float(rng.uniform(0.40, 0.78)); local = W._table(*lfp, h)
    else:
        local = W._make(ROTATED[kind], lfp, 'y0', rng); h = None
    top = []
    if ROTATED[kind] == 'table' and rng.random() < TOP_CLUTTER_P:
        top = _clutter_items(rng, 0., 0., max(0., width / 2 - 0.05), max(0., depth / 2 - 0.05), h)
    rot = rotate_local(local, theta)
    fx0, fy0, fx1, fy1 = footprint(rot)
    return fy1 - fy0, fx1 - fx0, False, dict(prims=rot, top=rotate_local(top, theta), off=(-fx0, -fy0), theta=theta)


def _build(variant, kind, fp, side, against, payload, f, rng, wall_h):
    """primitives of one object (and the attached objects: clutter on a table top) for footprint fp"""
    x0, y0, x1, y1 = fp
    if kind in W.SIZES_KIND and variant in ('ctrl', IDENTITY):
        return W._make(kind, fp, side if side else 'y0', rng) + _v420_chairs(kind, fp, side, against, rng), []
    if kind in W.SIZES_KIND and variant == 'scale':
        ax, ay = _anchor(fp, side, against)
        fp0 = (ax + (x0 - ax) / f, ay + (y0 - ay) / f, ax + (x1 - ax) / f, ay + (y1 - ay) / f)
        prims0 = W._make(kind, fp0, side if side else 'y0', rng) + _v420_chairs(kind, fp0, side, against, rng)
        prims = [q for q in (_scale_prim(p, ax, ay, f, wall_h) for p in prims0) if q is not None]
        return prims, []
    if kind in ROTATED:
        dx, dy = x0 + payload['off'][0], y0 + payload['off'][1]
        att = [('clutter', [shift(p, dx, dy) for p in payload['top']])] if payload['top'] else []
        return [shift(p, dx, dy) for p in payload['prims']], att
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    top_ok = wall_h - CEIL_GAP
    if kind == 'ball':
        r = (x1 - x0) / 2
        return [('ell', cx, cy, r, r, r, r)], []
    if kind == 'ellipsoid':
        rz = float(rng.uniform(0.2, 0.45))
        return [('ell', cx, cy, rz, (x1 - x0) / 2, (y1 - y0) / 2, rz)], []
    if kind == 'clutter':
        return _clutter_items(rng, cx, cy, (x1 - x0) / 2, (y1 - y0) / 2, 0.), []
    if kind == 'coat_rack':
        h = float(rng.uniform(1.6, 1.9)); rp = float(rng.uniform(*POLE_R)); rb = 0.45 * min(x1 - x0, y1 - y0)
        return [('cyl', cx, cy, rb, 0., 0.03), ('cyl', cx, cy, rp, 0., h),
                ('box', cx - 0.2, cy - 0.01, h - 0.12, cx + 0.2, cy + 0.01, h - 0.10), ('box', cx - 0.01, cy - 0.2, h - 0.12, cx + 0.01, cy + 0.2, h - 0.10)], []
    if kind == 'hanging':
        if rng.random() < 0.5:                                              # pendant: ellipsoid shade on a 1 cm cord
            rz = float(rng.uniform(0.10, 0.25)); cz = min(float(rng.uniform(1.6, 2.1)), wall_h - 0.15 - rz)
            return [('ell', cx, cy, cz, (x1 - x0) / 2, (y1 - y0) / 2, rz), ('cyl', cx, cy, 0.01, cz + rz, wall_h)], []
        z0 = float(rng.uniform(1.5, 2.0)); z1 = min(z0 + float(rng.uniform(0.2, 0.5)), wall_h - 0.10)
        return [('box', x0, y0, z0, x1, y1, z1), ('cyl', x0 + 0.1, cy, 0.01, z1, wall_h), ('cyl', x1 - 0.1, cy, 0.01, z1, wall_h)], []
    along, out = _along_out(fp, side)
    if kind == 'wall_shelf':
        zb = float(rng.uniform(*SHELF_Z))
        if rng.random() < 0.5:                                              # 1-3 boards, 3 cm thick, 0.3 m apart
            nb = int(rng.integers(1, 4))
            return [_wall_box(fp, side, 0., along, 0., out, zb + 0.3 * i, zb + 0.3 * i + 0.03) for i in range(nb) if zb + 0.3 * i + 0.03 <= top_ok] , []
        z1 = min(zb + float(rng.uniform(0.3, 0.6)), top_ok)
        return [_wall_box(fp, side, 0., along, 0., out, zb, z1)], []
    if kind == 'l_shape':
        dA = min(float(rng.uniform(0.45, 0.65)), out - 0.3); wB = min(float(rng.uniform(0.45, 0.7)), along / 2)
        hA, hB = float(rng.uniform(0.45, 0.95)), float(rng.uniform(0.45, 0.95))
        a0 = 0. if rng.random() < 0.5 else along - wB
        return [_wall_box(fp, side, 0., along, 0., dA, 0., hA), _wall_box(fp, side, a0, a0 + wB, dA, out, 0., hB)], []
    if kind == 'step_shape':
        n = int(rng.integers(2, 5)); h1 = float(rng.uniform(0.3, 0.6)); dh = float(rng.uniform(0.25, 0.45)); rev = rng.random() < 0.5
        out_p = []
        for i in range(n):
            hi = min(h1 + (n - 1 - i if rev else i) * dh, top_ok)
            out_p.append(_wall_box(fp, side, i * along / n, (i + 1) * along / n, 0., out, 0., hi))
        return out_p, []
    raise ValueError(kind)


# ----------------------------------------------------------------------------- world spec (v420's placement loop, library swapped)
def world_spec(variant, split, index, rooms=1):
    assert variant in VARIANTS or variant == IDENTITY, variant
    base = W.v371.world_spec(split, index, rooms)                     # house-fitted chart; raises on TEST
    L = base['chart_m']
    d = json.load(open(os.path.join(os.environ['HOUSEEXPO_JSON_DIR'], base['house_id'] + '.json')))
    v = np.asarray(d['verts'], np.float64); centre = (v.min(0) + v.max(0)) / 2
    shift_ = -centre + L / 2
    room_rects = [(cat, (b[0] + shift_[0], b[1] + shift_[1], b[2] + shift_[0], b[3] + shift_[1])) for cat, bl in sorted(d.get('room_category', {}).items()) for b in bl]
    v420_id = f"v420:{base['split']}:{base['house_id']}"
    world_id = v420_id if variant == IDENTITY else f"v420ood:{variant}:{base['split']}:{base['house_id']}"
    prng = np.random.default_rng(W._seed(v420_id + ':geometry'))     # O1: the plan (ceiling, door height) is v420's for every variant
    H_w = float(prng.uniform(2.4, 3.0)); h_d = float(prng.uniform(2.0, 2.15))
    rng = np.random.default_rng(W._seed(world_id + ':geometry'))
    rng.uniform(2.4, 3.0); rng.uniform(2.0, 2.15)                    # burnt (IDENTITY: the furniture rng state is exactly v420's)
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
            kind = variant_kind(variant, src)
            f = float(rng.choice(SCALE_FACTORS)) if variant == 'scale' else None
            for _ in range(20):                                         # placement attempts
                depth, width, against, payload = _plan(variant, kind, f, rng, H_w)
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
                prims, attached = _build(variant, kind, fp, side, against, payload, f, rng, H_w)
                if not prims:
                    continue
                push = {'x0': (1, 0), 'x1': (-1, 0), 'y0': (0, 1), 'y1': (0, -1)}.get(side, (0, 0)) if against else (0, 0)
                ok = False
                for k_push in range(0, 25 if against else 1):
                    dx, dy = push[0] * 0.05 * k_push, push[1] * 0.05 * k_push
                    fx0, fy0, fx1, fy1 = footprint(prims); fx0 += dx; fx1 += dx; fy0 += dy; fy1 += dy
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
                    prims = [shift(p, dx, dy) for p in prims]
                    attached = [(ak, [shift(p, dx, dy) for p in ap]) for ak, ap in attached]
                trial = taken.copy(); m = max(1, int(round(0.3 / cell))); trial[max(0, r0 - m):r1 + m, max(0, c0 - m):c1 + m] = True
                remaining = free & ~trial
                lab, _ = ndimage.label(remaining)
                if lab.max() == 0 or np.bincount(lab.ravel())[1:].max() < 0.95 * remaining.sum():
                    continue
                taken[win] = True
                o = dict(kind=kind, room=cat, prims=[list(p) for p in prims])
                if variant != IDENTITY:
                    o.update(src_kind=src, ood=variant != 'ctrl', family=family_of(variant, kind, f), scale=f, placement='floor',
                             angle_deg=(math.degrees(payload['theta']) if payload else None))
                objects.append(o)
                for ak, ap in attached:
                    objects.append(dict(kind=ak, room=cat, prims=[list(p) for p in ap], src_kind=src, ood=True, family=FAMILY[ak], scale=None,
                                        placement='table_top', angle_deg=None))
                break
    out = dict(version=W.VERSION, split=base['split'], index=int(index), rooms=int(rooms), house_id=base['house_id'], world_id=world_id,
               chart_m=L, height_m=W.HEIGHT_M, floor_t=W.FLOOR_T, wall_h=H_w, door_h=h_d, verts_m=base['verts_m'],
               room_rects=[(c, list(r)) for c, r in room_rects], objects=objects, base_spec=base)
    if variant != IDENTITY:
        out.update(ood_version=VERSION, variant=variant, plan_world_id=v420_id)
    return out


# ----------------------------------------------------------------------------- rasterisation
def _grid(spec, dims):
    NX, NY, NZ = dims
    cx, cz = spec['chart_m'] / NX, W.HEIGHT_M / NZ
    return cx, cz, (np.arange(NX) + .5) * cx, (np.arange(NY) + .5) * cx, (np.arange(NZ) + .5) * cz


def prim_mask(p, spec, dims):
    """voxels [NX, NY, NZ] of one primitive (before removing wall columns). box / cyl: v420 rasterize verbatim."""
    NX, NY, NZ = dims
    cx, cz, xc, yc, zc = _grid(spec, dims)
    m = np.zeros((NX, NY, NZ), bool)
    FT = W.FLOOR_T
    if p[0] == 'box':
        _, x0, y0, z0, x1, y1, z1 = p
        ix = (xc >= x0) & (xc < x1); iy = (yc >= y0) & (yc < y1); iz = (zc >= FT + z0) & (zc < FT + z1)
        ix = ix if ix.any() else W._cell_of((x0 + x1) / 2, cx, NX)
        iy = iy if iy.any() else W._cell_of((y0 + y1) / 2, cx, NY)
        iz = iz if iz.any() else W._cell_of(FT + (z0 + z1) / 2, cz, NZ)
        m[np.ix_(ix, iy, iz)] = True
        return m
    if p[0] == 'cyl':
        _, ccx, ccy, r, z0, z1 = p
        m2 = (xc[:, None] - ccx) ** 2 + (yc[None, :] - ccy) ** 2 <= r ** 2
        if not m2.any():
            m2[W._cell_of(ccx, cx, NX), W._cell_of(ccy, cx, NY)] = True
        iz = (zc >= FT + z0) & (zc < FT + z1)
        iz = iz if iz.any() else W._cell_of(FT + (z0 + z1) / 2, cz, NZ)
        return m2[:, :, None] & iz[None, None, :]
    if p[0] == 'obox':
        _, ccx, ccy, hx, hy, th, z0, z1 = p
        c, s = math.cos(th), math.sin(th)
        dx = xc[:, None] - ccx; dy = yc[None, :] - ccy
        u = dx * c + dy * s; v = -dx * s + dy * c
        thin_u, thin_v = 2 * hx < cx, 2 * hy < cx
        if thin_u and thin_v:
            m2 = np.zeros((NX, NY), bool)
        else:
            w = 0.5 * cx * (abs(c) + abs(s))                              # half-width of a cell's projection on a rotated axis
            mu = ((-w < u) & (u <= w)) if thin_u else ((u >= -hx) & (u < hx))
            mv = ((-w < v) & (v <= w)) if thin_v else ((v >= -hy) & (v < hy))
            m2 = mu & mv
        if not m2.any():
            m2 = np.zeros((NX, NY), bool); m2[W._cell_of(ccx, cx, NX), W._cell_of(ccy, cx, NY)] = True
        iz = (zc >= FT + z0) & (zc < FT + z1)
        iz = iz if iz.any() else W._cell_of(FT + (z0 + z1) / 2, cz, NZ)
        return m2[:, :, None] & iz[None, None, :]
    if p[0] == 'ell':
        _, ex, ey, ez, rx, ry, rz = p
        terms, sels = [], []
        for ctr, r, cc, n, cen in ((ex, rx, cx, NX, xc), (ey, ry, cx, NY, yc), (FT + ez, rz, cz, NZ, zc)):
            if 2 * r < cc:                                                  # thinner than a cell on this axis: the layer of the centre
                terms.append(np.zeros(n)); sels.append(W._cell_of(ctr, cc, n))
            else:
                terms.append(((cen - ctr) / r) ** 2); sels.append(np.ones(n, bool))
        q = terms[0][:, None, None] + terms[1][None, :, None] + terms[2][None, None, :]
        m = (q <= 1.0) & sels[0][:, None, None] & sels[1][None, :, None] & sels[2][None, None, :]
        if not m.any():
            m[W._cell_of(ex, cx, NX), W._cell_of(ey, cx, NY), W._cell_of(FT + ez, cz, NZ)] = True
        return m
    raise ValueError(p[0])


def walls2d(spec, dims):
    return (W.v371.rasterize(spec['base_spec'], dims[0])['walls'] != 0).T


def rasterize(spec, dims):
    """= v420 rasterize for box / cyl (called on the box / cyl primitives) + the new primitives; same outputs and conventions"""
    dims = tuple(dims)
    plain = dict(spec, objects=[dict(o, prims=[p for p in o['prims'] if p[0] in ('box', 'cyl')]) for o in spec['objects']])
    r = W.rasterize(plain, dims)
    new = np.zeros(dims, bool)
    for o in spec['objects']:
        for p in o['prims']:
            if p[0] in ('obox', 'ell'):
                new |= prim_mask(p, spec, dims)
    if new.any():
        new &= ~walls2d(spec, dims)[:, :, None]
        occ = r['occ'].astype(bool) | new
        return dict(occ=occ.astype(np.uint8), slab=r['slab'], obj=r['obj'] | (new & ~r['slab']), cell=r['cell'])
    return r


def object_masks(spec, dims):
    """per-object voxel masks (wall columns removed, slabs kept) in the order of spec['objects']"""
    dims = tuple(dims); wl = walls2d(spec, dims)[:, :, None]
    out = []
    for o in spec['objects']:
        m = np.zeros(dims, bool)
        for p in o['prims']:
            m |= prim_mask(p, spec, dims)
        out.append(m & ~wl)
    return out


def ood_arrays(spec, dims, r=None):
    """scoring-partition arrays: ood_mask (union of the flagged objects), per-object flat voxel indices, kinds, families, flags"""
    masks = object_masks(spec, dims)
    r = r if r is not None else rasterize(spec, dims)
    union = np.zeros(tuple(dims), bool)
    for m in masks:
        union |= m
    assert np.array_equal(union & ~r['slab'], r['obj']), 'per-object masks do not reproduce the object raster'
    flags = np.array([bool(o.get('ood', False)) for o in spec['objects']], bool)
    ood = np.zeros(tuple(dims), bool)
    for m, fl in zip(masks, flags):
        if fl:
            ood |= m
    idx = [np.flatnonzero(m).astype(np.int32) for m in masks]
    off = np.zeros(len(idx) + 1, np.int64); off[1:] = np.cumsum([len(i) for i in idx])
    return dict(ood_mask=ood, objvox_idx=np.concatenate(idx) if idx else np.zeros(0, np.int32), objvox_off=off,
                obj_kinds=np.array([o['kind'] for o in spec['objects']], dtype='U16'),
                obj_family=np.array([o.get('family', 'orig') for o in spec['objects']], dtype='U16'), obj_ood=flags)


def make_fixture(variant, split, index, rooms=1, dims=W.DIMS['R1'], agents=W.AGENTS):
    """= v420 make_fixture on the variant's world (same keys, same calls) + the scoring-partition arrays"""
    spec = world_spec(variant, split, index, rooms)
    r = rasterize(spec, tuple(dims)); occ = r['occ']; cell = r['cell']
    rng = np.random.default_rng(W._seed(spec['world_id'] + f':poses:{agents}'))
    ref = r if tuple(dims) == W.DIMS['R1'] else rasterize(spec, W.DIMS['R1'])
    centres = W.agent_positions(spec, ref['occ'].astype(bool), ref['cell'], agents, rng)
    jitter = rng.uniform(-.1, .1, (W.POSES, agents, 3)); jitter[:, :, 2] = 0.
    trajectory = centres[None] + jitter
    rays = W.fibonacci_sphere(W.BEAMS)
    private = np.stack([W.observe(occ, cell, trajectory[:, a], rays) for a in range(agents)])
    adj, edges = W.radio_forest(centres)
    reached = np.stack([W.forest_reached(edges, agents, h) for h in W.HOPS])
    fx = dict(spec=dict(split=split, index=int(index), rooms=int(rooms), dims=list(dims), agents=agents, version=W.VERSION, ood_version=VERSION,
                        variant=variant), world_id=spec['world_id'], chart_m=spec['chart_m'], occ=occ, slab=r['slab'], obj=r['obj'], cell=cell,
              private=private, positions_m=centres, trajectory_m=trajectory, adjacency=adj, edges=edges, reached=reached, hops=W.HOPS,
              n_objects=len(spec['objects']), wall_h=spec['wall_h'], variant=variant)
    fx.update(ood_arrays(spec, dims, r))
    fx['objects'] = [dict(kind=o['kind'], family=o.get('family', 'orig'), ood=bool(o.get('ood', False)), placement=o.get('placement', 'floor'),
                          scale=o.get('scale'), angle_deg=o.get('angle_deg'), n_prims=len(o['prims'])) for o in spec['objects']]
    return fx


def geometry(variant, split, index, rooms=1, dims=W.DIMS['R1'], agents=W.AGENTS, place_agents=True):
    """geometry-only summary (no ray casting, no scoring): objects, voxel counts, footprint share, agent placement"""
    spec = world_spec(variant, split, index, rooms); dims = tuple(dims)
    r = rasterize(spec, dims); oa = ood_arrays(spec, dims, r)
    cx, cz = r['cell'][0], r['cell'][2]
    zc = (np.arange(dims[2]) + .5) * cz
    fl = r['obj'][:, :, (zc >= W.FLOOR_T)].any(2)
    free_floor = ~(r['occ'].astype(bool)[:, :, (zc >= W.FLOOR_T) & (zc < W.FLOOR_T + 2.0)].all(2))
    nv = np.diff(oa['objvox_off'])
    out = dict(variant=variant, world_id=spec['world_id'], n_objects=len(spec['objects']), obj_voxels=int(r['obj'].sum()),
               ood_voxels=int(oa['ood_mask'].sum()), footprint_share=float(fl.sum() / max(1, free_floor.sum())),
               kinds={}, families={}, wall_h=spec['wall_h'], door_h=spec['door_h'])
    for o, k in zip(spec['objects'], nv):
        a = out['kinds'].setdefault(o['kind'], [0, 0]); a[0] += 1; a[1] += int(k)
        b = out['families'].setdefault(o.get('family', 'orig'), [0, 0]); b[0] += 1; b[1] += int(k)
    if place_agents:
        try:
            W.agent_positions(spec, r['occ'].astype(bool), r['cell'], agents, np.random.default_rng(W._seed(spec['world_id'] + f':poses:{agents}')))
            out['agents_ok'] = True
        except ValueError as e:
            out['agents_ok'] = False; out['agents_error'] = repr(e)[:200]
    return out, spec, r, oa


def module_for(variant):
    """a module object with every v420 attribute and the variant's world_spec / rasterize / make_fixture (for G.W / E.W)"""
    m = types.ModuleType(f'mapping_3d_storey_data_v420ood_{variant}')
    m.__dict__.update({k: v for k, v in vars(W).items() if not (k.startswith('__') and k.endswith('__'))})
    m.__file__ = os.path.abspath(__file__)
    m.OOD_VERSION = VERSION; m.VARIANT = variant
    m.world_spec = lambda split, index, rooms=1: world_spec(variant, split, index, rooms)
    m.rasterize = rasterize
    m.make_fixture = lambda split, index, rooms=1, dims=W.DIMS['R1'], agents=W.AGENTS: make_fixture(variant, split, index, rooms, dims, agents)
    return m
