"""R-A (accuracy-first) review probe: ladder extrapolation + noise-world headroom.

Pure arithmetic on numbers already in the materials; no training, no GPU.
Inputs:
  - ladder numbers transcribed from OVERNIGHT_REPORT_20261021.md D8-24/27/33/34/37
  - S0 / downsampling numbers from synthesis_C1.md (F13) and probes/A06/ceilings.out
  - noise numbers read from probes/C1/c1_noise_seed2.json and g4/review_e1/probe_summary.json
Outputs: ra_ladder_headroom.json and stdout (saved as ra_ladder_headroom.out).
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # analysis_20261027

out = {}

# ---------------------------------------------------------------- 1. ladder
# R8 furniture F1 (VAL-24, mean of the two seeds). q = horizontal quantum in voxels.
ladder_2cyc = [  # two-cycle recipes
    dict(rung="16x16x1", q=8, bytes=512, furn=(0.388 + 0.392) / 2, rule="v420", src="D8-27 control (3S-T 2-cycle D2)"),
    dict(rung="32x32x1", q=4, bytes=2048, furn=(0.572 + 0.563) / 2, rule="v420", src="D8-27 3S-X"),
    dict(rung="32x32x1", q=4, bytes=2048, furn=(0.5871 + 0.5888) / 2, rule="calibrated", src="D8-37 control"),
    dict(rung="64x64x1", q=2, bytes=8192, furn=(0.6794 + 0.6791) / 2, rule="calibrated", src="D8-37 3S-Z2"),
]
ladder_16k = [  # 16k-update recipes, all v420
    dict(rung="16x16x1", q=8, bytes=512, furn=(0.301 + 0.309) / 2, src="D8-24 D2 16k"),
    dict(rung="32x32x1", q=4, bytes=2048, furn=(0.421 + 0.429) / 2, src="D8-24 L1"),
    dict(rung="64x64x1", q=2, bytes=8192, furn=(0.5073 + 0.4978) / 2, src="D8-34 z1"),
]


def gap_ratio(a, b):
    return (1 - b) / (1 - a)


steps = {}
g1 = gap_ratio(ladder_2cyc[0]["furn"], ladder_2cyc[1]["furn"])  # v420 -> v420
g2 = gap_ratio(ladder_2cyc[2]["furn"], ladder_2cyc[3]["furn"])  # cal -> cal
h1 = gap_ratio(ladder_16k[0]["furn"], ladder_16k[1]["furn"])
h2 = gap_ratio(ladder_16k[1]["furn"], ladder_16k[2]["furn"])
steps["two_cycle"] = dict(
    inc_16_32_pp=100 * (ladder_2cyc[1]["furn"] - ladder_2cyc[0]["furn"]),
    inc_32_64_pp=100 * (ladder_2cyc[3]["furn"] - ladder_2cyc[2]["furn"]),
    gap_ratio_16_32=g1, gap_ratio_32_64=g2,
    alpha_16_32=-math.log(g1) / math.log(4), alpha_32_64=-math.log(g2) / math.log(4),
)
steps["16k"] = dict(
    inc_16_32_pp=100 * (ladder_16k[1]["furn"] - ladder_16k[0]["furn"]),
    inc_32_64_pp=100 * (ladder_16k[2]["furn"] - ladder_16k[1]["furn"]),
    gap_ratio_16_32=h1, gap_ratio_32_64=h2,
)
out["ladder_steps"] = steps

f64 = ladder_2cyc[3]["furn"]
gap64 = 1 - f64
inc_last = steps["two_cycle"]["inc_32_64_pp"]
inc_prev = steps["two_cycle"]["inc_16_32_pp"]
extrap = {
    "A_increment_shrinks_as_before": f64 + (inc_last * inc_last / inc_prev) / 100,
    "B_gap_ratio_trend(+0.07)": 1 - gap64 * min(0.99, g2 + (g2 - g1)),
    "B2_gap_ratio_constant": 1 - gap64 * g2,
    "C_16k_increment_ratio": f64 + (inc_last * (steps["16k"]["inc_32_64_pp"] / steps["16k"]["inc_16_32_pp"])) / 100,
    "D_optimistic_gap_ratio_back_to_16_32": 1 - gap64 * g1,
}
out["furn_128_extrapolation"] = extrap

# surface / K0 (calibrated rule)
surf_r8 = dict(x=(0.9075 + 0.9100) / 2, z2=(0.9293 + 0.9283) / 2)
k0_fresh = dict(x=(0.9371 + 0.9373) / 2, z2=(0.9505 + 0.9488) / 2)
k0_val = dict(x=(0.9287 + 0.9300) / 2, z2=(0.9436 + 0.9422) / 2)
other = {}
for name, d in [("R8_surface_VAL", surf_r8), ("K0_fresh", k0_fresh), ("K0_VAL", k0_val)]:
    r = gap_ratio(d["x"], d["z2"])
    other[name] = dict(x=d["x"], z2=d["z2"], gap_ratio=r,
                       next_if_ratio_same=1 - (1 - d["z2"]) * r,
                       next_if_ratio_0p88=1 - (1 - d["z2"]) * 0.88)
out["other_metrics_128_extrapolation"] = other

# where would the ladder be at raw-grid size (131072 B = 16x 8 KB)?
raw = 131072
out["ladder_at_raw_grid_size"] = {
    f"alpha={a:.2f}": 1 - gap64 * (raw / 8192) ** (-a)
    for a in (steps["two_cycle"]["alpha_16_32"], steps["two_cycle"]["alpha_32_64"])
}
# bytes needed to reach furniture .99 with the fitted alpha
out["bytes_for_furn_0p99"] = {
    f"alpha={a:.2f}": 8192 * (gap64 / 0.01) ** (1 / a)
    for a in (steps["two_cycle"]["alpha_16_32"], steps["two_cycle"]["alpha_32_64"])
}

# ---------------------------------------------------------------- 2. byte multiples
ref = dict(raw=131072, gzip_own=2733, ctx_own=1384, ctx_union=2813, S0_own=219, S0_conv_msg=401, S0_union=723,
           down2x2x1_union=1295)
msgs = {"3S-X 2KB": 2048, "3S-Z2 8KB": 8192, "128x128x1 32KB": 32768}
out["byte_multiples"] = {m: {k: b / v for k, v in ref.items()} for m, b in msgs.items()}
out["R_net_per_house_8rounds_14edges"] = {m: 8 * 14 * b for m, b in msgs.items()}

# marginal bytes per furniture pp
s0_curve = [(82, 0.667), (108, 0.857), (146, 0.941), (203, 0.994), (219, 1.0)]  # own map K0 (C1 F13, F6)
s0_marg = [((b2 - b1) / (100 * (f2 - f1))) for (b1, f1), (b2, f2) in zip(s0_curve, s0_curve[1:])]
lad = [(512, ladder_2cyc[0]["furn"]), (2048, ladder_2cyc[1]["furn"]), (8192, f64),
       (32768, extrap["B_gap_ratio_trend(+0.07)"])]
lad_marg = [((b2 - b1) / (100 * (f2 - f1))) for (b1, f1), (b2, f2) in zip(lad, lad[1:])]
out["marginal_bytes_per_furn_pp"] = dict(S0_lossy_curve=s0_marg, learned_ladder=lad_marg)

# downsampling ceilings at the same horizontal quantum (A06 ceilings.out, hop 8 = R8)
down = {4: 0.488, 2: 0.770, 1: 1.0}
out["learned_over_downsample_ceiling"] = {
    "q=4": ((0.5871 + 0.5888) / 2) / down[4],
    "q=2": f64 / down[2],
}

# ---------------------------------------------------------------- 3. noise headroom
c1 = json.load(open(os.path.join(ROOT, "probes", "C1", "c1_noise_seed2.json")))["conds"]
ps = json.load(open(os.path.join(ROOT, "g4", "review_e1", "probe_summary.json")))
r1b = ps["r1b"]
r1 = ps["r1"]


def g(d, k):
    return d.get(k)


noise = {}
noise["C1_seed2_V2_pooled_6houses"] = {
    c: dict(K0=c1[c]["K0_furn_f1"], R8_sum=c1[c]["furn_f1_pooled"]["sum"], R8_max=c1[c]["furn_f1_pooled"]["max"])
    for c in ("range_only", "pose_only", "cam_nf", "cam")
}
noise["E1review_V3_4houses_seed1"] = {
    c: dict(K0=r1b[f"{c}|V3_perbeam"]["K0_furn"], R8_sum=r1b[f"{c}|V3_perbeam"]["R8_furn_sum"],
            R8_max_occ=r1b[f"{c}|V3_perbeam"]["R8_furn_max_occ"])
    for c in ("range_only", "cam_nf", "cam")
}
noise["E1review_V2_4houses_seed1"] = {
    c: dict(K0=r1[f"{c}|V2_probe"]["K0_furn"], R8_sum=r1[f"{c}|V2_probe"]["R8_furn_sum"],
            R8_max_occ=r1[f"{c}|V2_probe"]["R8_furn_max_occ"])
    for c in ("range_only", "cam_nf", "cam")
}


def decomp(t):
    ro, nf, cam = t["range_only"], t["cam_nf"], t["cam"]
    mx = "R8_max_occ" if "R8_max_occ" in cam else "R8_max"
    return dict(
        R8_total_loss_cam_pp=100 * (1 - cam["R8_sum"]),
        R8_range_pp=100 * (1 - ro["R8_sum"]),
        R8_perframe_pose_pp=100 * (ro["R8_sum"] - nf["R8_sum"]),
        R8_interagent_registration_pp=100 * (nf["R8_sum"] - cam["R8_sum"]),
        R8_sum_minus_max_cam_pp=100 * (cam["R8_sum"] - cam[mx]),
        R8_sum_minus_max_camnf_pp=100 * (nf["R8_sum"] - nf[mx]),
        K0_total_loss_camnf_pp=100 * (1 - nf["K0"]) if nf["K0"] is not None else None,
        K0_range_pp=100 * (1 - ro["K0"]),
        K0_perframe_pose_pp=100 * (ro["K0"] - nf["K0"]) if nf["K0"] is not None else None,
    )


noise["decomposition_V3"] = decomp(noise["E1review_V3_4houses_seed1"])
noise["decomposition_V2"] = decomp(noise["E1review_V2_4houses_seed1"])
noise["integrator_cost_V2_minus_V3_pp"] = {
    c: dict(K0=100 * (noise["E1review_V2_4houses_seed1"][c]["K0"] - noise["E1review_V3_4houses_seed1"][c]["K0"]),
            R8_sum=100 * (noise["E1review_V2_4houses_seed1"][c]["R8_sum"]
                          - noise["E1review_V3_4houses_seed1"][c]["R8_sum"]))
    for c in ("range_only", "cam_nf", "cam")
}
# multiplicity strata (C1 seed 2, cam_nf, sum): F1 by number of observers m=1..>=5
noise["C1_camnf_sum_by_m"] = c1["cam_nf"]["furn_f1_by_m"]["sum"]
noise["C1_camnf_max_by_m"] = c1["cam_nf"]["furn_f1_by_m"]["max"]
noise["B4_bytes_per_house_median"] = {
    "V3_cam": None,  # reported in code_review_g4_e1.md table: 560 KB
    "V0_cam": r1["cam|V0_impl"]["B4_relay_median"],
    "V2_cam": r1["cam|V2_probe"]["B4_relay_median"],
}
out["noise"] = noise

# compare: old learned clean code vs classical noisy floor (cross-panel, indicative only)
out["clean_learned_vs_noisy_classical_R8_furn"] = dict(
    threeSZ2_clean_VAL24=f64,
    noisy_cam_sum_V3=noise["E1review_V3_4houses_seed1"]["cam"]["R8_sum"],
    noisy_cam_sum_V2=noise["E1review_V2_4houses_seed1"]["cam"]["R8_sum"],
    noisy_camnf_sum_V3=noise["E1review_V3_4houses_seed1"]["cam_nf"]["R8_sum"],
    noisy_camnf_sum_V2=noise["E1review_V2_4houses_seed1"]["cam_nf"]["R8_sum"],
)

json.dump(out, open(os.path.join(HERE, "ra_ladder_headroom.json"), "w"), indent=1)


def show(o, ind=0):
    for k, v in o.items():
        if isinstance(v, dict):
            print(" " * ind + str(k))
            show(v, ind + 2)
        elif isinstance(v, list):
            print(" " * ind + f"{k}: " + ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in v))
        elif isinstance(v, float):
            print(" " * ind + f"{k}: {v:.4f}")
        else:
            print(" " * ind + f"{k}: {v}")


show(out)
