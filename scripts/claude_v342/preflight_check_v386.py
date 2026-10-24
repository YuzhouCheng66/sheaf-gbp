"""Batch T preflight validator / projector (review code_review_v386_decoder.md M1-M3). Claude, 2026-10-23.
Reads one or more preflight run dirs (train_v386.py --max-updates N) and prints ONE line:
  VALID FIT|NOFIT rate=<s/update> load=<s> val=<s> need=<s> avail=<s>      (measurement valid; FIT if 1.05 x need <= avail and 2 x val <= final_window)
  INVALID <reason;...>                                                        (preflight did not run as required -> the chain must ABORT)
Validity (M1): history.json has exactly one epoch record with updates == --updates and finite train_loss; run.json has decoder_width ==
--width, parameters.decoder == --decoder-params and parameters.trunk/head equal to the control's run.json; the log has 'gen <s>s' and
'done <s>s' lines. Rate (M2) = float epoch_seconds / updates (includes warm-up => conservative); val = total - load - epoch_seconds
(clamped >= 30 s); need = load + 57,600 x rate + 4 x val. --mode parallel additionally requires rate >= --min-rate (batch Q's two-job
W=1 rate, 0.094 s/update): a lower value means the two preflights did not overlap and the measurement is meaningless.
Exit code 0 on VALID, 2 on INVALID (FIT/NOFIT is in the text).
"""
import argparse, json, math, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--preflight', nargs='+', required=True); ap.add_argument('--control-run', required=True)
ap.add_argument('--avail', type=float, required=True, help='seconds available before the train deadline')
ap.add_argument('--final-window', type=float, default=500., help='seconds between train deadline and total deadline for the final val+fresh evals')
ap.add_argument('--mode', choices=['parallel', 'solo'], required=True); ap.add_argument('--min-rate', type=float, default=0.094)
ap.add_argument('--updates', type=int, default=200); ap.add_argument('--total-updates', type=int, default=57600)
ap.add_argument('--width', type=float, default=2.0); ap.add_argument('--decoder-params', type=int, default=2491330)
ap.add_argument('--code-embed', type=int, default=None, help='if given, run.json code_embed must equal this (v388 batches)')
ap.add_argument('--frame-cond', type=int, default=None, help='if given (0/1), run.json frame_cond must equal it (v392 batches)')
a = ap.parse_args()
reasons = []; rates = []; loads = []; vals = []
ctrl = json.load(open(a.control_run))['parameters']
for d in a.preflight:
    d = Path(d)
    try:
        h = json.load(open(d / 'history.json')); r = json.load(open(d / 'run.json'))
        if len(h) != 1 or h[0].get('updates') != a.updates or h[0].get('train_loss') is None or not math.isfinite(h[0]['train_loss']):
            reasons.append(f'{d.name}: history {[(e.get("updates"), e.get("train_loss")) for e in h]} != one record of {a.updates} finite-loss updates')
        if float(r.get('decoder_width', r['args'].get('decoder_width', 1.0))) != a.width:
            reasons.append(f'{d.name}: decoder_width {r.get("decoder_width")} != {a.width}')
        if a.code_embed is not None and int(r.get('code_embed', r['args'].get('code_embed', 0))) != a.code_embed:
            reasons.append(f'{d.name}: code_embed {r.get("code_embed")} != {a.code_embed}')
        if a.frame_cond is not None and int(bool(r.get('frame_cond', r['args'].get('frame_cond', False)))) != a.frame_cond:
            reasons.append(f'{d.name}: frame_cond {r.get("frame_cond")} != {a.frame_cond}')
        p = r['parameters']
        if p.get('decoder') != a.decoder_params:
            reasons.append(f'{d.name}: decoder params {p.get("decoder")} != {a.decoder_params}')
        if p.get('trunk') != ctrl.get('trunk') or p.get('head') != ctrl.get('head'):
            reasons.append(f'{d.name}: encoder/head params {p.get("trunk")}/{p.get("head")} != control {ctrl.get("trunk")}/{ctrl.get("head")}')
        log = (d.parent / f'{d.name}.log').read_text()
        g = re.search(r' gen (\d+)s', log); t = re.search(r'\] done (\d+)s', log)
        if not g or not t:
            reasons.append(f'{d.name}: log lacks gen/done lines'); continue
        load = float(g.group(1)); total = float(t.group(1)); ep = float(h[0]['epoch_seconds'])
        rate = ep / a.updates; val = max(30., total - load - ep)
        if not (math.isfinite(rate) and rate > 0):
            reasons.append(f'{d.name}: rate {rate} not finite/positive')
        rates.append(rate); loads.append(load); vals.append(val)
    except Exception as e:
        reasons.append(f'{d.name}: {type(e).__name__}: {e}')
if a.mode == 'parallel' and len(a.preflight) < 2:
    reasons.append('parallel mode needs two preflight dirs')
if rates and a.mode == 'parallel' and max(rates) < a.min_rate:
    reasons.append(f'parallel rate {max(rates):.4f} < min {a.min_rate}: the two preflights did not overlap (measurement invalid)')
if reasons:
    print('INVALID ' + '; '.join(reasons)); raise SystemExit(2)
rate = max(rates); load = max(loads); val = max(vals)
need = load + a.total_updates * rate + 4 * val
fit = (1.05 * need <= a.avail) and (2 * val <= a.final_window)
print(f'VALID {"FIT" if fit else "NOFIT"} rate={rate:.4f} load={load:.0f} val={val:.0f} need={need:.0f} avail={a.avail:.0f} final_evals={2 * val:.0f}/{a.final_window:.0f}')
