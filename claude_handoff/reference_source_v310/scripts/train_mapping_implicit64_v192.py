"""Sealed, finite 5090 pilot: quantized global capacity, then masked MAP.

This is not the distributed GBP experiment or a 100x compression result.
Source data, per-world splits, optimizer/RNG, checkpoints and failures persist.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from scripts import mapping_implicit_data_v192 as data
from scripts.mapping_implicit_model_v192 import ImplicitMapCodec

GPU = 'GPU-238bdbd5-9f06-da6d-09f6-830fd831e817'
CONFIG = 'configs/mapping_implicit64_v192.json'
FILE = 'scripts/train_mapping_implicit64_v192.py'
FILES = [CONFIG, FILE] + ['scripts/' + stem + '_v192.py' for stem in (
    'mapping_implicit_data', 'mapping_implicit_model', 'mapping_scalar_gbp',
    'test_mapping_implicit_data', 'test_mapping_implicit_model', 'test_mapping_scalar_gbp')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf8')
    os.replace(tmp, path)


def guard():
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    allowed = {os.getpid()}
    viewer = Path('/proc/2863708/cmdline')
    if viewer.exists() and b'native-map-viewer-20261013/server.py' in viewer.read_bytes():
        allowed.add(2863708)
    foreign = [r for r in rows.splitlines() if r.split(',')[0].strip() == GPU and int(r.split(',')[1]) not in allowed]
    if foreign:
        raise RuntimeError('Other GPU workload; do not touch it: ' + str(foreign))


def prepare(root):
    root = Path(root).resolve(); repo = Path(__file__).resolve().parents[1]
    if root.exists():
        raise FileExistsError('No overwrite or implicit restart')
    p = read(repo / CONFIG)
    if p['gpu_uuid'] != GPU or p['test_opened']:
        raise ValueError('Wrong compute/split')
    root.mkdir(parents=True)
    for rel in FILES:
        dest = root / 'source' / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, dest)
    write(root / 'registered-protocol.json', p)
    predecessor = Path('/home/yuzhou/Desktop/sheaf-gbp/mapping-resume-c1024-visible-20261016')
    if read(predecessor / 'status.json')['state'] != 'complete':
        raise RuntimeError('Prior1024 not complete; explicit scheduling needed')
    selected = read(predecessor / 'runs/c1024/selection.json')['best']
    prior_path = predecessor / 'runs/c1024' / selected['path']
    if sha(prior_path) != selected['sha256']:
        raise RuntimeError('Prior checkpoint changed')
    write(root / 'preserved-prior.json', dict(path=str(prior_path), sha256=sha(prior_path),
        report=str(predecessor / 'c1024-report.json'), report_sha256=sha(predecessor / 'c1024-report.json'),
        old_4bit_experiment='/home/yuzhou/Desktop/sheaf-gbp/mapping-priority-q4-20261016', modified=False))
    write(root / 'source-manifest.json', dict(files={rel: sha(root / 'source' / rel) for rel in FILES},
        protocol_sha256=sha(root / 'registered-protocol.json'), preserved_sha256=sha(root / 'preserved-prior.json')))
    print(json.dumps(dict(prepared=str(root), test_opened=False)), flush=True)


def check(root):
    if Path(__file__).resolve() != root / 'source' / FILE:
        raise RuntimeError('Run sealed source only')
    manifest = read(root / 'source-manifest.json')
    for rel, expected in manifest['files'].items():
        if sha(root / 'source' / rel) != expected:
            raise RuntimeError('Changed sealed file: ' + rel)
    if sha(root / 'registered-protocol.json') != manifest['protocol_sha256']:
        raise RuntimeError('Changed registration')
    if sha(root / 'preserved-prior.json') != manifest['preserved_sha256']:
        raise RuntimeError('Changed prior pin')
    p = read(root / 'registered-protocol.json')
    if p['gpu_uuid'] != GPU or p['test_opened']:
        raise RuntimeError('Invalid protocol')
    return p


def make_bank(p):
    bank = []
    h = hashlib.sha256()
    for complexity in p['complexities']:
        for index in range(p['train_worlds_per_complexity']):
            item = data.sample_world('TRAIN', index, size=p['train_size'], complexity=complexity)
            h.update(item['walls'].tobytes()); h.update(item['private'].tobytes())
            bank.append(item)
    return bank, h.hexdigest()


def grid(size):
    return np.asarray(data.query_coords(size), dtype=np.float32).reshape(-1, 2)


def draw_batch(bank, p, rng, device):
    size = p['train_size']; Q = p['queries']; B = p['batch_size']
    maps, coordinates, targets, record = [], [], [], []
    public = grid(size)
    for _ in range(B):
        index = int(rng.integers(len(bank))); item = bank[index]; mode = int(rng.choice(3, p=[.5, .25, .25]))
        private = item['union'] if mode == 0 else item['private'][mode - 1]
        flat = private.reshape(-1)
        # Explicit boundary-focused objective. Selection only uses this source's own observations.
        wall = private == 1; boundary = np.zeros_like(wall)
        boundary[1:] |= wall[1:] != wall[:-1]; boundary[:-1] |= wall[1:] != wall[:-1]
        boundary[:, 1:] |= wall[:, 1:] != wall[:, :-1]; boundary[:, :-1] |= wall[:, 1:] != wall[:, :-1]
        candidates = np.flatnonzero((boundary & (private >= 0)).reshape(-1))
        uniform = int(Q * p['uniform_query_fraction'])
        first = rng.integers(size * size, size=uniform)
        second = rng.choice(candidates, Q - uniform, replace=True) if len(candidates) else rng.integers(size * size, size=Q - uniform)
        ids = np.concatenate([first, second]); maps.append(private); coordinates.append(public[ids]); targets.append(flat[ids])
        record.append([index, mode, hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()])
    return (torch.as_tensor(np.stack(maps), device=device), torch.as_tensor(np.stack(coordinates), device=device),
            torch.as_tensor(np.stack(targets), device=device), record)


def losses(output, labels, p):
    known = (labels >= 0).float(); wall = (labels == 1).float()
    lk = F.binary_cross_entropy_with_logits(output['known_logits'].float(), known)
    lw = F.binary_cross_entropy_with_logits(output['wall_logits'].float(), wall,
        pos_weight=wall.new_tensor(p['wall_positive_weight']), reduction='none')
    lw = (lw * known).sum() / known.sum().clamp_min(1)
    lz = output['latent_unquantized'].float().square().mean()
    total = p['known_loss_weight'] * lk + p['wall_loss_weight'] * lw + p['latent_l2_weight'] * lz
    return total, dict(known=float(lk.detach()), wall=float(lw.detach()), latent=float(lz.detach()))


@torch.no_grad()
def decode_map(model, latent, size, device):
    coords = torch.as_tensor(grid(size), device=device)
    kp, wp = [], []
    for block in coords.split(8192):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            out = model.decode(latent, block[None])
        kp.append(out['known_logits'].float().sigmoid().cpu().numpy().reshape(-1))
        wp.append(out['wall_logits'].float().sigmoid().cpu().numpy().reshape(-1))
    kp = np.concatenate(kp).reshape(size, size); wp = np.concatenate(wp).reshape(size, size)
    return np.where(kp >= .5, (wp >= .5).astype(np.int8), -1).astype(np.int8), kp, wp


def measures(pred, target):
    seen = target >= 0; walls = target == 1; unknown = ~seen; predicted_walls = (pred == 1) & seen
    def rate(mask, correct):
        return float(np.count_nonzero(correct & mask) / np.count_nonzero(mask)) if np.any(mask) else None
    return dict(union_accuracy=rate(seen, pred == target), wall_recall=rate(walls, pred == 1),
        wall_precision=rate(predicted_walls, walls), unsupported_false_known=rate(unknown, pred >= 0),
        known_recall=rate(seen, pred >= 0))


def door_measures(pred, sample):
    door = sample['door_mask'] & (sample['union'] >= 0)
    if not np.any(door):
        return dict(door_false_wall=None, door_free_recall=None)
    return dict(door_false_wall=float(np.mean(pred[door] == 1)), door_free_recall=float(np.mean(pred[door] == 0)))


def average(rows):
    result = {}
    for key in rows[0]:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def score(profiles):
    # A missing wall prediction scores zero precision, not an excluded easy case.
    values = []
    for m in profiles.values():
        values.append(sum((m.get(k) or 0.) for k in ('union_accuracy', 'wall_recall', 'wall_precision')) / 3.
                      - (m.get('unsupported_false_known') or 0.))
    return float(np.mean(values))


@torch.no_grad()
def evaluate(model, p, device, sizes=None, own=False):
    model.eval(); profiles = {}; records = []; qualities = {}
    for size in (sizes or [p['train_size']]):
        for complexity in p['complexities']:
            rows = []; own_rows = []; per_world_scores = []
            for index in range(p['validation_worlds_per_complexity']):
                sample = data.sample_world('VALIDATION', index, size=size, complexity=complexity)
                private = torch.as_tensor(sample['union'][None], device=device)
                z = model.latent_quantize(model.encode(private))
                pred, _, _ = decode_map(model, z, size, device)
                metrics = measures(pred, sample['union']); metrics.update(door_measures(pred, sample)); rows.append(metrics)
                per_world_scores.append(score({'world': metrics}))
                records.append(dict(size=size, complexity=complexity, index=index, source='union_oracle', metrics=metrics))
                if own:
                    for i in range(2):
                        z = model.latent_quantize(model.encode(torch.as_tensor(sample['private'][i:i+1], device=device)))
                        pred, _, _ = decode_map(model, z, size, device)
                        m = measures(pred, sample['private'][i]); own_rows.append(m)
                        records.append(dict(size=size, complexity=complexity, index=index, source='AB'[i], metrics=m))
            profiles[str(size) + '/' + complexity] = average(rows)
            qualities[str(size) + '/' + complexity] = float(np.mean(per_world_scores))
            if own:
                profiles[str(size) + '/' + complexity + '/own'] = average(own_rows)
    model.train(); return dict(profiles=profiles, records=records, quality=float(np.mean(list(qualities.values()))),
        quality_per_profile=qualities, encoder_precision='FP32', decoder_linear_precision='bfloat16' if device == 'cuda' else 'FP32',
        coordinate_quantizer_precision='FP32', interpretation='Larger-size results include encoder input-resolution shift, not decoder-only extrapolation.')


def save_checkpoint(path, model, optimizer, step, rng, p, draw_digest, validation=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, latent_dim=model.latent_dim,
        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        protocol=p, draw_digest=draw_digest, validation=validation, test_opened=False)
    tmp = path.with_suffix('.tmp'); torch.save(value, tmp); os.replace(tmp, path)


def train_arm(root, p, latent_dim, bank, global_start, reference_digest=None):
    torch.manual_seed(p['seed']); torch.cuda.manual_seed_all(p['seed'])
    rng = np.random.default_rng(p['seed'] + 1)
    run = root / ('capacity-' + str(latent_dim)); run.mkdir()
    model = ImplicitMapCodec(latent_dim=latent_dim).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    draw_hash = hashlib.sha256(); best = None; step = 0; started = time.monotonic()
    write(run / 'model.json', dict(parameters=sum(x.numel() for x in model.parameters()), latent_dim=latent_dim,
        quantized_latent_bytes=latent_dim, communication_claim=False))
    try:
        for step in range(1, p['updates_per_arm'] + 1):
            if time.monotonic() - global_start > p['max_active_seconds']:
                raise TimeoutError('Registered resource cap, not convergence')
            private, coords, labels, draw = draw_batch(bank, p, rng, 'cuda')
            draw_hash.update(json.dumps(draw, separators=(',', ':')).encode())
            lr = p['end_learning_rate'] + .5 * (p['learning_rate'] - p['end_learning_rate']) * (1 + math.cos(math.pi * (step - 1) / p['updates_per_arm']))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(private, coords, quantized=True)
            loss, components = losses(output, labels, p)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), p['gradient_clip'])
            if not torch.isfinite(grad):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            if step == 1 or step % 50 == 0:
                status = dict(state='training_capacity', latent_dim=latent_dim, step=step, loss=float(loss),
                    components=components, lr=lr, elapsed_s=time.monotonic()-started, test_opened=False)
                write(root / 'status.json', status)
                with (run / 'train.jsonl').open('a') as f:
                    f.write(json.dumps(status) + '\n')
                print(json.dumps(status), flush=True)
            if step % p['validation_every'] == 0:
                guard(); result = evaluate(model, p, 'cuda')
                path = run / ('validated-%06d.pt' % step)
                save_checkpoint(path, model, optimizer, step, rng, p, draw_hash.hexdigest(), result)
                write(run / ('validation-%06d.json' % step), result)
                if best is None or result['quality'] > best['quality']:
                    best = dict(path=str(path), sha256=sha(path), step=step, quality=result['quality'], profiles=result['profiles'])
                    write(run / 'selection.json', dict(best=best, selection=p['selection']))
                print(json.dumps(dict(validation_step=step, latent_dim=latent_dim, profiles=result['profiles'])), flush=True)
        if reference_digest is not None and draw_hash.hexdigest() != reference_digest:
            raise RuntimeError('Unmatched TRAIN sequence across capacity arms')
        write(run / 'complete.json', dict(actual_updates=step, draw_digest=draw_hash.hexdigest(), best=best,
            budget_completed_not_convergence=True, elapsed_s=time.monotonic()-started, test_opened=False))
        saved = torch.load(best['path'], map_location='cpu', weights_only=False); model.load_state_dict(saved['model'])
        return model, best, draw_hash.hexdigest()
    except BaseException:
        save_checkpoint(run / 'recovery.pt', model, optimizer, step, rng, p, draw_hash.hexdigest())
        write(run / 'interrupted.json', dict(reported_step=step, last_validated=best, mid_update_possible=True,
            implicit_resume_not_implemented=True, error=traceback.format_exc()))
        raise


def central_fusion(root, model, p):
    """Diagnostic MAP: real source observations only, no teacher labels elsewhere."""
    run = root / 'central-complementary'; run.mkdir()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cfg = p['stage_b']; size = p['train_size']; public = grid(size); records = []
    for complexity in p['complexities']:
        for index in range(cfg['worlds_per_complexity']):
            sample = data.sample_world('VALIDATION', index, size=size, complexity=complexity)
            privates = torch.as_tensor(sample['private'], device='cuda')
            with torch.no_grad():
                local = model.latent_quantize(model.encode(privates))
                initial = local.mean(0, keepdim=True).clamp(-.999, .999)
            # Exactly the same actual observed cells for all optimization steps.
            rng = np.random.default_rng(192501 + index + 1000 * p['complexities'].index(complexity))
            seen = np.flatnonzero(sample['union'].reshape(-1) >= 0)
            ids = rng.choice(seen, size=min(cfg['query_count'], len(seen)), replace=False)
            coords = torch.as_tensor(public[ids][None], device='cuda')
            labels = torch.as_tensor(sample['union'].reshape(-1)[ids][None], device='cuda')
            raw = torch.nn.Parameter(torch.atanh(initial)); optimizer = torch.optim.Adam([raw], lr=cfg['learning_rate'])
            curve = []
            for step in range(cfg['steps']):
                optimizer.zero_grad(set_to_none=True)
                z = raw.tanh(); out = model.decode(z, coords)
                wall_bce = F.binary_cross_entropy_with_logits(out['wall_logits'].float(), (labels == 1).float())
                known_bce = F.binary_cross_entropy_with_logits(out['known_logits'].float(), torch.ones_like(labels, dtype=torch.float32))
                loss = wall_bce + known_bce + cfg['latent_prior_weight'] * z.square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Central evidence objective')
                loss.backward(); optimizer.step()
                if step % 10 == 0 or step + 1 == cfg['steps']:
                    curve.append(dict(step=step+1, objective=float(loss.detach()),
                        wall_bce=float(wall_bce.detach()), known_positive_bce=float(known_bce.detach())))
            candidates = dict(A_only=local[0:1], B_only=local[1:2], latent_average=initial,
                evidence_MAP=raw.tanh().detach())
            with torch.no_grad():
                candidates['union_oracle'] = model.latent_quantize(model.encode(torch.as_tensor(sample['union'][None], device='cuda')))
            predictions = {}; metrics = {}
            for name, z in candidates.items():
                pred, _, _ = decode_map(model, z, size, 'cuda'); predictions[name] = pred
                m = measures(pred, sample['union']); peer = (sample['private'][0] < 0) & (sample['private'][1] >= 0)
                peerwall = peer & (sample['union'] == 1)
                m['A_peer_only_accuracy'] = float(np.mean(pred[peer] == sample['union'][peer])) if peer.any() else None
                m['A_peer_wall_recall'] = float(np.mean(pred[peerwall] == 1)) if peerwall.any() else None
                m.update(door_measures(pred, sample)); metrics[name] = m
            name = '%s-%03d' % (complexity, index)
            np.savez_compressed(run / (name + '.npz'), private=sample['private'], union=sample['union'], **predictions)
            record = dict(complexity=complexity, index=index, metrics=metrics, curve=curve,
                source_observed_query_count=len(ids), neural_weight_updates=0, validation_state_inference=True,
                query_index_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                communication_claim=False, inference_latent_quantized=False,
                scope='Unquantized centralized sampled-observation MAP diagnostic; not distributed result or exact global optimum.',
                knownness_caveat='Only observed-positive support constraints; unknown regions are unconstrained and may become falsely known. Overlap is deduplicated here; future distributed reference must match its own factor weighting.')
            write(run / (name + '.json'), record); records.append(record)
            write(root / 'status.json', dict(state='central_complementary_validation', completed=len(records),
                total=len(p['complexities'])*cfg['worlds_per_complexity'], test_opened=False))
    write(run / 'report.json', dict(records=records, test_opened=False, scope='Diagnostic only; no wireless compression claim'))
    return records


def run(root):
    p = check(root)
    if (root / 'started.json').exists():
        raise FileExistsError('No implicit restart')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise RuntimeError('5090 visibility must be configured before process creation')
    write(root / 'started.json', dict(pid=os.getpid(), time=time.time()))
    try:
        torch.set_num_threads(4)
        names = ['scripts.test_mapping_implicit_data_v192', 'scripts.test_mapping_implicit_model_v192', 'scripts.test_mapping_scalar_gbp_v192']
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise RuntimeError('CPU preflight tests failed; no GPU training')
        torch.set_num_threads(4)
        guard()
        if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Wrong GPU')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(p['seed']); np.random.seed(p['seed']); random.seed(p['seed'])
        write(root / 'gpu-started.json', dict(pid=os.getpid(), gpu=torch.cuda.get_device_name(0), uuid=GPU,
            torch_version=torch.__version__, precision=p['training_precision']))
        started = time.monotonic(); write(root / 'status.json', dict(state='building_train_bank', test_opened=False))
        bank, digest = make_bank(p)
        write(root / 'data-manifest.json', dict(train_worlds=len(bank), sha256=digest, split='TRAIN',
            validation_worlds_per_complexity=p['validation_worlds_per_complexity'], resolution=p['train_size'],
            family=p['data'], test_opened=False))
        raw_examples = []
        for size in p['validation_sizes']:
            for complexity in p['complexities']:
                sample = data.sample_world('VALIDATION', 2, size=size, complexity=complexity)
                raw_examples.append(dict(size=size, complexity=complexity,
                    source_A=data.raw_packet_counts(sample['private'][0]),
                    source_B=data.raw_packet_counts(sample['private'][1])))
        write(root / 'raw-codec-examples.json', dict(examples=raw_examples,
            scope='Illustrative per-source packet sizes, not multi-hop episode ratios; learned capacity latent has no communication claim.',
            note='Structured maps may already compress very well with lossless zlib; 100x versus dense raw is not 100x versus lossless raw.'))
        completed = {}; paired = None
        for dim in p['arms']:
            model, best, draw_digest = train_arm(root, p, dim, bank, started, paired)
            if paired is None:
                paired = draw_digest
            frozen = hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
            result = evaluate(model, p, 'cuda', sizes=p['validation_sizes'], own=True)
            write(root / ('capacity-%d/multiscale.json' % dim), result)
            gate = p['representation_gate']; checks = {}
            for key, m in best['profiles'].items():
                checks[key] = all((m.get(k) is not None and m[k] >= gate[k])
                    for k in ('union_accuracy', 'wall_recall', 'wall_precision')) and (m.get('unsupported_false_known') or 0.) <= gate['unsupported_false_known']
            write(root / ('capacity-%d/gate.json' % dim), dict(checks=checks, passed=all(checks.values()),
                scope='Gate for any subsequent neural recurrent training/expansion, not a prerequisite for central diagnostic.',
                central_diagnostic_runs_even_if_gate_fails=True, test_opened=False))
            if dim == 64:
                central_fusion(root, model, p)
            after = hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
            if frozen != after:
                raise RuntimeError('Validation changed learned weights')
            completed[str(dim)] = dict(selection=best, profiles=result['profiles'], validation_weight_hash=frozen,
                actual_updates=p['updates_per_arm'], draw_digest=draw_digest)
            del model; torch.cuda.empty_cache(); check(root)
        report = dict(state='complete_initial_capacity_and_central_diagnostics', arms=completed,
            matched_draws=True, prior_checkpoints_preserved=True, test_opened=False,
            trained_resolution=128, larger_resolutions_are_zero_shot=True,
            distributed_neural_GBP_started=False, compression_100x_demonstrated=False,
            automatic_followup=False, requires_evidence_based_phase_decision=True)
        write(root / 'report.json', report); write(root / 'status.json', dict(state=report['state'], report='report.json', test_opened=False))
    except BaseException:
        write(root / 'error.json', dict(error=traceback.format_exc(), automatic_restart=False))
        write(root / 'status.json', dict(state='error_or_resource_pause', test_opened=False, automatic_restart=False))
        raise


def launch(root):
    p = check(root); guard()
    if (root / 'launch.json').exists() or (root / 'started.json').exists():
        raise FileExistsError('Already launched')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONPATH='.', PYTHONDONTWRITEBYTECODE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (root / 'train.log').open('x') as f:
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root)],
            cwd=root / 'source', env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    write(root / 'launch.json', dict(pid=child.pid, time=time.time(), gpu_uuid=p['gpu_uuid']))
    print(json.dumps(dict(pid=child.pid, root=str(root))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--launch', action='store_true'); args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    root = Path(args.root).resolve()
    if args.prepare:
        prepare(root)
    elif args.launch:
        launch(root)
    else:
        run(root)
