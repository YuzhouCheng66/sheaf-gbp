"""Private-only v246 data, proper reconstruction losses and fail-closed gates.

Union labels exist ONLY in supervision/metrics. They are never source inputs.
Paid support determines knownness, not wall labels or extra unary factors.
"""
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F

from scripts.mapping_evidence_data_v208 import paired_private as _rectangles
from scripts.train_mapping_spatial_bottleneck_v226 import validated_bank
from scripts.train_mapping_implicit64_v192 import measures, door_measures, average
from scripts.mapping_topology_v211 import topology_metrics


MODES = ('disjoint_halves', 'rectangles')
PROFILES = tuple(f'128/{c}/{m}' for c in ('base', 'medium') for m in MODES)
TRAIN_SHA256 = '98c0a66e176d8813f489fa98868723a8148e8c2a381866a7aaf73ab3abbec026'


def paired_private(sample, mode, seed=230101):
    """Return only two declared sensor rasters, with v230 rectangle seeds."""
    metadata = sample['metadata']
    if str(metadata['split']).lower() not in ('train', 'validation'):
        raise ValueError('TRAIN/VALIDATION only; TEST closed')
    walls = np.asarray(sample['walls'])
    if walls.ndim != 2 or walls.shape[0] != walls.shape[1] or not np.isin(walls, [0, 1]).all():
        raise ValueError('A square binary sensor world is required')
    if metadata['complexity'] not in ('base', 'medium'):
        raise ValueError('Registered base/medium worlds only')
    if mode == 'disjoint_halves':
        result = np.full((2, *walls.shape), -1, dtype=np.int8)
        middle = walls.shape[1] // 2
        result[0, :, :middle] = walls[:, :middle]
        result[1, :, middle:] = walls[:, middle:]
        return result
    if mode != 'rectangles':
        raise ValueError('Registered footprint required')
    actual_seed = int(seed) + int(metadata['index']) + (100000 if metadata['complexity'] == 'medium' else 0)
    return _rectangles(sample, actual_seed, mode)


def validated_train_bank(p):
    """Reuse the exact v244 TRAIN2048 bank; no alternative geometry silently."""
    if (p.get('train_size') != 128 or p.get('train_worlds_per_complexity') != 1024
            or list(p.get('complexities', [])) != ['base', 'medium'] or p.get('test_opened', False)):
        raise ValueError('Registered native128 TRAIN2048 configuration required')
    bank, digest = validated_bank(p)
    if digest != TRAIN_SHA256 or p.get('train_bank_sha256', digest) != digest:
        raise ValueError('TRAIN bank differs from v244')
    if len(bank) != 2048 or any(s['metadata']['split'].lower() != 'train' for s in bank):
        raise ValueError('TRAIN-only bank required')
    return bank, digest


def draw_ids(rng, worlds, batch_size):
    """Rows (world index, mode index); caller hashes the actual shared draws."""
    if worlds < 1 or batch_size < 1:
        raise ValueError('Positive bank and batch size required')
    return np.stack((rng.integers(worlds, size=batch_size),
                     rng.integers(len(MODES), size=batch_size)), axis=1).astype('<i8')


def private_manifest(private):
    value = np.asarray(private)
    if value.ndim < 3 or not np.isin(value, [-1, 0, 1]).all():
        raise ValueError('Ternary private maps required')
    value = np.ascontiguousarray(value, dtype=np.int8)
    return dict(shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                actual_private_only=True, test_opened=False)


def reconstruction_terms(output, private):
    """Proper per-map known BCE and wall BCE divided by own observed count.

    Unknown cells have identically zero wall gradient. Labels are validated in
    CPU data assembly, not by accelerator-synchronizing checks on every update.
    """
    if private.ndim != 3:
        raise ValueError('Private [N,H,W] maps required')
    labels = private.flatten(1)
    known_logits, wall_logits = output['known_logits'], output['wall_logits']
    if known_logits.shape != labels.shape or wall_logits.shape != labels.shape:
        raise ValueError('Dense [N,H*W] logits required; no hidden padding/query labels')
    dtype = torch.float64 if wall_logits.dtype == torch.float64 else torch.float32
    seen = (labels >= 0).to(dtype)
    known = F.binary_cross_entropy_with_logits(known_logits.to(dtype), seen, reduction='none').mean(-1)
    wall_bce = F.binary_cross_entropy_with_logits(wall_logits.to(dtype), (labels == 1).to(dtype), reduction='none')
    wall = (wall_bce * seen).sum(-1) / seen.sum(-1).clamp_min(1)
    return dict(known=known.mean(), wall=wall.mean(), known_per_map=known, wall_per_map=wall)


def _flatten_output(output, batch, agents, pixels):
    result = {}
    for key in ('known_logits', 'wall_logits'):
        value = output[key]
        if value.shape == (batch, agents, pixels):
            value = value.reshape(batch * agents, pixels)
        if value.shape != (batch * agents, pixels):
            raise ValueError('Two node outputs per pair required')
        result[key] = value
    return result


def training_loss(own, private, fused=None):
    """Control own known+wall; candidate own known+mean(own wall,fused wall)."""
    if private.ndim == 3:
        private = private[None]
    if private.ndim != 4 or private.shape[1] != 2:
        raise ValueError('Private [2,H,W] or [B,2,H,W] pairs required')
    batch, agents, height, width = private.shape
    source = reconstruction_terms(_flatten_output(own, batch, agents, height * width),
                                  private.reshape(batch * agents, height, width))
    parts = dict(own_known=source['known'], own_wall=source['wall'])
    if fused is None:
        return source['known'] + source['wall'], parts
    # Supervision ONLY. This union must never be returned as an inference input.
    union = private.max(dim=1).values[:, None].expand(-1, agents, -1, -1)
    result = reconstruction_terms(_flatten_output(fused, batch, agents, height * width),
                                  union.reshape(batch * agents, height, width))
    parts['fused_wall'] = result['wall']
    return source['known'] + .5 * (source['wall'] + result['wall']), parts


def _wall_rates(prediction, target, mask):
    truth = (target == 1) & mask
    guess = (prediction == 1) & mask
    tp, actual, predicted = int((truth & guess).sum()), int(truth.sum()), int(guess.sum())
    # Empty denominators score zero, never disappear from profile averaging.
    return dict(wall_recall=tp / actual if actual else 0.,
                wall_precision=tp / predicted if predicted else 0.,
                wall_f1=2 * tp / (actual + predicted) if actual + predicted else 0.)


def _rows(predictions, private, sample):
    target = private.max(axis=0)
    interior = np.zeros_like(target, dtype=bool)
    interior[6:-6, 6:-6] = True
    rows = []
    for node, prediction in enumerate(predictions):
        seen = target >= 0
        own = private[node] >= 0
        peer = seen & ~own
        result = measures(prediction, target)
        result.update(_wall_rates(prediction, target, seen))
        for name, mask in (('own', own), ('peer', peer), ('interior', seen & interior),
                           ('own_interior', own & interior), ('peer_interior', peer & interior)):
            result.update({name + '_' + k: v for k, v in _wall_rates(prediction, target, mask).items()})
        for name, mask in (('own', own), ('peer', peer)):
            result[name + '_typed_accuracy'] = float(np.mean(prediction[mask] == target[mask])) if mask.any() else 0.
        unknown = ~seen
        result['unsupported_false_known'] = float(np.mean(prediction[unknown] >= 0)) if unknown.any() else 0.
        result['unsupported_known_count'] = int(((prediction >= 0) & unknown).sum())
        result['unobserved_cells'] = int(unknown.sum())
        result.update(door_measures(prediction, dict(sample, union=target)))
        result.update(topology_metrics(prediction, target))
        rows.append(result)
    return rows


def metrics(output, private, paid_known, sample):
    """Evaluate a pair; private/GT are labels only, support is received state."""
    private, paid_known = np.asarray(private), np.asarray(paid_known)
    if private.ndim != 3 or len(private) != 2 or not np.isin(private, [-1, 0, 1]).all():
        raise ValueError('Two actual ternary private maps required')
    if paid_known.shape != private.shape or paid_known.dtype != np.bool_:
        raise ValueError('Actual boolean received support per node required')
    overlap = (private[0] >= 0) & (private[1] >= 0)
    if np.any(private[0][overlap] != private[1][overlap]):
        raise ValueError('Conflicting exact sensor observations')
    probabilities = {}
    for name in ('known', 'wall'):
        logits = output[name + '_logits']
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().double().numpy()
        logits = np.asarray(logits)
        if logits.shape != (2, private.shape[1] * private.shape[2]) or not np.isfinite(logits).all():
            raise ValueError('Finite [2,H*W] decoded logits required')
        probabilities[name] = np.exp(-np.logaddexp(0., -logits)).reshape(private.shape)
    wall = probabilities['wall'] >= .5
    prediction = np.where(paid_known, wall.astype(np.int8), -1).astype(np.int8)
    raw_prediction = np.where(probabilities['known'] >= .5, wall.astype(np.int8), -1).astype(np.int8)
    rows, raw_rows = _rows(prediction, private, sample), _rows(raw_prediction, private, sample)
    return dict(metrics=average(rows), node_metrics=rows, prediction=prediction,
                raw_head_metrics=average(raw_rows), raw_head_node_metrics=raw_rows,
                raw_head_prediction=raw_prediction)


def gates(control, candidate):
    """Fixed-endpoint engineering gate; missing/None/NaN never passes."""
    required = ('union_accuracy', 'peer_wall_f1', 'own_wall_f1', 'own_wall_precision',
                'wall_precision', 'door_free_recall', 'pair_topology_accuracy', 'unsupported_false_known')
    finite = lambda x: isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(float(x)) and 0 <= x <= 1
    present = set(control) == set(PROFILES) and set(candidate) == set(PROFILES)
    invalid = {arm: {key: [m for m in required if not finite(profiles.get(key, {}).get(m))]
                     for key in PROFILES if any(not finite(profiles.get(key, {}).get(m)) for m in required)}
               for arm, profiles in (('control', control), ('candidate', candidate))}
    if not present or any(invalid.values()):
        return dict(passed=False, valid=False, missing_profiles={
            'control': sorted(set(PROFILES) - set(control)), 'candidate': sorted(set(PROFILES) - set(candidate))},
            unexpected_profiles={'control': sorted(set(control) - set(PROFILES)), 'candidate': sorted(set(candidate) - set(PROFILES))},
            invalid_fields=invalid)
    delta = {key: {m: float(candidate[key][m] - control[key][m]) for m in required} for key in PROFILES}
    macro = {m: float(np.mean([delta[key][m] for key in PROFILES])) for m in required}
    checks = dict(macro_union_gain=macro['union_accuracy'] >= .01 - 1e-12,
                  macro_peer_wall_f1_gain=macro['peer_wall_f1'] >= .05 - 1e-12,
                  medium_rectangles_peer_positive=delta['128/medium/rectangles']['peer_wall_f1'] > 0)
    protections = {}
    for key in PROFILES:
        protection = {m: delta[key][m] >= -limit - 1e-12 for m, limit in (
            ('own_wall_f1', .01), ('own_wall_precision', .01), ('wall_precision', .01),
            ('door_free_recall', .02), ('pair_topology_accuracy', .02))}
        protection['zero_unsupported_known'] = control[key]['unsupported_false_known'] == candidate[key]['unsupported_false_known'] == 0.
        protections[key] = protection
    passed = all(checks.values()) and all(all(row.values()) for row in protections.values())
    return dict(passed=passed, valid=True, checks=checks, protections=protections,
                macro_changes=macro, profile_changes=delta, high_fidelity_or_100x_claim=False)
