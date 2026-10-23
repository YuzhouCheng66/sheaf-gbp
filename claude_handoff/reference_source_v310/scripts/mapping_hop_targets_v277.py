"""CPU-only original-observation labels for static synchronous communication.

Labels are NOT inference inputs. After k rounds a receiver may depend only on
original sensors along directed paths of length <= k. This is an information
access bound, not a claim that quantized Gaussian messages retain that union.
Previously decoded maps never enter this module. Original sensor overlap must
agree exactly. Edges denote actual active positive-precision transmissions.
Repeated rounds take graph reachability, not repeated independent evidence.

The retained v246 TRAIN2048/native128 bank has no union/full-sensor model input.
The empty middle node in line3 is an empty sensor, not negative wall evidence.
Its nonzero public solver prior is not included in supervision as observation.
"""
import hashlib
from numbers import Integral

import numpy as np

from scripts.mapping_observed_data_v246 import (
    MODES, TRAIN_SHA256, paired_private, validated_train_bank,
)


GRAPH_KINDS = ('pair', 'line3')
FOOTPRINT_SEED = 230101
SAMPLING_SEED = 277102
WORLD_COUNT = 2048
DRAW_COLUMNS = ('world_index', 'mode_index', 'graph_index')


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _maps(own_maps):
    value = np.asarray(own_maps)
    if (value.ndim != 3 or min(value.shape) < 1
            or not np.issubdtype(value.dtype, np.integer)
            or not np.isin(value, (-1, 0, 1)).all()):
        raise ValueError('Original integer ternary own maps [N,H,W] required')
    # A source observation is exact in this registered static-world task.
    # Check all original sources, including currently disconnected sources.
    if np.any(np.any(value == 0, axis=0) & np.any(value == 1, axis=0)):
        raise ValueError('Conflicting original sensor observations')
    return value


def _edges(active_edges, agents):
    result = []
    try:
        incoming = list(active_edges)
    except TypeError as exc:
        raise ValueError('Actual directed edge pairs required') from exc
    for edge in incoming:
        if not isinstance(edge, (tuple, list, np.ndarray)) or len(edge) != 2:
            raise ValueError('Directed edges are (sender,receiver), no implicit weights')
        sender = _integer(edge[0], 'sender')
        receiver = _integer(edge[1], 'receiver')
        if sender >= agents or receiver >= agents or sender == receiver:
            raise ValueError('Valid distinct in-range sender and receiver required')
        result.append((sender, receiver))
    if len(set(result)) != len(result):
        raise ValueError('Duplicate directed edge; record each active route once')
    return tuple(result)


def reachable_sources(agents, directed_active_edges, rounds):
    """Boolean [receiver,original_source]; simultaneous updates, no labels."""
    agents = _integer(agents, 'agents', 1)
    rounds = _integer(rounds, 'rounds')
    edges = _edges(directed_active_edges, agents)
    reachable = np.eye(agents, dtype=bool)
    for _ in range(rounds):
        previous = reachable
        reachable = previous.copy()
        for sender, receiver in edges:
            reachable[receiver] |= previous[sender]
        if np.array_equal(previous, reachable):
            break
    return reachable


def hop_targets(own_maps, directed_active_edges, k):
    """Return CPU int8 [N,H,W] supervision from ORIGINAL sources only.

    Unknown stays -1. Paths are directed; no symmetrization is performed.
    This API assumes the declared static edges actually transmitted in every
    synchronous round with nonzero information precision. Dropout/asynchrony
    requires time-indexed causal ancestry, not this static k-hop shortcut.
    """
    value = _maps(own_maps)
    reachable = reachable_sources(len(value), directed_active_edges, k)
    result = np.empty(value.shape, dtype=np.int8)
    for receiver in range(len(value)):
        result[receiver] = value[reachable[receiver]].max(axis=0)
    return result


def make_graph(private_pair, kind):
    """Copy pair or [A,empty,B], returning own maps and directed adjacency.

    GBP v246/v213 instead accepts unique UNDIRECTED (i,j,weight) factors:
    the runner must canonicalize these reciprocal pairs once, not insert two
    factors. Keeping directed labels explicit avoids a false causal shortcut.
    """
    pair = _maps(private_pair)
    if len(pair) != 2 or kind not in GRAPH_KINDS:
        raise ValueError('Two original source maps and registered graph kind required')
    if kind == 'pair':
        return np.array(pair, dtype=np.int8, copy=True), ((0, 1), (1, 0))
    result = np.stack((pair[0], np.full(pair.shape[1:], -1, dtype=np.int8), pair[1])).astype(np.int8)
    return result, ((0, 1), (1, 0), (1, 2), (2, 1))


def build_train_bank(p):
    """Return original private cache [2048,2,2,128,128] and audit manifest.

    Existing validated_train_bank establishes geometry identity before views
    are made. No validation/test world is generated here. Native bank hash,
    original row identities and private-view byte hash remain explicit.
    """
    if p.get('footprint_seed', FOOTPRINT_SEED) != FOOTPRINT_SEED:
        raise ValueError('Preserve original footprint seed230101')
    bank, digest = validated_train_bank(p)
    if len(bank) != WORLD_COUNT or digest != TRAIN_SHA256:
        raise ValueError('Exact original TRAIN2048 bank required')
    private = np.empty((WORLD_COUNT, len(MODES), 2, 128, 128), dtype=np.int8)
    identities = []
    for world_index, sample in enumerate(bank):
        meta = sample['metadata']
        if str(meta['split']).upper() != 'TRAIN' or meta['complexity'] not in ('base', 'medium'):
            raise ValueError('TRAIN-only registered complexity required')
        index = _integer(meta['index'], 'original world index')
        identities.append(dict(world_index=world_index, split='TRAIN', index=index,
                               complexity=meta['complexity']))
        for mode_index, mode in enumerate(MODES):
            pair = paired_private(sample, mode, FOOTPRINT_SEED)
            if pair.shape != (2, 128, 128) or pair.dtype != np.int8:
                raise ValueError('Unmodified native128 int8 source pair required')
            _maps(pair)
            private[world_index, mode_index] = pair
    ids = [(row['complexity'], row['index']) for row in identities]
    if len(set(ids)) != WORLD_COUNT:
        raise ValueError('Duplicate original TRAIN world identity')
    if set(ids) != {(c, i) for c in ('base', 'medium') for i in range(1024)}:
        raise ValueError('Wrong original TRAIN world identity set')
    manifest = dict(bank_sha256=digest, private_sha256=hashlib.sha256(private.tobytes()).hexdigest(),
        shape=list(private.shape), dtype='int8', world_identities=identities,
        footprints=list(MODES), graph_kinds=list(GRAPH_KINDS), footprint_seed=FOOTPRINT_SEED,
        onlyTRAIN=True, test_opened=False, original_private_views_unmodified=True,
        privileged_union_or_full_sensor_inputs=False,
        labels_are_not_inference_inputs=True, unknown_value=-1,
        causal_assumption='Static active positive-precision directed routes, synchronous rounds; no decoded-source recursion',
        line3_semantics='sourceA,empty sensor,sourceB; relay may reach both after1, endpoints after2')
    return private, manifest


def draw_plan(updates, batch_graphs=2, seed=SAMPLING_SEED):
    """Fixed-seed TRAIN indices, balanced mode x graph in every four updates.

    Return little-endian int64 [updates,batch_graphs,3] and manifest. World
    indices are sampled uniformly with replacement from the retained2048 rows.
    Each update has one mode and graph kind for non-ragged GPU batching.
    Exact balance is over graph draws, not nodes: N3 has more node outputs;
    the runner must average loss per graph before averaging graphs.
    """
    updates = _integer(updates, 'updates', 1)
    batch_graphs = _integer(batch_graphs, 'batch_graphs', 1)
    if _integer(seed, 'seed') != SAMPLING_SEED:
        raise ValueError('Fixed sampling seed277102; no implicit seed sweep')
    count = updates * batch_graphs
    if updates % 4:
        raise ValueError('Update count must be divisible by4 for exact balance')
    rng = np.random.default_rng(seed)
    combos = np.array([(m, g) for m in range(2) for g in range(2)], dtype='<i8')
    plan = np.empty((updates, batch_graphs, 3), dtype='<i8')
    for first in range(0, updates, 4):
        order = combos[rng.permutation(4)]
        plan[first:first+4, :, 0] = rng.integers(WORLD_COUNT, size=(4, batch_graphs))
        plan[first:first+4, :, 1:] = order[:, None, :]
    manifest = dict(sha256=hashlib.sha256(plan.tobytes()).hexdigest(), shape=list(plan.shape),
        dtype='<i8', columns=list(DRAW_COLUMNS), seed=seed, worlds=WORLD_COUNT,
        graph_counts={name:count//2 for name in GRAPH_KINDS},
        footprint_counts={name:count//2 for name in MODES},
        mode_graph_counts={f'{m}/{g}':count//4 for m in MODES for g in GRAPH_KINDS},
        onlyTRAIN=True, test_opened=False, world_sampling='Uniform retained-bank row, with replacement',
        same_mode_and_graph_per_update=True,
        balance_unit='Graph draw, not node; train/eval loss must average per graph')
    return plan, manifest
