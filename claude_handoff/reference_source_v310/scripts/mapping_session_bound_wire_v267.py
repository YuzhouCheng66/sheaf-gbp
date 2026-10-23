"""Cold-reset isolation around the UNCHANGED v246 148-byte scalar radio.

Envelope = a 16-byte link-incarnation nonce + exact network-order float64
edge weight + original packet: 172 bytes. Nonces are installed by a trusted,
reliable external control plane, NEVER learned from data packets. Establishing
that agreement, authentication, and cross-process anti-replay are not supplied
here and their traffic is not zero. Carried reset-lineage tombstones prevent
reuse; this is not a process-global or persistent session authority.

This is not evidence deduplication, a graph-change solver, or a proof of loopy
GBP / learned-map accuracy / 100x communication. The current source replacement,
quantization, damping, and gradients are untouched. Explicit receive callers
must charge EVERY attempted byte, including stale, wrong-context, and errors.
"""
from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
import struct

import torch

from scripts import mapping_observed_wire_v246 as wire


CONTEXT_KEY = 'session_transport_v267'
ENVELOPE = struct.Struct('!16sd')
ENVELOPE_BYTES = 24
INNER_BYTES = 148
PACKET_BYTES = 172


@dataclass(frozen=True)
class _Binding:
    nodes: int
    # canonical undirected edge, nonce, exact weight bytes
    links: tuple
    used_nonces: frozenset


def _graph(state):
    if state['h'].ndim != 3 or state['h'].shape[0] != 1 or state['h'].shape[-1] != 64:
        raise ValueError('One world with 64-dimensional nodes required')
    n = state['h'].shape[1]
    directed = {}
    for i, j, weight in state['directed']:
        if (any(isinstance(x, bool) or not isinstance(x, Integral) for x in (i, j))
                or not 0 <= i < n or not 0 <= j < n or i == j
                or isinstance(weight, bool) or not isinstance(weight, Real)
                or not math.isfinite(float(weight)) or weight <= 0):
            raise ValueError('Invalid finite positive local edge weight / endpoints')
        if (i, j) in directed:
            raise ValueError('Duplicate directed edge')
        directed[(i, j)] = struct.pack('!d', float(weight))
    for (i, j), encoded_weight in directed.items():
        if directed.get((j, i)) != encoded_weight:
            raise ValueError('Both directions require exactly the same edge weight')
    if any(set(state[key]) != set(directed) for key in ('p', 'eta', 'message_versions')):
        raise ValueError('Cache keys must match the actual directed graph')
    return n, {(i, j): weight for (i, j), weight in directed.items() if i < j}


def _fresh(state):
    if CONTEXT_KEY in state or state['iteration'] != 0:
        raise ValueError('Bind/reset accepts a fresh, unbound solver state only')
    if any(v != -1 for v in state['message_versions'].values()):
        raise ValueError('Fresh message sequence state required')
    if any(torch.count_nonzero(value).item() for key in ('p', 'eta') for value in state[key].values()):
        raise ValueError('Fresh empty message caches required')


def _install(state, edge_contexts, used):
    _fresh(state)
    n, graph = _graph(state)
    if not isinstance(edge_contexts, dict) or set(edge_contexts) != set(graph):
        raise ValueError('Exactly one context for each canonical local edge required')
    new = set()
    links = []
    for key, weight in sorted(graph.items()):
        nonce = edge_contexts[key]
        if not isinstance(nonce, bytes) or len(nonce) != 16:
            raise ValueError('A pre-established 16-byte link nonce is required')
        if nonce in used or nonce in new:
            raise ValueError('A link incarnation nonce may never be reused')
        new.add(nonce)
        links.append((*key, nonce, weight))
    return dict(state, **{CONTEXT_KEY: _Binding(n, tuple(links), frozenset(used | new))})


def bind_context(state, edge_contexts):
    """First trusted installation; caller must use a genuinely new incarnation.

    Do not bypass reset_context by calling this on a second fresh state with an
    old nonce. Only the external reliable control plane can establish agreement
    across processes; this API intentionally does not implement a handshake.
    """
    return _install(state, edge_contexts, set())


def verify_context(state):
    binding = state.get(CONTEXT_KEY)
    if not isinstance(binding, _Binding):
        raise ValueError('Current edge contexts must be installed before data')
    n, graph = _graph(state)
    expected = {(i, j): weight for i, j, _, weight in binding.links}
    if n != binding.nodes or graph != expected:
        raise ValueError('Graph/weight mutation requires a fresh explicit reset')
    if any(nonce not in binding.used_nonces for _, _, nonce, _ in binding.links):
        raise ValueError('Context history is inconsistent')
    return binding


def reset_context(old_state, fresh_new_state, new_contexts):
    """Cold reset, not an incremental hot-graph update or source update.

    Clear caches with normal initialize first, then install NEW agreed nonces.
    Tombstones include removed links, preventing delete/re-add nonce reuse along
    this reset lineage. An empty graph retains those tombstones as well. The
    caller must pass its CURRENT active state, not fork a revoked old snapshot.
    """
    old = verify_context(old_state)
    return _install(fresh_new_state, new_contexts, set(old.used_nonces))


def _route(raw_packet):
    if not isinstance(raw_packet, bytes) or len(raw_packet) != INNER_BYTES:
        raise ValueError('Original 148-byte packet required')
    # Header-only routing: rejected contexts never invoke the old receiver or
    # decode its Gaussian values. Matching contexts are fully validated there.
    header = wire.ObservedCodec64.HEADER.unpack_from(raw_packet)
    return header[-3], header[-2]


def wrap_packet(state, raw_packet):
    binding = verify_context(state)
    decoded = wire.ObservedCodec64.decode(raw_packet)
    if decoded['mode'] != 'standardized' or decoded['precision_profile'] != 2:
        raise ValueError('Unchanged v246 standardized profile2 required')
    key = tuple(sorted((decoded['sender'], decoded['receiver'])))
    links = {(i, j): (nonce, weight) for i, j, nonce, weight in binding.links}
    if key not in links:
        raise ValueError('Cannot send on an inactive local edge')
    nonce, weight = links[key]
    return nonce + weight + raw_packet


def receive(state, packet):
    """Return (state, accepted); caller charges len(packet) even on rejection.

    Wrong session/edge/weight returns the ORIGINAL state, with no counter update.
    Matching-context duplicate/stale/conflict behavior is exactly v246 behavior.
    Malformed framing/payload or corrupted local contexts raise without mutation.
    """
    binding = verify_context(state)
    if not isinstance(packet, bytes) or len(packet) != PACKET_BYTES:
        raise ValueError('Exactly 172 serialized bytes required')
    nonce, weight, raw_packet = packet[:16], packet[16:24], packet[24:]
    i, j = _route(raw_packet)
    key = tuple(sorted((i, j)))
    links = {(a, b): (current_nonce, current_weight)
             for a, b, current_nonce, current_weight in binding.links}
    if key not in links or (i, j) not in state['p'] or links[key] != (nonce, weight):
        return state, False
    return wire.redeliver(state, raw_packet)


def advance(state, rounds=1, *, damping=.5, quantized=True, trace=False):
    """Same live differentiable solver, with actual serialization/accounting.

    Simultaneous packet proposals retain the native synchronous schedule. The
    old actual serializer checks hard numerical equality. The wrapper adds no
    quantization, detach, factor, or trainable parameter to the inference graph.
    Separate CPU tests reconstruct each sweep through receive, not just a ledger.
    """
    verify_context(state)
    if quantized is not True:
        raise ValueError('This wrapper counts the actual quantized v246 radio only')
    result = wire.advance(state, rounds, damping=damping, quantized=True, trace=True)
    audit = wire.audit_packets(result['trace'])
    packets = [wrap_packet(state, packet) for packet in audit['packets']]
    verify_context(result['state'])
    actual = sum(map(len, packets))
    return dict(result, trace=result['trace'] if trace else [], packets=packets,
        actual_serialized_application_bytes=actual,
        application_byte_budget_per_world=actual,
        inner_application_bytes=audit['actual_serialized_application_bytes'],
        envelope_bytes=ENVELOPE_BYTES * len(packets),
        inner_packet_sha256=audit['SHA256'],
        packet_sha256=hashlib.sha256(b''.join(packets)).hexdigest(),
        context_establishment_implemented=False, context_establishment_bytes=None,
        physical_network_headers_measured=False, full_process_communication_measured=False)
