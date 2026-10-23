"""Untrained two-round whole-map C64/scalar-q Gaussian GBP foundation.

All agents must use the SAME public global coordinate frame and latent chart.
The restriction maps are fixed identities, not learned or pose-conditioned.
For chosen unaries q_i,h_i=q_i*mu_i, this implements scalar Gaussian messages
for energy .5*sum_i q_i||x_i-mu_i||² + .5/.05*sum_edges||x_i-x_j||².
Scalar q is chosen Gaussian strength, not calibrated occupancy variance.

Adjacency is a public link-delivery subset of the bidirectional Euclidean
radius graph. Directed drops are permitted; every enabled direction must
independently satisfy the same physical radius and active-endpoint checks.
Radius is one public scalar or one public floating value per independent batch
world, never asymmetric per-agent/per-edge radii or a learned tensor.
The baseline graph may be symmetric. Directed drops are not claimed to yield
the posterior of that original symmetric factor graph.
These are exactly TWO synchronous iterations, not a convergence guarantee.
No source/fusion fidelity, trained restriction, or communication-rate claim is
made. Only q/h, public geometry, topology and positions enter this module.
"""
import hashlib
import math
from numbers import Real

import numpy as np
import torch

from occupancy_codec.gaussian import HEADER
from occupancy_codec.isotropic_patch_wire import BOUND_TOLERANCE, _integer, _numpy_codes
from occupancy_codec.lossless_frame import frame_packet, unframe_packet
from occupancy_codec.site_gaussian_wire import SiteGaussianQuantizer


VERSION = 15
CHANNELS = 64
BITS = 8
ROUNDS = 2
EDGE_VARIANCE = .05
PLAIN_BYTES = HEADER.size + 1 + 1 + CHANNELS


def _geometry(height, width):
    height, width = _integer(height, 'height', 128), _integer(width, 'width', 128)
    if min(height, width) < 8:
        raise ValueError('Whole-map public dimensions8..128 required')
    return height, width


def serialize_global_message(tick, sender, receiver, qcode, mean_codes, height, width, round_id):
    """Version15 scalar-q+64mean packet, exactly82B, or None for zero q."""
    height, width = _geometry(height, width)
    tick = _integer(tick, 'tick', (1 << 32)-1)
    sender, receiver = _integer(sender, 'sender', 65535), _integer(receiver, 'receiver', 65535)
    round_id = _integer(round_id, 'fusion round', ROUNDS-1)
    if sender == receiver:
        raise ValueError('Self-directed fusion packets are not physical edges')
    precision = _numpy_codes(qcode, (1,), 255, 'precision code')
    means = _numpy_codes(mean_codes, (CHANNELS,), 254, 'mean codes')
    if precision[0] == 0:
        return None
    return (HEADER.pack(tick, sender, receiver, CHANNELS, height, width, round_id)
            + bytes((VERSION, int(precision[0]))) + means.astype(np.uint8).tobytes())


def deserialize_global_message(packet):
    """Reject source-round0 v14/site v13: explicit two-round v15 semantics."""
    if not isinstance(packet, (bytes, bytearray, memoryview)) or len(packet) != PLAIN_BYTES:
        raise ValueError('Whole-map fusion packet must be exactly82 bytes')
    meta = HEADER.unpack_from(packet)
    _geometry(meta[4], meta[5])
    if (meta[3] != CHANNELS or meta[6] not in (0, 1) or meta[1] == meta[2]
            or packet[HEADER.size] != VERSION):
        raise ValueError('Whole-map fusion family/channels/round/address mismatch')
    qcode = np.frombuffer(packet, dtype=np.uint8, count=1, offset=HEADER.size+1).astype(np.int64)
    means = np.frombuffer(packet, dtype=np.uint8, count=CHANNELS, offset=HEADER.size+2).astype(np.int64)
    if qcode[0] == 0 or (means == 255).any():
        raise ValueError('Empty precision or reserved mean in nonempty fusion packet')
    return meta, qcode, means


def _validate(q, h, adjacency, positions, active, height, width, comm_radius_cells):
    height, width = _geometry(height, width)
    if (not isinstance(q, torch.Tensor) or not isinstance(h, torch.Tensor)
            or q.ndim != 3 or q.shape[-1] != 1 or min(q.shape[:2]) < 1
            or q.shape[1] > 65536 or h.shape != (*q.shape[:-1], CHANNELS)
            or not q.is_floating_point() or not h.is_floating_point()
            or q.dtype != h.dtype or q.device != h.device):
        raise ValueError('Matching floating scalarq[B,N,1] and naturalh[B,N,64] required')
    batch, agents = q.shape[:2]
    if (not isinstance(adjacency, torch.Tensor) or adjacency.shape != (batch, agents, agents)
            or adjacency.dtype != torch.bool or adjacency.device != q.device):
        raise ValueError('Boolean directed adjacency[B,sender,receiver] on matching device required')
    if (not isinstance(active, torch.Tensor) or active.shape != (batch, agents)
            or active.dtype != torch.bool or active.device != q.device):
        raise ValueError('Boolean activity[B,N] on matching device required')
    if (not isinstance(positions, torch.Tensor) or positions.shape != (batch, agents, 2)
            or positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            or positions.device != q.device):
        raise ValueError('Own integer public-frame positions[B,N,2] on matching device required')
    if isinstance(comm_radius_cells, torch.Tensor):
        if (comm_radius_cells.shape != (batch,) or not comm_radius_cells.is_floating_point()
                or comm_radius_cells.device != q.device or comm_radius_cells.requires_grad
                or not torch.isfinite(comm_radius_cells).all() or (comm_radius_cells < 0).any()):
            raise ValueError('Public physical communication radius tensor must be floating[B], finite, nonnegative, nonlearned, on matching device')
        declared_radius = comm_radius_cells.double()[:, None, None]
    else:
        if (not isinstance(comm_radius_cells, Real) or isinstance(comm_radius_cells, (bool, np.bool_))
                or not math.isfinite(float(comm_radius_cells)) or comm_radius_cells < 0):
            raise ValueError('One finite nonnegative public physical communication radius or floating[B] tensor required')
        declared_radius = float(comm_radius_cells)
    if (not torch.isfinite(q).all() or not torch.isfinite(h).all() or (q < 0).any()
            or ((q == 0).expand_as(h) & (h != 0)).any()):
        raise ValueError('Finite consistent nonnegative Gaussian q/h required')
    mean = h / torch.where(q > 0, q, torch.ones_like(q))
    if (mean.abs() > 1.+BOUND_TOLERANCE).any():
        raise ValueError('Own Gaussian means must lie in[-1,1]')
    if (q[~active] != 0).any() or (h[~active] != 0).any():
        raise ValueError('Inactive agents must have exactly zero own Gaussian information')
    if (positions < 0).any() or (positions[..., 0] >= height).any() or (positions[..., 1] >= width).any():
        raise ValueError('Every own position must be inside the public H/W extent')
    if torch.diagonal(adjacency, dim1=1, dim2=2).any():
        raise ValueError('Self-loop adjacency is not permitted')
    if (adjacency & ~(active[:, :, None] & active[:, None, :])).any():
        raise ValueError('Communication edges require two active endpoints')
    displacement = positions.long()[:, :, None] - positions.long()[:, None, :]
    distance_squared = displacement.square().sum(-1).double()
    # Shared with graph-data admission: sqrt of the exact integer squared
    # distance, in float64. Squaring a declared sqrt13 radius can round below13.
    physical = distance_squared.sqrt() <= declared_radius
    if (adjacency & ~physical).any():
        raise ValueError('Adjacency must be a subset of the public physical-radius graph')
    return height, width


def _outgoing(q, h, previous_q, previous_h, adjacency):
    # Slots [B,sender,receiver]; sum incoming sender axis, exclude j->i at i->j.
    cavity_q = q[:, :, None] + previous_q.sum(1)[:, :, None] - previous_q.transpose(1, 2)
    cavity_h = h[:, :, None] + previous_h.sum(1)[:, :, None] - previous_h.transpose(1, 2)
    denominator = 1. + EDGE_VARIANCE*cavity_q
    mask = adjacency[..., None].to(q.dtype)
    outgoing_q, outgoing_h = cavity_q/denominator*mask, cavity_h/denominator*mask
    return cavity_q, cavity_h, outgoing_q, outgoing_h


def _trace():
    return dict(rounds=[], slots='[B,sender,receiver]', rounds_count=ROUNDS,
        channels=CHANNELS, bits=BITS, global_scalar_precisions=1, wire_version=VERSION,
        plain_nonempty_packet_bytes=PLAIN_BYTES, edge_variance=EDGE_VARIANCE,
        own_unary_quantized=False, final_belief_quantized=False,
        common_coordinate_frame_required=True, restrictions='fixed identity global chart',
        learned_restriction_claim=False, calibrated_occupancy_variance_claim=False,
        fusion_training_performed=False, source_or_fusion_fidelity_claim=False,
        communication_rate_win_claim=False)


def tensor_fusion(q, h, adjacency, positions, active, height, width, comm_radius_cells):
    """Differentiable two-round GBP with immutable per-message b8 STE.

    Own unaries and final beliefs are NOT quantized. Only Schur messages obey
    q<=20; final belief q can exceed20. The trace retains differentiable round
    tensors, including prequantized Schur values and the exact integer codes.
    """
    _validate(q, h, adjacency, positions, active, height, width, comm_radius_cells)
    batch, agents = q.shape[:2]
    messages_q = q.new_zeros(batch, agents, agents, 1)
    messages_h = h.new_zeros(batch, agents, agents, CHANNELS)
    quantizer = SiteGaussianQuantizer().to(q.device)
    trace = _trace()
    trace['transport'] = 'tensor STE surrogate; not actual packet byte accounting'
    for round_id in range(ROUNDS):
        cavity_q, cavity_h, outgoing_q, outgoing_h = _outgoing(q, h, messages_q, messages_h, adjacency)
        qcodes, mean_codes, messages_q, messages_h = quantizer.encode(outgoing_q, outgoing_h)
        trace['rounds'].append(dict(round_id=round_id, cavity_q=cavity_q, cavity_h=cavity_h,
            prequant_q=outgoing_q, prequant_h=outgoing_h, qcodes=qcodes, mean_codes=mean_codes,
            q=messages_q, h=messages_h))
    return q+messages_q.sum(1), h+messages_h.sum(1), trace


@torch.no_grad()
def actual_wire_fusion(q, h, adjacency, positions, active, height, width, comm_radius_cells, tick=0):
    """Independent real v15/frame/parse transport; CPU or GPU inference only.

    Each B entry is a separate communication instance, not a wire side-channel.
    Receiver arrays are built ONLY from independently parsed packet codes. No
    sender's unencoded q/h is passed to its receiver. Empty messages cost zero
    packet bytes; actual generic framing/compression bytes are counted as sent.
    """
    height, width = _validate(q, h, adjacency, positions, active, height, width, comm_radius_cells)
    tick = _integer(tick, 'tick', (1 << 32)-1)
    batch, agents = q.shape[:2]
    messages_q = q.new_zeros(batch, agents, agents, 1)
    messages_h = h.new_zeros(batch, agents, agents, CHANNELS)
    sender_quantizer = SiteGaussianQuantizer().to(q.device)
    receiver_quantizer = SiteGaussianQuantizer().to(q.device)
    edges = adjacency.nonzero(as_tuple=False).cpu().tolist()
    trace = _trace()
    trace.update(transport='actual version15/frame/independent parse and decode',
        total_plain_bytes=0, total_framed_bytes=0, total_generic_wrapper_bytes=0,
        total_nonempty_packets=0, independent_packet_decode=True)
    for round_id in range(ROUNDS):
        cavity_q, cavity_h, outgoing_q, outgoing_h = _outgoing(q, h, messages_q, messages_h, adjacency)
        qcodes, mean_codes, _, _ = sender_quantizer.encode(outgoing_q, outgoing_h)
        received_q, received_h = torch.zeros_like(messages_q), torch.zeros_like(messages_h)
        records, plain_bytes, framed_bytes, nonempty = [], 0, 0, 0
        for batch_id, sender, receiver in edges:
            packet = serialize_global_message(tick, sender, receiver,
                qcodes[batch_id, sender, receiver].cpu().numpy(),
                mean_codes[batch_id, sender, receiver].cpu().numpy(), height, width, round_id)
            record = dict(batch=batch_id, sender=sender, receiver=receiver, round_id=round_id,
                sent=packet is not None, plain_bytes=0, framed_bytes=0, generic_wrapper_bytes=0,
                decoded_precision=0., decoded_natural_norm=0.)
            if packet is not None:
                framed = frame_packet(packet)
                restored = unframe_packet(framed)
                if restored != packet:
                    raise AssertionError('Generic lossless frame changed fusion packet bytes')
                meta, parsed_qcodes, parsed_means = deserialize_global_message(restored)
                if meta != (tick, sender, receiver, CHANNELS, height, width, round_id):
                    raise AssertionError('Fusion packet metadata does not match physical directed delivery')
                decoded_q, decoded_h = receiver_quantizer.decode(
                    torch.as_tensor(parsed_qcodes.copy(), device=q.device),
                    torch.as_tensor(parsed_means.copy(), device=q.device))
                received_q[batch_id, sender, receiver] = decoded_q.to(q.dtype)
                received_h[batch_id, sender, receiver] = decoded_h.to(h.dtype)
                record.update(plain_bytes=len(packet), framed_bytes=len(framed), generic_wrapper_bytes=9,
                    decoded_precision=float(decoded_q[0]), decoded_natural_norm=float(decoded_h.norm()),
                    qcode=int(parsed_qcodes[0]), packet_sha256=hashlib.sha256(packet).hexdigest(),
                    original_frame_bit_exact=True)
                plain_bytes += len(packet); framed_bytes += len(framed); nonempty += 1
            records.append(record)
        messages_q, messages_h = received_q, received_h
        trace['rounds'].append(dict(round_id=round_id, cavity_q=cavity_q, cavity_h=cavity_h,
            prequant_q=outgoing_q, prequant_h=outgoing_h, qcodes=qcodes, mean_codes=mean_codes,
            q=messages_q, h=messages_h, directed_delivery_attempts=records,
            plain_bytes=plain_bytes, framed_bytes=framed_bytes, nonempty_packets=nonempty))
        trace['total_plain_bytes'] += plain_bytes
        trace['total_framed_bytes'] += framed_bytes
        trace['total_nonempty_packets'] += nonempty
        trace['total_generic_wrapper_bytes'] += 9*nonempty
    return q+messages_q.sum(1), h+messages_h.sum(1), trace
