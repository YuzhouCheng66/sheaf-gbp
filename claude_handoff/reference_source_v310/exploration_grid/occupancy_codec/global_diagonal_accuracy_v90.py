"""Accuracy-first whole-map64 diagonal Gaussian source/GBP, untrained v90.

The64 coordinates represent ONE ENTIRE map in a fixed identity/common latent
chart, not64 patches or calibrated occupancy variances. The encoder/decoder
and same-parameter tied/diagonal129-parameter residual are imported unchanged
from v89. Only the quantizer/protocol changes:64 eight-bit means AND64 eight-bit
precisions,1024 data bits,146 plain bytes. This is not the old82-byte budget and
does not establish a communication win. Actual framed bytes must be measured.

Positive q uses255 fixed log-spaced FP32 entries .05..20; exact zero alone uses
code0. Mean codes0..254 span[-1,1], zero127; mean255 is reserved. Nonempty own
unaries remain strictly positive .05..4: there is no learned-zero/calibration
claim. Two quantized synchronous reverse-cavity Schur rounds approximate GBP
for the coordinatewise diagonal energy declared by v89, not an exact posterior.
Public directed link drops obey the exact shared physical-radius rule; they
are not claimed equivalent to a symmetric factor-graph posterior. No private
decoder skip, raw-map carrier, own overwrite, data/weights, or training here.
"""
import hashlib
import math

import numpy as np
import torch
from torch import nn

from occupancy_codec.gaussian import HEADER
from occupancy_codec.global_diagonal_gaussian_v89 import (
    GlobalDiagonalGaussianConfig as GlobalDiagonalAccuracyConfig,
    GlobalDiagonalGaussianSource as _Architecture,
    _numeric, _outgoing, _validate,
)
from occupancy_codec.global_spatial_source import _geometry
from occupancy_codec.isotropic_patch_wire import _integer, _numpy_codes, _torch_codes
from occupancy_codec.lossless_frame import frame_packet, unframe_packet


MODEL_KIND = 'global-diagonal-accuracy-v90'
SOURCE_VERSION, FUSION_VERSION, LAYOUT_TAG = 19, 20, 0x88
CHANNELS, MEAN_BITS, Q_BITS, ROUNDS = 64, 8, 8, 2
EDGE_VARIANCE, Q_MIN, Q_MAX = .05, .05, 20.
PLAIN_BYTES = HEADER.size+2+2*CHANNELS
# CPU/Python construction fixes table bits independently of receiver device.
_LOG_MIN, _LOG_RANGE = math.log(Q_MIN), math.log(Q_MAX/Q_MIN)
Q_LUT = tuple(np.asarray([0.]+[math.exp(_LOG_MIN+index*_LOG_RANGE/254)
                              for index in range(255)], dtype=np.float32).tolist())
Q_LUT = (0., float(np.float32(Q_MIN)))+Q_LUT[2:-1]+(Q_MAX,)


class GlobalDiagonalAccuracyQuantizer(nn.Module):
    """Fixed FP32 decode tables, arithmetic log-round q, exact-forward q/h STE."""
    def __init__(self):
        super().__init__()
        self.register_buffer('q_lut', torch.tensor(Q_LUT, dtype=torch.float32, device='cpu'))
        means = torch.linspace(-1., 1., 255, dtype=torch.float32, device='cpu'); means[127] = 0.
        self.register_buffer('mean_lut', means)

    def _tables(self, device):
        if (self.q_lut.device != device or self.mean_lut.device != device
                or self.q_lut.dtype != torch.float32 or self.mean_lut.dtype != torch.float32):
            raise ValueError('Independent fixed FP32 diagonal accuracy LUTs must share the tensor device')

    def positive_q_codes(self, q):
        """Clipped pre-Schur diagnostic also uses this rule, never for inference."""
        self._tables(q.device)
        code = ((q.clamp(Q_MIN, Q_MAX).log()-_LOG_MIN)/_LOG_RANGE*254).round().long()+1
        return torch.where(q > 0, code, torch.zeros_like(code))

    def decode(self, qcodes, mean_codes):
        if not isinstance(qcodes, torch.Tensor) or qcodes.ndim < 1 or qcodes.shape[-1] != CHANNELS:
            raise ValueError('Diagonal accuracy precision codes[...,64] required')
        qc = _torch_codes(qcodes, qcodes.shape, 255, 'diagonal accuracy precision codes')
        mc = _torch_codes(mean_codes, qcodes.shape, 254, 'diagonal accuracy mean codes')
        if qcodes.device != mean_codes.device:
            raise ValueError('Diagonal accuracy codes must share device')
        self._tables(qcodes.device)
        if ((qc == 0) & (mc != 127)).any():
            raise ValueError('Zero-q coordinate requires canonical mean127')
        q = self.q_lut[qc]
        return q, q*self.mean_lut[mc]

    def encode(self, q, h):
        _numeric(q, h, wire=True); self._tables(q.device)
        qc = self.positive_q_codes(q)
        mean = h/torch.where(q > 0, q, torch.ones_like(q))
        mc = ((mean.clamp(-1., 1.)+1.)*127).round().long()
        mc = torch.where(q > 0, mc, torch.full_like(mc, 127))
        rq, rh = self.decode(qc, mc)
        rq, rh = rq.to(q.dtype), rh.to(h.dtype)
        return qc, mc, rq.detach()+(q-q.detach()), rh.detach()+(h-h.detach())


def _serialize(version, max_round, tick, sender, receiver, qcodes, mean_codes, height, width, round_id):
    height, width = _geometry(height, width)
    tick = _integer(tick, 'tick', (1 << 32)-1)
    sender, receiver = _integer(sender, 'sender', 65535), _integer(receiver, 'receiver', 65535)
    round_id = _integer(round_id, 'diagonal accuracy round', max_round)
    if version == FUSION_VERSION and sender == receiver:
        raise ValueError('Self-directed diagonal accuracy fusion packet is not a physical edge')
    qc = _numpy_codes(qcodes, (CHANNELS,), 255, 'diagonal accuracy precision codes')
    mc = _numpy_codes(mean_codes, (CHANNELS,), 254, 'diagonal accuracy mean codes')
    if ((qc == 0) & (mc != 127)).any():
        raise ValueError('Zero-q coordinate requires canonical mean127')
    if not qc.any():
        return None
    return (HEADER.pack(tick, sender, receiver, CHANNELS, height, width, round_id)
            +bytes((version, LAYOUT_TAG))+qc.astype(np.uint8).tobytes()+mc.astype(np.uint8).tobytes())


def _deserialize(packet, version, max_round):
    if not isinstance(packet, (bytes, bytearray, memoryview)) or len(packet) != PLAIN_BYTES:
        raise ValueError('Whole-map diagonal accuracy packet must be exactly146 bytes')
    meta = HEADER.unpack_from(packet); _geometry(meta[4], meta[5])
    if (meta[3] != CHANNELS or not 0 <= meta[6] <= max_round
            or packet[HEADER.size] != version or packet[HEADER.size+1] != LAYOUT_TAG
            or (version == FUSION_VERSION and meta[1] == meta[2])):
        raise ValueError('Diagonal accuracy family/layout/channels/round/address mismatch')
    qc = np.frombuffer(packet, np.uint8, count=CHANNELS, offset=HEADER.size+2).astype(np.int64)
    mc = np.frombuffer(packet, np.uint8, count=CHANNELS, offset=HEADER.size+2+CHANNELS).astype(np.int64)
    if (mc == 255).any() or ((qc == 0) & (mc != 127)).any() or not qc.any():
        raise ValueError('Reserved/noncanonical/empty diagonal accuracy payload')
    return meta, qc, mc


def serialize_global_source(tick, sender, receiver, qcodes, mean_codes, height, width, round_id=0):
    return _serialize(SOURCE_VERSION, 0, tick, sender, receiver, qcodes, mean_codes, height, width, round_id)


def deserialize_global_source(packet):
    return _deserialize(packet, SOURCE_VERSION, 0)


def serialize_global_message(tick, sender, receiver, qcodes, mean_codes, height, width, round_id):
    return _serialize(FUSION_VERSION, 1, tick, sender, receiver, qcodes, mean_codes, height, width, round_id)


def deserialize_global_message(packet):
    return _deserialize(packet, FUSION_VERSION, 1)


def _trace():
    return dict(rounds=[], slots='[B,sender,receiver]', rounds_count=ROUNDS,
        channels=CHANNELS, diagonal_precisions=64, gaussian_payload_scalars=128,
        mean_bits=MEAN_BITS, q_bits=Q_BITS, packed_data_bits=1024,
        wire_version=FUSION_VERSION, layout_tag=LAYOUT_TAG, plain_nonempty_packet_bytes=PLAIN_BYTES,
        own_unary_quantized=False, final_belief_quantized=False, edge_variance=EDGE_VARIANCE,
        restrictions='fixed identity common global chart', exclusive_reverse_cavity_sums=True,
        calibrated_occupancy_variance_claim=False, learned_zero_unary_claim=False,
        quantized_message_passing_is_approximate=True, exact_posterior_claim=False,
        source_or_fusion_fidelity_claim=False, communication_rate_win_claim=False,
        old82byte_budget=False, accuracy_first=True)


def _fusion(q, h, adjacency, positions, active, height, width, comm_radius_cells, actual_wire, tick=0):
    height, width = _validate(q, h, adjacency, positions, active, height, width, comm_radius_cells)
    if actual_wire:
        tick = _integer(tick, 'tick', (1 << 32)-1)
    messages_q = q.new_zeros(*adjacency.shape, 64); messages_h = torch.zeros_like(messages_q)
    sender_quantizer = GlobalDiagonalAccuracyQuantizer().to(q.device)
    receiver_quantizer = GlobalDiagonalAccuracyQuantizer().to(q.device)
    trace = _trace()
    trace['transport'] = 'actual v20/frame/independent parse and decode' if actual_wire else 'tensor STE surrogate; no byte claim'
    if actual_wire:
        trace.update(total_plain_bytes=0, total_framed_bytes=0, total_nonempty_packets=0,
                     total_generic_wrapper_bytes=0, independent_packet_decode=True)
    for round_id in range(ROUNDS):
        cavity_q, cavity_h, outgoing_q, outgoing_h = _outgoing(q, h, messages_q, messages_h, adjacency)
        qc, mc, decoded_q, decoded_h = sender_quantizer.encode(outgoing_q, outgoing_h)
        clipped_cavity_codes = sender_quantizer.positive_q_codes(cavity_q.clamp(0, 20))
        present = adjacency[..., None] & (cavity_q > 0)
        attenuation_erased = present & (outgoing_q < cavity_q) & (qc == clipped_cavity_codes)
        wire = {}
        if actual_wire:
            decoded_q, decoded_h = torch.zeros_like(messages_q), torch.zeros_like(messages_h)
            parsed_qc, parsed_mc = torch.zeros_like(qc), torch.full_like(mc, 127)
            records, plain_bytes, framed_bytes, nonempty = [], 0, 0, 0
            for batch, sender, receiver in adjacency.nonzero(as_tuple=False).cpu().tolist():
                packet = serialize_global_message(tick, sender, receiver,
                    qc[batch, sender, receiver].cpu().numpy(), mc[batch, sender, receiver].cpu().numpy(),
                    height, width, round_id)
                record = dict(batch=batch, sender=sender, receiver=receiver, round_id=round_id,
                    sent=packet is not None, plain_bytes=0, framed_bytes=0, generic_wrapper_bytes=0,
                    decoded_positive_precision_coordinates=0, decoded_natural_norm=0.)
                if packet is not None:
                    framed = frame_packet(packet); restored = unframe_packet(framed)
                    if restored != packet:
                        raise AssertionError('Generic lossless frame changed diagonal accuracy packet')
                    meta, parsed_q, parsed_m = deserialize_global_message(restored)
                    if meta != (tick, sender, receiver, CHANNELS, height, width, round_id):
                        raise AssertionError('Diagonal accuracy packet metadata differs from physical directed delivery')
                    parsed_q = torch.as_tensor(parsed_q.copy(), device=q.device)
                    parsed_m = torch.as_tensor(parsed_m.copy(), device=q.device)
                    rq, rh = receiver_quantizer.decode(parsed_q, parsed_m)
                    decoded_q[batch, sender, receiver] = rq.to(q.dtype)
                    decoded_h[batch, sender, receiver] = rh.to(h.dtype)
                    parsed_qc[batch, sender, receiver] = parsed_q; parsed_mc[batch, sender, receiver] = parsed_m
                    record.update(plain_bytes=len(packet), framed_bytes=len(framed), generic_wrapper_bytes=9,
                        qcodes=parsed_q.cpu().tolist(), mean_codes=parsed_m.cpu().tolist(),
                        decoded_positive_precision_coordinates=int((rq > 0).sum()), decoded_natural_norm=float(rh.norm()),
                        packet_sha256=hashlib.sha256(packet).hexdigest(), original_frame_bit_exact=True)
                    plain_bytes += len(packet); framed_bytes += len(framed); nonempty += 1
                records.append(record)
            wire = dict(directed_delivery_attempts=records, plain_bytes=plain_bytes, framed_bytes=framed_bytes,
                        nonempty_packets=nonempty, parsed_qcodes=parsed_qc, parsed_mean_codes=parsed_mc)
            trace['total_plain_bytes'] += plain_bytes; trace['total_framed_bytes'] += framed_bytes
            trace['total_nonempty_packets'] += nonempty; trace['total_generic_wrapper_bytes'] += 9*nonempty
        messages_q, messages_h = decoded_q, decoded_h
        trace['rounds'].append(dict(round_id=round_id, cavity_q=cavity_q, cavity_h=cavity_h,
            prequant_q=outgoing_q, prequant_h=outgoing_h, qcodes=qc, mean_codes=mc,
            q=messages_q, h=messages_h, clipped_cavity_qcodes=clipped_cavity_codes,
            schur_attenuation_same_qcode=attenuation_erased,
            schur_attenuation_same_qcode_count=attenuation_erased.sum(), **wire))
    return q+messages_q.sum(1), h+messages_h.sum(1), trace


def tensor_fusion(q, h, adjacency, positions, active, height, width, comm_radius_cells):
    return _fusion(q, h, adjacency, positions, active, height, width, comm_radius_cells, False)


@torch.no_grad()
def actual_wire_fusion(q, h, adjacency, positions, active, height, width, comm_radius_cells, tick=0):
    return _fusion(q, h, adjacency, positions, active, height, width, comm_radius_cells, True, tick)


class GlobalDiagonalAccuracySource(_Architecture):
    """Unchanged v89 neural architecture, distinct accuracy-first wire family."""
    def __init__(self, config=GlobalDiagonalAccuracyConfig(), *, tied=False):
        super().__init__(config, tied=tied)
        self.quantizer = GlobalDiagonalAccuracyQuantizer()

    @classmethod
    def init_from_spatial(cls, parent, *, tied=False):
        model = super().init_from_spatial(parent, tied=tied)
        model.quantizer = GlobalDiagonalAccuracyQuantizer()
        return model

    def tensor_fusion(self, *args, **kwargs):
        return tensor_fusion(*args, **kwargs)

    def actual_wire_fusion(self, *args, **kwargs):
        return actual_wire_fusion(*args, **kwargs)

    def counts(self):
        result = super().counts()
        # Architecture statistics do not describe a checkpoint's training past.
        result.pop('source_only'); result.pop('fusion_training_performed')
        result.update(model_kind=MODEL_KIND, mean_bits=MEAN_BITS, q_bits=Q_BITS,
            packed_bits_per_coordinate=16, packed_data_bits=1024, gaussian_payload_scalars=128,
            source_wire_version=SOURCE_VERSION, fusion_wire_version=FUSION_VERSION, layout_tag=LAYOUT_TAG,
            plain_nonempty_packet_bytes=PLAIN_BYTES, uncompressed_framed_nonempty_bytes=PLAIN_BYTES+9,
            old82byte_budget=False, accuracy_first=True, exclusive_reverse_cavity_sums=True,
            training_status_recorded_in_checkpoint=True,
            precision_quantizer='fixed FP32 loground255positive .05..20; exactzero code0',
            mean_quantizer='fixed FP32 uniform255levels [-1,1]; zero127; reserved255')
        return result
