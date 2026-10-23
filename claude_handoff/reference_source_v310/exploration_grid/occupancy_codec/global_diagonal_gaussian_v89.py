"""Untrained whole-map64 diagonal Gaussian source/GBP causal pair, v89.

The64 learned coordinates still represent ONE ENTIRE map in one common latent
chart/public frame. They are not patches, cells, or a calibrated knownness mask.
Both arms copy v84 and add the same129-parameter spatial precision residual.
The tied arm averages residual logits; the diagonal arm retains their plane.
Nonempty unary precisions are strictly positive .05..4: no learned-zero claim.

Identity edges preserve diagonal Gaussian closure for the chosen energy
 .5*sum_ik q_ik*(x_ik-mu_ik)^2 + .5/.05*sum_edges,k (x_ik-x_jk)^2.
Zero-q coordinates require zero h and are unanchored/improper, not occupancy
unknown-cell indicators. Exactly two synchronous reverse-cavity Schur rounds
are quantized, so this is approximate GBP, not an exact posterior claim.
Directed public-radius link drops are allowed, without claiming equivalence to
the posterior of an original symmetric graph. Positions only admit edges.

One byte/coordinate packs3 precision bits and5 mean bits. The fixed coarse LUT
can erase Schur attenuation; traces report these code transitions. The tied arm
wastes redundant q bits for a causal comparison, not a rate win. Keep the old
efficient64mean/scalar-q8bit model as a separate reference. No private decoder
skip, own-map overwrite, training, checkpoint/data read, or rate claim here.
"""
import copy
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from occupancy_codec.gaussian import HEADER
from occupancy_codec.global_gaussian_fusion import _validate as _validate_public
from occupancy_codec.global_spatial_source import (
    GlobalSpatialSource, GlobalSpatialSourceConfig, _configuration, _geometry, _public,
)
from occupancy_codec.isotropic_patch_wire import BOUND_TOLERANCE, _integer, _numpy_codes, _torch_codes
from occupancy_codec.lossless_frame import frame_packet, unframe_packet


MODEL_KIND = 'global-diagonal-gaussian-v89'
SOURCE_VERSION, FUSION_VERSION, LAYOUT_TAG = 17, 18, 0x35
CHANNELS, MEAN_BITS, Q_BITS, ROUNDS = 64, 5, 3, 2
EDGE_VARIANCE = .05
Q_LUT = (0., .05, .15, .45, 1.35, 4., 12., 20.)
PLAIN_BYTES = HEADER.size+1+1+CHANNELS
GlobalDiagonalGaussianConfig = GlobalSpatialSourceConfig


def _numeric(q, h, wire=False):
    if (not isinstance(q, torch.Tensor) or not isinstance(h, torch.Tensor) or q.ndim < 1
            or q.shape[-1] != CHANNELS or h.shape != q.shape or not q.is_floating_point()
            or not h.is_floating_point() or q.dtype != h.dtype or q.device != h.device):
        raise ValueError('Matching floating diagonal q/h[...,64] required')
    if (not torch.isfinite(q).all() or not torch.isfinite(h).all() or (q < 0).any()
            or ((q == 0) & (h != 0)).any()):
        raise ValueError('Finite nonnegative diagonal q with zero-q/zero-h consistency required')
    if wire and (q > 20.+BOUND_TOLERANCE).any():
        raise ValueError('Outgoing diagonal q outside fixed[0,20] wire domain')
    mean = h/torch.where(q > 0, q, torch.ones_like(q))
    if (mean.abs() > 1.+BOUND_TOLERANCE).any():
        raise ValueError('Diagonal latent means must lie in[-1,1]')


class GlobalDiagonalQuantizer(nn.Module):
    """Fixed FP32 LUT, nearest log-positive q,31 mean levels, q/h STE.

    Zero q alone maps to code0; every positive q maps to one of seven positive
    entries. Zero-q coordinates canonically use mean15 and decode h=0. Learned
    unaries themselves cannot choose zero. Mean31 is always invalid.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('q_lut', torch.tensor(Q_LUT, dtype=torch.float32))
        means = torch.linspace(-1., 1., 31, dtype=torch.float32); means[15] = 0.
        self.register_buffer('mean_lut', means)

    def _tables(self, device):
        if (self.q_lut.device != device or self.mean_lut.device != device
                or self.q_lut.dtype != torch.float32 or self.mean_lut.dtype != torch.float32):
            raise ValueError('Independent fixed FP32 diagonal LUTs must share the tensor device')

    def positive_q_codes(self, q):
        """Also used to audit the clipped pre-Schur cavity, never for inference."""
        self._tables(q.device)
        distances = (q.clamp_min(.05).log()[..., None]-self.q_lut[1:].log()).abs()
        return torch.where(q > 0, distances.argmin(-1)+1, torch.zeros_like(q, dtype=torch.long))

    def decode(self, qcodes, mean_codes):
        if not isinstance(qcodes, torch.Tensor) or qcodes.ndim < 1 or qcodes.shape[-1] != CHANNELS:
            raise ValueError('Diagonal precision codes[...,64] required')
        qc = _torch_codes(qcodes, qcodes.shape, 7, 'diagonal precision codes')
        mc = _torch_codes(mean_codes, qcodes.shape, 30, 'diagonal mean codes')
        if qcodes.device != mean_codes.device:
            raise ValueError('Diagonal codes must share device')
        self._tables(qcodes.device)
        if ((qc == 0) & (mc != 15)).any():
            raise ValueError('Zero-q coordinate requires canonical mean15')
        q = self.q_lut[qc]
        return q, q*self.mean_lut[mc]

    def encode(self, q, h):
        _numeric(q, h, wire=True); self._tables(q.device)
        qc = self.positive_q_codes(q)
        mean = h/torch.where(q > 0, q, torch.ones_like(q))
        mc = ((mean.clamp(-1, 1)+1)*15).round().long()
        mc = torch.where(q > 0, mc, torch.full_like(mc, 15))
        rq, rh = self.decode(qc, mc)
        rq, rh = rq.to(q.dtype), rh.to(h.dtype)
        return qc, mc, rq.detach()+(q-q.detach()), rh.detach()+(h-h.detach())


def _serialize(version, max_round, tick, sender, receiver, qcodes, mean_codes, height, width, round_id):
    height, width = _geometry(height, width)
    tick = _integer(tick, 'tick', (1 << 32)-1)
    sender, receiver = _integer(sender, 'sender', 65535), _integer(receiver, 'receiver', 65535)
    round_id = _integer(round_id, 'diagonal round', max_round)
    if version == FUSION_VERSION and sender == receiver:
        raise ValueError('Self-directed diagonal fusion packet is not a physical edge')
    qc = _numpy_codes(qcodes, (CHANNELS,), 7, 'diagonal precision codes')
    mc = _numpy_codes(mean_codes, (CHANNELS,), 30, 'diagonal mean codes')
    if ((qc == 0) & (mc != 15)).any():
        raise ValueError('Zero-q coordinate requires canonical mean15')
    if not qc.any():
        return None
    packed = ((qc << 5) | mc).astype(np.uint8).tobytes()
    return HEADER.pack(tick, sender, receiver, CHANNELS, height, width, round_id)+bytes((version, LAYOUT_TAG))+packed


def _deserialize(packet, version, max_round):
    if not isinstance(packet, (bytes, bytearray, memoryview)) or len(packet) != PLAIN_BYTES:
        raise ValueError('Whole-map diagonal packet must be exactly82 bytes')
    meta = HEADER.unpack_from(packet); _geometry(meta[4], meta[5])
    if (meta[3] != CHANNELS or not 0 <= meta[6] <= max_round
            or packet[HEADER.size] != version or packet[HEADER.size+1] != LAYOUT_TAG
            or (version == FUSION_VERSION and meta[1] == meta[2])):
        raise ValueError('Diagonal family/layout/channels/round/address mismatch')
    packed = np.frombuffer(packet, np.uint8, count=CHANNELS, offset=HEADER.size+2).astype(np.int64)
    qc, mc = packed >> 5, packed & 31
    if (mc == 31).any() or ((qc == 0) & (mc != 15)).any() or not qc.any():
        raise ValueError('Reserved/noncanonical/empty diagonal payload')
    return meta, qc, mc


def serialize_global_source(tick, sender, receiver, qcodes, mean_codes, height, width, round_id=0):
    return _serialize(SOURCE_VERSION, 0, tick, sender, receiver, qcodes, mean_codes, height, width, round_id)


def deserialize_global_source(packet):
    return _deserialize(packet, SOURCE_VERSION, 0)


def serialize_global_message(tick, sender, receiver, qcodes, mean_codes, height, width, round_id):
    return _serialize(FUSION_VERSION, 1, tick, sender, receiver, qcodes, mean_codes, height, width, round_id)


def deserialize_global_message(packet):
    return _deserialize(packet, FUSION_VERSION, 1)


class GlobalDiagonalEncoder(nn.Module):
    def __init__(self, parent, tied):
        super().__init__()
        for name in ('layers', 'pool', 'mean', 'precision'):
            setattr(self, name, copy.deepcopy(getattr(parent, name)))
        self.q_residual = nn.Conv2d(128, 1, 1)
        nn.init.zeros_(self.q_residual.weight); nn.init.zeros_(self.q_residual.bias)
        self.tied = tied

    def forward(self, private, positions, active=None):
        if not isinstance(private, torch.Tensor) or private.ndim != 4:
            raise ValueError('Own private[B,N,H,W] required')
        b, n, height, width = private.shape; _geometry(height, width)
        if (private.dtype == torch.bool or private.is_complex() or not torch.isfinite(private).all()
                or not ((private == -1) | (private == 0) | (private == 1)).all()):
            raise ValueError('Own maps must be finite unknown-1/free0/wall1')
        if (not isinstance(positions, torch.Tensor) or positions.shape != (b, n, 2)
                or positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
                or positions.device != private.device):
            raise ValueError('Own integer positions[B,N,2] on matching device required')
        if active is not None and (not isinstance(active, torch.Tensor) or active.shape != (b, n)
                or active.dtype != torch.bool or active.device != private.device):
            raise ValueError('Public activity must be bool[B,N] on matching device')
        marker = torch.zeros(b, n, height*width, device=private.device, dtype=torch.float32)
        inside = ((positions[..., 0] >= 0) & (positions[..., 0] < height)
                  & (positions[..., 1] >= 0) & (positions[..., 1] < width))
        index = (positions.long()[..., 0]*width+positions.long()[..., 1]).clamp(0, height*width-1)
        marker.scatter_(-1, index[..., None], inside.float()[..., None])
        own = torch.stack(((private == 0).float(), (private == 1).float(), marker.reshape(b, n, height, width)), 2)
        public = _public(b*n, height, width, private.device, own.dtype, height, width)
        hidden = self.pool(self.layers(torch.cat((own.flatten(0, 1), public), 1)))
        mean = self.mean(hidden).flatten(1).tanh().reshape(b, n, 64)
        global_logits = self.precision(hidden.mean((-1, -2)))
        residual = self.q_residual(hidden)
        if self.tied:
            residual = residual.mean((-1, -2), keepdim=True).expand_as(residual)
        logits_plane = global_logits[:, :, None, None].expand_as(residual).contiguous()
        global_q = .05+3.95*global_logits.sigmoid()
        # Mathematically .05+3.95*sigmoid(global_logits+residual). Identically
        # shaped plane operations cancel EXACTLY at zero residual while keeping
        # live residual gradients; the scalar anchor matches original v84 bits.
        q = (global_q[:, :, None, None]+3.95*((logits_plane+residual).sigmoid()-logits_plane.sigmoid())).clamp(.05, 4.)
        nonempty = (private >= 0).any((-1, -2))
        if active is not None:
            nonempty = nonempty & active
        q = q.reshape(b, n, 64)*nonempty[..., None]
        return q, q*mean


class GlobalDiagonalDecoder(nn.Module):
    def __init__(self, parent):
        super().__init__()
        for name in ('lift', 'blocks', 'output'):
            setattr(self, name, copy.deepcopy(getattr(parent, name)))

    def forward(self, q, h, height, width):
        height, width = _geometry(height, width); _numeric(q, h)
        if q.ndim != 3:
            raise ValueError('Whole-map diagonal q/h[B,N,64] required')
        b, n = q.shape[:2]; batch = b*n
        mean = (h/torch.where(q > 0, q, torch.ones_like(q))).reshape(batch, 1, 8, 8)
        plane = q.reshape(batch, 1, 8, 8)
        first = plane[:, :, :1, :1].contiguous()
        baseline = first.expand_as(plane).contiguous()
        precision = first.log1p().expand_as(plane)+(plane.log1p()-baseline.log1p())
        hidden = self.lift(torch.cat((mean, precision, _public(batch, 8, 8, q.device, q.dtype, height, width)), 1))
        for size, block in zip((16, 32, 64, 128), self.blocks):
            sy, sx = min(size, height), min(size, width)
            hidden = F.interpolate(hidden, (sy, sx), mode='bilinear', align_corners=False)
            hidden = block(torch.cat((hidden, _public(batch, sy, sx, q.device, q.dtype, height, width)), 1))
        known, wall = self.output(hidden).reshape(b, n, 2, height, width).unbind(2)
        logprob = torch.stack((F.logsigmoid(-known), F.logsigmoid(known)+F.logsigmoid(-wall),
                               F.logsigmoid(known)+F.logsigmoid(wall)), -1)
        empty = torch.full_like(logprob, -80.); empty[..., 0] = 0.
        return torch.where((q == 0).all(-1)[..., None, None, None], empty, logprob)


def _validate(q, h, adjacency, positions, active, height, width, comm_radius_cells):
    _numeric(q, h)
    if q.ndim != 3:
        raise ValueError('Own diagonal q/h[B,N,64] required')
    # ONLY reuse admission. qmax has NO role in diagonal inference or decoding.
    return _validate_public(q.amax(-1, keepdim=True), h, adjacency, positions, active,
                            height, width, comm_radius_cells)


def _outgoing(q, h, previous_q, previous_h, adjacency):
    # Mathematically own+sum(incoming)-reverse. Exclusive sums avoid roundoff
    # from adding then subtracting the reverse value: at a5-bit midpoint that
    # could otherwise change an outgoing code despite exact reverse exclusion.
    # Two scans retain O(B*N^2*C) cost/storage; no extra sender/receiver axis.
    def exclude_reverse(previous):
        zero = torch.zeros_like(previous[:, :1])
        before = torch.cat((zero, previous.cumsum(1)[:, :-1]), 1)
        after = torch.cat((previous.flip(1).cumsum(1).flip(1)[:, 1:], zero), 1)
        return (before+after).transpose(1, 2)
    cavity_q = q[:, :, None]+exclude_reverse(previous_q)
    cavity_h = h[:, :, None]+exclude_reverse(previous_h)
    denominator = 1.+EDGE_VARIANCE*cavity_q
    mask = adjacency[..., None].to(q.dtype)
    return cavity_q, cavity_h, cavity_q/denominator*mask, cavity_h/denominator*mask


def _trace():
    return dict(rounds=[], slots='[B,sender,receiver]', rounds_count=ROUNDS,
        channels=CHANNELS, diagonal_precisions=64, mean_bits=MEAN_BITS, q_bits=Q_BITS,
        packed_data_bits=512, wire_version=FUSION_VERSION, layout_tag=LAYOUT_TAG,
        plain_nonempty_packet_bytes=PLAIN_BYTES, own_unary_quantized=False, final_belief_quantized=False,
        edge_variance=EDGE_VARIANCE, restrictions='fixed identity common global chart',
        calibrated_occupancy_variance_claim=False, learned_zero_unary_claim=False,
        quantized_message_passing_is_approximate=True, exact_posterior_claim=False,
        source_or_fusion_fidelity_claim=False, communication_rate_win_claim=False)


def _fusion(q, h, adjacency, positions, active, height, width, comm_radius_cells, actual_wire, tick=0):
    height, width = _validate(q, h, adjacency, positions, active, height, width, comm_radius_cells)
    if actual_wire:
        tick = _integer(tick, 'tick', (1 << 32)-1)
    messages_q = q.new_zeros(*adjacency.shape, 64); messages_h = torch.zeros_like(messages_q)
    sender_quantizer, receiver_quantizer = GlobalDiagonalQuantizer().to(q.device), GlobalDiagonalQuantizer().to(q.device)
    trace = _trace()
    trace['transport'] = 'actual v18/frame/independent parse and decode' if actual_wire else 'tensor STE surrogate; no byte claim'
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
            parsed_qc, parsed_mc = torch.zeros_like(qc), torch.full_like(mc, 15)
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
                        raise AssertionError('Generic lossless frame changed diagonal packet')
                    meta, parsed_q, parsed_m = deserialize_global_message(restored)
                    if meta != (tick, sender, receiver, CHANNELS, height, width, round_id):
                        raise AssertionError('Diagonal packet metadata differs from physical directed delivery')
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


class GlobalDiagonalGaussianSource(nn.Module):
    def __init__(self, config=GlobalDiagonalGaussianConfig(), *, tied=False):
        super().__init__(); _configuration(config)
        if type(tied) is not bool:
            raise ValueError('tied must be a boolean')
        parent = GlobalSpatialSource(config)
        self.config = copy.deepcopy(config)
        self.encoder = GlobalDiagonalEncoder(parent.encoder, tied)
        self.decoder = GlobalDiagonalDecoder(parent.decoder)
        self.quantizer = GlobalDiagonalQuantizer()

    @classmethod
    def init_from_spatial(cls, parent, *, tied=False):
        if type(parent) is not GlobalSpatialSource:
            raise TypeError('Explicit original v84 spatial source required')
        if type(tied) is not bool:
            raise ValueError('tied must be a boolean')
        if any(value.device.type != 'cpu' or value.dtype != torch.float32
               for value in (*parent.parameters(), *parent.buffers())):
            raise ValueError('Explicit FP32 CPU v84 source required before copying')
        model = cls.__new__(cls); nn.Module.__init__(model)
        model.config = copy.deepcopy(parent.config)
        model.encoder = GlobalDiagonalEncoder(parent.encoder, tied)
        model.decoder = GlobalDiagonalDecoder(parent.decoder)
        model.quantizer = GlobalDiagonalQuantizer()
        model.train(parent.training)
        return model

    @property
    def tied(self):
        return self.encoder.tied

    def source_views(self, private, positions, active=None):
        q, h = self.encoder(private, positions, active)
        _, _, wq, wh = self.quantizer.encode(q, h)
        height, width = private.shape[-2:]
        return self.decoder(q, h, height, width), self.decoder(wq, wh, height, width)

    def tensor_fusion(self, *args, **kwargs):
        return tensor_fusion(*args, **kwargs)

    def actual_wire_fusion(self, *args, **kwargs):
        return actual_wire_fusion(*args, **kwargs)

    def counts(self):
        encoder = sum(p.numel() for p in self.encoder.parameters())
        decoder = sum(p.numel() for p in self.decoder.parameters())
        return dict(model_kind=MODEL_KIND, encoder=encoder, decoder=decoder, total=encoder+decoder,
            q_residual=sum(p.numel() for p in self.encoder.q_residual.parameters()),
            mode='tied' if self.tied else 'diagonal', latent_mean_channels=64, diagonal_precisions=64,
            one_code_for_entire_map=True, mean_bits=5, q_bits=3, packed_bits_per_coordinate=8,
            packed_data_bits=512, source_wire_version=SOURCE_VERSION, fusion_wire_version=FUSION_VERSION,
            layout_tag=LAYOUT_TAG, plain_nonempty_packet_bytes=PLAIN_BYTES, generic_frame_overhead_bytes=9,
            uncompressed_framed_nonempty_bytes=PLAIN_BYTES+9, same_parameter_causal_tied_control=True,
            tied_control_wastes_redundant_precision_bits=True, original_scalar8bit_rate_baseline_retained=True,
            calibrated_occupancy_variance_claim=False, learned_zero_unary_claim=False,
            unary_positive_range=[.05, 4.], source_only=True, fusion_training_performed=False,
            no_private_decoder_context=True, no_own_projection=True, no_raw_map_carrier=True,
            no_rate_or_losslessness_claim=True, model_parameters_not_radio_traffic=True)
