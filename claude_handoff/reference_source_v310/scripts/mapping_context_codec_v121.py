"""Context-aware SOURCE codec and actual two-round Gaussian transport.

This is a new, untrained family. The Gaussian kernel follows the declared
diagonal surrogate; quantized finite-round inference is approximate, not a
theorem about categorical-map union. A latent
site is neutral when no observation lies in its CNN receptive field. Nonzero q
does NOT certify that every decoded cell is observed. No private decoder skip.

All public methods follow the old field's [B,N,H,W] ternary (-1/0/1) interface.
Layout is plane-major/y/x. Actual packets use the unchanged v90 quantizer and
the same generic lossless framing as the RAW comparator (all overhead charged).
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from occupancy_codec.gaussian import HEADER
from occupancy_codec.global_diagonal_accuracy_v90 import GlobalDiagonalAccuracyQuantizer
from occupancy_codec.global_diagonal_gaussian_v89 import _numeric as _numeric64, _outgoing
from occupancy_codec.global_gaussian_fusion import _validate as _validate_public
from occupancy_codec.global_spatial_source import _geometry, _public
from occupancy_codec.isotropic_patch_wire import _integer, _numpy_codes
from occupancy_codec.lossless_frame import frame_packet, unframe_packet
from occupancy_codec.spatial_evidence_field_v102 import _private


MODEL_KIND = 'mapping-context-codec-v121'
SOURCE_VERSION, FUSION_VERSION, LAYOUT_TAG = 121, 122, 0xC1
ROUNDS, EDGE_VARIANCE = 2, .05


@dataclass
class ContextCodecConfig:
    side: int = 16
    planes: int = 1
    bits: int = 8
    width: int = 128
    seed: int = 12101

    @property
    def channels(self):
        return self.side * self.side * self.planes


def _configuration(config):
    if (type(config) is not ContextCodecConfig
            or any(type(getattr(config, k)) is not int for k in ('side', 'planes', 'bits', 'width', 'seed'))
            or config.side != 16 or config.planes not in (1, 4)
            or config.bits != 8 or config.width != 128 or not 0 <= config.seed < 2**63):
        raise ValueError('Typed side16, planes1|4, bits8, width128 context configuration required')


def _numeric(q, h, channels, wire=False):
    if not isinstance(q, torch.Tensor) or q.ndim < 1 or q.shape[-1] != channels:
        raise ValueError('Invalid context natural-parameter dimensions')
    if not isinstance(h, torch.Tensor) or h.shape != q.shape:
        raise ValueError('Matching context q/h required')
    for qb, hb in zip(q.split(64, -1), h.split(64, -1)):
        _numeric64(qb, hb, wire=wire)


class ContextQuantizer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels not in (256, 1024):
            raise ValueError('Context channels must be 256 or 1024')
        self.channels = channels
        self.base = GlobalDiagonalAccuracyQuantizer()

    def encode(self, q, h):
        _numeric(q, h, self.channels, wire=True)
        banks = [self.base.encode(qb, hb) for qb, hb in zip(q.split(64, -1), h.split(64, -1))]
        return tuple(torch.cat([row[k] for row in banks], -1) for k in range(4))

    def decode(self, qcodes, mean_codes):
        if (not isinstance(qcodes, torch.Tensor) or not isinstance(mean_codes, torch.Tensor)
                or qcodes.ndim < 1 or qcodes.shape[-1] != self.channels or qcodes.shape != mean_codes.shape):
            raise ValueError('Matching context integer-code dimensions required')
        banks = [self.base.decode(qb, mb) for qb, mb in zip(qcodes.split(64, -1), mean_codes.split(64, -1))]
        return tuple(torch.cat([row[k] for row in banks], -1) for k in (0, 1))

    def positive_q_codes(self, q):
        if q.shape[-1] != self.channels:
            raise ValueError('Invalid q dimensions')
        return torch.cat([self.base.positive_q_codes(v) for v in q.split(64, -1)], -1)


class ContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__(); _configuration(config)
        self.config = copy.deepcopy(config)
        factory = dict(device='cpu', dtype=torch.float32)
        # free, wall, own-position marker, and four public coordinate channels.
        self.stem = nn.Conv2d(7, 32, 3, padding=1, **factory)
        self.residual = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, **factory), nn.SiLU(),
                                      nn.Conv2d(32, 32, 3, padding=1, **factory))
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1, **factory)
        self.down2 = nn.Conv2d(64, config.width, 3, stride=2, padding=1, **factory)
        self.mean = nn.Conv2d(config.width, config.planes, 1, **factory)
        self.precision = nn.Conv2d(config.width, config.planes, 1, **factory)

    def receptive_support(self, observed):
        """Exact structural receptive-field support, not cellwise visibility.

        3 stride1 convolutions (stem + two residual) and two stride2 convolutions
        yield a 13x13 receptive field before adaptive pooling. Residual identity
        paths are included in that superset. Adaptive up-pooling for tiny maps is
        permitted, declared, and never treated as newly observed information.
        """
        support = observed
        for stride in (1, 1, 1, 2, 2):
            support = F.max_pool2d(support, 3, stride=stride, padding=1)
        return F.adaptive_avg_pool2d(support, (self.config.side, self.config.side))

    def forward(self, private, positions, active=None):
        b, n, height, width = _private(private, active)
        weight = self.mean.weight
        if (weight.dtype != torch.float32 or private.device != weight.device
                or not isinstance(positions, torch.Tensor) or positions.shape != (b, n, 2)
                or positions.device != private.device
                or positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
                or (positions < 0).any() or (positions[..., 0] >= height).any()
                or (positions[..., 1] >= width).any()):
            raise ValueError('Inside integer positions and FP32 model on the input device required')
        dtype = weight.dtype
        marker = torch.zeros(b, n, height * width, device=private.device, dtype=dtype)
        index = positions.long()[..., 0] * width + positions.long()[..., 1]
        marker.scatter_(-1, index[..., None], 1.)
        own = torch.stack(((private == 0).to(dtype), (private == 1).to(dtype),
                           marker.reshape(b, n, height, width)), 2).flatten(0, 1)
        public = _public(b*n, height, width, private.device, dtype, height, width)
        features = F.silu(self.stem(torch.cat((own, public), 1)))
        features = F.silu(features + self.residual(features))
        features = F.silu(self.down1(features))
        features = F.silu(self.down2(features))
        features = F.adaptive_avg_pool2d(features, (self.config.side, self.config.side))
        support = self.receptive_support((private >= 0).to(dtype).reshape(b*n, 1, height, width))
        q = 4. * self.precision(features).sigmoid() * support
        if active is not None:
            q = q * active.reshape(b*n, 1, 1, 1)
        mean = self.mean(features).tanh()
        return q.reshape(b, n, -1), (q * mean).reshape(b, n, -1)


class ContextDecoder(nn.Module):
    def __init__(self, config):
        super().__init__(); _configuration(config)
        self.config = copy.deepcopy(config)
        factory = dict(device='cpu', dtype=torch.float32)
        self.lift = nn.Sequential(nn.Conv2d(2*config.planes+4, 128, 3, padding=1, **factory), nn.SiLU(),
                                  nn.Conv2d(128, 128, 3, padding=1, **factory), nn.SiLU())
        incoming, blocks = 128, []
        for outgoing in (96, 64, 48, 32):
            blocks.append(nn.Sequential(nn.Conv2d(incoming+4, outgoing, 3, padding=1, **factory), nn.SiLU(),
                                        nn.Conv2d(outgoing, outgoing, 3, padding=1, **factory), nn.SiLU()))
            incoming = outgoing
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv2d(32, 2, 1, **factory)

    def forward(self, q, h, height, width):
        height, width = _geometry(height, width); _numeric(q, h, self.config.channels)
        if q.ndim != 3 or min(q.shape[:2]) < 1:
            raise ValueError('Context q/h must have [B,N,K] shape')
        if q.device != self.output.weight.device or q.dtype != self.output.weight.dtype:
            raise ValueError('Decoder and q/h must share dtype/device')
        b, n = q.shape[:2]; count = b*n; side = self.config.side
        mean = (h / torch.where(q > 0, q, torch.ones_like(q))).reshape(count, self.config.planes, side, side)
        precision = q.log1p().reshape_as(mean)
        hidden = self.lift(torch.cat((mean, precision,
            _public(count, side, side, q.device, q.dtype, height, width)), 1))
        for extent, block in zip((16, 32, 64, 128), self.blocks):
            sy, sx = min(extent, height), min(extent, width)
            hidden = F.interpolate(hidden, (sy, sx), mode='bilinear', align_corners=False)
            hidden = block(torch.cat((hidden, _public(count, sy, sx, q.device, q.dtype, height, width)), 1))
        known, wall = self.output(hidden).reshape(b, n, 2, height, width).unbind(2)
        result = torch.stack((F.logsigmoid(-known), F.logsigmoid(known)+F.logsigmoid(-wall),
                              F.logsigmoid(known)+F.logsigmoid(wall)), -1)
        empty = torch.full_like(result, -80.); empty[..., 0] = 0.
        return torch.where((q == 0).all(-1)[..., None, None, None], empty, result)


def serialize_packet(config, tick, sender, receiver, qc, mc, height, width, round_id, *, source=False):
    _configuration(config); height, width = _geometry(height, width)
    tick = _integer(tick, 'tick', 2**32-1)
    sender, receiver = _integer(sender, 'sender', 65535), _integer(receiver, 'receiver', 65535)
    round_id = _integer(round_id, 'round', 0 if source else 1)
    if not source and sender == receiver:
        raise ValueError('A fusion packet cannot be self-directed')
    qc = _numpy_codes(qc, (config.channels,), 255, 'context q codes')
    mc = _numpy_codes(mc, (config.channels,), 254, 'context mean codes')
    if ((qc == 0) & (mc != 127)).any():
        raise ValueError('Zero precision requires canonical mean127')
    if not qc.any():
        return None
    prefix = bytes((SOURCE_VERSION if source else FUSION_VERSION, LAYOUT_TAG, config.side, config.planes))
    return HEADER.pack(tick, sender, receiver, config.channels, height, width, round_id) + prefix + qc.astype(np.uint8).tobytes() + mc.astype(np.uint8).tobytes()


def deserialize_packet(packet, config, *, source=False):
    _configuration(config)
    if not isinstance(packet, (bytes, bytearray, memoryview)) or len(packet) != HEADER.size+4+2*config.channels:
        raise ValueError('Invalid context packet length')
    meta = HEADER.unpack_from(packet); _geometry(meta[4], meta[5])
    prefix = bytes((SOURCE_VERSION if source else FUSION_VERSION, LAYOUT_TAG, config.side, config.planes))
    if (meta[3] != config.channels or bytes(packet[HEADER.size:HEADER.size+4]) != prefix
            or meta[6] not in ((0,) if source else (0, 1)) or (not source and meta[1] == meta[2])):
        raise ValueError('Context packet family/layout/round/address mismatch')
    qc = np.frombuffer(packet, np.uint8, config.channels, HEADER.size+4).astype(np.int64)
    mc = np.frombuffer(packet, np.uint8, config.channels, HEADER.size+4+config.channels).astype(np.int64)
    if (mc == 255).any() or ((qc == 0) & (mc != 127)).any() or not qc.any():
        raise ValueError('Reserved/noncanonical/empty context payload')
    return meta, qc, mc


class ContextCodec(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = ContextCodecConfig() if config is None else config
        _configuration(config); self.config = copy.deepcopy(config)
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.random.default_generator.manual_seed(config.seed)
            self.encoder = ContextEncoder(config)
            self.decoder = ContextDecoder(config)
            self.quantizer = ContextQuantizer(config.channels)

    def encode(self, private, positions, active=None):
        return self.encoder(private, positions, active)

    def decode(self, q, h, height, width):
        return self.decoder(q, h, height, width)

    def source_views(self, private, positions, active=None):
        q, h = self.encode(private, positions, active)
        _, _, wq, wh = self.quantizer.encode(q, h)
        return self.decode(q, h, *private.shape[-2:]), self.decode(wq, wh, *private.shape[-2:])

    def source_reconstruction(self, private, positions, active=None, *, quantized=True):
        return self.source_views(private, positions, active)[int(quantized)]

    def _delivery(self, qc, mc, slots, height, width, tick, round_id, *, source):
        receiver_quantizer = ContextQuantizer(self.config.channels).to(qc.device)
        parsed_qc, parsed_mc = torch.zeros_like(qc), torch.full_like(mc, 127)
        records, plain, framed, nonempty = [], 0, 0, 0
        for slot in slots:
            batch, sender = slot[:2]; receiver = sender if source else slot[2]
            index = tuple(slot)
            packet = serialize_packet(self.config, tick, sender, receiver, qc[index].detach().cpu().numpy(),
                mc[index].detach().cpu().numpy(), height, width, round_id, source=source)
            record = dict(batch=batch, sender=sender, receiver=receiver, round_id=round_id,
                          sent=packet is not None, plain_bytes=0, framed_bytes=0)
            if packet is not None:
                frame = frame_packet(packet); restored = unframe_packet(frame)
                meta, qcode, mcode = deserialize_packet(restored, self.config, source=source)
                if meta != (tick, sender, receiver, self.config.channels, height, width, round_id) or restored != packet:
                    raise AssertionError('Physical delivery metadata/lossless mismatch')
                parsed_qc[index] = torch.as_tensor(qcode.copy(), device=qc.device)
                parsed_mc[index] = torch.as_tensor(mcode.copy(), device=qc.device)
                record.update(plain_bytes=len(packet), framed_bytes=len(frame), independent_parse_and_decode=True)
                plain += len(packet); framed += len(frame); nonempty += 1
            records.append(record)
        q, h = receiver_quantizer.decode(parsed_qc, parsed_mc)
        return q, h, dict(directed_delivery_attempts=records, plain_bytes=plain, framed_bytes=framed,
            nonempty_packets=nonempty, generic_wrapper_bytes=9*nonempty,
            parsed_qcodes=parsed_qc, parsed_mean_codes=parsed_mc)

    def _fusion(self, q, h, adjacency, positions, active, height, width, comm_radius_cells, *, actual, tick=0):
        _numeric(q, h, self.config.channels)
        if q.ndim != 3:
            raise ValueError('Fusion q/h must be [B,N,K]')
        height, width = _validate_public(q.amax(-1, keepdim=True), h[..., :64], adjacency,
                                         positions, active, height, width, comm_radius_cells)
        tick = _integer(tick, 'tick', 2**32-1)
        previous_q = q.new_zeros(*adjacency.shape, self.config.channels); previous_h = torch.zeros_like(previous_q)
        trace = dict(rounds=[], rounds_count=2, channels=self.config.channels, edge_variance=EDGE_VARIANCE,
                     transport='actual independently parsed packets' if actual else 'tensor STE; no byte claim',
                     exclusive_reverse_cavity_sums=True, own_unary_quantized=False, final_belief_quantized=False)
        if actual:
            trace.update(total_plain_bytes=0, total_framed_bytes=0, total_nonempty_packets=0, total_generic_wrapper_bytes=0)
        for round_id in range(2):
            cq, ch, oq, oh = _outgoing(q, h, previous_q, previous_h, adjacency)
            qc, mc, rq, rh = self.quantizer.encode(oq, oh)
            wire = {}
            if actual:
                rq, rh, wire = self._delivery(qc, mc, adjacency.nonzero().cpu().tolist(),
                                              height, width, tick, round_id, source=False)
                rq, rh = rq.to(q.dtype), rh.to(h.dtype)
                for key in ('plain_bytes', 'framed_bytes', 'nonempty_packets', 'generic_wrapper_bytes'):
                    trace['total_'+key] += wire[key]
            previous_q, previous_h = rq, rh
            trace['rounds'].append(dict(round_id=round_id, cavity_q=cq, cavity_h=ch, prequant_q=oq,
                prequant_h=oh, qcodes=qc, mean_codes=mc, q=rq, h=rh, **wire))
        return q+previous_q.sum(1), h+previous_h.sum(1), trace

    def tensor_fusion(self, *args, **kwargs):
        return self._fusion(*args, **kwargs, actual=False)

    @torch.no_grad()
    def actual_wire_fusion(self, *args, **kwargs):
        return self._fusion(*args, **kwargs, actual=True)

    @torch.no_grad()
    def source_loopback(self, private, positions, active=None, tick=0):
        tick = _integer(tick, 'tick', 2**32-1)
        q, h = self.encode(private, positions, active)
        qc, mc, expected_q, expected_h = self.quantizer.encode(q, h)
        slots = [(b, n) for b in range(q.shape[0]) for n in range(q.shape[1])]
        rq, rh, trace = self._delivery(qc, mc, slots, *private.shape[-2:], tick, 0, source=True)
        if not torch.equal(rq, expected_q) or not torch.equal(rh, expected_h):
            raise AssertionError('Actual SOURCE differs from tensor quantization')
        for record in trace['directed_delivery_attempts']:
            record.update(diagnostic_loopback_only=True, included_in_radio_traffic=False)
        return rq, rh, trace['directed_delivery_attempts']

    def forward(self, private, positions, adjacency, active, comm_radius_cells, *, actual_wire=False, tick=0):
        q, h = self.encode(private, positions, active)
        fusion = self.actual_wire_fusion if actual_wire else self.tensor_fusion
        fq, fh, trace = fusion(q, h, adjacency, positions, active, *private.shape[-2:], comm_radius_cells, tick=tick)
        return self.decode(fq, fh, *private.shape[-2:]), trace

    def counts(self):
        enc = sum(p.numel() for p in self.encoder.parameters()); dec = sum(p.numel() for p in self.decoder.parameters())
        return dict(model_kind=MODEL_KIND, encoder=enc, decoder=dec, total=enc+dec,
            latent_mean_channels=self.config.channels, diagonal_precisions=self.config.channels,
            latent_chart=[16, 16], latent_feature_planes=self.config.planes,
            gaussian_payload_scalars=2*self.config.channels, mean_bits=8, q_bits=8,
            plain_nonempty_packet_bytes=HEADER.size+4+2*self.config.channels,
            generic_frame_overhead_bytes=9, rounds_count=2, no_private_decoder_context=True,
            zero_empty_receptive_field_information=True, encoder_receptive_field_before_pool=13,
            restrictions='identity in shared spatial chart', calibrated_precision_claim=False,
            exact_observed_union_claim=False, source_or_fusion_fidelity_claim=False,
            communication_rate_win_claim=False)
