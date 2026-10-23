"""Untrained observation-anchored whole-map spatial evidence field, v102.

Compare 16x16x1 with 8x8x4 at 256 diagonal Gaussian coordinates. Shared
pointwise pixel features are pooled ONLY within disjoint declared anchor
regions and ONLY over measured cells. No convolution, normalization, or global
pool mixes encoder regions. Empty regions contribute exactly zero observation
information, not a positive map-wide prior. This locality is an architectural
contract, not a calibrated-likelihood or exact observed-union claim.

One shared convolutional decoder consumes means/log1p(q)/public coordinates,
never private observations or a skip. Exactly two synchronous quantized Schur
rounds reuse the v89 exclusive-cavity kernel and canonical v90 FP32 tables.
The distinct v23/v24 actual transport lives in spatial_evidence_field_wire_v102.
There is no training, checkpoint/data access, accuracy or rate-win claim here.
"""
import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from occupancy_codec.global_diagonal_accuracy_v90 import GlobalDiagonalAccuracyQuantizer
from occupancy_codec.global_diagonal_gaussian_v89 import _numeric as _numeric64, _outgoing
from occupancy_codec.global_gaussian_fusion import _validate as _validate_public
from occupancy_codec.global_spatial_source import _geometry, _public


MODEL_KIND = 'spatial-evidence-field-v102'
CHANNELS, BANK_CHANNELS, ROUNDS, EDGE_VARIANCE = 256, 64, 2, .05
SOURCE_VERSION, FUSION_VERSION, LAYOUT_TAG = 23, 24, 0x88
PLAIN_BYTES = 16+4+2*CHANNELS


@dataclass
class SpatialEvidenceFieldConfig:
    side: int = 16
    planes: int = 1
    channels: int = CHANNELS
    bits: int = 8
    width: int = 128
    seed: int = 110201


def _configuration(config):
    if (type(config) is not SpatialEvidenceFieldConfig
            or any(type(getattr(config, key)) is not int
                   for key in ('side', 'planes', 'channels', 'bits', 'width', 'seed'))
            or (config.side, config.planes) not in ((16, 1), (8, 4))
            or (config.channels, config.bits, config.width) != (256, 8, 128)
            or not 0 <= config.seed < (1 << 63)):
        raise ValueError('Typed v102 16x16x1 OR 8x8x4/C256/b8/width128 configuration required')


def _banks(value):
    return value.split(BANK_CHANNELS, dim=-1)


def _numeric(q, h, wire=False):
    if (not isinstance(q, torch.Tensor) or not isinstance(h, torch.Tensor)
            or q.ndim < 1 or q.shape[-1] != CHANNELS or q.shape != h.shape):
        raise ValueError('Matching spatial-field q/h[...,256] required')
    for qb, hb in zip(_banks(q), _banks(h)):
        _numeric64(qb, hb, wire=wire)


class SpatialEvidenceQuantizer(nn.Module):
    """Four fixed64-coordinate calls to the unchanged v90 quantizer, one field."""
    def __init__(self):
        super().__init__()
        self.base = GlobalDiagonalAccuracyQuantizer()

    def encode(self, q, h):
        _numeric(q, h, wire=True)
        rows = [self.base.encode(qb, hb) for qb, hb in zip(_banks(q), _banks(h))]
        return tuple(torch.cat([row[index] for row in rows], -1) for index in range(4))

    def decode(self, qcodes, mean_codes):
        if (not isinstance(qcodes, torch.Tensor) or not isinstance(mean_codes, torch.Tensor)
                or qcodes.ndim < 1 or qcodes.shape[-1] != CHANNELS
                or mean_codes.shape != qcodes.shape):
            raise ValueError('Matching spatial-field code tensors[...,256] required')
        rows = [self.base.decode(qb, mb) for qb, mb in zip(_banks(qcodes), _banks(mean_codes))]
        return tuple(torch.cat([row[index] for row in rows], -1) for index in (0, 1))

    def positive_q_codes(self, q):
        if not isinstance(q, torch.Tensor) or q.ndim < 1 or q.shape[-1] != CHANNELS:
            raise ValueError('Spatial-field precision tensor[...,256] required')
        return torch.cat([self.base.positive_q_codes(bank) for bank in _banks(q)], -1)


def anchor_assignment(height, width, side, *, device=None):
    """Pixel->one anchor: floor(y*side/H), floor(x*side/W), never replication.

    Returns flattened anchor IDs[H*W] and populations[side*side]. Zero-area
    anchors are intentional when side exceeds an input dimension.
    """
    height, width = _geometry(height, width)
    if type(side) is not int or side not in (8, 16):
        raise ValueError('Anchor side must be integer8 or16')
    yy = torch.arange(height, device=device)*side//height
    xx = torch.arange(width, device=device)*side//width
    ids = (yy[:, None]*side+xx[None, :]).flatten()
    return ids, torch.bincount(ids, minlength=side*side)


def _private(private, active=None):
    if (not isinstance(private, torch.Tensor) or private.ndim != 4
            or min(private.shape[:2]) < 1 or private.dtype == torch.bool or private.is_complex()
            or not torch.isfinite(private).all()
            or not ((private == -1) | (private == 0) | (private == 1)).all()):
        raise ValueError('Own finite ternary private[B,N,H,W], unknown-1/free0/wall1 required')
    b, n, height, width = private.shape
    _geometry(height, width)
    if active is not None and (not isinstance(active, torch.Tensor) or active.shape != (b, n)
            or active.dtype != torch.bool or active.device != private.device):
        raise ValueError('Public activity bool[B,N] on matching device required')
    return b, n, height, width


def _region_sum(values, ids, sites):
    # No input-region axis, padding, adaptive pooling overlap, or global summary.
    result = values.new_zeros(values.shape[0], values.shape[1], sites)
    return result.scatter_add(2, ids[None, None].expand_as(values), values)


class SpatialEvidenceEncoder(nn.Module):
    def __init__(self, config):
        super().__init__(); _configuration(config)
        self.config = copy.deepcopy(config)
        # free,wall,marker; global y,x,1/H,1/W; within-anchor u,v.
        factory = dict(device='cpu', dtype=torch.float32)
        self.layers = nn.Sequential(nn.Conv2d(9, 64, 1, **factory), nn.SiLU(),
                                    nn.Conv2d(64, config.width, 1, **factory), nn.SiLU())
        self.mean = nn.Conv2d(config.width, config.planes, 1, **factory)
        self.precision = nn.Conv2d(config.width, config.planes, 1, **factory)

    def forward(self, private, positions, active=None):
        b, n, height, width = _private(private, active)
        weight = self.mean.weight
        if (private.device != weight.device or weight.dtype not in (torch.float32, torch.float64)
                or not isinstance(positions, torch.Tensor) or positions.shape != (b, n, 2)
                or positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
                or positions.device != private.device or (positions < 0).any()
                or (positions[..., 0] >= height).any() or (positions[..., 1] >= width).any()):
            raise ValueError('Inside integer public positions and FP32/64 model on input device required')
        dtype, side, sites = weight.dtype, self.config.side, self.config.side**2
        ids, populations = anchor_assignment(height, width, side, device=private.device)
        observed = (private >= 0).to(dtype).reshape(b*n, 1, height*width)
        marker = torch.zeros(b, n, height*width, dtype=dtype, device=private.device)
        index = positions.long()[..., 0]*width+positions.long()[..., 1]
        marker.scatter_(-1, index[..., None], 1.)
        own = torch.stack(((private == 0).to(dtype), (private == 1).to(dtype),
                           marker.reshape(b, n, height, width)), 2).flatten(0, 1)
        public = _public(b*n, height, width, private.device, dtype, height, width)
        # Anchor interval boundaries are the exact discrete assignment above.
        ys, xs = torch.arange(height, device=private.device), torch.arange(width, device=private.device)
        ybin, xbin = ys*side//height, xs*side//width
        ylo, xlo = (ybin*height+side-1)//side, (xbin*width+side-1)//side
        yhi, xhi = ((ybin+1)*height+side-1)//side, ((xbin+1)*width+side-1)//side
        u = 2*(ys.to(dtype)-ylo+.5)/(yhi-ylo)-1
        v = 2*(xs.to(dtype)-xlo+.5)/(xhi-xlo)-1
        local = torch.stack((u[:, None].expand(height, width), v[None, :].expand(height, width)))[None]
        features = self.layers(torch.cat((own, public, local.expand(b*n, -1, -1, -1)), 1))
        counts = _region_sum(observed, ids, sites)
        pooled = _region_sum(features.flatten(2)*observed, ids, sites)/counts.clamp_min(1)
        pooled = pooled.reshape(b*n, self.config.width, side, side)
        means = self.mean(pooled).tanh()
        fraction = (counts/populations.to(dtype).clamp_min(1)[None, None]).reshape(b*n, 1, side, side)
        q = 4.*self.precision(pooled).sigmoid()*fraction
        if active is not None:
            q = q*active.reshape(b*n, 1, 1, 1)
        q = q.reshape(b, n, CHANNELS)
        return q, q*means.reshape(b, n, CHANNELS)


class SpatialEvidenceDecoder(nn.Module):
    def __init__(self, config):
        super().__init__(); _configuration(config)
        self.config = copy.deepcopy(config)
        factory = dict(device='cpu', dtype=torch.float32)
        self.lift = nn.Sequential(nn.Conv2d(2*config.planes+4, config.width, 3, padding=1, **factory), nn.SiLU(),
                                  nn.Conv2d(config.width, config.width, 3, padding=1, **factory), nn.SiLU())
        incoming, blocks = config.width, []
        for outgoing in (96, 64, 48, 32):
            blocks.append(nn.Sequential(nn.Conv2d(incoming+4, outgoing, 3, padding=1, **factory), nn.SiLU(),
                                        nn.Conv2d(outgoing, outgoing, 3, padding=1, **factory), nn.SiLU()))
            incoming = outgoing
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv2d(32, 2, 1, **factory)

    def forward(self, q, h, height, width):
        height, width = _geometry(height, width); _numeric(q, h)
        if q.ndim != 3 or min(q.shape[:2]) < 1:
            raise ValueError('Spatial-field q/h[B,N,256] required')
        if q.device != self.output.weight.device or q.dtype != self.output.weight.dtype:
            raise ValueError('Spatial-field decoder and q/h must share dtype/device')
        b, n = q.shape[:2]; batch, side = b*n, self.config.side
        mean = (h/torch.where(q > 0, q, torch.ones_like(q))).reshape(batch, self.config.planes, side, side)
        precision = q.log1p().reshape_as(mean)
        hidden = self.lift(torch.cat((mean, precision,
            _public(batch, side, side, q.device, q.dtype, height, width)), 1))
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
        raise ValueError('Own spatial-field q/h[B,N,256] required')
    # Only public admission is reused; qmax never enters inference or decoding.
    return _validate_public(q.amax(-1, keepdim=True), h[..., :64], adjacency, positions,
                            active, height, width, comm_radius_cells)


def _trace(config):
    return dict(rounds=[], slots='[B,sender,receiver]', rounds_count=ROUNDS,
        channels=CHANNELS, diagonal_precisions=CHANNELS, latent_chart=[config.side]*2,
        latent_feature_planes=config.planes, coordinate_order='plane-major,y,x',
        one_code_for_entire_map=True, edge_variance=EDGE_VARIANCE,
        restrictions='fixed identity common spatial chart', exclusive_reverse_cavity_sums=True,
        own_unary_quantized=False, final_belief_quantized=False, mean_bits=8, q_bits=8,
        packed_data_bits=4096, source_wire_version=SOURCE_VERSION, wire_version=FUSION_VERSION,
        layout_tag=LAYOUT_TAG, plain_nonempty_packet_bytes=PLAIN_BYTES,
        calibrated_occupancy_variance_claim=False, exact_posterior_claim=False,
        exact_observed_union_claim=False, communication_rate_win_claim=False)


def _fusion(q, h, adjacency, positions, active, height, width, comm_radius_cells,
            config, delivery=None):
    _configuration(config)
    height, width = _validate(q, h, adjacency, positions, active, height, width, comm_radius_cells)
    previous_q = q.new_zeros(*adjacency.shape, CHANNELS); previous_h = torch.zeros_like(previous_q)
    quantizer = SpatialEvidenceQuantizer().to(q.device)
    trace = _trace(config)
    trace['transport'] = 'tensor STE surrogate; no byte accounting' if delivery is None else 'actual v24/frame/independent parse and decode'
    if delivery is not None:
        trace.update(total_plain_bytes=0, total_framed_bytes=0, total_nonempty_packets=0,
                     total_generic_wrapper_bytes=0, independent_packet_decode=True)
    for round_id in range(ROUNDS):
        cq, ch, oq, oh = _outgoing(q, h, previous_q, previous_h, adjacency)
        qc, mc, rq, rh = quantizer.encode(oq, oh)
        wire = {}
        if delivery is not None:
            rq, rh, wire = delivery(qc, mc, round_id)
            for key in ('plain_bytes', 'framed_bytes', 'nonempty_packets', 'generic_wrapper_bytes'):
                trace['total_'+key] += wire[key]
        previous_q, previous_h = rq, rh
        trace['rounds'].append(dict(round_id=round_id, cavity_q=cq, cavity_h=ch,
            prequant_q=oq, prequant_h=oh, qcodes=qc, mean_codes=mc, q=rq, h=rh, **wire))
    return q+previous_q.sum(1), h+previous_h.sum(1), trace


def tensor_fusion(q, h, adjacency, positions, active, height, width, comm_radius_cells,
                  config=SpatialEvidenceFieldConfig()):
    return _fusion(q, h, adjacency, positions, active, height, width, comm_radius_cells, config)


class SpatialEvidenceField(nn.Module):
    def __init__(self, config=SpatialEvidenceFieldConfig()):
        super().__init__(); _configuration(config)
        self.config = copy.deepcopy(config)
        # Deterministic CPU-only initialization without consuming the caller's
        # RNG or inheriting ambient device/dtype. No CUDA generator is queried.
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.random.default_generator.manual_seed(config.seed)
            self.encoder = SpatialEvidenceEncoder(config)
            self.decoder = SpatialEvidenceDecoder(config)
            self.quantizer = SpatialEvidenceQuantizer()

    def source_views(self, private, positions, active=None):
        """Bare and quantized-STE SOURCE logprobs; actual parse via source_loopback."""
        q, h = self.encoder(private, positions, active)
        _, _, wq, wh = self.quantizer.encode(q, h)
        height, width = private.shape[-2:]
        return self.decoder(q, h, height, width), self.decoder(wq, wh, height, width)

    def tensor_fusion(self, *args, **kwargs):
        return tensor_fusion(*args, **kwargs, config=self.config)

    def actual_wire_fusion(self, *args, **kwargs):
        from occupancy_codec.spatial_evidence_field_wire_v102 import actual_wire_fusion
        return actual_wire_fusion(*args, **kwargs, config=self.config)

    def source_loopback(self, *args, **kwargs):
        from occupancy_codec.spatial_evidence_field_wire_v102 import source_loopback
        return source_loopback(self, *args, **kwargs)

    def counts(self):
        encoder, decoder = (sum(p.numel() for p in module.parameters()) for module in (self.encoder, self.decoder))
        return dict(model_kind=MODEL_KIND, encoder=encoder, decoder=decoder, total=encoder+decoder,
            latent_mean_channels=CHANNELS, diagonal_precisions=CHANNELS, latent_chart=[self.config.side]*2,
            latent_feature_planes=self.config.planes, coordinate_order='plane-major,y,x',
            one_code_for_entire_map=True, local_pixel_kernels=[1, 1], disjoint_anchor_regions=True,
            observed_pixels_only_encoder_pool=True, zero_empty_region_observation_information=True,
            q_rule='4*sigmoid(local logits)*observed fraction', unary_range=[0., 4.],
            mean_bits=8, q_bits=8, packed_data_bits=4096, gaussian_payload_scalars=512,
            source_wire_version=SOURCE_VERSION, fusion_wire_version=FUSION_VERSION,
            layout_tag=LAYOUT_TAG, explicit_wire_side_and_planes=True,
            plain_nonempty_packet_bytes=PLAIN_BYTES, generic_frame_overhead_bytes=9,
            uncompressed_framed_nonempty_bytes=PLAIN_BYTES+9, actual_wire_implemented=True,
            restrictions='fixed identity common spatial chart', rounds_count=ROUNDS,
            edge_variance=EDGE_VARIANCE, exclusive_reverse_cavity_sums=True,
            calibrated_occupancy_variance_claim=False, exact_posterior_claim=False,
            exact_observed_union_claim=False, no_private_decoder_context=True,
            no_own_projection=True, no_raw_map_carrier=True, communication_rate_win_claim=False,
            source_or_fusion_fidelity_claim=False, spatial_vs_channel_not_exact_parameter_matched=True)
