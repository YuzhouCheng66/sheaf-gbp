"""Pure capacity expansion of the old pointwise spatial evidence field.

The old 9-channel pixel input, two 1x1 layers, disjoint observed-only region
pool and q=4*sigmoid(logit)*observed-fraction are retained verbatim. Only the
number of features per 16x16 anchor changes from one to four. Gaussian means
AND diagonal precisions are transmitted at 8 bits, with the same two-round
identity-restriction kernel. No private decoder skip or learned restriction.

Packet serialization and decoder are reused from v121, not its CNN encoder.
The generic packet layout explicitly carries side and planes; its version is
therefore v121/v122 rather than the old v23/v24. Both packet families contain
the same number of header bytes. Framed bytes must be measured, not inferred.

Warm start preserves the old first plane, duplicates its precision head,
leaves three distinct seeded mean heads, and zeros only their decoder input
weights. Hence the expanded model is mathematically function-preserving.
Changing convolution input width can change floating-point accumulation, so
bitwise whole-decoder equivalence is not claimed. Extra decoder weights can
learn at the first update; extra encoder heads receive gradients afterward.
"""
import copy

import torch
from torch import nn

from occupancy_codec import spatial_evidence_field_v102 as old
from occupancy_codec.global_spatial_source import _public
from scripts.mapping_context_codec_v121 import (
    ContextCodec, ContextCodecConfig, ContextDecoder, ContextQuantizer,
    _configuration,
)
from scripts.mapping_context_portable_wire_v123 import (
    install_shared_codebook_delivery, public_codebook_manifest,
)


MODEL_KIND = 'mapping-pointwise-capacity-v128'
# The transport and decoder require this exact typed configuration. It has
# identical fields for P1 and P4, with channels computed as side*side*planes.
PointwiseCapacityConfig = ContextCodecConfig


class PointwiseCapacityEncoder(nn.Module):
    def __init__(self, config):
        super().__init__(); _configuration(config)
        self.config = copy.deepcopy(config)
        factory = dict(device='cpu', dtype=torch.float32)
        self.layers = nn.Sequential(nn.Conv2d(9, 64, 1, **factory), nn.SiLU(),
                                    nn.Conv2d(64, config.width, 1, **factory), nn.SiLU())
        self.mean = nn.Conv2d(config.width, config.planes, 1, **factory)
        self.precision = nn.Conv2d(config.width, config.planes, 1, **factory)

    def forward(self, private, positions, active=None):
        b, n, height, width = old._private(private, active)
        weight = self.mean.weight
        if (private.device != weight.device or weight.dtype not in (torch.float32, torch.float64)
                or not isinstance(positions, torch.Tensor) or positions.shape != (b, n, 2)
                or positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
                or positions.device != private.device or (positions < 0).any()
                or (positions[..., 0] >= height).any() or (positions[..., 1] >= width).any()):
            raise ValueError('Inside integer public positions and FP32/64 model on input device required')
        dtype, side, sites = weight.dtype, self.config.side, self.config.side**2
        ids, populations = old.anchor_assignment(height, width, side, device=private.device)
        observed = (private >= 0).to(dtype).reshape(b*n, 1, height*width)
        marker = torch.zeros(b, n, height*width, dtype=dtype, device=private.device)
        index = positions.long()[..., 0]*width+positions.long()[..., 1]
        marker.scatter_(-1, index[..., None], 1.)
        own = torch.stack(((private == 0).to(dtype), (private == 1).to(dtype),
                           marker.reshape(b, n, height, width)), 2).flatten(0, 1)
        public = _public(b*n, height, width, private.device, dtype, height, width)
        ys, xs = torch.arange(height, device=private.device), torch.arange(width, device=private.device)
        ybin, xbin = ys*side//height, xs*side//width
        ylo, xlo = (ybin*height+side-1)//side, (xbin*width+side-1)//side
        yhi, xhi = ((ybin+1)*height+side-1)//side, ((xbin+1)*width+side-1)//side
        u = 2*(ys.to(dtype)-ylo+.5)/(yhi-ylo)-1
        v = 2*(xs.to(dtype)-xlo+.5)/(xhi-xlo)-1
        local = torch.stack((u[:, None].expand(height, width), v[None, :].expand(height, width)))[None]
        features = self.layers(torch.cat((own, public, local.expand(b*n, -1, -1, -1)), 1))
        counts = old._region_sum(observed, ids, sites)
        pooled = old._region_sum(features.flatten(2)*observed, ids, sites)/counts.clamp_min(1)
        pooled = pooled.reshape(b*n, self.config.width, side, side)
        means = self.mean(pooled).tanh()
        fraction = (counts/populations.to(dtype).clamp_min(1)[None, None]).reshape(b*n, 1, side, side)
        q = 4.*self.precision(pooled).sigmoid()*fraction
        if active is not None:
            q = q*active.reshape(b*n, 1, 1, 1)
        q = q.reshape(b, n, self.config.channels)
        return q, q*means.reshape(b, n, self.config.channels)


class PointwiseCapacityCodec(ContextCodec):
    def __init__(self, config=None):
        # Do not initialize (or retain) a context CNN. The inherited methods
        # operate exclusively on these pointwise modules and public latents.
        nn.Module.__init__(self)
        config = PointwiseCapacityConfig() if config is None else config
        _configuration(config); self.config = copy.deepcopy(config)
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.random.default_generator.manual_seed(config.seed)
            self.encoder = PointwiseCapacityEncoder(config)
            self.decoder = ContextDecoder(config)
            self.quantizer = ContextQuantizer(config.channels)

    def counts(self):
        result = super().counts()
        result.pop('zero_empty_receptive_field_information')
        result.pop('encoder_receptive_field_before_pool')
        result.update(model_kind=MODEL_KIND, input_pixel_channels=9,
            local_pixel_kernels=[1, 1], disjoint_anchor_regions=True,
            observed_pixels_only_encoder_pool=True,
            zero_empty_region_observation_information=True,
            q_rule='4*sigmoid(local logits)*observed fraction',
            coordinate_order='plane-major,y,x', unary_range=[0., 4.],
            same_old_pointwise_encoder_semantics=True,
            learned_restrictions=False, no_own_projection=True,
            no_raw_map_carrier=True, encoder_receptive_field_before_pool=1,
            decoder_structure='unchanged multiscale CNN; first input width 2*planes+4',
            public_checkpoint_codebook_required=True,
            packet_family='v121 SOURCE/v122 fusion generic side+plane layout',
            packed_data_bits=16*self.config.channels)
        return result


@torch.no_grad()
def expand_from_old(parent, seed=12801, planes=4):
    """Copy a loaded old 16x16x1 model; never mutate its weights or buffers.

    This is model-only initialization, not optimizer resume. The caller must
    retain inherited data/update budgets and identify/hash the parent file.
    P1 is also supported as a direct adapter to the same public transport.
    Returned model is CPU FP32 and has independent checkpoint-fixed receivers.
    """
    if (type(parent) is not old.SpatialEvidenceField
            or (parent.config.side, parent.config.planes, parent.config.channels,
                parent.config.bits, parent.config.width) != (16, 1, 256, 8, 128)
            or any(value.dtype != torch.float32 for value in parent.parameters())):
        raise ValueError('Original FP32 pointwise 16x16x1/C256 parent required')
    result = PointwiseCapacityCodec(PointwiseCapacityConfig(planes=planes, seed=seed))
    result.encoder.layers.load_state_dict(parent.encoder.layers.state_dict(), strict=True)
    result.encoder.mean.weight[:1].copy_(parent.encoder.mean.weight)
    result.encoder.mean.bias[:1].copy_(parent.encoder.mean.bias)
    result.encoder.precision.weight.copy_(parent.encoder.precision.weight.expand(planes, -1, -1, -1))
    result.encoder.precision.bias.copy_(parent.encoder.precision.bias.expand(planes))
    inherited = parent.decoder.state_dict()
    target = result.decoder.state_dict()
    for key, value in inherited.items():
        if key != 'lift.0.weight':
            target[key].copy_(value)
    target['lift.0.weight'].zero_()
    # Old input order: mean0, logq0, y,x,1/H,1/W. New: all means,
    # all logq, four coordinates. Extra planes remain initially ignored.
    columns = [0, planes, *range(2*planes, 2*planes+4)]
    for old_column, new_column in enumerate(columns):
        target['lift.0.weight'][:, new_column].copy_(inherited['lift.0.weight'][:, old_column])
    result.decoder.load_state_dict(target, strict=True)
    result.quantizer.load_state_dict(parent.quantizer.state_dict(), strict=True)
    install_shared_codebook_delivery(result)
    result.train(parent.training)
    result.initialization_manifest = dict(
        parent_architecture=old.MODEL_KIND, architecture=MODEL_KIND,
        old_planes=1, new_planes=planes, seed=seed,
        copied_pointwise_layers=True, copied_first_mean_plane=True,
        copied_precision_head_to_all_planes=True,
        extra_mean_initialization='distinct seeded PyTorch Conv2d initialization',
        decoder_inherited_column_mapping=columns,
        extra_decoder_input_weights_initially_zero=True,
        extra_precision_gate=False, new_modules_beyond_width_expansion=False,
        mathematical_function_preserving=True,
        bitwise_decoder_equivalence_claim=False,
        precision_head_and_first_plane_convolution_roundoff_possible=planes != 1,
        optimizer_state_copied=False,
        public_codebook=public_codebook_manifest(result),
        communication_dimension=result.config.channels)
    return result
