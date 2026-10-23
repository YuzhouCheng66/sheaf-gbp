"""Read already-earned marginal variance, without changing frozen GBP/radio.

The two readouts have identical parameters and initial functions. Both retain
the old hard64 mean, and a zero-initialized extra64 lift receives either a
normalized duplicate mean or normalized log marginal variance. The latter is
128 local statistics, NOT a pure64 autoencoder. Neither actual covariance nor
the decoder's deterministic1024 q-features are calibrated map provenance.
"""
import copy
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from scripts.mapping_context_codec_v121 import ContextDecoder
from scripts.mapping_implicit_model_v192 import latent_quantize
from scripts.mapping_local_capacity_v269 import _config, _state_hash, Q_MIN, Q_MAX
from scripts.mapping_recognition_hops_v277 import RecognitionHops64, _edges
from scripts import mapping_recurrent_gbp_v213 as recurrent
from scripts import mapping_observed_wire_v246 as wire
from scripts import mapping_session_bound_wire_v267 as transport


MODEL_KIND = 'frozen_belief_readout64_v282'
ARMS = ('duplicate_mean', 'posterior_diagonal')
STD_FLOOR = 1e-6


def _hard_mean(code):
    """Accept exactly the documented direct-divide OR reciprocal hard alphabet.

    FP64 division then FP32 cast recovers correctly rounded direct division on
    either device; the separate FP32 reciprocal product permits CUDA's known
    alternate expression. No tolerance, requantization, or mutation is used.
    """
    if (not isinstance(code, torch.Tensor) or code.ndim < 2 or code.shape[-1] != 64
            or min(code.shape[:-1]) < 1 or code.dtype != torch.float32
            or not bool(torch.isfinite(code).all()) or bool((code.abs() > 1).any())):
        raise ValueError('Finite FP32 hard8 posterior means required')
    integer = (code*127.).round()
    direct = (integer.double()/127.).float()
    reciprocal = integer*torch.tensor(1/127, dtype=torch.float32, device=code.device)
    if not bool(((code == direct) | (code == reciprocal)).all()):
        raise ValueError('Mean is not an exact documented hard8 integer-code decoding')


def belief_features(code, covariance):
    """Hardmean[B,N,64], C[B,N,64,64] -> detached FP32[B,N,128]."""
    _hard_mean(code)
    if (code.ndim != 3 or not isinstance(covariance, torch.Tensor)
            or covariance.shape != code.shape+(64,)
            or covariance.dtype != torch.float64 or covariance.device != code.device
            or not bool(torch.isfinite(covariance).all())):
        raise ValueError('Same-device finite FP64 local covariance of the actual belief required')
    diagonal = covariance.diagonal(dim1=-2, dim2=-1)
    if not bool((diagonal > 0).all()):
        raise ValueError('Positive posterior marginal variances required; no floor')
    result = torch.cat((code.detach(), diagonal.detach().log().float()), -1)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Nonfinite local belief features')
    return result


@torch.no_grad()
def belief_rollout(source, private, edges, rounds=2, serialize=False, edge_contexts=None):
    """Original synchronous K0/1/2 inference, without ANY old map decoding.

    Caller freezes source.requires_grad_(False).eval(); this helper neither
    mutates flags nor registers the source in a trainable readout. No source
    posterior/decoded map is re-encoded or added as new evidence.
    """
    if (type(source) is not RecognitionHops64 or any(p.requires_grad for p in source.parameters())
            or any(m.training for m in source.modules())):
        raise ValueError('Frozen eval original diagonal RecognitionHops64 source required')
    if type(rounds) is not int or rounds not in (0, 1, 2) or type(serialize) is not bool:
        raise ValueError('Only registered0/1/2 sweeps and explicit serialization allowed')
    factors = source.factors(private)
    batch, nodes = factors['h'].shape[:2]
    graph = _edges(edges, nodes)
    state = recurrent.initialize(factors['q'], factors['h'], graph)
    if serialize:
        if batch != 1 or edge_contexts is None:
            raise ValueError('Actual packets require B1 and externally established link contexts')
        state = transport.bind_context(state, edge_contexts)
    elif edge_contexts is not None:
        raise ValueError('Contexts are only used with actual serialization')
    codes, means, covariances, states, features = [], [], [], [], []
    trace, packets = [], []
    for hop in range(rounds+1):
        if hop:
            result = (transport.advance(state, 1, damping=.5, trace=True) if serialize
                      else wire.advance(state, 1, damping=.5, quantized=True, trace=True))
            state = result['state']
            trace.extend(result['trace'])
            if serialize:
                packets.extend(result['packets'])
            mean, covariance = result['mean'], result['covariance']
        else:
            mean, covariance = recurrent.belief(state)
        if state['q'] is not factors['q'] or state['h'] is not factors['h']:
            raise AssertionError('Original unary ownership changed')
        code = latent_quantize(mean.float())
        codes.append(code); means.append(mean); covariances.append(covariance); states.append(state)
        features.append(belief_features(code, covariance))
    return dict(source_factors=factors, codes=codes, means=means, covariances=covariances,
        states=states, state=state, trace=trace, packets=packets, features=features,
        initial=codes[0], final_code=codes[-1], mean=means[-1], covariance=covariances[-1],
        application_byte_budget_per_world=172*len(state['directed'])*rounds,
        inner_application_byte_budget_per_world=148*len(state['directed'])*rounds,
        actual_serialized_application_bytes=sum(map(len, packets)) if serialize else None,
        old_decoder_called=False, source_factors_fixed=True,
        context_establishment_implemented=False, context_establishment_bytes=None,
        physical_network_headers_measured=False, full_process_communication_measured=False,
        communication_100x_claim=False)


def fit_train_normalization(features, split='TRAIN'):
    """Fit population mean/std on actual TRAIN vectors, never labels/VAL/TEST."""
    if split != 'TRAIN':
        raise ValueError('Normalization is TRAIN-only')
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
            or features.shape[1] != 128 or features.shape[0] < 1
            or features.dtype != torch.float32 or not bool(torch.isfinite(features).all())):
        raise ValueError('Finite FP32 actual TRAIN features[rows,128] required')
    values = features.detach().cpu().contiguous()
    _hard_mean(values[:, :64])
    data = values.double()
    mean = data.mean(0).float()
    std = data.std(0, correction=0).clamp_min(STD_FLOOR).float()
    return dict(split='TRAIN', rows=len(values), dimension=128, ddof=0, std_floor=STD_FLOOR,
        mean=mean.tolist(), std=std.tolist(),
        input_sha256=hashlib.sha256(values.numpy().tobytes()).hexdigest(),
        feature_order='hard8_mean64_then_log_actual_marginal_variance64',
        fit_scope='All supplied legitimate TRAIN nodes/hops; never padded nodes or labels')


def _normalization(normalization):
    expected = {'split','rows','dimension','ddof','std_floor','mean','std','input_sha256','feature_order','fit_scope'}
    if (not isinstance(normalization, dict) or set(normalization) != expected
            or normalization['split'] != 'TRAIN' or type(normalization['rows']) is not int
            or normalization['rows'] < 1 or type(normalization['dimension']) is not int or normalization['dimension'] != 128
            or type(normalization['ddof']) is not int or normalization['ddof'] != 0
            or type(normalization['std_floor']) is not float or normalization['std_floor'] != STD_FLOOR
            or normalization['feature_order'] != 'hard8_mean64_then_log_actual_marginal_variance64'
            or normalization['fit_scope'] != 'All supplied legitimate TRAIN nodes/hops; never padded nodes or labels'
            or not isinstance(normalization['input_sha256'], str) or len(normalization['input_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in normalization['input_sha256'])
            or any(not isinstance(normalization[k], list) or len(normalization[k]) != 128 for k in ('mean','std'))):
        raise ValueError('Exact self-describing TRAIN normalization required')
    if any(type(v) not in (int, float) for k in ('mean','std') for v in normalization[k]):
        raise ValueError('Finite numeric normalization, not booleans, required')
    mean = torch.tensor(normalization['mean'], dtype=torch.float32)
    std = torch.tensor(normalization['std'], dtype=torch.float32)
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std >= STD_FLOOR).all()):
        raise ValueError('Invalid normalization statistics')
    if mean.tolist() != normalization['mean'] or std.tolist() != normalization['std']:
        raise ValueError('Normalization must record its actual FP32 values exactly')
    return mean, std


class BeliefReadout64(nn.Module):
    """Only old D and one expanded linear lift; frozen source is NOT a child."""
    resolution, latent_dim = 128, 64

    def _build(self, config, normalization, arm, source_construction, source_state_sha256):
        nn.Module.__init__(self)
        cfg = _config(config)
        if arm not in ARMS or not isinstance(source_construction, dict) or source_construction.get('model_kind') != 'recognition_hops64_v277':
            raise ValueError('Registered arm and original diagonal-source provenance required')
        if (not isinstance(source_state_sha256, str) or len(source_state_sha256) != 64
                or any(c not in '0123456789abcdef' for c in source_state_sha256)):
            raise ValueError('Source state SHA256 required')
        mean, std = _normalization(normalization)
        self.arm = arm
        self.decoder_config = copy.deepcopy(config)
        self.normalization = copy.deepcopy(normalization)
        self.source_construction = copy.deepcopy(source_construction)
        self.source_state_sha256 = source_state_sha256
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(282101)
            self.lift = nn.Linear(128, 2048, device='cpu', dtype=torch.float32)
            self.decoder = ContextDecoder(cfg)
        self.register_buffer('normalization_mean', mean)
        self.register_buffer('normalization_std', std)
        self.eval()

    @classmethod
    def from_parent_payload(cls, payload, normalization, arm):
        source = RecognitionHops64.from_payload(payload)
        result = cls.__new__(cls)
        result._build(source.capacity.decoder_config, normalization, arm,
                      source.construction_metadata(), source.weights_hash())
        result.decoder.load_state_dict(source.capacity.decoder.state_dict(), strict=True)
        with torch.no_grad():
            result.lift.weight[:, :64].copy_(source.capacity.lift.weight)
            result.lift.weight[:, 64:].zero_()
            result.lift.bias.copy_(source.capacity.lift.bias)
        return result

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get('construction'), dict) or not isinstance(payload.get('state_dict'), dict):
            raise ValueError('Self-contained readout payload required')
        meta = payload['construction']
        if meta.get('model_kind') != MODEL_KIND:
            raise ValueError('Wrong readout family')
        model = cls.__new__(cls)
        model._build(meta['decoder_config'], meta['normalization'], meta['arm'],
                     meta['source_construction'], meta['source_state_sha256'])
        model.load_state_dict(payload['state_dict'], strict=True)
        if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise ValueError('Nonfinite self-contained readout state')
        mean, std = _normalization(model.normalization)
        if not torch.equal(model.normalization_mean, mean) or not torch.equal(model.normalization_std, std):
            raise ValueError('Stored normalization buffers disagree with provenance')
        if model.construction_metadata() != meta:
            raise ValueError('Readout metadata disagrees with actual construction')
        return model

    def forward(self, features):
        if (not isinstance(features, torch.Tensor) or features.ndim not in (2, 3)
                or features.shape[-1] != 128 or min(features.shape[:-1]) < 1
                or features.dtype != torch.float32 or features.device != self.lift.weight.device
                or not bool(torch.isfinite(features).all())):
            raise ValueError('Finite detached-belief FP32 features[rows,128] or[B,N,128] required')
        values = features.detach()
        _hard_mean(values[..., :64])
        shape = values.shape[:-1]
        flat = values.reshape(-1, 128)
        code = flat[:, :64].contiguous()
        selection = slice(0, 64) if self.arm == 'duplicate_mean' else slice(64, 128)
        extra = (flat[:, selection]-self.normalization_mean[selection])/self.normalization_std[selection]
        with torch.autocast(device_type=values.device.type, enabled=False):
            # Same mathematical Linear128, but retain the old contiguous64 GEMM
            # path exactly instead of changing its floating summation schedule.
            lifted = (F.linear(code, self.lift.weight[:, :64].contiguous(), self.lift.bias)
                      + F.linear(extra, self.lift.weight[:, 64:].contiguous(), None))
            mean_raw, q_raw = lifted.chunk(2, -1)
            q = Q_MIN+(Q_MAX-Q_MIN)*q_raw.sigmoid()
            h = q*mean_raw.tanh()
            logp = self.decoder(q[:, None], h[:, None], 128, 128)[:, 0]
            known = (torch.logsumexp(logp[..., 1:], -1)-logp[..., 0]).flatten(1)
            wall = (logp[..., 2]-logp[..., 1]).flatten(1)
        return dict(known_logits=known.reshape(*shape, 16384), wall_logits=wall.reshape(*shape, 16384))

    decode = forward

    def trainable_parameters(self):
        for name, value in self.named_parameters():
            if not value.requires_grad:
                raise AssertionError('Only lift/D exist and all must train: '+name)
        return list(self.parameters())

    def weights_hash(self):
        return _state_hash(self.state_dict())

    def construction_metadata(self):
        counts = {name: sum(p.numel() for p in getattr(self, name).parameters()) for name in ('lift','decoder')}
        return dict(model_kind=MODEL_KIND, arm=self.arm, native_resolution=128,
            latent_dim=64, decoder_input_scalars=128, decoder_config=copy.deepcopy(self.decoder_config),
            normalization=copy.deepcopy(self.normalization), source_construction=copy.deepcopy(self.source_construction),
            source_state_sha256=self.source_state_sha256, source_checkpoint_pinned_by_caller=True,
            first64='Unmodified original hard8 mean; exact documented alphabet checked',
            extra64='Normalized duplicate mean' if self.arm=='duplicate_mean' else 'Normalized log actual posterior marginal variance',
            lift=[128,2048], lift_implementation='old64 F.linear plus extra64 F.linear; one expanded weight tensor',
            initial_lift='[oldW,zero64], unchanged old bias and all decoder weights',
            original_decoder_feature_q='Learned deterministic1024 q/h lift, not actual posterior precision',
            full_covariance_decoder_input=False, private_decoder_skip=False,
            source_registered_as_submodule=False, source_parameters_trainable=False,
            feature_gradients_stopped=True, pure64_autoencoder_claim=False,
            extra_radio_packets=0, session_packet_bytes=172,
            covariance_is_provenance=False, uncertainty_calibrated=False,
            parameters=counts, total_parameters=sum(counts.values()),
            optimizer_inherited=False, communication_100x_claim=False)
