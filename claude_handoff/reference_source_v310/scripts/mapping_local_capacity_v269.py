"""Independent local pure64 capacity branch, NOT a v266/checkpoint resume.

One shared hard8 code is the only private encoder/decoder boundary. Its learned
2048-feature expansion feeds a copied v121 ContextDecoder. The internal q/h
are deterministic decoder features, not transmitted or calibrated uncertainty.
The paired arms differ ONLY in decoder trainability, with identical weights.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from scripts.mapping_context_codec_v121 import ContextCodecConfig, ContextDecoder
from scripts.mapping_implicit_model_v192 import coordinate_grid, latent_quantize

MODEL_KIND = 'local_pure64_capacity_v269'
DEFAULT_SEED = 269101
Q_MIN, Q_MAX = 1e-4, 4.0


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)]).encode()
        h.update(len(header).to_bytes(8, 'little')); h.update(header)
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _config(config):
    if (not isinstance(config, dict) or set(config) != {'side', 'planes', 'bits', 'width', 'seed'}
            or any(type(v) is not int for v in config.values())
            or any(config[k] != v for k, v in dict(side=16, planes=4, bits=8, width=128).items())
            or not 0 <= config['seed'] < 2**63):
        raise ValueError('Typed v121 side16/planes4/bits8/width128 decoder configuration required')
    return ContextCodecConfig(**config)


class LocalCapacity64(nn.Module):
    resolution, latent_dim = 128, 64

    def __init__(self, decoder_state, decoder_config, parent_metadata, *, train_decoder=False, seed=DEFAULT_SEED):
        super().__init__()
        config = _config(decoder_config)
        if type(seed) is not int or not 0 <= seed < 2**63 or type(train_decoder) is not bool:
            raise ValueError('Integer CPU seed and boolean decoder trainability required')
        if (not isinstance(parent_metadata, dict) or parent_metadata.get('step') != 4000
                or parent_metadata.get('source_only') is not True or parent_metadata.get('joint_updates') != 0
                or not isinstance(parent_metadata.get('sha256'), str) or len(parent_metadata['sha256']) != 64):
            raise ValueError('Pinned source-only4000-step v121 parent provenance required')
        self.seed, self.train_decoder = seed, train_decoder
        self.decoder_config = dict(decoder_config)
        self.parent_metadata = dict(parent_metadata)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            factory = dict(device='cpu', dtype=torch.float32)
            layers = []; incoming = 5
            for width in (32, 64, 96, 128, 128):
                layers += [nn.Conv2d(incoming, width, 3, stride=2, padding=1, **factory), nn.SiLU()]
                incoming = width
            self.encoder = nn.Sequential(*layers)
            self.global_head = nn.Linear(2048, 64, **factory)
            self.lift = nn.Linear(64, 2048, **factory)
            self.decoder = ContextDecoder(config)
        if (not isinstance(decoder_state, dict) or set(decoder_state) != set(self.decoder.state_dict())
                or any(not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or
                       t.shape != self.decoder.state_dict()[k].shape or not bool(torch.isfinite(t).all())
                       for k, t in decoder_state.items())):
            raise ValueError('Complete finite FP32 v121 decoder weights required')
        self.decoder.load_state_dict(decoder_state, strict=True)
        for value in self.decoder.parameters(): value.requires_grad_(train_decoder)
        xy = coordinate_grid(128, 128, device='cpu').reshape(128, 128, 2).permute(2, 0, 1).contiguous()
        self.register_buffer('public_xy', xy, persistent=True)
        self.eval()

    @classmethod
    def from_parent(cls, path, *, expected_sha256, train_decoder=False, seed=DEFAULT_SEED):
        """Load a locally pinned self-describing parent; fresh optimizer/encoder."""
        if not isinstance(expected_sha256, str) or _sha(path) != expected_sha256.lower():
            raise ValueError('Parent checkpoint SHA256 mismatch')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (not isinstance(saved, dict) or saved.get('step') != 4000
                or saved.get('source_only') is not True or saved.get('joint_updates') != 0
                or not isinstance(saved.get('model_state'), dict)):
            raise ValueError('Expected source-only v121 step4000 checkpoint, not an inferred substitute')
        _config(saved.get('config'))
        decoder = {name.removeprefix('decoder.'): value for name, value in saved['model_state'].items()
                   if name.startswith('decoder.')}
        meta = dict(sha256=expected_sha256.lower(), step=4000, source_only=True, joint_updates=0,
                    family='mapping-context-codec-v121', optimizer_inherited=False)
        return cls(decoder, saved['config'], meta, train_decoder=train_decoder, seed=seed)

    @classmethod
    def from_payload(cls, payload):
        """Self-contained reconstruction: never opens the original parent path."""
        if not isinstance(payload, dict) or not isinstance(payload.get('construction'), dict):
            raise ValueError('Self-contained state_dict plus construction required')
        metadata, state = payload['construction'], payload.get('state_dict')
        if (metadata.get('model_kind') != MODEL_KIND or not isinstance(state, dict)
                or metadata.get('latent_dim') != 64 or metadata.get('native_resolution') != 128
                or metadata.get('lift_features') != 2048 or metadata.get('q_bounds') != [Q_MIN, Q_MAX]):
            raise ValueError('Wrong self-contained model semantics')
        decoder = {name.removeprefix('decoder.'): value for name, value in state.items() if name.startswith('decoder.')}
        model = cls(decoder, metadata['decoder_config'], metadata['parent'],
                    train_decoder=metadata['train_decoder'], seed=metadata['seed'])
        model.load_state_dict(state, strict=True)
        if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise ValueError('Nonfinite self-contained weights')
        if model.construction_metadata() != metadata:
            raise ValueError('Construction metadata disagrees with actual model')
        return model

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'decoder') and not self.train_decoder: self.decoder.eval()
        return self

    def _private_channels(self, private):
        if (not isinstance(private, torch.Tensor) or private.ndim != 3 or private.shape[0] < 1
                or private.shape[1:] != (128, 128) or private.is_complex() or private.dtype == torch.bool
                or private.device != self.public_xy.device or self.public_xy.dtype != torch.float32
                or not bool(((private == -1) | (private == 0) | (private == 1)).all())):
            raise ValueError('Same-device native128 ternary [B,H,W] private observations required')
        observed = torch.stack((private >= 0, private == 0, private == 1), 1).float()
        return torch.cat((observed, self.public_xy[None].expand(len(private), -1, -1, -1)), 1)

    def encode(self, private):
        features = self.encoder(self._private_channels(private))
        if features.shape[1:] != (128, 4, 4): raise RuntimeError('Wrong CNN features')
        return latent_quantize(self.global_head(features.flatten(1)).tanh())

    def lifted_features(self, code):
        if (not isinstance(code, torch.Tensor) or code.ndim != 2 or code.shape[1] != 64 or len(code) < 1
                or code.dtype != torch.float32 or code.device != self.lift.weight.device
                or not bool(torch.isfinite(code).all())):
            raise ValueError('Same-device finite FP32 [B,64] code required')
        mean_raw, q_raw = self.lift(code).chunk(2, -1)
        mean = mean_raw.tanh()
        q = Q_MIN + (Q_MAX - Q_MIN) * q_raw.sigmoid()
        return q[:, None], (q * mean)[:, None]

    def decode_continuous(self, code):
        """Original categorical decoder rewritten as equivalent proper heads.

        logp = [log(1-k), log(k)+log(1-w), log(k)+log(w)]. Therefore
        known_logit = logsumexp(logp[1:])-logp[0], wall_logit=logp[2]-logp[1].
        This path has no quantization: encoder applies the sole hard8 boundary.
        """
        q, h = self.lifted_features(code)
        logp = self.decoder(q, h, 128, 128)[:, 0]
        return dict(known_logits=(torch.logsumexp(logp[..., 1:], -1)-logp[..., 0]).flatten(1),
                    wall_logits=(logp[..., 2]-logp[..., 1]).flatten(1))

    def forward(self, private, positions=None):
        code = self.encode(private)
        return self.decode_continuous(code), code

    def trainable_parameters(self):
        values = []
        for name, parameter in self.named_parameters():
            expected = self.train_decoder or not name.startswith('decoder.')
            if parameter.requires_grad != expected: raise AssertionError('Wrong trainability: '+name)
            if expected: values.append(parameter)
            elif parameter.grad is not None: raise AssertionError('Frozen decoder gradient: '+name)
        if not self.train_decoder and any(m.training for m in self.decoder.modules()):
            raise AssertionError('Frozen decoder must remain eval')
        return values

    def decoder_hash(self): return _state_hash(self.decoder.state_dict())

    def weights_hash(self): return _state_hash(self.state_dict())

    def construction_metadata(self):
        return dict(model_kind=MODEL_KIND, seed=self.seed, native_resolution=128, latent_dim=64,
            source_code_bytes=64, codec='symmetric255-level hard8 STE, fixed[-1,1]',
            encoder_widths=[32,64,96,128,128], global_head=[2048,64], lift_features=2048,
            lift_order='mean_raw1024 then q_raw1024; both plane-major/y/x', q_bounds=[Q_MIN,Q_MAX],
            q_interpretation='Positive deterministic decoder feature, NOT communicated/calibrated uncertainty',
            train_decoder=self.train_decoder, decoder_config=dict(self.decoder_config), parent=dict(self.parent_metadata),
            parameters={name:sum(p.numel() for p in getattr(self,name).parameters())
                        for name in ('encoder','global_head','lift','decoder')},
            total_parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameter_count=sum(p.numel() for p in self.trainable_parameters()),
            private_skip=False, uses_agent_position=False, only_code_and_public_decoder_inputs=True,
            parent_encoder_or_optimizer_loaded=False, is_v266_resume=False,
            gbp_implemented=False, communication_100x_claim=False)
