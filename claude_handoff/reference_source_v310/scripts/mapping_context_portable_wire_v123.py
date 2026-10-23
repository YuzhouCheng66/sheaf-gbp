"""Checkpoint-public fixed codebooks for v121 context packet receivers.

Call ``install_shared_codebook_delivery(model)`` AFTER loading the checkpoint.
The model's existing non-learned quantizer tables define the public protocol;
receivers get independent exact copies, not a fresh platform-specific linspace.
No private map, latent state, parameter, gradient or training data is copied.
Only this model instance's delivery method is bound; state_dict is unchanged.

Packets, codes, quantizer bins, encoder/decoder and both GBP updates are intact.
2044 existing checkpoint bytes (256+255 FP32 entries) are public deployment
constants, not per-message payload. They must be provisioned identically to all
agents. The helper does not claim this setup or the model deployment is free.
"""
from __future__ import annotations

import copy
import hashlib
import types

import numpy as np
import torch


_ATTRIBUTE = '_portable_public_codebook_delivery_v123'
_TABLES = ('base.q_lut', 'base.mean_lut')


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().astype('<f4', copy=False).tobytes()


def _tables(quantizer):
    if list(quantizer.parameters()):
        raise ValueError('Public quantizer must have no learned parameters')
    buffers = dict(quantizer.named_buffers())
    if set(buffers) != set(_TABLES):
        raise ValueError('Only registered fixed q/mean codebooks are permitted')
    q, mean = (buffers[name] for name in _TABLES)
    if (q.shape != (256,) or mean.shape != (255,)
            or any(t.dtype != torch.float32 or t.requires_grad or not torch.isfinite(t).all()
                   for t in (q, mean))):
        raise ValueError('Finite non-gradient FP32 public codebooks required')
    if (q[0] != 0 or q[1] != np.float32(.05) or q[-1] != 20
            or mean[0] != -1 or mean[127] != 0 or mean[-1] != 1
            or not (q[1:] > q[:-1]).all() or not (mean[1:] > mean[:-1]).all()):
        raise ValueError('Codebook endpoints/order differ from the fixed v90 protocol')
    return buffers


class _PublicDelivery:
    """Not an nn.Module: no extra model buffer/parameter/state_dict entries."""
    def __init__(self, model, original):
        self.original = original
        self.tables = {name: value.detach().cpu().clone() for name, value in _tables(model.quantizer).items()}
        # Independent receiver initialized by copying only the public fixed
        # quantizer; never copy the encoder, decoder, or agent state.
        self.prototype = copy.deepcopy(model.quantizer).cpu().eval()
        self.receivers = {}
        self.functions = {name: original.__func__.__globals__[name] for name in (
            'serialize_packet', 'deserialize_packet', 'frame_packet', 'unframe_packet')}
        payload = b''.join(_bytes(self.tables[name]) for name in _TABLES)
        self.manifest = dict(version='checkpoint-public-codebook-delivery-v123',
            codebook_sha256=hashlib.sha256(payload).hexdigest(),
            table_sha256={name: hashlib.sha256(_bytes(value)).hexdigest() for name, value in self.tables.items()},
            public_table_bytes=len(payload), public_scalar_count=511,
            codebook_origin='existing non-learned checkpoint buffers shared by all agents',
            provisioning='identical constants delivered with each agent model; 2044 bytes per agent before deduplication',
            additional_per_message_bytes=0, packet_format_changed=False,
            encoder_decoder_or_gbp_changed=False, private_agent_data_in_codebook=False,
            learned_codebook_claim=False, model_state_dict_changed=False)

    def check(self, quantizer):
        actual = _tables(quantizer)
        for name, expected in self.tables.items():
            if _bytes(actual[name]) != _bytes(expected):
                raise AssertionError('Public codebook differs from registered checkpoint copy: '+name)

    def receiver(self, model, device):
        self.check(model.quantizer)
        key = str(device)
        if key not in self.receivers:
            self.receivers[key] = copy.deepcopy(self.prototype).to(device).eval()
        receiver = self.receivers[key]
        self.check(receiver)
        return receiver


def _shared_delivery(self, qc, mc, slots, height, width, tick, round_id, *, source):
    """Original v121 delivery; only receiver codebook construction differs."""
    public = getattr(self, _ATTRIBUTE)
    receiver_quantizer = public.receiver(self, qc.device)
    serialize, deserialize, frame_packet, unframe_packet = (public.functions[name] for name in (
        'serialize_packet', 'deserialize_packet', 'frame_packet', 'unframe_packet'))
    parsed_qc, parsed_mc = torch.zeros_like(qc), torch.full_like(mc, 127)
    records, plain, framed, nonempty = [], 0, 0, 0
    for slot in slots:
        batch, sender = slot[:2]; receiver = sender if source else slot[2]
        index = tuple(slot)
        packet = serialize(self.config, tick, sender, receiver, qc[index].detach().cpu().numpy(),
            mc[index].detach().cpu().numpy(), height, width, round_id, source=source)
        record = dict(batch=batch, sender=sender, receiver=receiver, round_id=round_id,
                      sent=packet is not None, plain_bytes=0, framed_bytes=0)
        if packet is not None:
            frame = frame_packet(packet); restored = unframe_packet(frame)
            meta, qcode, mcode = deserialize(restored, self.config, source=source)
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


def install_shared_codebook_delivery(model):
    """Bind exact public-checkpoint-codebook delivery; return the same model.

    Install after state_dict loading. Later codebook changes fail closed. Each
    separately loaded agent must install from the SAME checkpoint, checked via
    ``public_codebook_manifest(model)['codebook_sha256']`` at deployment.
    No per-packet hash/negotiation is added to the existing packet format.
    """
    if hasattr(model, _ATTRIBUTE):
        getattr(model, _ATTRIBUTE).check(model.quantizer)
        if getattr(model._delivery, '__func__', None) is not _shared_delivery:
            raise RuntimeError('Portable delivery binding was replaced')
        return model
    if (not hasattr(model, 'config') or getattr(model.config, 'channels', None) not in (256, 1024)
            or not hasattr(model, '_delivery') or not isinstance(model._delivery, types.MethodType)
            or not hasattr(model, 'quantizer')):
        raise TypeError('Loaded v121-compatible context codec required')
    original = model._delivery
    if not all(name in original.__func__.__globals__ for name in (
            'serialize_packet', 'deserialize_packet', 'frame_packet', 'unframe_packet')):
        raise TypeError('Expected original v121-compatible serializer/parser delivery')
    public = _PublicDelivery(model, original)
    setattr(model, _ATTRIBUTE, public)
    model._delivery = types.MethodType(_shared_delivery, model)
    return model


def public_codebook_manifest(model):
    """Read-only metadata; no table/latent/private input is returned."""
    if not hasattr(model, _ATTRIBUTE):
        raise ValueError('Install checkpoint-public delivery first')
    public = getattr(model, _ATTRIBUTE)
    public.check(model.quantizer)
    return copy.deepcopy(public.manifest)
