"""Registered low-bit wire for chosen isotropic patch Gaussian factors.

The model chooses one scalar precision and C natural coordinates per public
32x32 atlas patch. This is exact isotropy for that model, not an approximation
of arbitrary dense/diagonal covariance. Gamma=.05 and coisometric identity
restrictions imply outgoing q<20; bounded source means and convex cavity means
give |h/q|<=1. Quantization is declared lossy and is applied during training.

Versions 11:(C32,6bit) and 12:(C48,4bit) both transmit 24 mean bytes plus one
precision byte and a uint16 patch address. The 19-byte application header,
version and count are explicit. Optional generic framing adds its actual bytes
separately. No factor mass or occupancy code enters this wire.
"""
import struct

import numpy as np
import torch
from torch import nn

from occupancy_codec.gaussian import HEADER, PRECISION_LUT
from occupancy_codec.pair_vocabulary import pack_indices, unpack_indices


PATCH_SIDE = 32
VERSIONS = {(32, 6): 11, (48, 4): 12}
COUNT = struct.Struct('<H')
ENTRY = struct.Struct('<HB')  # public atlas patch id, scalar precision code
BOUND_TOLERANCE = 1e-5


def _variant(channels, bits):
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in (channels, bits)) or (channels, bits) not in VERSIONS:
        raise ValueError('Registered isotropic patch channel/bit pair required')
    return VERSIONS[(channels, bits)]


def _integer(value, name, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not 0 <= value <= maximum:
        raise ValueError(name + ' outside wire domain')
    return int(value)


def _geometry(height, width):
    height = _integer(height, 'height', 128)
    width = _integer(width, 'width', 128)
    if min(height, width) < 8:
        raise ValueError('Registered grid dimensions are 8 through128')
    return ((height + PATCH_SIDE - 1) // PATCH_SIDE) * ((width + PATCH_SIDE - 1) // PATCH_SIDE)


def _numpy_codes(value, shape, maximum, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in 'iuf':
        raise ValueError(name + ' has invalid shape or numeric type')
    if not np.isfinite(value).all() or not np.equal(value, np.round(value)).all():
        raise ValueError(name + ' must contain finite integers')
    if (value < 0).any() or (value > maximum).any():
        raise ValueError(name + ' outside wire domain')
    return value.astype(np.int64)


def _torch_codes(value, shape, maximum, name):
    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype == torch.bool or value.is_complex():
        raise ValueError(name + ' has invalid tensor shape or type')
    if not torch.isfinite(value).all() or (value != value.to(torch.float64).round()).any():
        raise ValueError(name + ' must contain finite integers')
    if (value < 0).any() or (value > maximum).any():
        raise ValueError(name + ' outside wire domain')
    return value.long()


class IsotropicPatchQuantizer(nn.Module):
    """q/code/rq retain [...,1]; h/mean-code/rh retain [...,C].

    Precision uses the pre-existing installed float32 log LUT. Means use the
    fixed endpoint-inclusive uniform [-1,1] grid with 2**bits-1 levels (63 or15).
    The midpoint represents exact zero; the maximal b-bit code is reserved and
    rejected as malformed. Storage remains24mean bytes for either version.
    Codes are not learned or fit using validation maps.
    """
    def __init__(self, channels, bits):
        super().__init__()
        _variant(channels, bits)
        self.channels, self.bits = int(channels), int(bits)
        self.register_buffer('precision_lut', torch.from_numpy(PRECISION_LUT.copy()))
        self.register_buffer('mean_lut', torch.linspace(-1., 1., (1 << bits) - 1, dtype=torch.float32))
        self.mean_lut[((1 << bits) - 2) // 2] = 0.

    def decode(self, qcode, meancodes):
        if not isinstance(qcode, torch.Tensor) or qcode.ndim < 1 or qcode.shape[-1] != 1:
            raise ValueError('Scalar precision codes require a final singleton')
        qc = _torch_codes(qcode, qcode.shape, 255, 'precision codes')
        mc = _torch_codes(meancodes, (*qcode.shape[:-1], self.channels), (1 << self.bits) - 2, 'mean codes')
        if qcode.device != meancodes.device or qcode.device != self.precision_lut.device:
            raise ValueError('Quantizer and codes must be on one device')
        q = self.precision_lut[qc]
        mean = self.mean_lut[mc]
        return q, q * mean

    def encode(self, q, h):
        if not isinstance(q, torch.Tensor) or not isinstance(h, torch.Tensor) or q.ndim < 1 or q.shape[-1] != 1:
            raise ValueError('Isotropic precision must have shape [...,1]')
        if h.shape != (*q.shape[:-1], self.channels) or not q.is_floating_point() or not h.is_floating_point():
            raise ValueError('Isotropic natural vector has invalid shape or type')
        if q.device != h.device or q.dtype != h.dtype or q.device != self.precision_lut.device:
            raise ValueError('Statistics and quantizer must share dtype/device')
        if not torch.isfinite(q).all() or not torch.isfinite(h).all() or (q < 0).any() or (q > 20. + BOUND_TOLERANCE).any():
            raise ValueError('Gaussian precision outside declared finite [0,20] domain')
        positive = q > 0
        if ((~positive).expand_as(h) & (h != 0)).any():
            raise ValueError('Zero precision cannot carry natural information')
        mean = h / torch.where(positive, q, torch.ones_like(q))
        if (mean.abs() > 1. + BOUND_TOLERANCE).any():
            raise ValueError('Gaussian mean outside declared [-1,1] domain')
        qmin, qmax = .001, 20.
        logcode = ((q.clamp(qmin, qmax).log() - np.log(qmin)) / (np.log(qmax) - np.log(qmin)) * 254).round().long() + 1
        qcode = torch.where(positive, logcode, torch.zeros_like(logcode))
        meancodes = ((mean.clamp(-1., 1.) + 1.) * ((1 << self.bits) - 2) / 2).round().long()
        meancodes = torch.where(positive, meancodes, torch.zeros_like(meancodes))
        rq, rh = self.decode(qcode, meancodes)
        rq, rh = rq.to(q.dtype), rh.to(h.dtype)
        return qcode, meancodes, rq.detach() + (q - q.detach()), rh.detach() + (h - h.detach())


def serialize_iso(tick, sender, receiver, qcodes, meancodes, height, width, round_id, channels, bits):
    version = _variant(channels, bits)
    tiles = _geometry(height, width)
    tick = _integer(tick, 'tick', (1 << 32) - 1)
    sender = _integer(sender, 'sender', 65535)
    receiver = _integer(receiver, 'receiver', 65535)
    round_id = _integer(round_id, 'round', 1)
    qcodes = _numpy_codes(qcodes, (tiles,), 255, 'precision codes')
    meancodes = _numpy_codes(meancodes, (tiles, channels), (1 << bits) - 2, 'mean codes')
    active = np.flatnonzero(qcodes)
    if not len(active):
        return None
    entries = [ENTRY.pack(int(tile), int(qcodes[tile])) + pack_indices(meancodes[tile], bits) for tile in active]
    return HEADER.pack(tick, sender, receiver, channels, height, width, round_id) + bytes([version]) + COUNT.pack(len(active)) + b''.join(entries)


def deserialize_iso(packet, tiles, channels, bits):
    version = _variant(channels, bits)
    tiles = _integer(tiles, 'tiles', 16)
    if not isinstance(packet, (bytes, bytearray, memoryview)) or len(packet) < HEADER.size + 1 + COUNT.size:
        raise ValueError('Truncated isotropic patch header')
    meta = HEADER.unpack_from(packet)
    expected = _geometry(meta[4], meta[5])
    if tiles != expected or meta[3] != channels or meta[6] not in (0, 1):
        raise ValueError('Isotropic patch geometry/channels/round mismatch')
    if packet[HEADER.size] != version:
        raise ValueError('Wrong isotropic patch wire version')
    count = COUNT.unpack_from(packet, HEADER.size + 1)[0]
    entry_bytes = ENTRY.size + (channels * bits + 7) // 8
    at = HEADER.size + 1 + COUNT.size
    if not 1 <= count <= tiles or len(packet) != at + count * entry_bytes:
        raise ValueError('Invalid patch count or packet length')
    entries, seen = [], set()
    for _ in range(count):
        tile, qcode = ENTRY.unpack_from(packet, at)
        at += ENTRY.size
        if tile >= tiles or tile in seen or qcode == 0:
            raise ValueError('Invalid, repeated or empty isotropic patch entry')
        seen.add(tile)
        means = unpack_indices(packet[at:at + entry_bytes - ENTRY.size], channels, bits)
        at += entry_bytes - ENTRY.size
        if (means == (1 << bits) - 1).any():
            raise ValueError('Reserved isotropic mean code')
        entries.append((tile, qcode, means))
    return meta, entries
