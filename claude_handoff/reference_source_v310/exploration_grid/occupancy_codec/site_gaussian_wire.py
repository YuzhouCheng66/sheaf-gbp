"""High-capacity source-first v78 Gaussian wire on public8x8 atlas sites.

Each active site carries one scalar precision byte and64 mean bytes. The
existing installed logarithmic precision LUT and an exact-zero255-level mean
grid are fixed, not fit to data. Mean code255 is reserved/invalid. This is
declared lossy Gaussian quantization, not a lossless arbitrary occupancy codec.
All site IDs,19 application bytes and optional external framing are charged.
No mass, source map, occupancy labels or hidden side-channel is serialized.
"""
import numpy as np
import torch
from torch import nn

from occupancy_codec.gaussian import HEADER, PRECISION_LUT
from occupancy_codec.isotropic_patch_wire import (
    COUNT, ENTRY, IsotropicPatchQuantizer, _integer, _numpy_codes,
)
from occupancy_codec.pair_vocabulary import pack_indices, unpack_indices


SITE_SIDE = 8
CHANNELS = 64
BITS = 8
VERSION = 13


def _family(channels, bits):
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in (channels, bits)):
        raise ValueError('Integer registered site Gaussian channels/bits required')
    if (channels, bits) != (CHANNELS, BITS):
        raise ValueError('Only registered C64/8bit site Gaussian family')


def _geometry(height, width):
    height = _integer(height, 'height', 128)
    width = _integer(width, 'width', 128)
    if min(height, width) < 8:
        raise ValueError('Registered grid dimensions are8 through128')
    return ((height + SITE_SIDE - 1) // SITE_SIDE) * ((width + SITE_SIDE - 1) // SITE_SIDE)


class SiteGaussianQuantizer(IsotropicPatchQuantizer):
    """C64/b8 specialization; inherited encode/decode and STE are unchanged.

    q/qcode/rq have shape [...,1]; h/meancodes/rh have shape [...,64]. Means
    use255 valid levels spanning [-1,1], midpoint127 is exact0, and255 is
    rejected even though it would fit in storage. Scalar q is Gaussian
    confidence, not integer evidence mass or calibrated occupancy certainty.
    """
    def __init__(self, channels=CHANNELS, bits=BITS):
        _family(channels, bits)
        nn.Module.__init__(self)
        self.channels, self.bits = CHANNELS, BITS
        self.register_buffer('precision_lut', torch.from_numpy(PRECISION_LUT.copy()))
        mean_lut = torch.linspace(-1., 1., 255, dtype=torch.float32)
        mean_lut[127] = 0.
        self.register_buffer('mean_lut', mean_lut)


def serialize_site(tick, sender, receiver, qcodes, meancodes, height, width, round_id,
                   channels=CHANNELS, bits=BITS):
    """Return version13 packet (19+67*active-sites bytes), or None if empty."""
    _family(channels, bits)
    sites = _geometry(height, width)
    tick = _integer(tick, 'tick', (1 << 32) - 1)
    sender = _integer(sender, 'sender', 65535)
    receiver = _integer(receiver, 'receiver', 65535)
    round_id = _integer(round_id, 'round', 1)
    qcodes = _numpy_codes(qcodes, (sites,), 255, 'precision codes')
    means = _numpy_codes(meancodes, (sites, CHANNELS), 254, 'mean codes')
    active = np.flatnonzero(qcodes)
    if not len(active):
        return None
    entries = [ENTRY.pack(int(site), int(qcodes[site])) + pack_indices(means[site], BITS) for site in active]
    return HEADER.pack(tick, sender, receiver, CHANNELS, height, width, round_id) + bytes([VERSION]) + COUNT.pack(len(active)) + b''.join(entries)


def deserialize_site(packet, sites, channels=CHANNELS, bits=BITS):
    """Validate all boundaries and return(header,[(site,qcode,mean-codes)])."""
    _family(channels, bits)
    sites = _integer(sites, 'sites', 256)
    if not isinstance(packet, (bytes, bytearray, memoryview)) or len(packet) < HEADER.size + 1 + COUNT.size:
        raise ValueError('Truncated site Gaussian header')
    meta = HEADER.unpack_from(packet)
    if sites != _geometry(meta[4], meta[5]) or meta[3] != CHANNELS or meta[6] not in (0, 1):
        raise ValueError('Site Gaussian geometry/channels/round mismatch')
    if packet[HEADER.size] != VERSION:
        raise ValueError('Wrong site Gaussian version')
    count = COUNT.unpack_from(packet, HEADER.size + 1)[0]
    at = HEADER.size + 1 + COUNT.size
    entry_bytes = ENTRY.size + CHANNELS
    if not 1 <= count <= sites or len(packet) != at + count * entry_bytes:
        raise ValueError('Invalid site count or Gaussian packet length')
    entries, seen = [], set()
    for _ in range(count):
        site, qcode = ENTRY.unpack_from(packet, at)
        at += ENTRY.size
        if site >= sites or site in seen or qcode == 0:
            raise ValueError('Invalid, repeated or empty Gaussian site')
        seen.add(site)
        means = unpack_indices(packet[at:at + CHANNELS], CHANNELS, BITS)
        at += CHANNELS
        if (means == 255).any():
            raise ValueError('Reserved site Gaussian mean code')
        entries.append((site, qcode, means))
    return meta, entries
