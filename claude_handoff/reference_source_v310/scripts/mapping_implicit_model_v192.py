"""Fresh, whole-map latent capacity pilot; this module does not implement GBP.

Only ``z`` crosses the encoder/decoder boundary. The decoder has no access to
the private map, a per-map feature grid, an observation skip, or per-map weights.
Unknown/free/wall are distinct encoder channels. Spatial aggregation preserves
a fixed 16 x 16 layout before the global bottleneck; it is NOT a global average.

The 8-bit codec has 255 symmetric values, including exact zero, and one reserved
code. Fixed public bounds are [-1, 1]; there is no uncharged per-map scale.
``encode`` returns the continuous latent; ``forward`` quantizes by default.
Coordinates are normalized cell-centre (x, y), in [-1, 1].
The codec and coordinate/Fourier calculations are always FP32, including under
AMP. In particular, BF16 cannot distinguish all 512-wide cell-centre queries.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def coordinate_grid(height: int, width: int, *, device=None,
                    dtype=torch.float32) -> Tensor:
    """Return H*W cell-centre queries in row-major order, shape [Q, 2]."""
    if height < 1 or width < 1:
        raise ValueError("Map dimensions must be positive")
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def latent_codes(z: Tensor) -> Tensor:
    """Fixed 8-bit alphabet: 0..254 represent -1..1; 127 is exact zero."""
    return (z.float().clamp(-1.0, 1.0) * 127.0).round().to(torch.int16).add(127).to(torch.uint8)


def codes_to_latent(codes: Tensor, *, dtype=torch.float32) -> Tensor:
    """Decode trusted codes 0..254; call validate_codes at external boundaries."""
    if codes.dtype != torch.uint8 or dtype != torch.float32:
        raise ValueError("Latent codes must be uint8 and the public decoder uses FP32")
    return (codes.float() - 127.0) / 127.0


def validate_codes(codes: Tensor) -> None:
    """Explicit external-input validation; intentionally outside hot forwards."""
    if codes.dtype != torch.uint8 or bool((codes == 255).any()):
        raise ValueError("Invalid latent code; code 255 is reserved")


def latent_quantize(z: Tensor) -> Tensor:
    """Hard-codec forward, identity STE inside fixed clipping bounds."""
    bounded = z.float().clamp(-1.0, 1.0)
    hard = codes_to_latent(latent_codes(bounded))
    # This ordering retains the exact hard values in forward, including zero.
    return hard.detach() + (bounded - bounded.detach())


def private_channels(private: Tensor, *, dtype=torch.float32) -> Tensor:
    """Construct unknown/free/wall/x/y without caching any private contents.

    The caller owns value validation of its dataset (-1, 0, 1). Avoid per-step
    GPU synchronization for known, preflight-validated training tensors.
    """
    if private.ndim != 3 or private.shape[-2] < 16 or private.shape[-1] < 16:
        raise ValueError("private must have shape [B,H,W] with H,W >= 16")
    batch, height, width = private.shape
    observed = torch.stack((private == -1, private == 0, private == 1), dim=1).to(dtype)
    xy = coordinate_grid(height, width, device=private.device, dtype=dtype)
    xy = xy.reshape(height, width, 2).permute(2, 0, 1).unsqueeze(0).expand(batch, -1, -1, -1)
    return torch.cat((observed, xy), dim=1)


class _SpatialResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x + self.net(x))


class _FiLMResidual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        x = x.float()
        branch = self.norm(x) * (1.0 + 0.5 * torch.tanh(scale.float()[:, None, :]))
        branch = F.silu(branch + shift.float()[:, None, :])
        branch = self.second(F.silu(self.first(branch).float())).float()
        return x + branch


class ImplicitMapCodec(nn.Module):
    """Shared CNN encoder and coordinate-conditioned implicit map decoder.

    ``latent_dim`` is the sole per-map transmitted representation (64/128/256
    in the paired capacity pilot). Learned tensors belong to the global model,
    not to individual maps. Decoder heads factor knownness and occupancy:
    ``known_logits`` predicts observed support; ``wall_logits`` is conditional
    on known. Occupancy supervision must therefore be masked by known support.
    """

    def __init__(self, latent_dim: int = 64, width: int = 128,
                 fourier_bands: int = 8, query_chunk_size: int = 8192):
        super().__init__()
        if latent_dim < 1 or width < 8 or fourier_bands < 0 or query_chunk_size < 1:
            raise ValueError("Invalid architecture dimensions")
        self.latent_dim = int(latent_dim)
        self.width = int(width)
        self.fourier_bands = int(fourier_bands)
        self.query_chunk_size = int(query_chunk_size)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.SiLU(),
            _SpatialResidual(64),
            nn.Conv2d(64, 16, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
        )
        self.encoder_head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 16 * 16, 512),
                                          nn.SiLU(), nn.Linear(512, self.latent_dim))
        frequencies = math.pi * torch.pow(2.0, torch.arange(fourier_bands, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies)
        coordinate_dim = 2 + 4 * fourier_bands
        self.query_projection = nn.Linear(coordinate_dim, width)
        self.latent_projection = nn.Linear(self.latent_dim, width)
        self.condition = nn.Linear(self.latent_dim, 3 * 2 * width)
        self.blocks = nn.ModuleList([_FiLMResidual(width) for _ in range(3)])
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 2))

    def encode(self, private: Tensor) -> Tensor:
        inputs = private_channels(private, dtype=torch.float32)
        return torch.tanh(self.encoder_head(self.encoder_cnn(inputs)).float())

    @staticmethod
    def latent_quantize(z: Tensor) -> Tensor:
        return latent_quantize(z)

    def _decode_chunk(self, z_projected: Tensor, conditioning: Tensor,
                      coords: Tensor) -> Tensor:
        # Never round cell-centre queries to BF16 (adjacent 512 cells can merge).
        coords = coords.float()
        angles = coords.unsqueeze(-1) * self.frequencies.float()
        features = torch.cat((coords, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)
        hidden = F.silu(self.query_projection(features).float() + z_projected.float()[:, None, :])
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, conditioning[:, index, 0, :], conditioning[:, index, 1, :])
        return self.readout(hidden).float()

    def decode(self, z: Tensor, coords: Tensor, *,
               chunk_size: Optional[int] = None) -> Dict[str, Tensor]:
        """Decode only supplied latent and queries; return two [B,Q] logits."""
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError("z must have shape [B, latent_dim]")
        if coords.ndim != 3 or coords.shape[0] != z.shape[0] or coords.shape[-1] != 2:
            raise ValueError("coords must have shape [B,Q,2], normalized (x,y)")
        if z.device != coords.device:
            raise ValueError("Latent and coordinates must be on the same device")
        size = self.query_chunk_size if chunk_size is None else int(chunk_size)
        if size < 1:
            raise ValueError("chunk_size must be positive")
        z = z.float()
        coords = coords.float()
        projected = self.latent_projection(z)
        conditioning = self.condition(z).reshape(z.shape[0], 3, 2, self.width)
        if coords.shape[1] == 0:
            empty = z.new_empty((z.shape[0], 0))
            return {"known_logits": empty, "wall_logits": empty.clone()}
        logits = torch.cat([self._decode_chunk(projected, conditioning, part)
                            for part in coords.split(size, dim=1)], dim=1)
        return {"known_logits": logits[..., 0], "wall_logits": logits[..., 1]}

    def forward(self, private: Tensor, coords: Tensor,
                quantized: bool = True) -> Dict[str, Tensor]:
        continuous = self.encode(private)
        latent = self.latent_quantize(continuous) if quantized else continuous
        result = self.decode(latent, coords)
        return dict(result, latent=latent, latent_unquantized=continuous)

    def count_parameters(self) -> Dict[str, int]:
        encoder = sum(p.numel() for module in (self.encoder_cnn, self.encoder_head)
                      for p in module.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": encoder, "decoder": total - encoder, "total": total}


__all__ = ["ImplicitMapCodec", "coordinate_grid", "private_channels", "latent_codes",
           "codes_to_latent", "validate_codes", "latent_quantize"]
