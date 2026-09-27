"""Model 2: UNet-TT, a U-Net for the map with a small time transformer joined at the bottleneck.

Map part: a ResNet U-Net reads the latest flood map, how old each pixel's observation is, and the
terrain layers. Time part: a 4-layer transformer (~0.8M parameters) reads 30 days of daily river
flow, rain and soil moisture plus 7 days of forecasts, one token per day, with a "missing" flag for
every value. Joining: the bottleneck features attend to the day tokens (cross-attention), and a
lead-time token scales them (FiLM), so one network answers "next pass", +1, +2, +3, +5 and +7 days.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from sailab.nn.unet import ResNetUNet


class CrossAttention2d(nn.Module):
    """Every bottleneck pixel attends to the sequence of day tokens."""

    def __init__(self, channels: int, token_dim: int, heads: int = 8, dim: int = 256) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, dim)
        self.kv = nn.Linear(token_dim, 2 * dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Linear(dim, channels)
        nn.init.zeros_(self.out.weight)  # starts as identity: the map path works on its own first
        nn.init.zeros_(self.out.bias)

    def forward(self, fmap: torch.Tensor, tokens: torch.Tensor, token_pad: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = fmap.shape
        q = self.q(self.norm_q(fmap.flatten(2).transpose(1, 2)))
        k, v = self.kv(tokens).chunk(2, dim=-1)
        att, _ = self.attn(q, k, v, key_padding_mask=token_pad, need_weights=False)
        return fmap + self.out(att).transpose(1, 2).reshape(b, c, h, w)


class TimeEncoder(nn.Module):
    def __init__(self, n_features: int, n_tokens: int, d_model: int = 128, layers: int = 4, heads: int = 4,
                 ff: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        # each value arrives with its missing flag, plus one flag saying "this day is a forecast"
        self.embed = nn.Linear(2 * n_features + 1, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.lead = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values: torch.Tensor, missing: torch.Tensor, is_forecast: torch.Tensor,
                lead_days: torch.Tensor) -> torch.Tensor:
        x = torch.cat([values * (1 - missing), missing, is_forecast[..., None]], dim=-1)
        tokens = self.embed(x) + self.pos
        lead = self.lead(lead_days[:, None, None] / 7.0)
        return self.norm(self.encoder(torch.cat([lead, tokens], dim=1)))  # token 0 is the lead token


class UNetTT(nn.Module):
    def __init__(self, map_channels: int, series_features: int, n_tokens: int = 37, encoder: str = "resnet18",
                 d_model: int = 128, layers: int = 4, heads: int = 4, ff: int = 512, dropout: float = 0.1,
                 pretrained: bool = False) -> None:
        super().__init__()
        self.unet = ResNetUNet(map_channels, 1, encoder=encoder, pretrained=pretrained)
        self.time = TimeEncoder(series_features, n_tokens, d_model, layers, heads, ff, dropout)
        c = self.unet.encoder_channels[-1]
        self.cross = CrossAttention2d(c, d_model)
        self.film = nn.Linear(d_model, 2 * c)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.config = {"map_channels": map_channels, "series_features": series_features, "n_tokens": n_tokens,
                       "encoder": encoder, "d_model": d_model, "layers": layers, "heads": heads, "ff": ff,
                       "dropout": dropout}

    def forward(self, maps: torch.Tensor, values: torch.Tensor, missing: torch.Tensor,
                is_forecast: torch.Tensor, lead_days: torch.Tensor) -> torch.Tensor:
        """Returns flood logits (B, H, W)."""
        feats = self.unet.encode(maps)
        tokens = self.time(values, missing, is_forecast, lead_days)
        bottleneck = self.cross(feats[-1], tokens)
        gamma, beta = self.film(tokens[:, 0]).chunk(2, dim=-1)
        feats[-1] = bottleneck * (1 + gamma[..., None, None]) + beta[..., None, None]
        return self.unet.decode(feats)[:, 0]


def time_encoder_parameters(model: UNetTT) -> int:
    return sum(p.numel() for p in model.time.parameters())


def sinusoid(n: int, d: int) -> torch.Tensor:  # kept for experiments with fixed positions
    pos = torch.arange(n)[:, None]
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    out = torch.zeros(n, d)
    out[:, 0::2] = torch.sin(pos * div)
    out[:, 1::2] = torch.cos(pos * div)
    return out
