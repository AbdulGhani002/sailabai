"""Model 2 comparison (phase 3): the same U-Net and inputs as UNet-TT, with a ConvLSTM as the time part.

The day tokens are projected to a small feature vector, broadcast over the bottleneck grid next to
the map features, and run through a ConvLSTM cell day by day; its last hidden state is added back
to the bottleneck. Everything else (inputs, loss, training, evaluation) is shared with UNet-TT, so
the comparison isolates the temporal model.

    sailab train forecast --arch unet_clstm --out-dir runs/model2_clstm
    sailab evaluate forecast --split val --model-dir runs/model2_clstm --experiment model2-clstm
"""

from __future__ import annotations

import torch
from torch import nn

from sailab.nn.unet import ResNetUNet


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden: int, kernel: int = 3) -> None:
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(in_ch + hidden, 4 * hidden, kernel, padding=kernel // 2)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class UNetConvLSTM(nn.Module):
    def __init__(self, map_channels: int, series_features: int, n_tokens: int = 37, encoder: str = "resnet18",
                 hidden: int = 64, token_dim: int = 32, pretrained: bool = False, **_: object) -> None:
        super().__init__()
        self.unet = ResNetUNet(map_channels, 1, encoder=encoder, pretrained=pretrained)
        c = self.unet.encoder_channels[-1]
        self.token = nn.Sequential(nn.Linear(2 * series_features + 2, token_dim), nn.GELU())  # values, flags, lead
        self.squeeze = nn.Conv2d(c, hidden, 1)
        self.cell = ConvLSTMCell(hidden + token_dim, hidden)
        self.expand = nn.Conv2d(hidden, c, 1)
        nn.init.zeros_(self.expand.weight)  # starts as the plain U-Net
        nn.init.zeros_(self.expand.bias)
        self.config = {"arch": "unet_clstm", "map_channels": map_channels, "series_features": series_features,
                       "n_tokens": n_tokens, "encoder": encoder, "hidden": hidden, "token_dim": token_dim}

    def forward(self, maps: torch.Tensor, values: torch.Tensor, missing: torch.Tensor,
                is_forecast: torch.Tensor, lead_days: torch.Tensor) -> torch.Tensor:
        feats = self.unet.encode(maps)
        b = feats[-1]
        n, _, h, w = b.shape
        lead = (lead_days / 7.0)[:, None, None].expand(-1, values.shape[1], 1)
        tokens = self.token(torch.cat([values * (1 - missing), missing, is_forecast[..., None], lead], dim=-1))
        base = self.squeeze(b)
        hs = torch.zeros_like(base)
        cs = torch.zeros_like(base)
        for t in range(tokens.shape[1]):
            day = tokens[:, t, :, None, None].expand(-1, -1, h, w)
            hs, cs = self.cell(torch.cat([base, day], dim=1), (hs, cs))
        feats[-1] = b + self.expand(hs)
        return self.unet.decode(feats)[:, 0]
