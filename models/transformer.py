from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Transformer classifier for pre-extracted sequence features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.3,
        pooling: str = "mean",
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "max", "cls"}:
            raise ValueError("pooling must be one of: mean, max, cls")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.pooling = pooling
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position = SinusoidalPositionalEncoding(hidden_dim, max_len=max_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_projection(features)

        if lengths is None:
            lengths = torch.full(
                (x.size(0),),
                x.size(1),
                dtype=torch.long,
                device=x.device,
            )
        else:
            lengths = lengths.to(x.device)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.position(x)

        padding_mask = self._padding_mask(lengths, x.size(1))
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        pooled = self._pool(encoded, lengths, padding_mask)
        return self.classifier(pooled)

    def _padding_mask(self, lengths: torch.Tensor, sequence_len: int) -> torch.Tensor:
        positions = torch.arange(sequence_len, device=lengths.device).unsqueeze(0)
        valid = positions <= lengths.unsqueeze(1)
        valid[:, 0] = True
        return ~valid

    def _pool(
        self,
        encoded: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return encoded[:, 0]

        tokens = encoded[:, 1:]
        token_mask = ~padding_mask[:, 1:]

        if self.pooling == "mean":
            masked = tokens * token_mask.unsqueeze(-1)
            return masked.sum(dim=1) / lengths.clamp(min=1).unsqueeze(1)

        masked = tokens.masked_fill(~token_mask.unsqueeze(-1), float("-inf"))
        return masked.max(dim=1).values
