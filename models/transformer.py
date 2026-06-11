from __future__ import annotations

import torch
from torch import nn

from .encoder import (
    build_padding_mask,
    build_positional_encoding,
    build_transformer_encoder,
    resolve_lengths,
)


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
        position_encoding: str = "sinusoidal",
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
        self.position = build_positional_encoding(
            position_encoding,
            hidden_dim,
            max_len=max_len + 1,
        )

        self.encoder = build_transformer_encoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_projection(features)
        lengths = resolve_lengths(lengths, x)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.position(x)

        padding_mask = build_padding_mask(
            lengths,
            x.size(1),
            prefix_tokens=1,
        )
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        pooled = self._pool(encoded, lengths, padding_mask)
        return self.classifier(pooled)

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
