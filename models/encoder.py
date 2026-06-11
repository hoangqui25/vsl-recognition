from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal position information."""

    def __init__(self, hidden_dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, hidden_dim, 2)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_len, hidden_dim)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer(
            "pe",
            encoding.unsqueeze(0),
            persistent=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.pe[:, : features.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """Trainable position information."""

    def __init__(self, hidden_dim: int, max_len: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Parameter(
            torch.zeros(1, max_len, hidden_dim)
        )
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.embedding[:, : features.size(1)]


def build_positional_encoding(
    encoding_type: str,
    hidden_dim: int,
    max_len: int,
) -> nn.Module:
    if encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(hidden_dim, max_len)
    if encoding_type == "learned":
        return LearnedPositionalEncoding(hidden_dim, max_len)
    raise ValueError(
        "position_encoding must be one of: sinusoidal, learned"
    )


def build_transformer_encoder(
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    feedforward_dim: int,
    dropout: float,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=feedforward_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        encoder_layer=layer,
        num_layers=num_layers,
        enable_nested_tensor=False,
    )


def resolve_lengths(
    lengths: torch.Tensor | None,
    features: torch.Tensor,
) -> torch.Tensor:
    if lengths is None:
        return torch.full(
            (features.size(0),),
            features.size(1),
            dtype=torch.long,
            device=features.device,
        )
    return lengths.to(features.device)


def build_padding_mask(
    lengths: torch.Tensor,
    sequence_len: int,
    prefix_tokens: int = 0,
) -> torch.Tensor:
    positions = torch.arange(
        sequence_len,
        device=lengths.device,
    ).unsqueeze(0)
    return positions >= (lengths + prefix_tokens).unsqueeze(1)
