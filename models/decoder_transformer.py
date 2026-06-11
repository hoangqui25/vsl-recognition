from __future__ import annotations

import torch
from torch import nn

from .encoder import (
    build_padding_mask,
    build_positional_encoding,
    build_transformer_encoder,
    resolve_lengths,
)


class ClassQueryDecoderLayer(nn.Module):
    """Cross-attention decoder layer for a single learned class query."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        attended, _ = self.cross_attention(
            normalized_query,
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        query = query + self.dropout(attended)
        return query + self.dropout(
            self.feedforward(self.feedforward_norm(query))
        )


class DecoderTransformerClassifier(nn.Module):
    """Transformer encoder with a learned class-query decoder."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        decoder_layers: int = 2,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.3,
        max_len: int = 512,
        position_encoding: str = "sinusoidal",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if decoder_layers < 1:
            raise ValueError("decoder_layers must be positive")

        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.position = build_positional_encoding(
            position_encoding,
            hidden_dim,
            max_len=max_len,
        )

        self.encoder = build_transformer_encoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.class_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.decoder = nn.ModuleList(
            ClassQueryDecoderLayer(
                hidden_dim,
                num_heads,
                feedforward_dim,
                dropout,
            )
            for _ in range(decoder_layers)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.trunc_normal_(self.class_query, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.position(self.input_projection(features))
        lengths = resolve_lengths(lengths, memory)
        padding_mask = build_padding_mask(lengths, memory.size(1))
        memory = self.encoder(
            memory,
            src_key_padding_mask=padding_mask,
        )

        query = self.class_query.expand(memory.size(0), -1, -1)
        for decoder_layer in self.decoder:
            query = decoder_layer(query, memory, padding_mask)
        return self.classifier(query[:, 0])
