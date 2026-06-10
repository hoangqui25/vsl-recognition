from __future__ import annotations

import torch
from torch import nn

from .transformer import SinusoidalPositionalEncoding


POSE_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)

HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def mediapipe_graph_mask(num_joints: int = 75) -> torch.Tensor:
    """Return an attention mask for 33 pose and two 21-joint hands."""
    if num_joints != 75:
        raise ValueError("MediaPipe graph requires exactly 75 joints")

    connected = torch.eye(num_joints, dtype=torch.bool)
    edges = list(POSE_EDGES)
    edges.extend((left + 33, right + 33) for left, right in HAND_EDGES)
    edges.extend((left + 54, right + 54) for left, right in HAND_EDGES)
    edges.extend(((15, 33), (16, 54)))

    for source, target in edges:
        connected[source, target] = True
        connected[target, source] = True
    return ~connected


class GraphTransformerBlock(nn.Module):
    """Graph-masked self-attention over the joints of one frame."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=graph_mask,
            need_weights=False,
        )
        x = x + self.dropout(attended)
        return x + self.dropout(self.feedforward(self.norm2(x)))


class GraphTransformerClassifier(nn.Module):
    """Spatial graph attention followed by temporal Transformer encoding."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        graph_layers: int = 2,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.3,
        pooling: str = "mean",
        max_len: int = 512,
        num_joints: int = 75,
        coord_dim: int = 4,
    ) -> None:
        super().__init__()
        if input_dim != num_joints * coord_dim:
            raise ValueError(
                "Graph Transformer expects skeleton features with "
                f"{num_joints} joints x {coord_dim} values = "
                f"{num_joints * coord_dim} input dimensions, got {input_dim}"
            )
        if pooling not in {"mean", "max", "cls"}:
            raise ValueError("pooling must be one of: mean, max, cls")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if graph_layers < 1:
            raise ValueError("graph_layers must be positive")

        self.num_joints = num_joints
        self.coord_dim = coord_dim
        self.pooling = pooling
        self.joint_projection = nn.Linear(coord_dim, hidden_dim)
        self.joint_embedding = nn.Parameter(
            torch.zeros(1, 1, num_joints, hidden_dim)
        )
        self.graph_blocks = nn.ModuleList(
            GraphTransformerBlock(
                hidden_dim,
                num_heads,
                feedforward_dim,
                dropout,
            )
            for _ in range(graph_layers)
        )
        self.register_buffer(
            "graph_mask",
            mediapipe_graph_mask(num_joints),
            persistent=False,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position = SinusoidalPositionalEncoding(hidden_dim, max_len + 1)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_frames, _ = features.shape
        joints = features.reshape(
            batch_size,
            num_frames,
            self.num_joints,
            self.coord_dim,
        )
        x = self.joint_projection(joints) + self.joint_embedding
        x = x.reshape(batch_size * num_frames, self.num_joints, -1)

        for block in self.graph_blocks:
            x = block(x, self.graph_mask)

        frame_tokens = x.mean(dim=1).reshape(batch_size, num_frames, -1)
        lengths = self._resolve_lengths(lengths, batch_size, num_frames, features.device)

        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = self.position(torch.cat([cls, frame_tokens], dim=1))
        padding_mask = self._padding_mask(lengths, sequence.size(1))
        encoded = self.temporal_encoder(
            sequence,
            src_key_padding_mask=padding_mask,
        )
        pooled = self._pool(encoded, lengths, padding_mask)
        return self.classifier(pooled)

    @staticmethod
    def _resolve_lengths(
        lengths: torch.Tensor | None,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        if lengths is None:
            return torch.full(
                (batch_size,),
                num_frames,
                dtype=torch.long,
                device=device,
            )
        return lengths.to(device)

    @staticmethod
    def _padding_mask(
        lengths: torch.Tensor,
        sequence_len: int,
    ) -> torch.Tensor:
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
