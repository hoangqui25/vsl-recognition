from __future__ import annotations

import torch
from torch import nn


class RNNClassifier(nn.Module):
    """Vanilla RNN classifier for pre-extracted feature sequences."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pooling: str = "last",
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        if pooling not in {"last", "mean", "max"}:
            raise ValueError("pooling must be one of: last, mean, max")
        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError("nonlinearity must be one of: tanh, relu")

        self.pooling = pooling
        self.bidirectional = bidirectional
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                features,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.rnn(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=features.size(1),
            )
        else:
            output, _ = self.rnn(features)

        return self.classifier(self._pool(output, lengths))

    def _pool(
        self,
        output: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        if lengths is None:
            if self.pooling == "mean":
                return output.mean(dim=1)
            if self.pooling == "max":
                return output.max(dim=1).values
            return output[:, -1]

        lengths = lengths.to(output.device)
        if self.pooling == "last":
            index = (lengths - 1).clamp(min=0).view(-1, 1, 1)
            index = index.expand(-1, 1, output.size(-1))
            return output.gather(1, index).squeeze(1)

        mask = torch.arange(output.size(1), device=output.device).unsqueeze(0)
        mask = mask < lengths.unsqueeze(1)
        if self.pooling == "mean":
            masked = output * mask.unsqueeze(-1)
            return masked.sum(dim=1) / lengths.clamp(min=1).unsqueeze(1)

        masked = output.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return masked.max(dim=1).values
