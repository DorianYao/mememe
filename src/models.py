from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def _mlp_trunk(
    input_dim: int,
    hidden_dims: Iterable[int],
    dropout: float,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        in_dim = hidden_dim
    return nn.Sequential(*layers), in_dim


class MLPClassifier(nn.Module):
    """方案 1：把 (window, features) 拉平后过几层 MLP，输出涨跌 logit。"""

    def __init__(
        self,
        window_size: int,
        num_features: int,
        hidden_dims: Iterable[int] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        input_dim = window_size * num_features
        trunk, out_dim = _mlp_trunk(input_dim, hidden_dims, dropout)
        self.net = nn.Sequential(trunk, nn.Linear(out_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"MLPClassifier expects (batch, window, features); got {tuple(x.shape)}"
            )
        flat = x.reshape(x.size(0), -1)
        return self.net(flat).squeeze(-1)


class MLPTimeDecayClassifier(nn.Module):
    """Flatten with exponential time decay (recent bars weighted higher)."""

    def __init__(
        self,
        window_size: int,
        num_features: int,
        hidden_dims: Iterable[int] = (256, 128, 64),
        dropout: float = 0.3,
        time_decay_gamma: float = 0.95,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        weights = time_decay_gamma ** torch.arange(
            window_size - 1, -1, -1, dtype=torch.float32
        )
        weights = weights / weights.sum()
        self.register_buffer("time_weights", weights)

        input_dim = window_size * num_features
        trunk, out_dim = _mlp_trunk(input_dim, hidden_dims, dropout)
        self.net = nn.Sequential(trunk, nn.Linear(out_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"MLPTimeDecayClassifier expects (batch, window, features); "
                f"got {tuple(x.shape)}"
            )
        w = self.time_weights.view(1, self.window_size, 1)
        weighted = x * w
        flat = weighted.reshape(x.size(0), -1)
        return self.net(flat).squeeze(-1)


class GRUClassifier(nn.Module):
    """Single-layer GRU + MLP head (~109k params at w=192, F=15)."""

    def __init__(
        self,
        window_size: int,
        num_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"GRUClassifier expects (batch, window, features); got {tuple(x.shape)}"
            )
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class CNN1DClassifier(nn.Module):
    """方案 2：1D CNN，把每个特征当一个通道，沿时间维度卷积。

    输入: (batch, window, features) → 内部转置成 (batch, features, window)。
    """

    def __init__(
        self,
        window_size: int,
        num_features: int,
        channels: Iterable[int] = (32, 64),
        kernel_size: int = 3,
        fc_hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features

        conv_layers: list[nn.Module] = []
        in_channels = num_features
        current_length = window_size
        channels_list = list(channels)
        for i, out_channels in enumerate(channels_list):
            conv_layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            if i < len(channels_list) - 1 and current_length >= 4:
                conv_layers.append(nn.MaxPool1d(kernel_size=2))
                current_length //= 2
            in_channels = out_channels
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        conv_layers.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_layers)

        self.head = nn.Sequential(
            nn.Linear(in_channels, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"CNN1DClassifier expects (batch, window, features); got {tuple(x.shape)}"
            )
        x = x.transpose(1, 2)
        features = self.conv(x)
        return self.head(features).squeeze(-1)


def build_model(name: str, window_size: int, num_features: int, config) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLPClassifier(
            window_size=window_size,
            num_features=num_features,
            hidden_dims=tuple(config.mlp_hidden),
            dropout=config.dropout,
        )
    if name in {"mlp_decay", "mlp_time_decay"}:
        gamma = float(getattr(config, "time_decay_gamma", 0.95))
        return MLPTimeDecayClassifier(
            window_size=window_size,
            num_features=num_features,
            hidden_dims=tuple(config.mlp_hidden),
            dropout=config.dropout,
            time_decay_gamma=gamma,
        )
    if name == "gru":
        return GRUClassifier(
            window_size=window_size,
            num_features=num_features,
            hidden_dim=int(getattr(config, "gru_hidden", 64)),
            dropout=config.dropout,
        )
    if name == "cnn":
        return CNN1DClassifier(
            window_size=window_size,
            num_features=num_features,
            channels=tuple(config.cnn_channels),
            kernel_size=config.cnn_kernel_size,
            fc_hidden=config.cnn_fc_hidden,
            dropout=config.dropout,
        )
    raise ValueError(
        f"Unknown model name: {name}. Expected mlp, mlp_decay, gru, or cnn."
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
