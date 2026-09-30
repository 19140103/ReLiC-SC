"""Compact EEG retrieval baselines used by the ReLiC-SC release utilities."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class EEGNetRetrieval(nn.Module):
    """EEGNet-style encoder that predicts a normalized visual embedding."""

    def __init__(
        self,
        channels: int = 32,
        samples: int = 250,
        embedding_dim: int = 512,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        spatial_filters = temporal_filters * depth_multiplier
        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, 63),
                padding=(0, 31),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, 15),
                padding=(0, 7),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            feature_dim = int(
                self.features(torch.zeros(1, 1, channels, samples)).numel()
            )
        self.projection = nn.Linear(feature_dim, embedding_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG shaped [batch, channels, samples], got {eeg.shape}")
        features = self.features(eeg.unsqueeze(1)).flatten(1)
        return F.normalize(self.projection(features), dim=-1)


class ENIGMASpatioTemporalCNN(nn.Module):
    """Spatio-temporal CNN encoder used by the public ENIGMA codebase.

    The module follows the MIT-licensed implementation in
    ``external/ENIGMA/source/models.py`` but avoids an ``einops`` dependency in
    the core package.  For 32 channels and 250 samples it produces a 184-d
    hidden vector, matching the public Alljoined configuration.
    """

    def __init__(
        self,
        channels: int = 32,
        samples: int = 250,
        embedding_size: int = 4,
        temporal_filters: int = 40,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.temporal = nn.Linear(samples, samples)
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 5), stride=(1, 1)),
            nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 5)),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.Conv2d(
                temporal_filters,
                temporal_filters,
                kernel_size=(channels, 1),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(
            temporal_filters,
            embedding_size,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG shaped [batch, channels, samples], got {eeg.shape}")
        transformed = self.temporal(eeg)
        features = self.tsconv(transformed.unsqueeze(1))
        projected = self.projection(features)
        return projected.flatten(1)


class ResidualAdd(nn.Module):
    """Residual wrapper used by ENIGMA's projector."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class ENIGMAProjector(nn.Sequential):
    """Residual MLP projector from the public ENIGMA retrieval backbone."""

    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(
            nn.Linear(hidden_dim, embedding_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.Dropout(dropout),
                )
            ),
            nn.LayerNorm(embedding_dim),
        )


class ENIGMARetrieval(nn.Module):
    """Public ENIGMA EEG backbone adapted to the unified retrieval protocol."""

    def __init__(
        self,
        channels: int = 32,
        samples: int = 250,
        embedding_dim: int = 512,
        dropout: float = 0.5,
        normalize_output: bool = True,
        learn_logit_scale: bool = False,
    ) -> None:
        super().__init__()
        self.normalize_output = normalize_output
        self.logit_scale = (
            nn.Parameter(torch.tensor(1.0 / 0.07).log())
            if learn_logit_scale
            else None
        )
        self.encoder = ENIGMASpatioTemporalCNN(
            channels=channels,
            samples=samples,
            dropout=dropout,
        )
        with torch.no_grad():
            hidden_dim = int(self.encoder(torch.zeros(1, channels, samples)).shape[-1])
        self.projector = ENIGMAProjector(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        projected = self.projector(self.encoder(eeg))
        if self.normalize_output:
            projected = F.normalize(projected, dim=-1)
        return projected


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]


class ATMSEncoderLayer(nn.Module):
    """Channel-token transformer block used in the public ATM-S retrieval model."""

    def __init__(
        self,
        d_model: int = 250,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.attention = ATMSAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x)
        x = self.norm1(x + self.dropout(attended))
        y = self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y)


class ATMSAttention(nn.Module):
    """ATM-S attention projection allowing d_model not divisible by n_heads."""

    def __init__(self, d_model: int = 250, n_heads: int = 4, dropout: float = 0.25) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.scale = self.d_keys**-0.5
        inner_dim = self.d_keys * n_heads
        self.query_projection = nn.Linear(d_model, inner_dim)
        self.key_projection = nn.Linear(d_model, inner_dim)
        self.value_projection = nn.Linear(d_model, inner_dim)
        self.out_projection = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        queries = self.query_projection(x).view(batch, length, self.n_heads, self.d_keys)
        keys = self.key_projection(x).view(batch, length, self.n_heads, self.d_keys)
        values = self.value_projection(x).view(batch, length, self.n_heads, self.d_keys)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * self.scale
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.einsum("bhls,bshd->blhd", weights, values)
        return self.out_projection(attended.reshape(batch, length, -1))


class ATMSChannelTransformer(nn.Module):
    """ATM-S-style channel-token transformer adapted to arbitrary channel counts."""

    def __init__(
        self,
        channels: int = 32,
        samples: int = 250,
        d_model: int = 250,
        depth: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.value_embedding = nn.Linear(samples, d_model)
        self.position = SinusoidalPosition(d_model=d_model, max_len=channels + 4)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [ATMSEncoderLayer(d_model=d_model, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG shaped [batch, channels, samples], got {eeg.shape}")
        tokens = self.value_embedding(eeg)
        tokens = self.dropout(tokens + self.position(tokens.shape[1]).to(tokens.dtype))
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm(tokens[:, : self.channels])


class ATMSPatchEmbedding(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        embedding_size: int = 40,
        temporal_filters: int = 40,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 25), stride=(1, 1)),
            nn.AvgPool2d(kernel_size=(1, 51), stride=(1, 5)),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.Conv2d(
                temporal_filters,
                temporal_filters,
                kernel_size=(channels, 1),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(
            temporal_filters,
            embedding_size,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.projection(self.tsconv(eeg.unsqueeze(1))).flatten(1)


class ATMSRetrieval(nn.Module):
    """ATM-S retrieval backbone adapted from the public eeg_image_decode code."""

    def __init__(
        self,
        channels: int = 32,
        samples: int = 250,
        embedding_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.channel_transformer = ATMSChannelTransformer(
            channels=channels,
            samples=samples,
            dropout=0.25,
        )
        self.patch_embedding = ATMSPatchEmbedding(
            channels=channels,
            dropout=dropout,
        )
        with torch.no_grad():
            hidden_dim = int(self.patch_embedding(torch.zeros(1, channels, samples)).shape[-1])
        self.projector = ENIGMAProjector(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        transformed = self.channel_transformer(eeg)
        return F.normalize(self.projector(self.patch_embedding(transformed)), dim=-1)


class CEVRCumulativeEncoder(nn.Module):
    """CEVR shared causal additive encoder adapted to Alljoined channel count.

    Ported from the author's MIT-licensed CEVR implementation recorded in
    ``docs/CEVR_REUSE_AUDIT.md``. Each horizon uses only its observed EEG prefix.
    """

    def __init__(
        self,
        channels: int = 32,
        hidden_dim: int = 1024,
        embedding_dim: int = 512,
        max_samples: int = 250,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_samples = max_samples
        self.temporal_weight = nn.Parameter(
            torch.empty(hidden_dim, max_samples, channels)
        )
        self.temporal_bias = nn.Parameter(torch.empty(hidden_dim))
        nn.init.kaiming_uniform_(
            self.temporal_weight.view(hidden_dim, -1), a=5**0.5
        )
        bound = 1.0 / (max_samples * channels) ** 0.5
        nn.init.uniform_(self.temporal_bias, -bound, bound)
        self.state_update = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.shared_readout = nn.Linear(hidden_dim, embedding_dim)

    def encode_trajectory(
        self,
        eeg: torch.Tensor,
        horizon_ends: tuple[int, ...] = (25, 50, 75, 100, 125, 150, 200, 250),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected horizon states and their pre-nonlinear accumulators."""

        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG shaped [batch, channels, samples], got {eeg.shape}")
        if not horizon_ends or min(horizon_ends) < 1 or max(horizon_ends) > eeg.shape[-1]:
            raise ValueError(f"Invalid horizon endpoints {horizon_ends} for {eeg.shape[-1]} samples")
        if max(horizon_ends) > self.max_samples:
            raise ValueError("Horizon exceeds configured additive context")
        if any(left >= right for left, right in zip(horizon_ends, horizon_ends[1:])):
            raise ValueError("Horizon endpoints must be strictly increasing")

        cumulative = self.temporal_bias.unsqueeze(0).expand(eeg.shape[0], -1)
        states: list[torch.Tensor] = []
        accumulators: list[torch.Tensor] = []
        previous_end = 0
        for end in horizon_ends:
            segment = eeg[:, :, previous_end:end].transpose(1, 2).reshape(eeg.shape[0], -1)
            weights = self.temporal_weight[:, previous_end:end, :].reshape(
                self.hidden_dim, -1
            )
            cumulative = cumulative + F.linear(segment, weights)
            state = cumulative + self.state_update(cumulative)
            states.append(self.output_norm(state))
            accumulators.append(cumulative)
            previous_end = end
        return (
            self.shared_readout(torch.stack(states, dim=1)),
            torch.stack(accumulators, dim=1),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        horizon_ends: tuple[int, ...] = (25, 50, 75, 100, 125, 150, 200, 250),
    ) -> torch.Tensor:
        projected, _ = self.encode_trajectory(eeg, horizon_ends)
        return projected


class CEVRRetrieval(nn.Module):
    """CEVR EEG branch with parameter-free cumulative-window mean fusion."""

    def __init__(
        self,
        channels: int = 32,
        samples: int = 250,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = CEVRCumulativeEncoder(
            channels=channels,
            max_samples=samples,
            embedding_dim=embedding_dim,
        )
        self.adapter = nn.Linear(embedding_dim, embedding_dim)

    def encode_trajectory(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose CEVR evidence dynamics without changing its fused prediction.

        ``accumulator_increments`` are exact additions between nested temporal
        prefixes before the nonlinear state update.  They let the early gate
        test marginal evidence without claiming that nested windows are
        independent observations.
        """

        horizon_states, accumulators = self.encoder.encode_trajectory(eeg)
        stage_embeddings = F.normalize(self.adapter(horizon_states), dim=-1)
        accumulator_increments = torch.cat(
            (accumulators[:, :1], accumulators[:, 1:] - accumulators[:, :-1]), dim=1
        )
        fused = F.normalize(self.adapter(horizon_states.mean(dim=1)), dim=-1)
        return {
            "fused_embedding": fused,
            "horizon_states": horizon_states,
            "stage_embeddings": stage_embeddings,
            "accumulators": accumulators,
            "accumulator_increments": accumulator_increments,
        }

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.encode_trajectory(eeg)["fused_embedding"]


def multipositive_retrieval_loss(
    predicted: torch.Tensor,
    target_rows: torch.Tensor,
    embedding_bank: torch.Tensor,
    temperature: float = 0.07,
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    """Contrast against unique batch images without treating repeats as negatives."""
    unique_rows, labels = torch.unique(target_rows, sorted=True, return_inverse=True)
    candidates = F.normalize(embedding_bank.index_select(0, unique_rows), dim=-1)
    logits = predicted @ candidates.transpose(0, 1) / temperature
    contrastive = F.cross_entropy(logits, labels)
    paired = candidates.index_select(0, labels)
    cosine = 1.0 - (predicted * paired).sum(dim=-1).mean()
    return contrastive + cosine_weight * cosine


@torch.no_grad()
def retrieval_topk(
    predicted: torch.Tensor,
    target_labels: torch.Tensor,
    candidates: torch.Tensor,
    topk: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Return single-trial cosine-retrieval accuracy for candidate labels."""
    requested = sorted(set(int(k) for k in topk))
    if not requested or requested[0] < 1:
        raise ValueError("topk values must be positive")
    maximum = min(requested[-1], candidates.shape[0])
    scores = F.normalize(predicted, dim=-1) @ F.normalize(candidates, dim=-1).T
    ranking = scores.topk(maximum, dim=1).indices
    result: dict[str, float] = {}
    for k in requested:
        effective = min(k, candidates.shape[0])
        correct = ranking[:, :effective].eq(target_labels[:, None]).any(dim=1)
        result[f"top{k}"] = float(correct.float().mean().item())
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
