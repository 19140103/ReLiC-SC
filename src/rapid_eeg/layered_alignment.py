"""Layered EEG-to-vision alignment pilots for Alljoined-1.6M."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from rapid_eeg.p1_baseline import CEVRCumulativeEncoder


class MultiScaleEEGEncoder(nn.Module):
    """Parallel temporal receptive fields that return one token per scale."""

    def __init__(
        self,
        channels: int = 32,
        samples: int = 250,
        token_dim: int = 512,
        branch_filters: int = 16,
        kernels: tuple[int, ...] = (7, 15, 31, 47, 63, 95, 127, 191),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.kernels = tuple(int(k) for k in kernels)
        if not self.kernels or any(k < 1 for k in self.kernels):
            raise ValueError("kernels must be a non-empty tuple of positive integers")
        self.branches = nn.ModuleList()
        self.projections = nn.ModuleList()
        for kernel in self.kernels:
            padding = kernel // 2
            branch = nn.Sequential(
                nn.Conv2d(1, branch_filters, kernel_size=(1, kernel), padding=(0, padding), bias=False),
                nn.BatchNorm2d(branch_filters),
                nn.GELU(),
                nn.Conv2d(
                    branch_filters,
                    branch_filters,
                    kernel_size=(channels, 1),
                    groups=branch_filters,
                    bias=False,
                ),
                nn.BatchNorm2d(branch_filters),
                nn.GELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(
                    branch_filters,
                    branch_filters,
                    kernel_size=(1, 15),
                    padding=(0, 7),
                    groups=branch_filters,
                    bias=False,
                ),
                nn.Conv2d(branch_filters, branch_filters, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_filters),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 8)),
                nn.Dropout(dropout),
            )
            with torch.no_grad():
                width = int(branch(torch.zeros(1, 1, channels, samples)).numel())
            self.branches.append(branch)
            self.projections.append(nn.Linear(width, token_dim))
        self.token_norm = nn.LayerNorm(token_dim)

    @property
    def token_count(self) -> int:
        return len(self.kernels)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG shaped [batch, channels, samples], got {eeg.shape}")
        x = eeg.unsqueeze(1)
        tokens = [
            projection(branch(x).flatten(1))
            for branch, projection in zip(self.branches, self.projections, strict=True)
        ]
        return F.normalize(self.token_norm(torch.stack(tokens, dim=1)), dim=-1)


class DepthTemporalBlock(nn.Module):
    """Residual temporal block used by DepthEEGEncoder."""

    def __init__(self, width: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=kernel, padding=padding, dilation=dilation, groups=width, bias=False),
            nn.Conv1d(width, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DepthEEGEncoder(nn.Module):
    """Stacked temporal encoder that exposes intermediate hidden states."""

    def __init__(
        self,
        channels: int = 32,
        token_dim: int = 512,
        width: int = 96,
        layers: int = 8,
        kernel: int = 15,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.input_projection = nn.Sequential(
            nn.Conv1d(channels, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        dilations = [1, 1, 2, 2, 4, 4, 8, 8]
        while len(dilations) < layers:
            dilations.extend(dilations)
        self.blocks = nn.ModuleList(
            DepthTemporalBlock(width, kernel=kernel, dilation=dilations[index], dropout=dropout)
            for index in range(layers)
        )
        self.projections = nn.ModuleList(nn.Linear(width, token_dim) for _ in range(layers))
        self.token_norm = nn.LayerNorm(token_dim)

    @property
    def token_count(self) -> int:
        return len(self.blocks)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG shaped [batch, channels, samples], got {eeg.shape}")
        x = self.input_projection(eeg)
        tokens: list[torch.Tensor] = []
        for block, projection in zip(self.blocks, self.projections, strict=True):
            x = block(x)
            pooled = x.mean(dim=-1)
            tokens.append(projection(pooled))
        return F.normalize(self.token_norm(torch.stack(tokens, dim=1)), dim=-1)


class CEVRStageEEGEncoder(nn.Module):
    """CEVR cumulative horizons exposed as eight EEG evidence tokens."""

    def __init__(
        self,
        channels: int = 32,
        samples: int = 250,
        token_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        horizons: tuple[int, ...] = (25, 50, 75, 100, 125, 150, 200, 250),
    ) -> None:
        super().__init__()
        self.horizons = tuple(int(value) for value in horizons)
        self.encoder = CEVRCumulativeEncoder(
            channels=channels,
            hidden_dim=hidden_dim,
            embedding_dim=token_dim,
            max_samples=samples,
            dropout=dropout,
        )
        self.adapter = nn.Linear(token_dim, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)

    @property
    def token_count(self) -> int:
        return len(self.horizons)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.encoder.encode_trajectory(eeg, self.horizons)
        return F.normalize(self.token_norm(self.adapter(tokens)), dim=-1)


class LayeredAlignmentModel(nn.Module):
    """Scores EEG tokens against layered visual embeddings with learned alignment."""

    def __init__(
        self,
        encoder: nn.Module,
        image_layers: int = 8,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if not hasattr(encoder, "token_count"):
            raise TypeError("encoder must expose token_count")
        self.encoder = encoder
        self.image_layers = int(image_layers)
        self.temperature = float(temperature)
        self.alignment_logits = nn.Parameter(torch.zeros(encoder.token_count, image_layers))

    def alignment(self) -> torch.Tensor:
        return F.softmax(self.alignment_logits.flatten(), dim=0).view_as(self.alignment_logits)

    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.encoder(eeg)

    def score_tokens(self, eeg_tokens: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        if eeg_tokens.ndim != 3:
            raise ValueError("eeg_tokens must be shaped [batch, eeg_layers, dim]")
        if image_tokens.ndim != 3:
            raise ValueError("image_tokens must be shaped [items, image_layers, dim]")
        if eeg_tokens.shape[-1] != image_tokens.shape[-1]:
            raise ValueError("EEG and image token dimensions differ")
        if eeg_tokens.shape[1] != self.alignment_logits.shape[0]:
            raise ValueError("EEG token count does not match alignment matrix")
        if image_tokens.shape[1] != self.image_layers:
            raise ValueError("Image token count does not match alignment matrix")
        eeg_tokens = F.normalize(eeg_tokens, dim=-1)
        image_tokens = F.normalize(image_tokens, dim=-1)
        pair_scores = torch.einsum("bed,nld->benl", eeg_tokens, image_tokens)
        return torch.einsum("benl,el->bn", pair_scores, self.alignment())

    def forward(self, eeg: torch.Tensor, candidate_image_tokens: torch.Tensor) -> torch.Tensor:
        return self.score_tokens(self.encode(eeg), candidate_image_tokens) / self.temperature


def layered_multipositive_loss(
    model: LayeredAlignmentModel,
    eeg: torch.Tensor,
    target_rows: torch.Tensor,
    layered_bank: torch.Tensor,
) -> torch.Tensor:
    """Contrast each EEG trial against unique image candidates in its batch."""

    unique_rows, labels = torch.unique(target_rows, sorted=True, return_inverse=True)
    candidates = layered_bank.index_select(0, unique_rows)
    logits = model(eeg, candidates)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def topk_from_scores(
    scores: torch.Tensor,
    target_labels: torch.Tensor,
    topk: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    requested = sorted(set(int(k) for k in topk))
    if not requested or requested[0] < 1:
        raise ValueError("topk values must be positive")
    maximum = min(requested[-1], scores.shape[1])
    ranking = scores.topk(maximum, dim=1).indices
    result: dict[str, float] = {}
    for k in requested:
        effective = min(k, scores.shape[1])
        correct = ranking[:, :effective].eq(target_labels[:, None]).any(dim=1)
        result[f"top{k}"] = float(correct.float().mean().item())
    target_scores = scores.gather(1, target_labels[:, None])
    result["mean_rank"] = float((scores > target_scores).sum(dim=1).float().add(1).mean().item())
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
