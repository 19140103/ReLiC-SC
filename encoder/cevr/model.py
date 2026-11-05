from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedCausalAdditiveEncoder(nn.Module):
    """Shared cumulative-window encoder used by the ReLiC-SC experiments."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        embed_dim: int,
        max_samples: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if max_samples < 1:
            raise ValueError("max_samples must be positive.")
        self.hidden_dim = int(hidden_dim)
        self.max_samples = int(max_samples)
        self.temporal_weight = nn.Parameter(torch.empty(hidden_dim, max_samples, channels))
        self.temporal_bias = nn.Parameter(torch.empty(hidden_dim))
        nn.init.kaiming_uniform_(self.temporal_weight.view(hidden_dim, -1), a=math.sqrt(5))
        bound = 1.0 / math.sqrt(max_samples * channels)
        nn.init.uniform_(self.temporal_bias, -bound, bound)
        self.state_update = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.shared_projection = nn.Linear(hidden_dim, embed_dim)

    def forward(self, eeg: torch.Tensor, window_ends: list[int]) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG shape (B,C,T), got {tuple(eeg.shape)}.")
        if not window_ends:
            raise ValueError("At least one cumulative EEG window endpoint is required.")
        if min(window_ends) < 1 or max(window_ends) > eeg.shape[-1]:
            raise ValueError(f"Invalid window endpoints {window_ends} for EEG length {eeg.shape[-1]}.")
        if max(window_ends) > self.max_samples:
            raise ValueError(f"Endpoint {max(window_ends)} exceeds configured max_samples={self.max_samples}.")
        if any(left >= right for left, right in zip(window_ends, window_ends[1:])):
            raise ValueError("Cumulative EEG window endpoints must be strictly increasing.")

        cumulative = self.temporal_bias.unsqueeze(0).expand(eeg.shape[0], -1)
        states = []
        previous_end = 0
        for end in window_ends:
            segment = eeg[:, :, previous_end:end].transpose(1, 2).reshape(eeg.shape[0], -1)
            weights = self.temporal_weight[:, previous_end:end, :].reshape(self.hidden_dim, -1)
            cumulative = cumulative + F.linear(segment, weights)
            state = cumulative + self.state_update(cumulative)
            states.append(self.output_norm(state))
            previous_end = int(end)
        return self.shared_projection(torch.stack(states, dim=1))


class ParameterMatchedSlotHeads(nn.Module):
    """Compatibility heads retained to reproduce the paper parameter count."""

    def __init__(self, embed_dim: int, num_heads: int, num_slots: int, dropout: float):
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(num_slots, embed_dim) * 0.02)
        self.slot_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.slot_norm = nn.LayerNorm(embed_dim)
        self.slot_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(num_slots)
            ]
        )
        self.assignment_logits = nn.Parameter(torch.eye(num_slots) * 2.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        queries = self.slot_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        slots, _ = self.slot_attention(queries, tokens, tokens, need_weights=False)
        slots = self.slot_norm(slots)
        return torch.stack([head(slots[:, idx, :]) for idx, head in enumerate(self.slot_heads)], dim=1)


class CEVRModel(nn.Module):
    def __init__(
        self,
        channels: int = 17,
        embed_dim: int = 512,
        window_ends: list[int] | None = None,
        additive_hidden_dim: int = 1024,
        dropout: float = 0.1,
        eeg_fusion_mode: str = "mean",
        current_slot_weight: float = 0.0,
        parameter_matched_heads: bool = True,
        num_heads: int = 8,
        num_slots: int = 5,
    ):
        super().__init__()
        self.window_ends = [25, 50, 75, 100, 125, 150, 200, 250] if window_ends is None else [int(x) for x in window_ends]
        self.eeg_fusion_mode = str(eeg_fusion_mode)
        if self.eeg_fusion_mode not in {"mean", "last"}:
            raise ValueError(f"Unknown eeg_fusion_mode: {self.eeg_fusion_mode}")
        self.current_slot_weight = float(current_slot_weight)
        self.encoder = SharedCausalAdditiveEncoder(
            channels=channels,
            hidden_dim=additive_hidden_dim,
            embed_dim=embed_dim,
            max_samples=max(self.window_ends),
            dropout=dropout,
        )
        self.window_heads = nn.ModuleList([nn.Identity() for _ in self.window_ends])
        self.fusion = nn.Linear(embed_dim, embed_dim)
        self.slot_heads = (
            ParameterMatchedSlotHeads(embed_dim, num_heads, num_slots, dropout)
            if parameter_matched_heads
            else None
        )

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        window_tokens = self.encoder(eeg, self.window_ends)
        window_outputs = [head(window_tokens[:, idx, :]) for idx, head in enumerate(self.window_heads)]
        if self.eeg_fusion_mode == "last":
            fused_input = window_tokens[:, -1, :]
        else:
            fused_input = window_tokens.mean(dim=1)
        current = self.fusion(fused_input)
        if self.slot_heads is not None and self.current_slot_weight != 0.0:
            slots = self.slot_heads(window_tokens)
            current = current + self.current_slot_weight * slots[:, 2, :]
        return {
            "current": current,
            "window_embeddings": window_tokens,
            "window_outputs": window_outputs,
        }

    def eeg_component_weights(self) -> list[float]:
        if self.eeg_fusion_mode == "last":
            return [0.0] * (len(self.window_ends) - 1) + [1.0]
        return [1.0 / len(self.window_ends)] * len(self.window_ends)


class MultiLevelVisualProjector(nn.Module):
    """Global visual target, with optional G adapter and H stage supervision.

    Stage targets mix the *raw* visual layers before the shared stage
    projection. They are separate from the projected layers used by the
    global target. Returned targets are unnormalized; the contrastive loss
    applies normalization to both sides.
    """

    def __init__(
        self,
        image_dim: int = 512,
        embed_dim: int = 512,
        num_layers: int = 8,
        fusion_indices: list[int] | None = None,
        parameter_matched_heads: bool = True,
        num_relations: int = 5,
        current_target_adapter: str = "linear",
        current_target_bottleneck_dim: int = 128,
        current_target_dropout: float = 0.0,
        num_eeg_levels: int | None = None,
        stage_calibration_mode: str = "none",
        stage_calibration_temperature: float = 1.0,
        stage_calibration_init_indices: list[int] | None = None,
        stage_calibration_init_strength: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        self.image_dim = int(image_dim)
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.current_target_adapter = str(current_target_adapter)
        if self.current_target_adapter not in {"linear", "bottleneck"}:
            raise ValueError(f"Unknown current_target_adapter: {self.current_target_adapter}")
        self.current_target_bottleneck_dim = int(current_target_bottleneck_dim)
        if self.current_target_bottleneck_dim < 1:
            raise ValueError("current_target_bottleneck_dim must be positive.")
        self.current_target_dropout = float(current_target_dropout)
        if not 0.0 <= self.current_target_dropout <= 1.0:
            raise ValueError("current_target_dropout must be between 0 and 1.")
        self.num_eeg_levels = self.num_layers if num_eeg_levels is None else int(num_eeg_levels)
        if self.num_eeg_levels < 1:
            raise ValueError("num_eeg_levels must be positive.")
        self.stage_calibration_mode = str(stage_calibration_mode)
        if self.stage_calibration_mode not in {"none", "stage_scalar"}:
            raise ValueError(f"Unknown stage_calibration_mode: {self.stage_calibration_mode}")
        self.stage_calibration_temperature = float(stage_calibration_temperature)
        if not math.isfinite(self.stage_calibration_temperature) or self.stage_calibration_temperature <= 0.0:
            raise ValueError("stage_calibration_temperature must be finite and positive.")
        self.fusion_indices = list(range(num_layers)) if fusion_indices is None else [int(x) for x in fusion_indices]
        if not self.fusion_indices:
            raise ValueError("At least one visual fusion index is required.")
        for index in self.fusion_indices:
            if index < 0 or index >= num_layers:
                raise ValueError(f"Visual fusion index {index} is out of range for {num_layers} layers.")
        self.level_projectors = nn.ModuleList([nn.Linear(image_dim, embed_dim) for _ in range(num_layers)])
        if self.current_target_adapter == "linear":
            self.current_fusion_projector = nn.Linear(embed_dim, embed_dim)
        else:
            self.current_fusion_projector = nn.Sequential(
                nn.Linear(embed_dim, self.current_target_bottleneck_dim),
                nn.GELU(),
                nn.Dropout(self.current_target_dropout),
                nn.Linear(self.current_target_bottleneck_dim, embed_dim),
            )
        if parameter_matched_heads or self.stage_calibration_mode != "none":
            self.stage_calibration_projector = nn.Linear(image_dim, embed_dim)
        else:
            self.stage_calibration_projector = None
        if parameter_matched_heads:
            self.relation_level_logits = nn.Parameter(torch.zeros(num_relations, num_layers))
        else:
            self.register_parameter("relation_level_logits", None)
        if self.stage_calibration_mode == "stage_scalar":
            self.stage_calibration_logits = nn.Parameter(torch.zeros(self.num_eeg_levels, self.num_layers))
            with torch.no_grad():
                for stage, layer in enumerate((stage_calibration_init_indices or [])[: self.num_eeg_levels]):
                    if not 0 <= int(layer) < self.num_layers:
                        raise ValueError(f"Stage calibration index {layer} is out of range for {self.num_layers} layers.")
                    self.stage_calibration_logits[stage, int(layer)] = float(stage_calibration_init_strength)
        else:
            self.register_parameter("stage_calibration_logits", None)

    def project_layers(self, visual_features: torch.Tensor) -> torch.Tensor:
        if visual_features.ndim != 3:
            raise ValueError(f"Expected multi-level visual feature shape (B,L,D), got {tuple(visual_features.shape)}.")
        if visual_features.shape[1:] != (self.num_layers, self.image_dim):
            raise ValueError(f"Expected {self.num_layers} visual layers of dimension {self.image_dim}, got {tuple(visual_features.shape)}.")
        return torch.stack(
            [projector(visual_features[:, idx, :]) for idx, projector in enumerate(self.level_projectors)],
            dim=1,
        )

    def forward(self, visual_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.project_layers(visual_features)
        indices = torch.tensor(self.fusion_indices, device=projected.device, dtype=torch.long)
        fused = projected.index_select(1, indices).mean(dim=1)
        return self.current_fusion_projector(fused), projected

    def calibrate_stage_targets(
        self, visual_features: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return H's raw-layer mixture targets and learned stage/layer weights."""
        if self.stage_calibration_mode == "none":
            return None, None
        if visual_features.ndim != 3 or visual_features.shape[1:] != (self.num_layers, self.image_dim):
            raise ValueError(
                f"Expected multi-level visual features (B,{self.num_layers},{self.image_dim}), "
                f"got {tuple(visual_features.shape)}."
            )
        assert self.stage_calibration_logits is not None
        assert self.stage_calibration_projector is not None
        temperature = max(self.stage_calibration_temperature, 1e-6)
        weights = F.softmax(self.stage_calibration_logits / temperature, dim=1).to(
            device=visual_features.device, dtype=visual_features.dtype
        )
        mixed = torch.einsum("tl,bld->btd", weights, visual_features)
        return self.stage_calibration_projector(mixed), weights

    @torch.no_grad()
    def encode_numpy(self, visual_features: torch.Tensor | "np.ndarray", device: torch.device, batch_size: int = 1024):
        import numpy as np

        self.eval()
        out = []
        for start in range(0, visual_features.shape[0], batch_size):
            chunk = torch.as_tensor(visual_features[start : start + batch_size], dtype=torch.float32, device=device)
            fused, _ = self(chunk)
            out.append(fused.detach().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32, copy=False)

    def fusion_weights(self) -> list[float]:
        weights = torch.zeros(self.num_layers)
        indices = torch.tensor(self.fusion_indices, dtype=torch.long)
        weights[indices] = 1.0 / max(len(self.fusion_indices), 1)
        return weights.numpy().astype(float).tolist()
