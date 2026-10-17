"""
FiLM Generator Heads for IDM-VTON FiLM Extension.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class FiLMGeneratorHead(nn.Module):
    """Single FiLM head: maps a measurement embedding → (γ, β) for one feature map."""

    def __init__(self, input_dim: int, channel_dim: int) -> None:
        """
        Args:
            input_dim:   dimension of the input measurement embedding (e.g. 2048 for SDXL)
            channel_dim: number of feature-map channels to modulate (e.g. 1280)
        """
        super().__init__()
        self.gamma_proj = nn.Linear(input_dim, channel_dim)
        self.beta_proj  = nn.Linear(input_dim, channel_dim)
        # Near-identity init: γ ≈ 1, β ≈ 0
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embedding: [B, input_dim]
        Returns:
            gamma: [B, channel_dim]
            beta:  [B, channel_dim]
        """
        return self.gamma_proj(embedding), self.beta_proj(embedding)


class FiLMGeneratorSet(nn.Module):
    """Collection of FiLM generator heads, one per target ResNet block."""

    def __init__(self, input_dim: int, channel_dims: List[int]) -> None:
        """
        Args:
            input_dim:    measurement embedding dimension shared by all heads
            channel_dims: list of channel counts, one entry per target ResNet block
        """
        super().__init__()
        self.heads = nn.ModuleList([
            FiLMGeneratorHead(input_dim, c) for c in channel_dims
        ])

    def __len__(self) -> int:
        return len(self.heads)
