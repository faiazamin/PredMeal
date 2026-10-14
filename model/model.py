"""User tower plus one learned vector per recipe.

Scores are dot products over the full catalog. Nothing masks a recipe out.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from schema import feature_dim


class UserTower(nn.Module):
    def __init__(self, input_dim: int | None = None, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        input_dim = feature_dim() if input_dim is None else input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=-1)


class MealRanker(nn.Module):
    def __init__(self, recipe_count: int, input_dim: int | None = None, embed_dim: int = 32):
        super().__init__()
        self.user_tower = UserTower(input_dim=input_dim, embed_dim=embed_dim)
        self.recipe_embeddings = nn.Embedding(recipe_count, embed_dim)
        # Cosine scores sit in [-1, 1]. A scale lets the softmax peak on the top grades.
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def recipe_vectors(self) -> torch.Tensor:
        return F.normalize(self.recipe_embeddings.weight, dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        cosine = self.user_tower(features) @ self.recipe_vectors().T
        return cosine * self.logit_scale.clamp(1.0, 100.0)
