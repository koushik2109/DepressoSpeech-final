from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(0.2),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class QuestionHead(nn.Module):
    def __init__(self, input_dim: int, num_questions: int = 8):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(0.2),
            nn.Linear(input_dim, num_questions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(0.2),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(0.2),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)
