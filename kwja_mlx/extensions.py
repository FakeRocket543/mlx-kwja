"""KWJA-specific extensions to OpenMed's DeBERTa-v2 MLX implementation.

Adds:
1. ConvLayer (1D conv for local context mixing after embeddings)
2. Two-layer classifier head (matching KWJA's tagger structure)
"""

import mlx.core as mx
import mlx.nn as nn


class ConvLayer(nn.Module):
    """1D Convolution layer used in DeBERTa-v2 encoder for local context."""
    
    def __init__(self, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        # Conv1D: (hidden_size, kernel_size, hidden_size) weight format from KWJA
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.LayerNorm = nn.LayerNorm(hidden_size)
    
    def __call__(self, x):
        # x: (batch, seq_len, hidden) → transpose for conv1d → back
        out = self.conv(x)  # MLX Conv1d expects (B, L, C) — NLC format
        out = self.LayerNorm(out)
        out = nn.gelu(out)
        return out


class TwoLayerClassifier(nn.Module):
    """Two-layer classifier head matching KWJA's tagger structure.
    
    layers.0 = Linear(hidden → hidden)  (dense)
    layers.3 = Linear(hidden → num_labels) (classifier)
    """
    
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def __call__(self, x):
        x = nn.gelu(self.dense(x))
        return self.classifier(x)
