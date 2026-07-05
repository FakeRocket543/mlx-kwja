"""KWJA-specific extensions to OpenMed's DeBERTa-v2 MLX implementation.

Adds:
1. ConvLayer (1D conv for local context mixing after embeddings)
2. Two-layer classifier head (matching KWJA's tagger structure)
"""

import mlx.core as mx
import mlx.nn as nn


class ConvLayer(nn.Module):
    """1D Convolution layer used in DeBERTa-v2 encoder for local context.
    
    Matches HuggingFace transformers ConvLayer:
    1. Conv1d(hidden, hidden, kernel=3, padding=1)
    2. tanh activation
    3. Residual: LayerNorm(residual + conv_output)
    """
    
    def __init__(self, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.LayerNorm = nn.LayerNorm(hidden_size)
    
    def __call__(self, hidden_states, residual_states, input_mask=None):
        # Conv1d: MLX expects (B, L, C) - NLC format
        # But Conv1d weight from PyTorch is (out, kernel, in) → need transpose
        out = self.conv(hidden_states)
        out = mx.tanh(out)
        # Residual + LayerNorm
        output = self.LayerNorm(residual_states + out)
        # Apply mask
        if input_mask is not None:
            mask = input_mask[:, :, None] if input_mask.ndim == 2 else input_mask
            output = output * mask
        return output


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
