"""KWJA-specific extensions to OpenMed's DeBERTa-v2 MLX implementation.

Adds:
1. ConvLayer (1D conv for local context mixing after embeddings)
2. Two-layer classifier head (matching KWJA's tagger structure)
"""

import mlx.core as mx
import mlx.nn as nn


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
