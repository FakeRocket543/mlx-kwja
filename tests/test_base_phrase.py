"""Weight-free unit tests for base-phrase feature head + clause decoding."""

import mlx.core as mx
import numpy as np

from kwja_mlx.analyzer import LoRAMultiLabelHead, _decode_clauses
from kwja_mlx.constants import BASE_PHRASE_FEATURES


def _bp_head():
    h = LoRAMultiLabelHead(hidden_size=16, num_labels=len(BASE_PHRASE_FEATURES), rank=4)
    assert len(BASE_PHRASE_FEATURES) == 64
    return h


class TestBasePhraseHeadShape:
    def test_output_shape_and_range(self):
        head = _bp_head()
        x = mx.array(np.random.RandomState(0).randn(2, 5, 16).astype(np.float32))
        probs = np.array(head(x))
        assert probs.shape == (2, 5, 64)
        assert ((probs >= 0.0) & (probs <= 1.0)).all()


def _one_hot_bp(n_words, hot: dict[int, list[str]]):
    """Build (n_words, 64) prob array with listed features hot per word."""
    arr = np.zeros((n_words, len(BASE_PHRASE_FEATURES)), dtype=np.float32)
    for i, feats in hot.items():
        for f in feats:
            arr[i, BASE_PHRASE_FEATURES.index(f)] = 0.9
    return arr


class TestDecodeClauses:
    def test_single_clause_no_boundary(self):
        arr = _one_hot_bp(3, {})
        arr[:, BASE_PHRASE_FEATURES.index("節-主辞")] = [0.1, 0.9, 0.2]
        assert _decode_clauses(3, arr) == [(0, 3, 1)]

    def test_boundary_splits_and_heads(self):
        arr = _one_hot_bp(4, {1: ["節-区切"], 3: ["節-区切:連体修飾"]})
        arr[:, BASE_PHRASE_FEATURES.index("節-主辞")] = [0.8, 0.1, 0.2, 0.9]
        assert _decode_clauses(4, arr) == [(0, 2, 0), (2, 4, 3)]

    def test_empty(self):
        arr = np.zeros((0, 64), dtype=np.float32)
        assert _decode_clauses(0, arr) == []
