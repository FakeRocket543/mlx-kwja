"""Weight-free unit tests for kwja_mlx.cohesion (head shapes + decoding)."""
import math

import mlx.core as mx
import numpy as np
import pytest

from kwja_mlx.cohesion import (
    LoRARelationWiseWordSelectionHead,
    WordSelectionHead,
    analyze_heads,
    decode_cohesion,
    decode_discourse,
)
from kwja_mlx.constants import (
    COHESION_RELATIONS,
    DISCOURSE_RELATIONS,
    EXOPHORA_REFERENTS,
    SPECIALS,
)


class TestLabelTables:
    def test_cohesion_relations_count(self):
        """9 relations: 7 PAS cases + bridging ノ + coreference =."""
        assert len(COHESION_RELATIONS) == 9
        assert COHESION_RELATIONS == ('ガ', 'ヲ', 'ニ', 'ガ２', 'デ', 'ト', '時間', 'ノ', '=')

    def test_discourse_relations(self):
        assert len(DISCOURSE_RELATIONS) == 7
        assert DISCOURSE_RELATIONS[0] == '談話関係なし'

    def test_specials(self):
        assert len(SPECIALS) == 7
        assert {f'[{ref}]' for ref in EXOPHORA_REFERENTS} <= set(SPECIALS)
        assert {'[NULL]', '[NA]', '[ROOT]'} <= set(SPECIALS)


class TestHeadShapes:
    HIDDEN = 32

    def test_cohesion_logits_shape(self):
        """cohesion_analyzer emits (b, n, n, 9)."""
        head = LoRARelationWiseWordSelectionHead(9, self.HIDDEN, rank=1)
        pooled = mx.random.normal((2, 5, self.HIDDEN))
        logits = head(pooled)
        assert logits.shape == (2, 5, 5, 9)

    def test_discourse_logits_shape(self):
        """discourse_relation_analyzer emits (b, n, n, 7)."""
        head = WordSelectionHead(7, self.HIDDEN)
        pooled = mx.random.normal((2, 5, self.HIDDEN))
        logits = head(pooled)
        assert logits.shape == (2, 5, 5, 7)

    def test_cohesion_head_matches_formula(self):
        """logits[i,j,r] = cls[r]·GELU((Ws+Δs[r])hi + (Wt+Δt[r])hj)."""
        rng = np.random.default_rng(0)
        h = self.HIDDEN
        head = LoRARelationWiseWordSelectionHead(9, h, rank=1)
        weights = []
        Ws = rng.normal(size=(h, h)).astype(np.float32)
        Wt = rng.normal(size=(h, h)).astype(np.float32)
        da_s = rng.normal(size=(h, 1, 9)).astype(np.float32)
        db_s = rng.normal(size=(1, h, 9)).astype(np.float32)
        da_t = rng.normal(size=(h, 1, 9)).astype(np.float32)
        db_t = rng.normal(size=(1, h, 9)).astype(np.float32)
        cls = rng.normal(size=(h, 9)).astype(np.float32)
        weights = [
            ('l_source.weight', Ws), ('l_source.bias', np.zeros(h, np.float32)),
            ('l_target.weight', Wt), ('l_target.bias', np.zeros(h, np.float32)),
            ('delta_source.dense_a', da_s), ('delta_source.dense_b', db_s),
            ('delta_target.dense_a', da_t), ('delta_target.dense_b', db_t),
            ('classifier', cls),
        ]
        head.load_weights([(k, mx.array(v)) for k, v in weights])

        pooled_np = rng.normal(size=(4, h)).astype(np.float32)
        pooled = mx.array(pooled_np[None])
        logits = np.array(head(pooled)[0])  # (n, n, 9)

        gelu = np.vectorize(lambda x: 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0))))
        hs = pooled_np @ Ws.T  # (n, h)
        ht = pooled_np @ Wt.T
        ds = np.einsum('hrl,ril->hil', da_s, db_s)  # (h, h, 9)
        dt = np.einsum('hrl,ril->hil', da_t, db_t)
        for r in range(9):
            src = hs + pooled_np @ ds[:, :, r]  # (n, h)
            tgt = ht + pooled_np @ dt[:, :, r]
            for i in range(4):
                for j in range(4):
                    hidden = gelu(src[i] + tgt[j])
                    expected = float(np.sum(cls[:, r] * hidden))
                    assert logits[i, j, r] == pytest.approx(expected, abs=1e-4)


class TestCohesionDecode:
    N = 3

    def _logits(self):
        """(n+7, n+7, 9); every (target, rel) defaults to argmax [NULL] (no
        relation) so a raised 2.0 cell is the sole decoded entry."""
        logits = np.full((self.N + len(SPECIALS),) * 2 + (9,), -1.0, dtype=np.float32)
        logits[:, self.N + SPECIALS.index('[NULL]'), :] = 1.0
        return logits

    def test_argmax_on_morpheme_is_endophora(self):
        logits = self._logits()
        logits[0, 2, 0] = 2.0  # target 0, ガ (rel 0) → morpheme 2
        result = decode_cohesion(logits, self.N)
        assert len(result) == self.N
        assert result[0] == {'ガ': 2}

    def test_argmax_on_special_is_exophora(self):
        logits = self._logits()
        author_col = self.N + SPECIALS.index('[著者]')
        logits[1, author_col, 8] = 2.0  # target 1, = (rel 8) → 著者
        result = decode_cohesion(logits, self.N)
        assert result[1] == {'=': '著者'}
        assert '著者' in EXOPHORA_REFERENTS

    def test_argmax_on_null_is_no_relation(self):
        logits = self._logits()
        null_col = self.N + SPECIALS.index('[NULL]')
        logits[2, null_col, 3] = 2.0  # target 2, ガ２ (rel 3) → [NULL] → absent
        result = decode_cohesion(logits, self.N)
        assert result[2] == {}

    def test_all_specials_decode(self):
        rel = COHESION_RELATIONS[5]  # ト
        for special in SPECIALS:
            logits = self._logits()
            col = self.N + SPECIALS.index(special)
            logits[0, col, 5] = 2.0
            result = decode_cohesion(logits, self.N)
            if special[1:-1] in EXOPHORA_REFERENTS:
                assert result[0] == {rel: special[1:-1]}
            else:
                assert result[0] == {}


class TestDiscourseDecode:
    N = 2

    def test_argmax_none_class_omitted(self):
        logits = np.zeros((self.N, self.N, 7), dtype=np.float32)
        logits[:, :, 0] = 1.0  # 談話関係なし everywhere
        assert decode_discourse(logits, self.N, threshold=0.0) == []

    def test_argmax_relation_kept_with_score(self):
        logits = np.zeros((self.N, self.N, 7), dtype=np.float32)
        logits[0, 1, 2] = 2.0  # modifier 0 → head 1: 目的
        result = decode_discourse(logits, self.N, threshold=0.0)
        assert len(result) == 1
        assert result[0]['modifier'] == 0
        assert result[0]['head'] == 1
        assert result[0]['relation'] == '目的'
        expected = math.exp(2.0) / (math.exp(2.0) + 6.0)
        assert result[0]['score'] == pytest.approx(expected, rel=1e-5)

    def test_threshold_forces_none(self):
        """max prob < threshold forces 談話関係なし (kwja predict_step)."""
        logits = np.zeros((self.N, self.N, 7), dtype=np.float32)
        logits[0, 1, 2] = 2.0  # p(目的) = e²/(e²+6) ≈ 0.577
        assert decode_discourse(logits, self.N, threshold=0.7) == []
        # below threshold they survive
        assert len(decode_discourse(logits, self.N, threshold=0.5)) == 1


class TestAnalyzeHeadsWiring:
    def test_returns_per_morpheme_and_sentence_lists(self):
        hidden = 16
        n = 3
        cohesion_head = LoRARelationWiseWordSelectionHead(9, hidden, rank=1)
        discourse_head = WordSelectionHead(7, hidden)
        pooled = mx.random.normal((1, n + len(SPECIALS), hidden))
        mx.eval(cohesion_head.parameters(), discourse_head.parameters())
        cohesion, discourse = analyze_heads(cohesion_head, discourse_head, pooled, n)
        assert len(cohesion) == n
        assert all(isinstance(c, dict) for c in cohesion)
        assert isinstance(discourse, list)
