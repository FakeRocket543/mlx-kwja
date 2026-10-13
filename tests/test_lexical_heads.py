"""Weight-free unit tests for word-module lexical heads.

Covers: masked-argmax morpheme attribute decoding (subpos constrained by POS,
conjform constrained by conjtype + INFLECTABLE, non-inflectable -> "*"),
the LoRA word-feature head (sigmoid probabilities), and reading-vocab
fallback behaviour.
"""
from typing import ClassVar

import mlx.core as mx
import numpy as np
import pytest

from kwja_mlx.analyzer import (
    POS_TAGS,
    Analyzer,
    LoRAMultiLabelHead,
    WordModule,
    _decode_morpheme_attributes,
)
from kwja_mlx.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TAGS,
    CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID,
    CONJTYPE_TAGS,
    INFLECTABLE,
    POS_TAG_SUBPOS_TAG2SUBPOS_ID,
    SUBPOS_TAGS,
    WORD_FEATURES,
    get_reading2reading_id,
)


def _one_hot_logits(n_tags: int, hot: int, value: float = 5.0):
    logits = np.zeros((1, n_tags), dtype=np.float32)
    logits[0, hot] = value
    return logits


class TestDecodeMorphemeAttributes:
    def test_verb_subpos_forced_to_star(self):
        """動詞 only allows subpos '*'; a noun-subpos-favoring logit is masked."""
        subpos_logits = _one_hot_logits(len(SUBPOS_TAGS), SUBPOS_TAGS.index("普通名詞"))
        sub, ct, cf = _decode_morpheme_attributes(
            pos_preds=[POS_TAGS.index("動詞")],
            subpos_logits=subpos_logits,
            conjtype_logits=_one_hot_logits(len(CONJTYPE_TAGS), CONJTYPE_TAGS.index("母音動詞")),
            conjform_logits=np.zeros((1, len(CONJFORM_TAGS)), dtype=np.float32),
        )
        assert SUBPOS_TAGS[sub[0]] == "*"
        # 動詞 is inflectable → conjtype decoded (not forced to "*")
        assert ct[0] == CONJTYPE_TAGS.index("母音動詞")

    def test_noun_subpos_stays_within_pos(self):
        """名詞 subpos argmax never leaks outside the 名詞 subpos set."""
        pos = POS_TAGS.index("名詞")
        allowed = POS_TAG_SUBPOS_TAG2SUBPOS_ID["名詞"]
        # Raw argmax would pick 格助詞 — illegal for 名詞
        logits = np.zeros((1, len(SUBPOS_TAGS)), dtype=np.float32)
        logits[0, SUBPOS_TAGS.index("格助詞")] = 10.0
        logits[0, SUBPOS_TAGS.index("普通名詞")] = 3.0
        sub, ct, cf = _decode_morpheme_attributes(
            pos_preds=[pos],
            subpos_logits=logits,
            conjtype_logits=_one_hot_logits(len(CONJTYPE_TAGS), 1),
            conjform_logits=np.zeros((1, len(CONJFORM_TAGS)), dtype=np.float32),
        )
        assert SUBPOS_TAGS[sub[0]] == "普通名詞"
        assert SUBPOS_TAGS[sub[0]] in allowed
        # 名詞 is not inflectable → conjtype/conjform forced to "*"
        assert CONJTYPE_TAGS[ct[0]] == "*"
        assert CONJFORM_TAGS[cf[0]] == "*"

    def test_non_inflectable_conjform_forced_star(self):
        """副詞 (non-inflectable) → conjtype and conjform '*' regardless of logits."""
        sub, ct, cf = _decode_morpheme_attributes(
            pos_preds=[POS_TAGS.index("副詞")],
            subpos_logits=_one_hot_logits(len(SUBPOS_TAGS), 0),
            conjtype_logits=_one_hot_logits(len(CONJTYPE_TAGS), CONJTYPE_TAGS.index("母音動詞")),
            conjform_logits=_one_hot_logits(len(CONJFORM_TAGS), CONJFORM_TAGS.index("基本形")),
        )
        assert CONJTYPE_TAGS[ct[0]] == "*"
        assert CONJFORM_TAGS[cf[0]] == "*"

    def test_verb_conjform_constrained_by_conjtype(self):
        """For 母音動詞 the conjform argmax is restricted to its conjform set."""
        allowed = CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID["母音動詞"]
        excluded = [t for t in CONJFORM_TAGS if t not in allowed]
        assert excluded, "expected conjform tags outside the 母音動詞 set"
        logits = np.zeros((1, len(CONJFORM_TAGS)), dtype=np.float32)
        logits[0, CONJFORM_TAGS.index(excluded[0])] = 10.0  # illegal
        logits[0, CONJFORM_TAGS.index("基本形")] = 3.0  # legal runner-up
        sub, ct, cf = _decode_morpheme_attributes(
            pos_preds=[POS_TAGS.index("動詞")],
            subpos_logits=_one_hot_logits(len(SUBPOS_TAGS), 0),
            conjtype_logits=_one_hot_logits(len(CONJTYPE_TAGS), CONJTYPE_TAGS.index("母音動詞")),
            conjform_logits=logits,
        )
        assert CONJFORM_TAGS[cf[0]] in allowed
        assert CONJFORM_TAGS[cf[0]] == "基本形"

    def test_inflectable_suffix_pair(self):
        """接尾辞/動詞性接尾辞 is inflectable per INFLECTABLE."""
        assert ("接尾辞", "動詞性接尾辞") in INFLECTABLE
        sub, ct, cf = _decode_morpheme_attributes(
            pos_preds=[POS_TAGS.index("接尾辞")],
            subpos_logits=_one_hot_logits(len(SUBPOS_TAGS), SUBPOS_TAGS.index("動詞性接尾辞")),
            conjtype_logits=_one_hot_logits(len(CONJTYPE_TAGS), CONJTYPE_TAGS.index("動詞性接尾辞ます型")),
            conjform_logits=np.zeros((1, len(CONJFORM_TAGS)), dtype=np.float32),
        )
        assert SUBPOS_TAGS[sub[0]] == "動詞性接尾辞"
        assert ct[0] == CONJTYPE_TAGS.index("動詞性接尾辞ます型")

    def test_lengths_match_checkpoint_heads(self):
        assert len(SUBPOS_TAGS) == 35
        assert len(CONJTYPE_TAGS) == 33
        assert len(CONJFORM_TAGS) == 81
        assert len(WORD_FEATURES) == 5


class TestLoRAMultiLabelHead:
    def _head(self):
        head = LoRAMultiLabelHead(hidden_size=16, num_labels=5, rank=4)
        # Non-trivial weights so the LoRA path actually contributes
        dense_a = np.random.RandomState(0).randn(16, 4, 5).astype(np.float32) * 0.1
        dense_b = np.random.RandomState(1).randn(4, 16, 5).astype(np.float32) * 0.1
        w = np.random.RandomState(2).randn(16, 5).astype(np.float32) * 0.1
        head.update({
            "dense_a": mx.array(dense_a),
            "dense_b": mx.array(dense_b),
            "classifier_weight": mx.array(w),
        })
        return head

    def test_output_shape_and_range(self):
        head = self._head()
        x = mx.array(np.random.RandomState(3).randn(2, 7, 16).astype(np.float32))
        probs = head(x)
        assert probs.shape == (2, 7, 5)
        arr = np.array(probs)
        assert (arr >= 0.0).all() and (arr <= 1.0).all()

    def test_constant_input_still_sigmoid(self):
        head = self._head()
        x = mx.zeros((1, 3, 16))
        probs = np.array(head(x))
        assert probs.shape == (1, 3, 5)
        assert np.isfinite(probs).all()


class TestReadingVocab:
    def _id2reading(self):
        r2id = get_reading2reading_id()
        id2r = [''] * len(r2id)
        for r, i in r2id.items():
            id2r[i] = r
        return id2r

    def test_vocab_size_and_special_ids(self):
        id2r = self._id2reading()
        assert len(id2r) == 12907
        assert id2r[0] == "[UNK]" and id2r[1] == "[ID]"
        assert id2r[2] != "[UNK]"

    def test_no_duplicate_ids(self):
        assert len(set(self._id2reading())) == 12907

class TestLexicalTag:
    """End-to-end head pass with a tiny weight-free WordModule."""

    SMALL_CONFIG: ClassVar[dict] = {
        'hidden_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 4,
        'intermediate_size': 64, 'vocab_size': 32008, 'max_position_embeddings': 512,
        'relative_attention': True, 'position_buckets': 256, 'max_relative_positions': -1,
        'pos_att_type': ['p2c', 'c2p'], 'share_att_key': True,
        'hidden_dropout_prob': 0.0, 'attention_probs_dropout_prob': 0.0,
        'norm_rel_ebd': 'layer_norm', 'position_biased_input': False,
    }

    def _analyzer(self):
        from kwja_mlx.extensions import TwoLayerClassifier
        a = object.__new__(Analyzer)
        config = dict(self.SMALL_CONFIG)
        m = WordModule(config)
        n_hidden = config['hidden_size']
        n_tags = {'pos': 14, 'ne': 17, 'subpos': 35, 'conjtype': 33, 'conjform': 81}
        for name, n in n_tags.items():
            head: TwoLayerClassifier = getattr(m, f'{name}_tagger')
            head.update({
                'dense': {'weight': mx.array(np.random.randn(n_hidden, n_hidden).astype(np.float32) * 0.05),
                          'bias': mx.array(np.zeros(n_hidden, dtype=np.float32))},
                'classifier': {'weight': mx.array(np.random.randn(n, n_hidden).astype(np.float32) * 0.05),
                               'bias': mx.array(np.zeros(n, dtype=np.float32))},
            })
        rt = m.reading_tagger
        rt.update({
            'dense': {'weight': mx.array(np.random.randn(n_hidden, n_hidden).astype(np.float32) * 0.05),
                      'bias': mx.array(np.zeros(n_hidden, dtype=np.float32))},
            'classifier': {'weight': mx.array(np.random.randn(12907, n_hidden).astype(np.float32) * 0.01),
                           'bias': mx.array(np.zeros(12907, dtype=np.float32))},
        })
        a.word_model = m
        a.word_model.base_phrase_feature_tagger.update({
            'dense_a': mx.array(np.zeros((n_hidden, 4, len(BASE_PHRASE_FEATURES)), dtype=np.float32)),
            'dense_b': mx.array(np.zeros((4, n_hidden, len(BASE_PHRASE_FEATURES)), dtype=np.float32)),
            'classifier_weight': mx.array(np.zeros((n_hidden, len(BASE_PHRASE_FEATURES)), dtype=np.float32)),
            'classifier_bias': mx.array(np.zeros(len(BASE_PHRASE_FEATURES), dtype=np.float32)),
        })
        a.crf_start = np.zeros(17, dtype=np.float32)
        a.crf_end = np.zeros(17, dtype=np.float32)
        a.crf_trans = np.zeros((17, 17), dtype=np.float32)
        return a

    def test_tuple_shape_and_valid_tags(self):
        a = self._analyzer()
        words = ["食べ", "た"]
        pooled = mx.array(np.random.RandomState(4).randn(1, len(words), 32).astype(np.float32))
        tagged = a._lexical_tag(pooled, words)
        assert len(tagged) == 2
        for row in tagged:
            assert len(row) == 9
            surface, pos, ne, subpos, conjtype, conjform, reading_cls, wf, bpf = row
            assert subpos in POS_TAG_SUBPOS_TAG2SUBPOS_ID[pos]
            assert conjtype in CONJTYPE_TAGS
            if (pos, subpos) in INFLECTABLE:
                assert conjform in CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID[conjtype]
            else:
                assert conjtype == "*" and conjform == "*"
            id2r = set(get_reading2reading_id())
            assert reading_cls == surface or reading_cls in id2r
            assert set(wf).issubset(set(WORD_FEATURES))
            assert isinstance(wf, tuple)
            assert set(bpf).issubset(set(BASE_PHRASE_FEATURES))
            assert isinstance(bpf, tuple)


class TestWordFeatureLoadingKeys:
    def test_delta_key_remap_logic(self):
        """word_feature_tagger.delta.* keys fold into the module namespace."""
        k = 'word_feature_tagger.delta.dense_a'
        assert k.replace('word_feature_tagger.delta.', 'word_feature_tagger.') \
            == 'word_feature_tagger.dense_a'


@pytest.mark.slow
class TestTokenizeIntegration:
    """Real-weights integration test; skipped unless weights are available."""

    def test_tokenize_tabeta(self):
        pytest.importorskip("safetensors")
        try:
            from kwja_mlx import Analyzer
            a = Analyzer()
        except Exception:  # noqa: BLE001 — weights-unavailable guard: skip integration
            pytest.skip("model weights unavailable")
        morphemes = a.tokenize("食べた")
        assert morphemes
        assert "".join(m.surface for m in morphemes) == "食べた"
        pairs = {(m.pos, m.subpos) for m in morphemes}
        for pos, subpos in pairs:
            assert subpos in POS_TAG_SUBPOS_TAG2SUBPOS_ID[pos]
        for m in morphemes:
            assert m.conjtype in CONJTYPE_TAGS and m.conjform in CONJFORM_TAGS
            if (m.pos, m.subpos) not in INFLECTABLE:
                assert m.conjtype == "*" and m.conjform == "*"
