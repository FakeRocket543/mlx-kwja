"""Weight-free tests for char-module heads: key remap, sentence splitting, word norm op decoding."""
import numpy as np
import pytest


class TestRemapCharKeys:
    def test_head_keys(self):
        from kwja_mlx.analyzer import _remap_char_key
        assert _remap_char_key('word_segmentation_tagger.layers.0.weight') == 'word_seg.dense.weight'
        assert _remap_char_key('sent_segmentation_tagger.layers.0.weight') == 'sent_seg.dense.weight'
        assert _remap_char_key('sent_segmentation_tagger.layers.3.bias') == 'sent_seg.classifier.bias'
        assert _remap_char_key('word_norm_op_tagger.layers.0.weight') == 'norm_op.dense.weight'
        assert _remap_char_key('word_norm_op_tagger.layers.3.bias') == 'norm_op.classifier.bias'

    def test_encoder_keys_unchanged(self):
        from kwja_mlx.analyzer import _remap_char_key
        assert _remap_char_key('encoder.embeddings.word_embeddings.weight') == 'embeddings.word_embeddings.weight'
        assert _remap_char_key('encoder.encoder.LayerNorm.weight') == 'encoder.LayerNorm.weight'


class TestSentenceDecode:
    def _fake_sent_logits(self, length, b_positions):
        """Deterministic arange-based fake logits: B (0) at b_positions, I (1) elsewhere."""
        idx = np.arange(length)
        b = np.isin(idx, list(b_positions)).astype(np.float32)
        return np.stack([b * 10.0 + 0.5, 5.0 - b * 10.0], axis=1)

    def test_two_sentences(self):
        from kwja_mlx.analyzer import _decode_sentences
        # [CLS] こんにちは。 さようなら。 [SEP]
        tokens = ['[CLS]', 'こ', 'ん', 'に', 'ち', 'は', '。', 'さ', 'よ', 'う', 'な', 'ら', '。', '[SEP]']
        ids = list(range(len(tokens)))
        tags = np.argmax(self._fake_sent_logits(len(tokens), {1, 7}), axis=-1)
        sents = _decode_sentences(tokens, tags, ids, special_ids={0, 13})
        assert sents == [['こ', 'ん', 'に', 'ち', 'は', '。'], ['さ', 'よ', 'う', 'な', 'ら', '。']]

    def test_filters_by_token_id_not_position(self):
        from kwja_mlx.analyzer import _decode_sentences
        # Realistic ids (char-wwm tokenizer): [PAD]=0, [CLS]=1, [SEP]=2, [MASK]=4,
        # extra ' '=22012. Filtering must key off token IDS, not position indices
        # (positions 0/1/2/4 are real chars here).
        tokens = ['[CLS]', 'こ', 'ん', 'に', 'ち', '[SEP]']
        ids = [1, 37, 57, 8, 105, 2]
        tags = np.argmax(self._fake_sent_logits(len(tokens), set()), axis=-1)
        sents = _decode_sentences(tokens, tags, ids, special_ids={0, 1, 2, 4, 22012})
        assert sents == [['こ', 'ん', 'に', 'ち']]

    def test_no_split_without_b(self):
        from kwja_mlx.analyzer import _decode_sentences
        tokens = ['[CLS]', '東', '京', '[SEP]']
        tags = np.argmax(self._fake_sent_logits(len(tokens), set()), axis=-1)
        assert _decode_sentences(tokens, tags, [0, 1, 2, 3], {0, 3}) == [['東', '京']]

    def test_consecutive_b_single_chars(self):
        from kwja_mlx.analyzer import _decode_sentences
        tokens = ['[CLS]', 'あ', 'い', 'う', '[SEP]']
        tags = np.argmax(self._fake_sent_logits(len(tokens), {1, 2, 3}), axis=-1)
        assert _decode_sentences(tokens, tags, [0, 1, 2, 3, 4], {0, 4}) == [['あ'], ['い'], ['う']]

    def test_space_special_filtered(self):
        from kwja_mlx.analyzer import _decode_sentences
        # ' ' is the extra word-boundary special token: skipped, not part of the text
        tokens = ['[CLS]', 'あ', ' ', 'い', '[SEP]']
        tags = np.argmax(self._fake_sent_logits(len(tokens), set()), axis=-1)
        assert _decode_sentences(tokens, tags, [0, 1, 22012, 2, 3], {0, 22012, 3}) == [['あ', 'い']]


class TestNormalizedSurf:
    def test_keep(self):
        from kwja_mlx.analyzer import get_normalized_surf
        assert get_normalized_surf('今日', ['K', 'K']) == '今日'

    def test_delete(self):
        from kwja_mlx.analyzer import get_normalized_surf
        # っ (hatsuon) is removable
        assert get_normalized_surf('よっか', ['K', 'D', 'K']) == 'よか'

    def test_devoice(self):
        from kwja_mlx.analyzer import get_normalized_surf
        # V on word-initial voiced kana: ば -> は
        assert get_normalized_surf('ばす', ['V', 'K']) == 'はす'

    def test_small_to_upper(self):
        from kwja_mlx.analyzer import get_normalized_surf
        assert get_normalized_surf('でぇ', ['K', 'S']) == 'でえ'

    def test_prolonged_p(self):
        from kwja_mlx.analyzer import get_normalized_surf
        # よー -> よう (P converts the prolonged sign after K)
        assert get_normalized_surf('よー', ['K', 'P']) == 'よう'

    def test_prolonged_e(self):
        from kwja_mlx.analyzer import get_normalized_surf
        # てー -> てえ (E uses the e-row map)
        assert get_normalized_surf('みてー', ['K', 'K', 'E']) == 'みてえ'

    def test_nonstrict_relaxes_constraints(self):
        from kwja_mlx.analyzer import get_normalized_surf
        # V off word-initial position still applies in non-strict mode
        assert get_normalized_surf('あば', ['K', 'V']) == 'あは'
        # S on a non-small kana keeps the char
        assert get_normalized_surf('か', ['S']) == 'か'
        # P with no preceding kana keeps the char
        assert get_normalized_surf('ー', ['P']) == 'ー'
        # P on a non-prolonged char keeps the char
        assert get_normalized_surf('あよ', ['K', 'P']) == 'あよ'

    def test_length_mismatch_raises(self):
        from kwja_mlx.analyzer import get_normalized_surf
        with pytest.raises(AssertionError):
            get_normalized_surf('あ', ['K', 'K'])

    def test_unknown_tag_raises(self):
        from kwja_mlx.analyzer import get_normalized_surf
        with pytest.raises(NotImplementedError):
            get_normalized_surf('あ', ['X'])


class TestCharModuleHeads:
    def test_head_shapes(self):
        import mlx.core as mx

        from kwja_mlx.analyzer import CharModule
        from kwja_mlx.constants import SENT_SEGMENTATION_TAGS, WORD_NORM_OP_TAGS
        config = {
            'hidden_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 2,
            'intermediate_size': 32, 'vocab_size': 100, 'max_position_embeddings': 32,
            'relative_attention': True, 'position_buckets': 16, 'max_relative_positions': -1,
            'pos_att_type': ['p2c', 'c2p'], 'share_att_key': True,
            'hidden_dropout_prob': 0.0, 'attention_probs_dropout_prob': 0.0,
            'norm_rel_ebd': 'layer_norm', 'position_biased_input': False,
        }
        m = CharModule(config)
        ids = mx.array(np.array([[1, 2, 3, 4, 5]]))
        h = m(ids)
        assert h.shape == (1, 5, 16)
        assert m.word_seg(h).shape == (1, 5, len(SENT_SEGMENTATION_TAGS))
        assert m.sent_seg(h).shape == (1, 5, len(SENT_SEGMENTATION_TAGS))
        assert m.norm_op(h).shape == (1, 5, len(WORD_NORM_OP_TAGS))
