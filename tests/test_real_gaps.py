"""Real-weight gap tests: sentences, strict norm, batch paths, toggles, clauses.

Requires cached weights (kevinzhow/kwja-mlx base). Skips cleanly otherwise.
"""
import numpy as np
import pytest

pytest.importorskip("safetensors")

from kwja_mlx.analyzer import _decode_clauses, get_normalized_surf


@pytest.fixture(scope="module")
def analyzer():
    from kwja_mlx import Analyzer
    try:
        return Analyzer(quantize="q8")
    except Exception:
        pytest.skip("model weights unavailable")


class TestSentences:
    def test_two_sentences(self, analyzer):
        out = analyzer.sentences("こんにちは。さようなら。")
        assert len(out) == 2
        assert "".join(out[0]) == "こんにちは。"
        assert "".join(out[1]) == "さようなら。"

    def test_single_no_split(self, analyzer):
        out = analyzer.sentences("東京都庁に行きました")
        assert len(out) == 1 and "".join(out[0]) == "東京都庁に行きました"

    def test_empty(self, analyzer):
        assert analyzer.sentences("") == []
        assert analyzer.sentences("   ") == []


class TestStrictNorm:
    def test_strict_raises_on_bad_delete(self, analyzer):
        with pytest.raises(ValueError):
            get_normalized_surf("あか", ["K", "D"], strict=True)

    def test_strict_ok_path(self, analyzer):
        # っ ∈ HATSUON_SET → strict-deletable; がっこう → がこう
        assert get_normalized_surf("がっこう", ["K", "D", "K", "K"], strict=True) == "がこう"


class TestBatchPaths:
    def test_tokenize_batch_matches_single(self, analyzer):
        texts = ["東京に行く", "天気がいい"]
        batched = analyzer.tokenize_batch(texts, batch_size=2)
        singles = [analyzer.tokenize(t) for t in texts]
        for b, s in zip(batched, singles):
            assert [m.surface for m in b] == [m.surface for m in s]
            assert [m.pos for m in b] == [m.pos for m in s]

    def test_tag_batch_tuple_width(self, analyzer):
        out = analyzer.tag_batch([["東京", "に", "行く"]])
        assert len(out[0][0]) == 9


class TestAnalyzeToggles:
    def test_no_reading_no_cohesion(self, analyzer):
        r = analyzer.analyze("東京に行く", with_reading=False, with_cohesion=False)
        assert r.morphemes
        assert all(m.reading is None for m in r.morphemes)
        assert all(m.cohesion == {} for m in r.morphemes)

    def test_cohesion_off_keeps_lexical(self, analyzer):
        r = analyzer.analyze("食べた", with_reading=False, with_cohesion=False)
        assert r.morphemes[0].pos == "動詞"
        assert r.morphemes[0].conjform == "タ形"


class TestRealClauses:
    def test_clause_split_fires(self, analyzer):
        words = analyzer.segment("天気がいいから、散歩しよう")
        import mlx.core as mx
        enc = analyzer.word_tok(words, is_split_into_words=True, return_tensors="np",
                                truncation=True, max_length=256)
        h = analyzer.word_model(mx.array(enc["input_ids"]))
        mx.eval(h)
        word_ids = enc.word_ids(0)
        first_idx = [0] * len(words)
        for ti, wid in enumerate(word_ids):
            if wid is not None and wid < len(words) and first_idx[wid] == 0:
                first_idx[wid] = ti
        pooled = h[0, mx.array(np.array(first_idx, dtype=np.int32))][None, :, :]
        bp = np.array(analyzer.word_model.base_phrase_feature_tagger(pooled)[0])
        clauses = _decode_clauses(len(words), bp)
        assert len(clauses) >= 2  # いい｜しよう at minimum
        total = sum(e - s for s, e, _ in clauses)
        assert total == len(words)
