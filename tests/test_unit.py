"""Unit tests for kwja-mlx (no model weights required for most tests)."""
import pytest
import numpy as np


class TestViterbiDecode:
    """Test CRF Viterbi decoding with known inputs."""

    def _make_analyzer(self, start, end, trans):
        """Create a minimal Analyzer mock with CRF parameters only."""
        from kwja_mlx import Analyzer
        a = object.__new__(Analyzer)
        a.crf_start = start
        a.crf_end = end
        a.crf_trans = trans
        return a

    def test_simple_path(self):
        """With identity transitions, should pick argmax per step."""
        # Mock: 3 tags, 4 timesteps
        emissions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        start = np.zeros(3, dtype=np.float32)
        end = np.zeros(3, dtype=np.float32)
        trans = np.zeros((3, 3), dtype=np.float32)

        a = self._make_analyzer(start, end, trans)
        path = a._viterbi_decode(emissions)
        assert path == [0, 1, 2, 0]

    def test_transition_influence(self):
        """Transitions should override emission preferences."""
        emissions = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],  # emission says tag 0, but transition forces tag 1
        ], dtype=np.float32)
        start = np.zeros(3, dtype=np.float32)
        end = np.zeros(3, dtype=np.float32)
        trans = np.array([
            [-10.0, 5.0, -10.0],  # from 0 → strongly prefer 1
            [-10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0],
        ], dtype=np.float32)

        a = self._make_analyzer(start, end, trans)
        path = a._viterbi_decode(emissions)
        assert path[0] == 0  # first step: emission dominates
        assert path[1] == 1  # second step: transition overrides


class TestMorpheme:
    def test_repr(self):
        from kwja_mlx.analyzer import Morpheme
        m = Morpheme(surface='東京', pos='名詞', ne='B-LOCATION')
        assert '東京' in repr(m)
        assert 'B-LOCATION' in repr(m)

    def test_default_ne(self):
        from kwja_mlx.analyzer import Morpheme
        m = Morpheme(surface='は', pos='助詞')
        assert m.ne == 'O'


class TestAnalysisResult:
    def test_to_text(self):
        from kwja_mlx.analyzer import Morpheme, AnalysisResult
        morphemes = [
            Morpheme(surface='東京', pos='名詞'),
            Morpheme(surface='に', pos='助詞'),
        ]
        result = AnalysisResult(text='東京に', morphemes=morphemes)
        assert result.to_text() == '東京/名詞 に/助詞'


class TestEmptyInput:
    """Test that empty/whitespace input returns empty results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Only load model once for this class."""
        from kwja_mlx import Analyzer
        self.a = Analyzer(quantize='q8')

    def test_empty_segment(self):
        assert self.a.segment('') == []

    def test_whitespace_segment(self):
        assert self.a.segment('   ') == []

    def test_empty_tag(self):
        assert self.a.tag([]) == []

    def test_empty_tokenize(self):
        assert self.a.tokenize('') == []

    def test_empty_analyze(self):
        result = self.a.analyze('')
        assert result.morphemes == []


class TestBasicFunctionality:
    """Test basic functionality with loaded model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from kwja_mlx import Analyzer
        self.a = Analyzer(quantize='q8')

    def test_segment_basic(self):
        words = self.a.segment('東京都庁に行きました')
        assert len(words) > 0
        assert ''.join(words) == '東京都庁に行きました'

    def test_tokenize_pos(self):
        morphemes = self.a.tokenize('東京に行く')
        assert len(morphemes) > 0
        # Check POS tags are valid
        valid_pos = {'特殊', '動詞', '形容詞', '判定詞', '助動詞', '名詞', '指示詞', '副詞', '助詞', '接続詞', '連体詞', '感動詞', '接頭辞', '接尾辞'}
        for m in morphemes:
            assert m.pos in valid_pos

    def test_ner_bio_valid(self):
        morphemes = self.a.tokenize('田中太郎さんが東京スカイツリーを訪れた')
        # Check B/I sequence validity
        prev_type = None
        for m in morphemes:
            if m.ne.startswith('I-'):
                assert prev_type == m.ne[2:], f'Invalid I-{m.ne[2:]} after {prev_type}'
            if m.ne.startswith('B-'):
                prev_type = m.ne[2:]
            elif m.ne.startswith('I-'):
                pass
            else:
                prev_type = None

    def test_segment_batch_matches_sequential(self):
        sents = ['東京に行く', '天気がいい', 'テスト']
        seq = [self.a.segment(s) for s in sents]
        batch = self.a.segment_batch(sents, batch_size=2)
        # Characters should be identical even if splits differ slightly
        for s, b in zip(seq, batch):
            assert ''.join(s) == ''.join(b)
