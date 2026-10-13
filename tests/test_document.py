"""Weight-free unit tests for Analyzer.analyze_document (merged cohesion window)."""


from kwja_mlx.analyzer import AnalysisResult, Analyzer


def _stub_analyzer(monkeypatch=None):
    a = object.__new__(Analyzer)
    sents = [["彼", "が", "走る"], ["彼", "は", "速い"]]
    a.segment_batch = lambda texts, batch_size=64: [["彼", "が", "走る"], ["彼", "は", "速い"]][:len(texts)]
    a.tag_batch = lambda lists, batch_size=32: [
        [(w, "名詞", "O", "*", "*", "*", w, (), ()) for w in words] for words in lists
    ]
    a.word_model = object()
    a.word_tok = object()
    captured = {}

    def fake_cohesion(words, discourse_threshold=0.0):
        captured["n"] = len(words)
        # ガ of 走る(2) -> 彼(0); ガ of 速い(5) -> 彼(3, cross-sentence)
        rels = [{} for _ in words]
        rels[2] = {"ガ": 0}
        rels[5] = {"ガ": 3}
        return rels, [{"head": 1, "dep": 4, "rel": "原因・理由"}]

    a.analyze_cohesion = fake_cohesion
    a.read = lambda words: [(w, w, w, w) for w in words]
    import kwja_mlx.analyzer as mod
    orig_dep = mod.add_dependency
    mod.add_dependency = lambda *args, **kwargs: None
    a._restore_dep = lambda: setattr(mod, "add_dependency", orig_dep)
    a._captured = captured
    return a


class TestAnalyzeDocument:
    def test_empty(self):
        a = _stub_analyzer()
        try:
            out = a.analyze_document(["", "   "])
            assert len(out) == 2 and all(r.morphemes == [] for r in out)
        finally:
            a._restore_dep()

    def test_single_falls_back(self):
        a = _stub_analyzer()
        try:
            a.analyze = lambda t, with_reading=True: AnalysisResult(text=t, morphemes=[])
            out = a.analyze_document(["hello", ""])
            assert out[0].text == "hello" and out[1].morphemes == []
            assert "n" not in a._captured  # merged window not used
        finally:
            a._restore_dep()

    def test_merged_window_cross_sentence(self):
        a = _stub_analyzer()
        try:
            out = a.analyze_document(["彼が走る", "彼は速い"])
            assert a._captured["n"] == 6  # flat window spans both sentences
            # cross-sentence antecedent: 速い.ガ -> flat index 3 (second 彼)
            assert out[1].morphemes[2].cohesion == {"ガ": 3}
            assert out[0].morphemes[2].cohesion == {"ガ": 0}
            # discourse attached to last result only
            assert out[1].discourse == [{"head": 1, "dep": 4, "rel": "原因・理由"}]
            assert out[0].discourse == []
        finally:
            a._restore_dep()
