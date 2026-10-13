"""Weight-free unit tests for dependency parsing (kwja_mlx.dependency).

Decoding is verified with hand-built fake score matrices,
mirroring upstream KWJA DependencyManager + add_dependency semantics.
"""
import mlx.core as mx
import numpy as np
import pytest

from kwja_mlx.dependency import (
    DependencyTypeHead,
    WordSelectionHead,
    add_dependency,
    base_phrase_heads,
    decode_dependency,
    has_cycle,
    split_base_phrases,
)


class TestHasCycle:
    def test_two_node_cycle(self):
        assert has_cycle({0: {1}, 1: {0}})

    def test_chain_no_cycle(self):
        assert not has_cycle({0: {1}, 1: {2}})

    def test_self_loop(self):
        assert has_cycle({0: {0}})

    def test_diamond_no_cycle(self):
        # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3: shared descendant is not a cycle
        assert not has_cycle({0: {1, 2}, 1: {3}, 2: {3}})

    def test_long_cycle(self):
        assert has_cycle({0: {1}, 1: {2}, 2: {3}, 3: {0}})


class TestSplitBasePhrases:
    def test_basic_split(self):
        # 文節-区切 flags the phrase-FINAL word: 0, 1, 3 end phrases
        assert split_base_phrases([True, True, False, True, False]) == [[0], [1], [2, 3], [4]]

    def test_bunsetsu_pattern(self):
        # 私は | 学生です: flags on は and です
        assert split_base_phrases([False, True, False, True]) == [[0, 1], [2, 3]]

    def test_all_false_is_single_phrase(self):
        assert split_base_phrases([False, False, False]) == [[0, 1, 2]]

    def test_single_word(self):
        assert split_base_phrases([True]) == [[0]]


class TestBasePhraseHeads:
    def test_flagged_head_wins(self):
        groups = [[0, 1], [2, 3, 4]]
        head_flags = [False, True, False, True, False]
        assert base_phrase_heads(groups, head_flags) == [1, 3]

    def test_last_flagged_wins(self):
        groups = [[0, 1, 2]]
        assert base_phrase_heads(groups, [True, False, True]) == [2]

    def test_fallback_to_phrase_final_word(self):
        groups = [[0, 1], [2, 3]]
        assert base_phrase_heads(groups, [False, False, False, False]) == [1, 3]


class TestDecodeDependency:
    """Fake score matrices (last column = root)."""

    @staticmethod
    def _type_preds(shape, overrides=None):
        tp = np.zeros(shape, dtype=np.int64)
        for (i, j), t in (overrides or {}).items():
            tp[i, j] = t
        return tp

    def test_right_chain(self):
        # 0 -> 1 -> 2 -> root
        s = np.full((3, 4), -1.0)
        s[0, 1], s[1, 2], s[2, 3] = 9.0, 8.0, 7.0
        parents, types = decode_dependency(s, self._type_preds((3, 4)))
        assert parents == [1, 2, -1]
        assert types == ['D', 'D', 'D']

    def test_star_attachment(self):
        # 0 -> 2 <- 1, 2 is root
        s = np.full((3, 4), -1.0)
        s[0, 2], s[1, 2], s[2, 3] = 9.0, 8.0, 7.0
        parents, types = decode_dependency(s, self._type_preds((3, 4)))
        assert parents == [2, 2, -1]
        assert len([p for p in parents if p == -1]) == 1

    def test_nested_chain(self):
        # 0 -> 1 <- root, 2 -> 1
        s = np.full((3, 4), -1.0)
        s[0, 1], s[1, 3], s[2, 1] = 9.0, 8.0, 7.0
        parents, _ = decode_dependency(s, self._type_preds((3, 4)))
        assert parents == [1, -1, 1]

    def test_dependency_types_respected(self):
        # type_preds carry D(0)/P(1)/A(2)/I(3) indices
        s = np.full((3, 4), -1.0)
        s[0, 1], s[1, 2], s[2, 3] = 9.0, 8.0, 7.0
        tp = self._type_preds((3, 4), {(0, 1): 1, (1, 2): 3, (2, 3): 2})
        _, types = decode_dependency(s, tp)
        assert types == ['P', 'I', 'A']
        assert all(t in ('D', 'P', 'A', 'I') for t in types)

    def test_cycle_rejected_next_candidate_taken(self):
        # 0's best is 1; 1's best is 0 (cycle!) — must fall to root instead
        s = np.full((3, 4), -1.0)
        s[0, 1], s[1, 0], s[1, 3], s[2, 1] = 9.0, 9.0, 8.0, 5.0
        parents, _ = decode_dependency(s, self._type_preds((3, 4)))
        assert parents[0] == 1
        assert parents[1] == -1  # cycle edge 1 -> 0 rejected
        assert parents[2] == 1
        # resulting tree is acyclic: following parents from any node terminates
        for i in range(3):
            seen = set()
            j = i
            while parents[j] != -1:
                assert j not in seen
                seen.add(j)
                j = parents[j]

    def test_double_root_rejected(self):
        # both 0 and 1 prefer root; only the first may take it
        s = np.full((3, 4), -1.0)
        s[0, 3], s[1, 3], s[1, 2], s[2, 1] = 9.0, 8.0, 6.0, 5.0
        parents, _ = decode_dependency(s, self._type_preds((3, 4)))
        assert parents[0] == -1
        assert parents[1] == 2  # second root claim rejected, next candidate 2
        assert parents.count(-1) == 1

    def test_last_phrase_takes_root_when_missing(self):
        # root scores low everywhere; child 2 still gets it as the only option
        s = np.full((3, 4), -1.0)
        s[0, 1], s[1, 2], s[2, 3] = 9.0, 8.0, 1.0
        parents, types = decode_dependency(s, self._type_preds((3, 4)))
        assert parents == [1, 2, -1]
        assert types[2] == 'D'

    def test_forced_root_branch_for_last_phrase(self):
        # 0 -> 1 -> 2 -> 3 -> 4 chain; child 4's top-4 candidates (3..0) all
        # close cycles and root is outside topk -> forced root with type 'D'.
        n = 5
        s = np.full((n, n + 1), -1.0)
        for i in range(n - 1):
            s[i, i + 1] = 9.0
            s[n - 1, i] = 8.0  # child 4 prefers earlier phrases, all cyclic
        parents, types = decode_dependency(s, self._type_preds((n, n + 1)), topk=4)
        assert parents == [1, 2, 3, 4, -1]
        assert types[-1] == 'D'
        assert parents.count(-1) == 1

    def test_rightward_fallback_after_topk_failure(self):
        # child 1's only top-1 candidate (0) closes a cycle -> fallback picks 2
        s = np.full((3, 4), -1.0)
        s[0, 1], s[1, 0], s[2, 3] = 9.0, 9.0, 6.0
        parents, types = decode_dependency(s, self._type_preds((3, 4)), topk=1)
        assert parents[0] == 1
        assert parents[1] == 2  # rightward fallback (not root, not 0)
        assert parents[2] == -1
        assert types[1] == 'D'  # fallback edges get 'D'

    def test_resolve_dependency_raises_when_trapped(self):
        """Sequential decoding always finds a fallback (right targets have no
        outgoing edges yet), so the raise is defensive — upstream marks it
        pragma: no cover. Exercise it directly: every fallback target of
        source 3 already points at 3."""
        from kwja_mlx.dependency import _resolve_dependency
        graph = {0: {3}, 1: {3}, 2: {3}}
        with pytest.raises(RuntimeError):
            _resolve_dependency(graph, 3, 4)

    def test_self_column_never_selected(self):
        # even when the self cell carries the top score, it must be skipped
        s = np.full((2, 3), -1.0)
        s[0, 0] = 99.0  # self
        s[1, 1] = 99.0  # self
        s[0, 1], s[1, 2] = 5.0, 5.0
        parents, _ = decode_dependency(s, self._type_preds((2, 3)))
        assert parents == [1, -1]


class TestHeadsWeightFree:
    def test_word_selection_head_shape(self):
        head = WordSelectionHead(8, 1)
        pooled = mx.random.normal((1, 3, 8))
        assert head(pooled).shape == (1, 3, 3)

    def test_word_selection_head_no_output_bias(self):
        head = WordSelectionHead(8, 1)
        assert 'weight' in head.output_layer and 'bias' not in head.output_layer

    def test_dependency_type_head_shape(self):
        head = DependencyTypeHead(8, 4)
        child = mx.random.normal((1, 2, 8))
        head_h = mx.random.normal((1, 2, 8))
        assert head(child, head_h).shape == (1, 2, 4)


class TestAddDependencyGuard:
    def test_empty_morphemes_noop(self):
        add_dependency(None, [], None)
