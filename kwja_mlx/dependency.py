"""Dependency parsing for the word module (ported from the KWJA word module).

Base phrases (文節) are split with the word-feature tagger's '文節-区切'
probability (>= 0.5 ⇔ membership in Morpheme.word_features). The dedicated
base_phrase_feature_tagger (64-label LoRA matrix) is intentionally left
unloaded for now; '基本句-主辞' from the same word-feature head stands in for
the base-phrase head word.

Heads (upstream keys in word_deberta-v2-base.safetensors):
  - dependency_parser: WordSelectionHead —
        score(i, j) = w · GELU(Ws·hi + Wt·hj), no output bias
  - dependency_type_parser: concat(child, head) 1536 → 1536 → GELU → 4 types
        D(係り) / P(並列) / A(同格) / I(不良並列)

Decoding ports kwja.utils.dependency_parsing.DependencyManager +
callbacks.utils.add_dependency semantics: top-k candidate edges per base
phrase head, DFS cycle rejection, single-root rejection, right-then-left
fallback resolution (Japanese attaches rightward first).
"""
from collections import defaultdict
from typing import Optional

import mlx.core as mx
import numpy as np
from mlx import nn

from kwja_mlx.constants import DEPENDENCY_TYPES


class WordSelectionHead(nn.Module):
    """KWJA WordSelectionHead: pairwise word selection score.

    score(i, j) = output_layer(GELU(l_source(hi) + l_target(hj)))
    with i the child (modifier) and j the head (dependent-on).
    """

    def __init__(self, hidden_size: int, num_labels: int = 1):
        super().__init__()
        self.l_source = nn.Linear(hidden_size, hidden_size)
        self.l_target = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_labels, bias=False)

    def __call__(self, pooled: mx.array) -> mx.array:
        h_source = self.l_source(pooled)  # (B, n, H)
        h_target = self.l_target(pooled)  # (B, n, H)
        hidden = nn.gelu(h_source[:, :, None, :] + h_target[:, None, :, :])  # (B, n, n, H)
        return self.output_layer(hidden).squeeze(-1)  # (B, n, n)


class DependencyTypeHead(nn.Module):
    """KWJA SequenceLabelingHead(4, hidden*2) applied to concat(child, head)."""

    def __init__(self, hidden_size: int, num_types: int = 4):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, num_types)

    def __call__(self, child: mx.array, head: mx.array) -> mx.array:
        # child/head: (..., H) -> logits (..., num_types)
        x = mx.concat([child, head], axis=-1)
        return self.classifier(nn.gelu(self.dense(x)))


class DependencyHead(nn.Module):
    """Dependency parsing heads attached to WordModule."""

    def __init__(self, hidden_size: int, num_types: int = 4):
        super().__init__()
        self.dependency_parser = WordSelectionHead(hidden_size, 1)
        self.dependency_type_parser = DependencyTypeHead(hidden_size, num_types)


def has_cycle(graph: dict[int, set[int]]) -> bool:
    """DFS cycle check over a child→head graph (DependencyManager semantics).

    Mirrors upstream is_cyclic/has_cycle: results are cached per source, a
    back-edge to a node already on the visited path means a cycle.
    """
    visited: set[int] = set()
    cache: dict[int, bool] = {}

    def is_cyclic(source: int) -> bool:
        if source in cache:
            return cache[source]
        if source in visited:
            return True
        visited.add(source)
        ret = any(is_cyclic(target) for target in graph.get(source, ()))
        cache[source] = ret
        return ret

    return any(is_cyclic(source) for source in list(graph))


def split_base_phrases(kugiri_flags: list[bool]) -> list[list[int]]:
    """Group word indices into base phrases.

    kugiri_flags[i] is True when word i ENDS a base phrase ('文節-区切'
    probability >= 0.5 marks the phrase-final word, e.g. 私は|学生です flags
    は and です). The last phrase keeps any trailing unflagged words.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    for i, flag in enumerate(kugiri_flags):
        current.append(i)
        if flag:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def base_phrase_heads(groups: list[list[int]], head_flags: list[bool]) -> list[int]:
    """Head word index per base phrase: the last '基本句-主辞' flagged word
    (>= 0.5), falling back to the phrase-final word."""
    heads = []
    for group in groups:
        flagged = [i for i in group if head_flags[i]]
        heads.append(flagged[-1] if flagged else group[-1])
    return heads


def decode_dependency(
    scores: np.ndarray,
    type_preds: np.ndarray,
    topk: int = 4,
    root_col: Optional[int] = None,
) -> tuple[list[int], list[str]]:
    """Decode a base-phrase dependency tree from selection scores.

    Args:
        scores: (n, m) score(i, j) for child base phrase i, candidate column j.
        type_preds: (n, m) argmax dependency-type index per (child, candidate).
        topk: number of candidate heads tried per child, in score order.
        root_col: column index standing for the root (default: last column).

    Returns:
        (parents, types): parents[i] = -1 (root) or parent base-phrase index;
        types[i] ∈ DEPENDENCY_TYPES ('D'/'P'/'A'/'I').

    Rejection rules (upstream add_dependency): an edge is rejected when it
    closes a cycle or when a second child claims the root; then the next
    candidate is tried. If all top-k candidates fail, the rightward-then-
    leftward fallback (upstream _resolve_dependency) attaches the phrase with
    type 'D'; the last phrase becomes root if none exists.
    """
    n = scores.shape[0]
    root = scores.shape[1] - 1 if root_col is None else root_col
    parents: list[Optional[int]] = [None] * n
    types: list[Optional[str]] = [None] * n
    graph: dict[int, set[int]] = defaultdict(set)
    has_root = False
    for i in range(n):
        row = np.asarray(scores[i], dtype=np.float64).copy()
        row[i] = -np.inf  # a phrase never attaches to itself
        order = np.argsort(-row, kind="stable")[:topk]
        for j in order:
            j = int(j)
            if j == i:
                continue
            graph[i].add(j)
            if has_cycle(graph) or (j == root and has_root):
                graph[i].discard(j)
                continue
            parents[i] = -1 if j == root else j
            types[i] = DEPENDENCY_TYPES[int(type_preds[i, j])]
            if j == root:
                has_root = True
            break
        else:
            if i == n - 1 and not has_root:
                parents[i] = -1
                types[i] = DEPENDENCY_TYPES[0]
                has_root = True
            else:
                parents[i], types[i] = _resolve_dependency(graph, i, n)
    return parents, types


def _resolve_dependency(graph: dict[int, set[int]], source: int, n: int) -> tuple[int, str]:
    """Fallback attachment: rightward first (Japanese is head-initial),
    then leftward; attach with the plain 'D' type."""
    for target in range(source + 1, n):
        graph.setdefault(source, set()).add(target)
        if not has_cycle(graph):
            return target, DEPENDENCY_TYPES[0]
        graph[source].discard(target)
    for target in range(source - 1, -1, -1):
        graph.setdefault(source, set()).add(target)
        if not has_cycle(graph):
            return target, DEPENDENCY_TYPES[0]
        graph[source].discard(target)
    raise RuntimeError("couldn't resolve dependency")


def add_dependency(word_model, morphemes: list, word_tok) -> None:
    """Attach base-phrase dependency to morphemes (parent_index / dep_type).

    parent_index is the parent BASE PHRASE index (-1 = root); dep_type is one
    of DEPENDENCY_TYPES. Every morpheme inherits its base phrase's attachment.
    """
    n = len(morphemes)
    if n == 0:
        return
    kugiri_flags = ["文節-区切" in m.word_features for m in morphemes]
    groups = split_base_phrases(kugiri_flags)
    head_flags = ["基本句-主辞" in m.word_features for m in morphemes]
    heads = base_phrase_heads(groups, head_flags)
    nb = len(groups)
    word_of_col = heads + [n]  # bp column → word index; last column = [ROOT]
    bp_of_word = {w: b for b, group in enumerate(groups) for w in group}

    # Encode words + [ROOT] with the word encoder, first-subword pooling.
    enc = word_tok(
        [[m.surface for m in morphemes] + ["[ROOT]"]],
        is_split_into_words=True,
        truncation=True,
        max_length=256,
        return_tensors="np",
    )
    input_ids = mx.array(enc["input_ids"])
    h = word_model(input_ids)  # (1, L, H)
    mx.eval(h)
    word_ids = enc.word_ids(0)
    first_idx = [0] * (n + 1)
    for ti, wid in enumerate(word_ids):
        if wid is not None and wid <= n and first_idx[wid] == 0:
            first_idx[wid] = ti
    pooled = h[0, mx.array(np.array(first_idx, dtype=np.int32))][None, :, :]  # (1, n+1, H)

    dependency_head = word_model.dependency_head
    # Selection scores over base-phrase head words + root.
    scores = np.array(dependency_head.dependency_parser(pooled)[0])  # (n+1, n+1)
    child_scores = scores[np.ix_(heads, word_of_col)].copy()  # (nb, nb+1)

    # Dependency types for the top-k candidates of each child.
    cand = np.argsort(-child_scores, axis=1, kind="stable")[:, :4]  # (nb, topk)
    type_preds = np.zeros((nb, nb + 1), dtype=np.int64)
    for b in range(nb):
        cand_words = [word_of_col[j] for j in cand[b]]
        child_h = pooled[0, heads[b]]  # (H,)
        cand_h = pooled[0, mx.array(np.array(cand_words, dtype=np.int32))]  # (topk, H)
        child_expanded = mx.broadcast_to(child_h[None, :], cand_h.shape)  # (topk, H)
        logits = dependency_head.dependency_type_parser(child_expanded, cand_h)  # (topk, 4)
        type_preds[b, cand[b]] = np.array(mx.argmax(logits, axis=-1))

    parents, types = decode_dependency(child_scores, type_preds, topk=4, root_col=nb)
    for w, m in enumerate(morphemes):
        b = bp_of_word[w]
        m.parent_index = parents[b]
        m.dep_type = types[b]
