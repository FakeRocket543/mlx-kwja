"""Cohesion analysis (照応・談話関係) heads for kwja-mlx.

Ports KWJA v2.5.1 word-module heads:
- LoRARelationWiseWordSelectionHead → cohesion_analyzer (9 relations, LoRA rank 1)
- WordSelectionHead → discourse_relation_analyzer (7 discourse relations)

logits[i, j, r] = cls[r] · GELU((W_s + Δ_s[r]) h_i + (W_t + Δ_t[r]) h_j)
where i is the target (anaphor/modifier) index and j the source (candidate/head)
index. Source: ku-nlp/kwja v2.5.1 src/kwja/modules/components/head.py.
"""
import mlx.core as mx
import numpy as np
from mlx import nn

from kwja_mlx.constants import (
    COHESION_RELATIONS,
    DISCOURSE_RELATIONS,
    EXOPHORA_REFERENTS,
    SPECIALS,
)


class LoRADelta(nn.Module):
    """Rank-decomposed per-relation weight delta: (hid, hid, rel)."""

    def __init__(self, num_relations: int, hidden_size: int, rank: int):
        super().__init__()
        self.dense_a = mx.zeros((hidden_size, rank, num_relations))
        self.dense_b = mx.zeros((rank, hidden_size, num_relations))

    def __call__(self) -> mx.array:
        return mx.einsum("hrl,ril->hil", self.dense_a, self.dense_b)


class LoRARelationWiseWordSelectionHead(nn.Module):
    """Cohesion analyzer head: shared l_source/l_target + rank-1 LoRA per relation."""

    def __init__(self, num_relations: int, hidden_size: int, rank: int = 1):
        super().__init__()
        self.l_source = nn.Linear(hidden_size, hidden_size)
        self.l_target = nn.Linear(hidden_size, hidden_size)
        self.delta_source = LoRADelta(num_relations, hidden_size, rank)
        self.delta_target = LoRADelta(num_relations, hidden_size, rank)
        self.classifier = mx.zeros((hidden_size, num_relations))

    def __call__(self, pooled: mx.array) -> mx.array:
        h_source = self.l_source(pooled)  # (b, seq, hid)
        h_target = self.l_target(pooled)  # (b, seq, hid)
        delta_source_out = mx.einsum("bsh,hil->bsli", pooled, self.delta_source())  # (b, seq, rel, hid)
        delta_target_out = mx.einsum("bsh,hil->bsli", pooled, self.delta_target())
        source = h_source[:, :, None, :] + delta_source_out  # (b, seq, rel, hid)
        target = h_target[:, :, None, :] + delta_target_out
        hidden = nn.gelu(source[:, :, None, :, :] + target[:, None, :, :, :])  # (b, t, s, rel, hid)
        return mx.einsum("bstlh,hl->bstl", hidden, self.classifier)  # (b, t, s, rel)


class WordSelectionHead(nn.Module):
    """Pairwise selection head (discourse relations; also used for dependency)."""

    def __init__(self, num_labels: int, hidden_size: int):
        super().__init__()
        self.l_source = nn.Linear(hidden_size, hidden_size)
        self.l_target = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_labels, bias=False)

    def __call__(self, pooled: mx.array) -> mx.array:
        h_source = self.l_source(pooled)  # (b, seq, hid)
        h_target = self.l_target(pooled)  # (b, seq, hid)
        hidden = nn.gelu(h_source[:, :, None, :] + h_target[:, None, :, :])  # (b, t, s, hid)
        return self.output_layer(hidden)  # (b, t, s, label)


def decode_cohesion(
    logits: np.ndarray,
    num_morphemes: int,
    relations: list = COHESION_RELATIONS,
    exophora_referents: list = EXOPHORA_REFERENTS,
    specials: list = SPECIALS,
) -> list:
    """Decode cohesion logits (seq, seq, rel) for one sentence.

    For every target morpheme i and relation r, argmax over endophora
    candidates (morphemes 0..num_morphemes-1) plus special-token columns
    (num_morphemes..num_morphemes+len(specials)-1), mirroring KWJA's
    _to_rel_tag. A morpheme argmax yields an endophoric antecedent (morpheme
    index), an exophora-referent special ([著者] etc.) yields the referent
    string, and [NULL]/[NA]/[ROOT] yield no relation (absent from the dict).

    Returns one dict {relation: int | str} per morpheme.
    """
    exophora_set = {f"[{ref}]" for ref in exophora_referents}
    results = []
    for i in range(num_morphemes):
        rel2target = {}
        for r, rel in enumerate(relations):
            scores = logits[i, : num_morphemes + len(specials), r]
            j = int(np.argmax(scores))
            if j < num_morphemes:
                rel2target[rel] = j  # endophora: antecedent morpheme index
            else:
                special = specials[j - num_morphemes]
                if special in exophora_set:
                    rel2target[rel] = special[1:-1]  # exophora: 著者 / 読者 / …
                # [NULL] / [NA] / [ROOT] → no relation
        results.append(rel2target)
    return results


def decode_discourse(
    logits: np.ndarray,
    num_morphemes: int,
    threshold: float = 0.0,
    relations: list = DISCOURSE_RELATIONS,
) -> list:
    """Decode discourse logits (seq, seq, rel) for one sentence.

    For every modifier/head morpheme pair (i, j), softmax → argmax; pairs whose
    max probability is below `threshold` are forced to 談話関係なし, as are
    argmax-0 pairs (KWJA predict_step semantics). Returns
    [{"modifier": i, "head": j, "relation": rel, "score": p}, ...].
    """
    relations = list(relations)
    none_index = relations.index("談話関係なし")
    probs = _softmax(logits[:num_morphemes, :num_morphemes])
    predictions = np.argmax(probs, axis=2)
    max_probs = np.max(probs, axis=2)
    results = []
    for i in range(num_morphemes):
        for j in range(num_morphemes):
            pred = int(predictions[i, j])
            if max_probs[i, j] < threshold:
                pred = none_index
            if pred == none_index:
                continue
            results.append({
                "modifier": i,
                "head": j,
                "relation": relations[pred],
                "score": float(max_probs[i, j]),
            })
    return results


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def analyze_heads(
    cohesion_analyzer,
    discourse_analyzer,
    pooled: "mx.array",
    num_morphemes: int,
    discourse_threshold: float = 0.0,
):
    """Run both heads over pooled states (1, n+7, hid) and decode.

    Returns (per-morpheme cohesion dicts, sentence-level discourse list).
    """
    cohesion_logits = np.array(cohesion_analyzer(pooled)[0])  # (seq, seq, 9)
    discourse_logits = np.array(discourse_analyzer(pooled)[0])  # (seq, seq, 7)
    cohesion = decode_cohesion(cohesion_logits, num_morphemes)
    discourse = decode_discourse(discourse_logits, num_morphemes, threshold=discourse_threshold)
    return cohesion, discourse
