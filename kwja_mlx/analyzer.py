"""kwja-mlx: Full integrated Japanese analyzer (char → word → seq2seq)."""

import re
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from mlx import nn
from safetensors.numpy import load_file
from transformers import AutoTokenizer

from kwja_mlx.cohesion import (
    LoRARelationWiseWordSelectionHead,
    WordSelectionHead,
    analyze_heads,
)
from kwja_mlx.constants import (
    BASE_PHRASE_FEATURES,
    CHOON_SET,
    COHESION_RELATIONS,
    CONJFORM_TAGS,
    CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID,
    CONJTYPE_TAGS,
    DISCOURSE_RELATIONS,
    HATSUON_SET,
    INFLECTABLE,
    LOWER2UPPER,
    POS_TAG_SUBPOS_TAG2SUBPOS_ID,
    PROLONGED_MAP,
    PROLONGED_MAP_FOR_EROW,
    SENT_SEGMENTATION_TAGS,
    SPECIALS,
    SUBPOS_TAGS,
    VOICED2VOICELESS,
    WORD_FEATURES,
    WORD_NORM_OP_TAGS,
    WORD_SEGMENTATION_TAGS,
    get_reading2reading_id,
)
from kwja_mlx.deberta_v2_mlx import (
    DebertaV2Embeddings,
    DebertaV2Encoder,
    build_relative_position,
)
from kwja_mlx.dependency import DependencyHead, add_dependency
from kwja_mlx.extensions import TwoLayerClassifier
from kwja_mlx.t5_mlx import T5ForConditionalGeneration

# KWJA label constants (from kwja.utils.constants)
POS_TAGS = ['特殊', '動詞', '形容詞', '判定詞', '助動詞', '名詞', '指示詞', '副詞', '助詞', '接続詞', '連体詞', '感動詞', '接頭辞', '接尾辞']
NE_TAGS = ['O', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-PERSON', 'I-PERSON', 'B-LOCATION', 'I-LOCATION',
           'B-ARTIFACT', 'I-ARTIFACT', 'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME', 'B-MONEY', 'I-MONEY',
           'B-PERCENT', 'I-PERCENT']


@dataclass
class Morpheme:
    surface: str
    pos: str
    ne: str = "O"
    reading: Optional[str] = None
    lemma: Optional[str] = None
    canon: Optional[str] = None
    subpos: str = "*"
    conjtype: str = "*"
    conjform: str = "*"
    reading_cls: Optional[str] = None
    word_features: tuple[str, ...] = ()
    base_phrase_features: tuple[str, ...] = ()
    # {relation: antecedent morpheme index (endophora) | exophora referent str}
    cohesion: dict = field(default_factory=dict)
    parent_index: Optional[int] = None  # parent base-phrase index (-1 = root)
    dep_type: Optional[str] = None  # 'D' / 'P' / 'A' / 'I'

    def __repr__(self):
        parts = [f"{self.surface}/{self.pos}"]
        if self.ne != "O":
            parts[0] += f"[{self.ne}]"
        if self.reading:
            parts.append(f"reading={self.reading}")
        return f"Morpheme({', '.join(parts)})"


@dataclass
class AnalysisResult:
    text: str
    morphemes: list[Morpheme]
    # [{modifier, head, relation, score}] decoded between morpheme pairs
    discourse: list = field(default_factory=list)

    def to_text(self) -> str:
        return " ".join(f"{m.surface}/{m.pos}" for m in self.morphemes)


# Weight key mapping
def _remap_deberta_key(k: str) -> str:
    k = re.sub(r'^encoder\.embeddings\.', 'embeddings.', k)
    k = re.sub(r'^encoder\.encoder\.layer\.', 'encoder.layer.', k)
    k = re.sub(r'^encoder\.encoder\.rel_embeddings\.', 'encoder.rel_embeddings.', k)
    k = re.sub(r'^encoder\.encoder\.LayerNorm\.', 'encoder.LayerNorm.', k)
    k = re.sub(r'^encoder\.encoder\.conv\.', 'encoder.conv.', k)
    k = k.replace('.attention.output.dense.', '.attention.out_proj.')
    k = k.replace('.attention.output.LayerNorm.', '.ln1.')
    k = k.replace('.output.LayerNorm.', '.ln2.')
    k = k.replace('.intermediate.dense.', '.linear1.')
    k = k.replace('.output.dense.', '.linear2.')
    return k


def _remap_char_key(k: str) -> str:
    """Map a char-module checkpoint key to the MLX CharModule parameter path."""
    k = _remap_deberta_key(k)
    for src, dst in (
        ('word_segmentation_tagger.layers.0.', 'word_seg.dense.'),
        ('word_segmentation_tagger.layers.3.', 'word_seg.classifier.'),
        ('sent_segmentation_tagger.layers.0.', 'sent_seg.dense.'),
        ('sent_segmentation_tagger.layers.3.', 'sent_seg.classifier.'),
        ('word_norm_op_tagger.layers.0.', 'norm_op.dense.'),
        ('word_norm_op_tagger.layers.3.', 'norm_op.classifier.'),
    ):
        k = k.replace(src, dst)
    return k


def get_normalized_surf(surf: str, word_norm_op_tags: list[str], strict: bool = False) -> str:
    """Rebuild a normalized surface from chars and their per-char norm op tags.

    Port of kwja.utils.word_normalization.get_normalized_surf. K=keep, D=delete,
    V=devoice (word-initial kana only in strict mode), S=small kana -> regular,
    P/E=prolonged sign "ー" -> long vowel of the preceding kana (E uses the
    e-row map). In non-strict mode an inapplicable op keeps the original char
    instead of raising.
    """
    assert len(surf) == len(word_norm_op_tags)
    norm = ''
    for i, (c, op) in enumerate(zip(surf, word_norm_op_tags)):
        if op == 'K':
            norm += c
        elif op == 'V':
            if strict and i != 0:
                raise ValueError(f'not an initial kana {c} in {surf}')
            norm += VOICED2VOICELESS.get(c, c)
        elif op == 'D':
            if strict and c not in CHOON_SET and c not in HATSUON_SET and c not in LOWER2UPPER:
                raise ValueError(f'not a removable kana {c} in {surf}')
        elif op == 'S':
            if c in LOWER2UPPER:
                norm += LOWER2UPPER[c]
            else:
                if strict:
                    raise ValueError(f'not a small kana {c} in {surf}')
                norm += c
        elif op in ('P', 'E'):
            table = PROLONGED_MAP if op == 'P' else PROLONGED_MAP_FOR_EROW
            if len(norm) == 0:
                if strict:
                    raise ValueError(f'no preceding kana for {c} in {surf}')
                norm += c
            elif c not in CHOON_SET:
                if strict:
                    raise ValueError(f'not a prolonged sign for {c} in {surf}')
                norm += c
            elif norm[-1] in table:
                norm += table[norm[-1]]
            else:
                if strict:
                    raise ValueError(f'not a valid preceding kana {norm[-1]} in {surf}')
                norm += c
        else:
            raise NotImplementedError(f'unknown word norm op tag {op}')
    return norm


def _special_token_ids(tok) -> set[int]:
    """Token ids whose positions are skipped when decoding char-module outputs:
    bos/eos/sep/pad/cls/mask plus the extra word-boundary ' ' token. [UNK] is
    real text and stays (mirrors upstream CharModuleWriter)."""
    ids = {getattr(tok, f'{prefix}_token_id') for prefix in ('bos', 'eos', 'sep', 'pad', 'cls', 'mask')}
    ids.discard(None)
    space_id = tok.convert_tokens_to_ids(' ')
    if space_id is not None:
        ids.add(space_id)
    return ids


def _decode_sentences(tokens: list[str], tags, input_ids: list[int], special_ids: set) -> list[list[str]]:
    """Group char tokens into sentences from per-char B/I tags (0 = B).

    A B at a non-empty buffer starts a new sentence (upstream set_sentences);
    positions whose *token id* is in special_ids are skipped.
    """
    sentences: list[list[str]] = []
    cur: list[str] = []
    for i, tok in enumerate(tokens):
        if input_ids[i] in special_ids:
            continue
        if tags[i] == 0 and cur:
            sentences.append(cur)
            cur = []
        cur.append(tok)
    if cur:
        sentences.append(cur)
    return sentences


class ConvLayerMLX(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def __call__(self, hidden_states, residual_states):
        return self.LayerNorm(residual_states + nn.gelu(self.conv(hidden_states)))


class DeBERTaWithConv(DebertaV2Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.conv = ConvLayerMLX(config['hidden_size'])

    def __call__(self, h, special_token_indices=None):
        mask = self.get_attention_mask(mx.ones(h.shape[:2]))
        rel_emb = self.get_rel_embedding()
        rel_pos = None
        if special_token_indices is not None and self.relative_attention:
            # kwja deberta_v2.get_rel_pos: rows/cols of special tokens share one
            # reserved relative-position bucket (-position_buckets), which after
            # the +position_buckets embedding shift resolves to bucket 0.
            length = h.shape[1]
            rel_pos = build_relative_position(
                h, h, bucket_size=self.position_buckets, max_position=self.max_relative_positions,
            )
            idx = mx.array(np.array(special_token_indices, dtype=np.int32))
            special_mask = mx.zeros((1, length, length), dtype=mx.bool_)
            special_mask[:, idx, :] = True
            special_mask[:, :, idx] = True
            special_pos = mx.full((1, length, length), -self.position_buckets, dtype=rel_pos.dtype)
            rel_pos = mx.where(special_mask, special_pos, rel_pos)
        out = h
        embed = h
        for i, layer in enumerate(self.layer):
            out = layer(out, mask, query_states=out, relative_pos=rel_pos, rel_embeddings=rel_emb)
            if i == 0:
                out = self.conv(embed, out)
        return out


class CharModule(nn.Module):
    """Char-level modules: word segmentation / sentence segmentation / word norm ops."""

    def __init__(self, config):
        super().__init__()
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DeBERTaWithConv(config)
        self.word_seg = TwoLayerClassifier(config['hidden_size'], len(WORD_SEGMENTATION_TAGS))
        self.sent_seg = TwoLayerClassifier(config['hidden_size'], len(SENT_SEGMENTATION_TAGS))
        self.norm_op = TwoLayerClassifier(config['hidden_size'], len(WORD_NORM_OP_TAGS))

    def __call__(self, input_ids):
        # Returns encoder hidden states; callers apply only the heads they need
        # (same pattern as WordModule, whose heads are applied in Analyzer).
        return self.encoder(self.embeddings(input_ids))


class WordModule(nn.Module):
    """Word-level POS tagging and NER."""

    def __init__(self, config):
        super().__init__()
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DeBERTaWithConv(config)
        self.pos_tagger = TwoLayerClassifier(config['hidden_size'], 14)
        self.ne_tagger = TwoLayerClassifier(config['hidden_size'], 17)
        self.subpos_tagger = TwoLayerClassifier(config['hidden_size'], len(SUBPOS_TAGS))
        self.conjtype_tagger = TwoLayerClassifier(config['hidden_size'], len(CONJTYPE_TAGS))
        self.conjform_tagger = TwoLayerClassifier(config['hidden_size'], len(CONJFORM_TAGS))
        self.reading_tagger = TwoLayerClassifier(config['hidden_size'], len(_reading_id2reading()))
        self.word_feature_tagger = LoRAMultiLabelHead(config['hidden_size'], len(WORD_FEATURES), rank=4)
        # Base-phrase features (64 KNP features incl 節-区切/節-主辞): same LoRA multi-label head
        self.base_phrase_feature_tagger = LoRAMultiLabelHead(config['hidden_size'], len(BASE_PHRASE_FEATURES), rank=4)
        self.dependency_head = DependencyHead(config['hidden_size'])
        # Cohesion analysis: 9 relations (7 PAS cases + ノ + coref "="), LoRA rank 1
        self.cohesion_analyzer = LoRARelationWiseWordSelectionHead(
            len(COHESION_RELATIONS), config['hidden_size'], rank=1,
        )
        # Discourse relation analysis: 7 classes (談話関係なし + 6 relations)
        self.discourse_relation_analyzer = WordSelectionHead(
            len(DISCOURSE_RELATIONS), config['hidden_size'],
        )

    def __call__(self, input_ids, special_token_indices=None):
        return self.encoder(self.embeddings(input_ids), special_token_indices=special_token_indices)


class LoRAMultiLabelHead(nn.Module):
    """KWJA LoRASequenceMultiLabelingHead: sigmoid probabilities with a
    per-label low-rank delta (dense_a/dense_b, rank 4) added to the dense path."""

    def __init__(self, hidden_size: int, num_labels: int, rank: int = 4):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        # NOTE: plain mx.array attrs are NOT registered params in this mlx
        # version (no nn.Parameter); load_weights must set them explicitly.
        self.dense_a = mx.zeros((hidden_size, rank, num_labels))
        self.dense_b = mx.zeros((rank, hidden_size, num_labels))
        self.classifier_weight = mx.zeros((hidden_size, num_labels))
        self.classifier_bias = mx.zeros((num_labels,))

    def __call__(self, x):
        dense_out = nn.gelu(self.dense(x))  # (B, L, H)
        delta = mx.einsum('hrl,ril->hil', self.dense_a, self.dense_b)  # (H, H, labels)
        delta_out = mx.einsum('bsh,hil->bsil', x, delta)  # (B, L, H, labels)
        hidden = nn.gelu(dense_out[..., :, None] + delta_out)  # (B, L, H, labels)
        logits = mx.einsum('bshl,hl->bsl', hidden, self.classifier_weight) + self.classifier_bias
        return mx.sigmoid(logits)


MASKED = -np.inf
_reading_id2reading_cache: Optional[list[str]] = None


def _reading_id2reading() -> list[str]:
    """Reading vocab: id 0 = [UNK], 1 = [ID], then vocab.txt lines (deduped)."""
    global _reading_id2reading_cache
    if _reading_id2reading_cache is None:
        reading2id = get_reading2reading_id()
        id2reading = [''] * len(reading2id)
        for reading, rid in reading2id.items():
            id2reading[rid] = reading
        _reading_id2reading_cache = id2reading
    return _reading_id2reading_cache


def _decode_morpheme_attributes(pos_preds, subpos_logits, conjtype_logits, conjform_logits):
    """Masked-argmax decoding (mirrors kwja get_morpheme_attribute_predictions).

    subpos is restricted to the subpos set of the predicted POS; conjform is
    restricted to the conjform set of the predicted conjtype and only decoded
    for inflectable (pos, subpos) pairs — otherwise conjtype/conjform → "*".
    Returns (subpos_preds, conjtype_preds, conjform_preds) label index lists.
    """
    n = len(pos_preds)
    conjtype_preds = [int(x) for x in np.argmax(np.asarray(conjtype_logits), axis=1)]
    subpos_preds: list[int] = []
    conjform_preds: list[int] = []
    for i in range(n):
        pos_tag = POS_TAGS[pos_preds[i]]
        possible_subpos = set(POS_TAG_SUBPOS_TAG2SUBPOS_ID[pos_tag])
        subpos_scores = np.array(subpos_logits[i], dtype=np.float64)
        for idx, tag in enumerate(SUBPOS_TAGS):
            if tag not in possible_subpos:
                subpos_scores[idx] = MASKED
        subpos_idx = int(np.argmax(subpos_scores))
        subpos_preds.append(subpos_idx)
        subpos_tag = SUBPOS_TAGS[subpos_idx]
        if (pos_tag, subpos_tag) in INFLECTABLE:
            conjtype_tag = CONJTYPE_TAGS[conjtype_preds[i]]
            possible_conjform = set(CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID[conjtype_tag])
            conjform_scores = np.array(conjform_logits[i], dtype=np.float64)
            for idx, tag in enumerate(CONJFORM_TAGS):
                if tag not in possible_conjform:
                    conjform_scores[idx] = MASKED
            conjform_idx = int(np.argmax(conjform_scores))
        else:
            conjtype_preds[i] = CONJTYPE_TAGS.index('*')
            conjform_idx = CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID['*']['*']
        conjform_preds.append(conjform_idx)
    return subpos_preds, conjtype_preds, conjform_preds


def _decode_clauses(n_words: int, bp_array: np.ndarray) -> list[tuple[int, int, int]]:
    """Split word indices into clauses from 節-区切* (mirrors kwja add_base_phrase_features).

    Returns [(start, end_exclusive, head_idx), ...] where head_idx has max 節-主辞 prob.
    Last clause is force-closed if no boundary fires (upstream behaviour).
    """
    try:
        head_col = BASE_PHRASE_FEATURES.index("節-主辞")
    except ValueError:
        return [(0, n_words, 0)] if n_words else []
    bound_cols = [k for k, f in enumerate(BASE_PHRASE_FEATURES) if f.startswith("節-区切")]
    clauses = []
    start = 0
    for i in range(n_words):
        if any(bp_array[i, c] >= 0.5 for c in bound_cols):
            clauses.append((start, i + 1, start + int(np.argmax(bp_array[start:i + 1, head_col]))))
            start = i + 1
    if start < n_words:
        clauses.append((start, n_words, start + int(np.argmax(bp_array[start:, head_col]))))
    return clauses


class Analyzer:
    """Full KWJA-MLX pipeline: char → word → seq2seq."""

    def __init__(self, model_size: str = "base", quantize: Optional[str] = None):
        """
        Args:
            model_size: "base" (only size available in MLX currently)
            quantize: None, "fp16", "q8", or "q4"
        """
        self.model_size = model_size
        self.quantize_mode = quantize
        self._load_models()

    def _load_models(self) -> None:
        repo = "kevinzhow/kwja-mlx"
        snapshot_base = f"{self.model_size}/"

        # Char module
        char_config = {
            'hidden_size': 768, 'num_hidden_layers': 12, 'num_attention_heads': 12,
            'intermediate_size': 3072, 'vocab_size': 22013, 'max_position_embeddings': 512,
            'relative_attention': True, 'position_buckets': 256, 'max_relative_positions': -1,
            'pos_att_type': ['p2c', 'c2p'], 'share_att_key': True,
            'hidden_dropout_prob': 0.0, 'attention_probs_dropout_prob': 0.0,
            'norm_rel_ebd': 'layer_norm', 'position_biased_input': False,
        }
        char_path = hf_hub_download(repo, f"{snapshot_base}char_deberta-v2-base-wwm.mlx.safetensors")
        char_weights = load_file(char_path)
        char_mapped = {}
        for k, v in char_weights.items():
            char_mapped[_remap_char_key(k)] = v
        self.char_model = CharModule(char_config)
        self.char_model.load_weights([(k, mx.array(v)) for k, v in char_mapped.items()])
        self.char_tok = AutoTokenizer.from_pretrained(
            'ku-nlp/deberta-v2-base-japanese-char-wwm',
            do_word_tokenize=False, additional_special_tokens=[' ']
        )
        self._char_special_ids = _special_token_ids(self.char_tok)

        # Word module
        word_config = {
            'hidden_size': 768, 'num_hidden_layers': 12, 'num_attention_heads': 12,
            'intermediate_size': 3072, 'vocab_size': 32008, 'max_position_embeddings': 512,
            'relative_attention': True, 'position_buckets': 256, 'max_relative_positions': -1,
            'pos_att_type': ['p2c', 'c2p'], 'share_att_key': True,
            'hidden_dropout_prob': 0.0, 'attention_probs_dropout_prob': 0.0,
            'norm_rel_ebd': 'layer_norm', 'position_biased_input': False,
        }
        word_path = hf_hub_download(repo, f"{snapshot_base}word_deberta-v2-base.mlx.safetensors")
        word_weights = load_file(word_path)
        skip_prefixes: list = []  # all word heads now loaded (incl base_phrase)
        word_mapped = {}
        for k, v in word_weights.items():
            new_k = _remap_deberta_key(k)
            new_k = new_k.replace('pos_tagger.layers.0.', 'pos_tagger.dense.')
            new_k = new_k.replace('pos_tagger.layers.3.', 'pos_tagger.classifier.')
            new_k = new_k.replace('ne_tagger.layers.0.', 'ne_tagger.dense.')
            new_k = new_k.replace('ne_tagger.layers.3.', 'ne_tagger.classifier.')
            for name in ('subpos_tagger', 'conjtype_tagger', 'conjform_tagger', 'reading_tagger'):
                new_k = new_k.replace(f'{name}.layers.0.', f'{name}.dense.')
                new_k = new_k.replace(f'{name}.layers.3.', f'{name}.classifier.')
            new_k = new_k.replace('word_feature_tagger.delta.', 'word_feature_tagger.')
            new_k = new_k.replace('base_phrase_feature_tagger.delta.', 'base_phrase_feature_tagger.')
            new_k = new_k.replace('dependency_type_parser.layers.0.', 'dependency_head.dependency_type_parser.dense.')
            new_k = new_k.replace('dependency_type_parser.layers.3.', 'dependency_head.dependency_type_parser.classifier.')
            new_k = new_k.replace('dependency_parser.', 'dependency_head.dependency_parser.')
            if any(x in new_k for x in skip_prefixes):
                continue
            word_mapped[new_k] = v

        # Load CRF parameters as numpy arrays (small tensors, no need for MLX)
        self.crf_start = word_weights['crf.start_transitions']   # (17,)
        self.crf_end = word_weights['crf.end_transitions']       # (17,)
        self.crf_trans = word_weights['crf.transitions']         # (17, 17)

        # Remove CRF keys from MLX model loading (they're not part of the nn.Module)
        word_mapped = {k: v for k, v in word_mapped.items() if not k.startswith('crf.')}

        self.word_model = WordModule(word_config)
        self.word_model.load_weights([(k, mx.array(v)) for k, v in word_mapped.items()])
        self.word_tok = AutoTokenizer.from_pretrained(
            'ku-nlp/deberta-v2-base-japanese',
            additional_special_tokens=['[著者]', '[読者]', '[不特定:人]', '[不特定:物]', '[NULL]', '[NA]', '[ROOT]', ' ']
        )

        # Seq2seq module (T5)
        t5_config = {
            'hidden_size': 768, 'num_heads': 12, 'ff_size': 2048,
            'num_encoder_layers': 12, 'num_decoder_layers': 12, 'vocab_size': 32100,
        }
        t5_path = hf_hub_download(repo, f"{snapshot_base}seq2seq_t5-base.mlx.safetensors")
        t5_weights = load_file(t5_path)
        self.seq2seq_model = T5ForConditionalGeneration(t5_config)
        self.seq2seq_model.load_weights([(k, mx.array(v)) for k, v in t5_weights.items()])
        self.seq2seq_tok = AutoTokenizer.from_pretrained('retrieva-jp/t5-base-long')

        # Apply quantization
        if self.quantize_mode:
            self._apply_quantization()

        mx.eval(self.char_model.parameters(), self.word_model.parameters(), self.seq2seq_model.parameters())

    def _apply_quantization(self) -> None:
        def _is_linear(_, m):
            return isinstance(m, nn.Linear)

        if self.quantize_mode == "fp16":
            for model in [self.char_model, self.word_model, self.seq2seq_model]:
                params = dict(nn.utils.tree_map(
                    lambda x: x.astype(mx.float16) if x.dtype == mx.float32 else x,
                    model.parameters()
                ))
                model.update(params)
        elif self.quantize_mode == "q8":
            for model in [self.char_model, self.word_model, self.seq2seq_model]:
                nn.quantize(model, bits=8, group_size=64, class_predicate=_is_linear)
        elif self.quantize_mode == "q4":
            for model in [self.char_model, self.word_model, self.seq2seq_model]:
                nn.quantize(model, bits=4, group_size=64, class_predicate=_is_linear)

    def _viterbi_decode(self, emissions: np.ndarray) -> list[int]:
        """Viterbi decode NER tags with CRF transitions.

        emissions: (seq_len, n_tags) numpy array of emission scores
        Returns: list of tag indices (best path)
        """
        seq_len, n_tags = emissions.shape

        # score[t] holds the best score arriving at each tag at time t
        score = self.crf_start + emissions[0]  # (n_tags,)
        # backpointers[t] stores the best previous tag for each current tag
        backpointers = []

        for t in range(1, seq_len):
            # score_prev[:, None] + trans[i, j] = score of coming from tag i to tag j
            # shape: (n_tags, n_tags) where [i, j] = score[i] + trans[i, j]
            next_scores = score[:, None] + self.crf_trans  # (n_tags, n_tags)
            best_prev = np.argmax(next_scores, axis=0)  # (n_tags,)
            score = next_scores[best_prev, np.arange(n_tags)] + emissions[t]
            backpointers.append(best_prev)

        # Add end transitions
        score += self.crf_end
        best_last = int(np.argmax(score))

        # Backtrack
        best_path = [best_last]
        for bp in reversed(backpointers):
            best_path.append(int(bp[best_path[-1]]))
        best_path.reverse()

        return best_path

    def segment(self, text: str) -> list[str]:
        """Segment text into words using char module."""
        if not text or not text.strip():
            return []
        enc = self.char_tok(text, return_tensors='np', truncation=True, max_length=512)
        h = self.char_model(mx.array(enc['input_ids']))
        logits = self.char_model.word_seg(h)
        mx.eval(logits)
        tags = np.array(mx.argmax(logits[0], axis=-1))
        tokens = self.char_tok.convert_ids_to_tokens(enc['input_ids'][0])

        words = []
        cur = ''
        for i in range(1, len(tokens)):
            if tokens[i] in ('[SEP]', '[PAD]'):
                break
            if tags[i] == 0:  # B (begin new word)
                if cur:
                    words.append(cur)
                cur = tokens[i]
            else:  # I (inside word)
                cur += tokens[i]
        if cur:
            words.append(cur)
        return words

    def segment_batch(self, texts: list[str], batch_size: int = 64) -> list[list[str]]:
        """Batch word segmentation. Much faster than calling segment() in a loop.

        Args:
            texts: list of input strings
            batch_size: number of sentences per batch

        Returns:
            list of word lists, one per input text
        """
        results = [[] for _ in range(len(texts))]

        # Filter out empty strings, track original indices
        valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not valid:
            return results

        for batch_start in range(0, len(valid), batch_size):
            batch = valid[batch_start:batch_start + batch_size]
            batch_texts = [t for _, t in batch]

            # Batch encode with padding
            enc = self.char_tok(
                batch_texts, return_tensors='np', truncation=True,
                max_length=512, padding=True,
            )
            input_ids = mx.array(enc['input_ids'])  # (B, L)
            h = self.char_model(input_ids)  # (B, L, H)
            logits = self.char_model.word_seg(h)  # (B, L, 2)
            mx.eval(logits)
            all_tags = np.array(mx.argmax(logits, axis=-1))  # (B, L)

            # Decode each sentence in the batch
            for j, (orig_idx, text) in enumerate(batch):
                ids_j = enc['input_ids'][j]
                tags_j = all_tags[j]
                tokens_j = self.char_tok.convert_ids_to_tokens(ids_j)

                words = []
                cur = ''
                for k in range(1, len(tokens_j)):
                    if tokens_j[k] in ('[SEP]', '[PAD]'):
                        break
                    if tags_j[k] == 0:  # B (begin new word)
                        if cur:
                            words.append(cur)
                        cur = tokens_j[k]
                    else:  # I (inside word)
                        cur += tokens_j[k]
                if cur:
                    words.append(cur)
                results[orig_idx] = words

        return results

    def sentences(self, text: str) -> list[list[str]]:
        """Split text into sentences with the char module's sentence segmentation head.

        Returns a list of sentences, each the list of its character tokens
        (special tokens filtered out; ''.join(sentence) gives the text).
        """
        if not text or not text.strip():
            return []
        enc = self.char_tok(text, return_tensors='np', truncation=True, max_length=512)
        h = self.char_model(mx.array(enc['input_ids']))
        logits = self.char_model.sent_seg(h)
        mx.eval(logits)
        tags = np.array(mx.argmax(logits[0], axis=-1))  # 0 = B (new sentence), 1 = I
        tokens = self.char_tok.convert_ids_to_tokens(enc['input_ids'][0])
        return _decode_sentences(tokens, tags, enc['input_ids'][0].tolist(), self._char_special_ids)

    def tokenize_batch(self, texts: list[str], batch_size: int = 64) -> list[list[Morpheme]]:
        """Batch tokenization (segment + POS + NER). Much faster than loop.

        Args:
            texts: list of input strings
            batch_size: number of sentences per batch

        Returns:
            list of Morpheme lists, one per input text
        """
        # Step 1: batch segmentation
        all_words = self.segment_batch(texts, batch_size=batch_size)

        # Step 2: batch POS + NER using true batched DeBERTa inference
        valid = [(i, w) for i, w in enumerate(all_words) if w]
        if not valid:
            return [[] for _ in range(len(texts))]

        indices = [i for i, _ in valid]
        word_lists = [w for _, w in valid]
        tagged_all = self.tag_batch(word_lists, batch_size=batch_size)

        results = [[] for _ in range(len(texts))]
        for orig_idx, tagged in zip(indices, tagged_all):
            results[orig_idx] = [
                Morpheme(
                    surface=w, pos=p, ne=n,
                    subpos=sp, conjtype=ct, conjform=cf,
                    reading_cls=r, word_features=wf,
                    base_phrase_features=bpf,
                )
                for w, p, n, sp, ct, cf, r, wf, bpf in tagged
            ]
            add_dependency(self.word_model, results[orig_idx], self.word_tok)

        return results

    def tag_batch(self, word_lists: list[list[str]], batch_size: int = 32) -> list[list[tuple]]:
        """Batch tag pre-segmented sentences: POS, NER, subpos, conjtype,
        conjform, reading and word features.

        Batches the DeBERTa word model forward pass by padding inputs.
        CRF Viterbi decoding remains per-sentence (fast, O(L×K²)).

        Args:
            word_lists: list of word lists (each is a pre-segmented sentence)
            batch_size: number of sentences per DeBERTa forward pass

        Returns:
            list of [(word, pos, ne, subpos, conjtype, conjform, reading_cls,
            word_features), ...] for each sentence
        """
        all_results: list[list[tuple]] = []

        for batch_start in range(0, len(word_lists), batch_size):
            batch_words = word_lists[batch_start:batch_start + batch_size]

            # Batch tokenize all sentences in this batch
            enc = self.word_tok(
                batch_words,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='np',
            )

            # Run DeBERTa on the full batch: (B, L) -> (B, L, H)
            input_ids = mx.array(enc['input_ids'])
            h = self.word_model(input_ids)
            mx.eval(h)

            # Per-sentence: first-subword pooling, POS argmax, NER Viterbi
            for j, words in enumerate(batch_words):
                n_words = len(words)

                # First-subword pooling using word_ids
                word_ids = enc.word_ids(j)
                first_idx: list[int] = [0] * n_words
                for ti, wid in enumerate(word_ids):
                    if wid is not None and wid < n_words and first_idx[wid] == 0:
                        first_idx[wid] = ti

                idx = mx.array(np.array(first_idx, dtype=np.int32))
                pooled = h[j, idx][None, :, :]  # (1, n_words, H)
                all_results.append(self._lexical_tag(pooled, words))

        return all_results

    def _lexical_tag(self, pooled: 'mx.array', words: list[str]) -> list[tuple]:
        """Run all word-level heads on pooled hidden states and decode.

        Returns [(word, pos, ne, subpos, conjtype, conjform, reading_cls,
        word_features, base_phrase_features), ...].
        """
        n_words = len(words)
        pos_logits = self.word_model.pos_tagger(pooled)
        ne_logits = self.word_model.ne_tagger(pooled)
        subpos_logits = self.word_model.subpos_tagger(pooled)
        conjtype_logits = self.word_model.conjtype_tagger(pooled)
        conjform_logits = self.word_model.conjform_tagger(pooled)
        reading_logits = self.word_model.reading_tagger(pooled)
        wf_probs = self.word_model.word_feature_tagger(pooled)
        bp_probs = self.word_model.base_phrase_feature_tagger(pooled)
        mx.eval(pos_logits, ne_logits, subpos_logits, conjtype_logits,
                conjform_logits, reading_logits, wf_probs, bp_probs)

        pos_preds = np.array(mx.argmax(pos_logits[0], axis=-1))
        ne_emissions = np.array(ne_logits[0])
        ne_preds = np.array(self._viterbi_decode(ne_emissions))

        subpos_preds, conjtype_preds, conjform_preds = _decode_morpheme_attributes(
            pos_preds,
            np.array(subpos_logits[0]),
            np.array(conjtype_logits[0]),
            np.array(conjform_logits[0]),
        )

        id2reading = _reading_id2reading()
        reading_preds = np.array(mx.argmax(reading_logits[0], axis=-1))
        wf_array = np.array(wf_probs[0])  # (n_words, n_features)
        bp_array = np.array(bp_probs[0])  # (n_words, 64)

        tagged = []
        for i in range(n_words):
            reading_id = int(reading_preds[i])
            # UNK / ID → fall back to the surface string
            reading_cls = words[i] if reading_id in (0, 1) else id2reading[reading_id]
            word_features = tuple(
                WORD_FEATURES[k] for k in range(len(WORD_FEATURES)) if wf_array[i, k] >= 0.5
            )
            base_phrase_features = tuple(
                BASE_PHRASE_FEATURES[k] for k in range(len(BASE_PHRASE_FEATURES)) if bp_array[i, k] >= 0.5
            )
            tagged.append((
                words[i],
                POS_TAGS[pos_preds[i]],
                NE_TAGS[ne_preds[i]],
                SUBPOS_TAGS[subpos_preds[i]],
                CONJTYPE_TAGS[conjtype_preds[i]],
                CONJFORM_TAGS[conjform_preds[i]],
                reading_cls,
                word_features,
                base_phrase_features,
            ))
        return tagged

    def tag(self, words: list[str]) -> list[tuple]:
        """Tag pre-segmented words: POS, NER, subpos, conjtype, conjform,
        reading and word features. Returns [(word, pos, ne, subpos, conjtype,
        conjform, reading_cls, word_features), ...]."""
        if not words:
            return []
        return self.tag_batch([words])[0]

    def analyze_cohesion(self, words: list[str], discourse_threshold: float = 0.0) -> tuple:
        """Run the cohesion (PAS/橋渡し/共参照) and discourse relation heads.

        Single-sentence input: antecedent candidates are the sentence's own
        morphemes plus the 7 special tokens (exophora referents, [NULL],
        [NA], [ROOT]), appended to the encoder input like upstream.

        Returns (per-morpheme cohesion dicts, sentence-level discourse list).

        TODO: document-level analysis. KWJA splits long documents into
        overlapping sub-documents (document_split_stride=2: each sub-document
        reuses the last 2 sentences of the previous one as left context and
        only decodes those trailing "target" sentences); this port decodes
        each sentence independently.
        """
        if not words:
            return [], []
        # Token layout mirrors upstream: [CLS] + morpheme subwords + [SEP]
        # + 7 special tokens (+ padding), truncated so the total fits 256.
        enc = self.word_tok(
            words, is_split_into_words=True, return_tensors='np',
            truncation=True, max_length=256 - len(SPECIALS),
        )
        special_ids = self.word_tok.convert_tokens_to_ids(list(SPECIALS))
        sub_ids = enc['input_ids'][0]
        input_ids = np.concatenate([sub_ids, np.array(special_ids, dtype=sub_ids.dtype)])
        special_token_indices = list(range(len(sub_ids), len(input_ids)))
        h = self.word_model(mx.array(input_ids[None, :]), special_token_indices=special_token_indices)
        mx.eval(h)

        # First-subword pooling for morphemes; last positions for specials.
        word_ids = enc.word_ids(0)
        first_idx: list[int] = [0] * len(words)
        for ti, wid in enumerate(word_ids):
            if wid is not None and wid < len(words) and first_idx[wid] == 0:
                first_idx[wid] = ti
        pool_idx = mx.array(np.array(first_idx + special_token_indices, dtype=np.int32))
        pooled = h[0, pool_idx][None, :, :]  # (1, n_words + 7, H)
        return analyze_heads(
            self.word_model.cohesion_analyzer,
            self.word_model.discourse_relation_analyzer,
            pooled, len(words), discourse_threshold=discourse_threshold,
        )

    def read(self, words: list[str]) -> list[tuple[str, str, str, str]]:
        """Get reading, lemma, canon for words using seq2seq T5.
        Returns [(surface, reading, lemma, canon), ...]."""
        # Morphemes joined by <extra_id_5> sentinel; drop stray word-boundary
        # markers (▁) so joins stay glued when retokenized by SentencePiece.
        src_text = '<extra_id_5>'.join(words)
        tokens = [x for x in self.seq2seq_tok.tokenize(src_text) if x != '▁']
        input_ids = self.seq2seq_tok.convert_tokens_to_ids(tokens) + [self.seq2seq_tok.eos_token_id]

        input_mx = mx.array([input_ids])
        attention_mask = mx.ones_like(input_mx)

        generated = self.seq2seq_model.generate(
            input_mx, attention_mask, max_length=len(input_ids) * 5,
            eos_token_id=1, decoder_start_token_id=0,
        )
        decoded = self.seq2seq_tok.decode(generated[1:])

        # Parse: <extra_id_0>surf<extra_id_1>reading<extra_id_2>lemma<extra_id_3>canon
        results = []
        morphemes = decoded.split('<extra_id_0>')[1:]
        for m in morphemes:
            try:
                parts1 = m.split('<extra_id_1>')
                surf = parts1[0]
                parts2 = parts1[1].split('<extra_id_2>')
                reading = parts2[0]
                parts3 = parts2[1].split('<extra_id_3>')
                lemma = parts3[0]
                canon = parts3[1] if len(parts3) > 1 else ""
                results.append((surf, reading, lemma, canon))
            except (IndexError, KeyError):
                results.append((m.split('<extra_id_')[0], '', '', ''))
        return results

    def analyze(self, text: str, with_reading: bool = True, with_cohesion: bool = True) -> AnalysisResult:
        """Full analysis pipeline: segmentation → POS/NER → reading/lemma
        → cohesion/discourse relations."""
        if not text or not text.strip():
            return AnalysisResult(text=text or "", morphemes=[])

        # Step 1: Segment
        words = self.segment(text)

        # Step 2: POS + NER
        tagged = self.tag(words)

        # Step 3: Reading/lemma (optional, slower due to autoregressive T5)
        readings = None
        if with_reading:
            readings = self.read(words)

        # Step 4: Cohesion + discourse relations (optional, one DeBERTa pass)
        cohesion_rels: list = []
        discourse: list = []
        if with_cohesion and words:
            cohesion_rels, discourse = self.analyze_cohesion(words)

        # Build result
        morphemes = []
        for i, (word, pos, ne, subpos, conjtype, conjform, reading_cls, word_features, base_phrase_features) in enumerate(tagged):
            m = Morpheme(
                surface=word, pos=pos, ne=ne,
                subpos=subpos, conjtype=conjtype, conjform=conjform,
                reading_cls=reading_cls, word_features=word_features,
                base_phrase_features=base_phrase_features,
            )
            if readings and i < len(readings):
                _, reading, lemma, canon = readings[i]
                m.reading = reading
                m.lemma = lemma
                m.canon = canon
            if i < len(cohesion_rels):
                m.cohesion = cohesion_rels[i]
            morphemes.append(m)
        add_dependency(self.word_model, morphemes, self.word_tok)

        return AnalysisResult(text=text, morphemes=morphemes, discourse=discourse)

    def tokenize(self, text: str) -> list[Morpheme]:
        """Quick segmentation + lexical tagging (no seq2seq reading). Fastest mode."""
        if not text or not text.strip():
            return []
        words = self.segment(text)
        tagged = self.tag(words)
        morphemes = [
            Morpheme(
                surface=w, pos=p, ne=n,
                subpos=sp, conjtype=ct, conjform=cf,
                reading_cls=r, word_features=wf,
                base_phrase_features=bpf,
            )
            for w, p, n, sp, ct, cf, r, wf, bpf in tagged
        ]
        add_dependency(self.word_model, morphemes, self.word_tok)
        return morphemes
