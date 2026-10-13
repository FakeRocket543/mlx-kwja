"""kwja-mlx: Full integrated Japanese analyzer (char → word → seq2seq)."""

import re
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from mlx import nn
from safetensors.numpy import load_file
from transformers import AutoTokenizer

from kwja_mlx.deberta_v2_mlx import DebertaV2Embeddings, DebertaV2Encoder
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

    def __call__(self, h, **kw):
        mask = self.get_attention_mask(mx.ones(h.shape[:2]))
        rel_emb = self.get_rel_embedding()
        out = h
        embed = h
        for i, layer in enumerate(self.layer):
            out = layer(out, mask, query_states=out, relative_pos=None, rel_embeddings=rel_emb)
            if i == 0:
                out = self.conv(embed, out)
        return out


class CharModule(nn.Module):
    """Character-level word segmentation."""

    def __init__(self, config):
        super().__init__()
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DeBERTaWithConv(config)
        self.word_seg = TwoLayerClassifier(config['hidden_size'], 2)

    def __call__(self, input_ids):
        return self.word_seg(self.encoder(self.embeddings(input_ids)))


class WordModule(nn.Module):
    """Word-level POS tagging and NER."""

    def __init__(self, config):
        super().__init__()
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DeBERTaWithConv(config)
        self.pos_tagger = TwoLayerClassifier(config['hidden_size'], 14)
        self.ne_tagger = TwoLayerClassifier(config['hidden_size'], 17)

    def __call__(self, input_ids):
        return self.encoder(self.embeddings(input_ids))


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
            new_k = _remap_deberta_key(k)
            new_k = new_k.replace('word_segmentation_tagger.layers.0.', 'word_seg.dense.')
            new_k = new_k.replace('word_segmentation_tagger.layers.3.', 'word_seg.classifier.')
            if 'sent_seg' in new_k or 'norm_op' in new_k:
                continue
            char_mapped[new_k] = v
        self.char_model = CharModule(char_config)
        self.char_model.load_weights([(k, mx.array(v)) for k, v in char_mapped.items()])
        self.char_tok = AutoTokenizer.from_pretrained(
            'ku-nlp/deberta-v2-base-japanese-char-wwm',
            do_word_tokenize=False, additional_special_tokens=[' ']
        )

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
        skip_prefixes = ['reading', 'subpos', 'conjtype', 'conjform', 'word_feature',
                         'base_phrase', 'cohesion', 'dependency', 'discourse']
        word_mapped = {}
        for k, v in word_weights.items():
            new_k = _remap_deberta_key(k)
            new_k = new_k.replace('pos_tagger.layers.0.', 'pos_tagger.dense.')
            new_k = new_k.replace('pos_tagger.layers.3.', 'pos_tagger.classifier.')
            new_k = new_k.replace('ne_tagger.layers.0.', 'ne_tagger.dense.')
            new_k = new_k.replace('ne_tagger.layers.3.', 'ne_tagger.classifier.')
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
        logits = self.char_model(mx.array(enc['input_ids']))
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
            logits = self.char_model(input_ids)  # (B, L, 2)
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
            results[orig_idx] = [Morpheme(surface=w, pos=p, ne=n) for w, p, n in tagged]

        return results

    def tag_batch(self, word_lists: list[list[str]], batch_size: int = 32) -> list[list[tuple[str, str, str]]]:
        """Batch POS tag and NER for multiple pre-segmented sentences.

        Batches the DeBERTa word model forward pass by padding inputs.
        CRF Viterbi decoding remains per-sentence (fast, O(L×K²)).

        Args:
            word_lists: list of word lists (each is a pre-segmented sentence)
            batch_size: number of sentences per DeBERTa forward pass

        Returns:
            list of [(word, pos, ne), ...] for each sentence
        """
        all_results: list[list[tuple[str, str, str]]] = []

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

                pos_logits = self.word_model.pos_tagger(pooled)
                ne_logits = self.word_model.ne_tagger(pooled)
                mx.eval(pos_logits, ne_logits)

                pos_preds = np.array(mx.argmax(pos_logits[0], axis=-1))
                ne_emissions = np.array(ne_logits[0])
                ne_preds = np.array(self._viterbi_decode(ne_emissions))

                tagged = [
                    (words[i], POS_TAGS[pos_preds[i]], NE_TAGS[ne_preds[i]])
                    for i in range(n_words)
                ]
                all_results.append(tagged)

        return all_results

    def tag(self, words: list[str]) -> list[tuple[str, str, str]]:
        """POS tag and NER pre-segmented words. Returns [(word, pos, ne), ...]."""
        if not words:
            return []
        enc = self.word_tok(words, is_split_into_words=True, return_tensors='np',
                            truncation=True, max_length=256)
        h = self.word_model(mx.array(enc['input_ids']))
        mx.eval(h)

        # First-subword pooling
        word_ids = enc.word_ids(0)
        first_idx: list[int] = [0] * len(words)
        for ti, wid in enumerate(word_ids):
            if wid is not None and wid < len(words) and first_idx[wid] == 0:
                first_idx[wid] = ti

        idx = mx.array(np.array(first_idx, dtype=np.int32))
        pooled = h[0, idx][None, :, :]

        pos_logits = self.word_model.pos_tagger(pooled)
        ne_logits = self.word_model.ne_tagger(pooled)
        mx.eval(pos_logits, ne_logits)

        pos_preds = np.array(mx.argmax(pos_logits[0], axis=-1))
        ne_emissions = np.array(ne_logits[0])
        ne_preds = np.array(self._viterbi_decode(ne_emissions))

        return [
            (words[i], POS_TAGS[pos_preds[i]], NE_TAGS[ne_preds[i]])
            for i in range(len(words))
        ]

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

    def analyze(self, text: str, with_reading: bool = True) -> AnalysisResult:
        """Full analysis pipeline: segmentation → POS/NER → reading/lemma."""
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

        # Build result
        morphemes = []
        for i, (word, pos, ne) in enumerate(tagged):
            m = Morpheme(surface=word, pos=pos, ne=ne)
            if readings and i < len(readings):
                _, reading, lemma, canon = readings[i]
                m.reading = reading
                m.lemma = lemma
                m.canon = canon
            morphemes.append(m)

        return AnalysisResult(text=text, morphemes=morphemes)

    def tokenize(self, text: str) -> list[Morpheme]:
        """Quick segmentation + POS (no reading). Fastest mode."""
        if not text or not text.strip():
            return []
        words = self.segment(text)
        tagged = self.tag(words)
        return [Morpheme(surface=w, pos=p, ne=n) for w, p, n in tagged]
