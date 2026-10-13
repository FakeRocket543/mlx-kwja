# kwja-mlx — KWJA Japanese Analyzer on Apple Silicon MLX

Complete MLX port of [KWJA](https://github.com/ku-nlp/kwja) (Kyoto-Waseda Japanese Analyzer).  
All three modules running on Metal GPU with full accuracy.

## Features

- **Word segmentation** (char module, DeBERTa-v2) + **sentence split** (`sentences()`, sent_seg B/I)
- **Surface normalization** (norm_op K/D/V/S/P/E → lemma)
- **POS tagging** — 14 JUMAN++ POS classes + **subpos** (35), **conjtype** (33), **conjform** (81, masked-argmax)
- **Named Entity Recognition** — PERSON, LOCATION, ORGANIZATION, ARTIFACT, DATE, TIME, MONEY, PERCENT (+ CRF Viterbi)
- **Reading prediction** ×2 — word-classifier (12907 vocab) + seq2seq T5 — 東京→とうきょう, 行った→おこなった
- **Lemmatization & canonicalization** (seq2seq T5)
- **Word/base-phrase features** — 文節-区切/基本句-主辞 (sigmoid) + 64 KNP base-phrase features (節-区切/節-主辞 → clause split)
- **Dependency parsing** — score `w·GELU(Ws·hi+Wt·hj)`, topk-4 + cycle/single-root decode, types D/P/A/I (`parent_index` = base-phrase index, -1 = root)
- **Cohesion** — PAS (ガ/ヲ/ニ/ガ２/デ/ト/時間) + BR (ノ) + coreference (=), exophora 著者/読者/不特定 (`m.cohesion`)
- **Discourse relations** — 7 classes incl 原因・理由/条件/逆接 (`result.discourse`)
- **Document mode** — `analyze_document(texts)`: lexical per sentence, merged-window cross-sentence cohesion
- **Quantization**: FP32, FP16, Q8, Q4 — all with near-perfect accuracy

## Quick Start

```python
from kwja_mlx import Analyzer

analyzer = Analyzer(quantize="q8")  # q8 recommended: fast + accurate

# Quick mode (segmentation + POS + NER, no reading)
morphemes = analyzer.tokenize("東京都庁に行きました")
for m in morphemes:
    print(f"{m.surface}/{m.pos}")
# 東京/名詞  都庁/名詞  に/助詞  行き/動詞  ました/接尾辞

# Full mode (+ reading/lemma/canon via T5, + dependency/cohesion)
result = analyzer.analyze("東京都庁に行きました")
for m in result.morphemes:
    print(f"{m.surface}/{m.pos} [{m.reading}] parent={m.parent_index}/{m.dep_type} {m.cohesion}")
# 東京/名詞 [とうきょう]  都庁/名詞 [とちょう]  に/助詞 [に]  行き/動詞 [いき]  ました/接尾辞 [ました]

# Document mode (cross-sentence coreference)
results = analyzer.analyze_document(["田中さんが来た。", "彼は笑った。"])
# 笑った ガ→彼 (flat cross-sentence index)

# Sentence split
analyzer.sentences("こんにちは。さようなら。")  # → [["こ","ん","に","ち","は","。"], [...]]
```

## Performance

Tested on Apple Silicon (M-series Mac).

### Quick Mode (char → word)

| Variant | Speed (sent/sec) | Segmentation F1 | POS Accuracy |
|---------|------------------|-----------------|--------------|
| FP32    | 78               | 1.000           | baseline     |
| FP16    | ~150             | 1.000           | 1.000        |
| Q8      | ~170             | 1.000           | 1.000        |
| Q4      | ~180             | 0.997           | 1.000        |

### Full Mode (char → word → T5)

| Variant | Speed (sent/sec) | Output Match |
|---------|------------------|--------------|
| FP32    | 5.9              | baseline     |
| Q8      | ~8               | ✓ identical  |
| Q4      | ~9               | ✓ identical  |

### vs PyTorch KWJA

| Backend | Quick Mode | Full Mode |
|---------|-----------|-----------|
| PyTorch CPU | ~10 sent/sec | ~3 sent/sec |
| PyTorch MPS | ~30 sent/sec | ~8 sent/sec |
| **MLX FP32** | **78 sent/sec** | **5.9 sent/sec** |
| **MLX Q8** | **~170 sent/sec** | **~8 sent/sec** |

## Architecture

```
Input text
    ↓
[char module] DeBERTa-v2-base-japanese-char-wwm (22K vocab)
    → Word segmentation (per-char B/I tagging)
    ↓
[word module] DeBERTa-v2-base-japanese (32K vocab)
    → POS tagging + NER (first-subword pooling → classification)
    ↓
[seq2seq module] T5-base (32K vocab, encoder-decoder)
    → Reading, lemma, canonical form (autoregressive generation)
    ↓
Output: Morpheme(surface, pos, ne, reading, lemma, canon)
```

## Requirements

```
mlx >= 0.20
numpy
safetensors
transformers
huggingface_hub
sentencepiece
```

## Models

Weights from [kevinzhow/kwja-mlx](https://huggingface.co/kevinzhow/kwja-mlx) (auto-downloaded on first use).

Based on KWJA v2.4 checkpoints from Kyoto University:
- char: `ku-nlp/deberta-v2-base-japanese-char-wwm`
- word: `ku-nlp/deberta-v2-base-japanese`
- seq2seq: `retrieva-jp/t5-base-long`

## Citation

```bibtex
@InProceedings{Ueda2023a,
  author    = {Nobuhiro Ueda and Kazumasa Omura and Takashi Kodama and Hirokazu Kiyomaru and Yugo Murawaki and Daisuke Kawahara and Sadao Kurohashi},
  title     = {KWJA: A Unified Japanese Analyzer Based on Foundation Models},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
  year      = {2023},
  address   = {Toronto, Canada},
}
```

## License

MIT (same as KWJA)
