# kwja-mlx — KWJA (日文形態素解析) の MLX 移植

日文版 CKIP MLX。京大 KWJA の DeBERTa encoder を MLX 化し、Apple Silicon で高速推理。

## 背景

| 工具 | 架構 | 速度 | 精度 |
|------|------|------|------|
| MeCab (fugashi) | CRF + 辭典 | 270K sent/sec | ⭐⭐ |
| Sudachi | lattice + 正規化 | 244K sent/sec | ⭐⭐⭐ |
| **KWJA** | **DeBERTa neural** | **~10 sent/sec** | **⭐⭐⭐⭐** |
| **kwja-mlx (目標)** | **MLX Metal GPU** | **~100+ sent/sec** | **⭐⭐⭐⭐** |

目標：KWJA 的精度 + MeCab 級の速度（10x 加速 over PyTorch CPU）

## KWJA 架構分析

```
Input text
    ↓
[1] Typo correction module (DeBERTa seq2seq) — optional
    ↓
[2] Character module (DeBERTa token classification)
    → 文字單位：每字預測 B/I/O boundary
    ↓
[3] Word module (DeBERTa token classification)
    → 詞單位：POS tag + 基本形 + 讀み
    ↓
[4] Seq2seq module (DeBERTa/T5)
    → NER + 依存關係 + 共參照
    ↓
Output: 分詞 + POS + NER + 依存 + 共參照
```

## MLX 化計畫

### Phase 1: Encoder 轉換（最大加速）

```
PyTorch DeBERTa weights → MLX safetensors
推理: MLX forward pass + @mx.compile
```

這跟 GLiNER-Forge 的 `to_mlx.py` 幾乎一樣。DeBERTa 是相同架構。

### Phase 2: Character module （分詞核心）

```
Input: 文字列 → DeBERTa encoder → per-character BIO tag
Output: 分詞結果
```

這是最有用的部分——只做分詞不做 NER/依存的話，只需要 char module。

### Phase 3: Word module（POS tagging）

```
Input: 分詞結果 → DeBERTa encoder → per-token POS/活用形
Output: 品詞付き分詞
```

### Phase 4: Full pipeline（NER + 依存）

完整移植。可能不值得——GiNZA/spaCy 已經能做。

## 量化實驗計畫

| 量化 | 預期 | 風險 |
|------|------|------|
| FP32 → FP16 | 2x 速度, 精度無損 | 無 |
| FP16 → INT8 | 2x 速度, 精度微降 | DeBERTa disentangled attention 可能受影響 |
| INT4 / Q4 | 4x 速度 | encoder 可能崩潰（同 GLiNER Forge 的教訓） |

從 GLiNER-Forge 學到的：
- FP16 安全
- INT8 需要 fp16→fp32 清理（153 個隱藏 tensor）再量化
- INT4 對 encoder 通常毀滅性

## 實驗步驟

```
1. 下載 KWJA PyTorch weights (ku-nlp/kwja-char-base)
2. 轉換 → MLX safetensors (to_mlx.py 複用 GLiNER-Forge 的)
3. 寫 MLX forward pass (DeBERTa encoder + BIO classification head)
4. 驗證：跟 KWJA PyTorch 比較分詞結果一致性
5. Benchmark：MLX vs PyTorch CPU vs PyTorch MPS
6. 量化：FP16 → INT8 → 測精度/速度 trade-off
7. 整合：output 格式跟 fugashi 相容（surface + POS）
```

## 預期結果

| Backend | Speed (est.) | Accuracy |
|---------|-------------|----------|
| KWJA PyTorch CPU | 10 sent/sec | baseline |
| KWJA PyTorch MPS | 30 sent/sec | same |
| **kwja-mlx FP16** | **100-200 sent/sec** | same |
| **kwja-mlx INT8** | **200-400 sent/sec** | ~99% |
| fugashi (MeCab) | 270K sent/sec | lower |

即使 MLX 化後仍比 MeCab 慢 1000x，但精度差距足以 justify（特別是口語/網路語言）。

## Target API

```python
from kwja_mlx import Analyzer

analyzer = Analyzer(model="base", quantize="fp16")

# 分詞 only（最快）
tokens = analyzer.tokenize("ピアノ上手ですね")
# → [Token(surface='ピアノ', pos='名詞'), Token(surface='上手', pos='形容詞'), ...]

# Full analysis
result = analyzer.analyze("ピアノ上手ですね")
# → Analysis(tokens=[...], ner=[...], dependencies=[...])
```

## 依賴

```
mlx >= 0.20
numpy
safetensors
sentencepiece (for DeBERTa tokenizer)
```

## 與 GLiNER-Forge 共用

- `convert/to_mlx.py` — 權重轉換（DeBERTa 同架構）
- `convert/fix_fp16.py` — fp16 隱藏 tensor 清理
- `convert/quantize.py` — INT8 量化
- Benchmark 方法論

## License

MIT (KWJA itself is MIT)
