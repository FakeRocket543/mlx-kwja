# kwja-mlx Benchmark Results

**Date**: 2026-07-05
**Platform**: Apple M4 Max, macOS, MLX 0.31.2
**Model**: ku-nlp/deberta-v2-base-japanese-char-wwm (104M params)
**Weights**: kevinzhow/kwja-mlx base

## Speed & Accuracy

### vs PyTorch (same model)

| Backend | Speed | vs CPU | Memory |
|---------|-------|--------|--------|
| PyTorch CPU | 39 sent/sec | 1.0x | ~850 MB |
| PyTorch MPS | 91 sent/sec | 2.4x | ~850 MB |
| **MLX FP32** | **90 sent/sec** | **2.3x** | 424 MB |
| **MLX FP16** | **193-200 sent/sec** | **5.1x** | 212 MB |
| **MLX Q8** | **210-221 sent/sec** | **5.5x** | 106 MB |
| **MLX Q4** | **211-225 sent/sec** | **5.7x** | 53 MB |

### Accuracy (35 sentences, Wikipedia-style Japanese)

| Variant | F1 | Exact Match | Mismatches |
|---------|-----|-------------|-----------|
| FP32 | 1.0000 | 100.0% | 0/35 |
| FP16 | 1.0000 | 100.0% | 0/35 |
| Q8 | 1.0000 | 100.0% | 0/35 |
| Q4 | 0.9969 | 97.1% | 1/35 |

### Q4 唯一 mismatch

```
Input: 日本の面積は約三十七万平方キロメートルである。
FP32: 日本 / の / 面積 / は / 約 / 三十七万 / 平方キロメートル / である / 。
Q4:   日本 / の / 面積 / は / 約 / 三十七万 / 平方キロ / メートル / である / 。
```

差異：「平方キロメートル」vs「平方キロ / メートル」— 複合單位切割粒度，兩種皆合理。

## Sample Outputs (FP32 = Q8)

```
日本国は東アジアに位置する島国である。
→ 日本 / 国 / は / 東 / アジア / に / 位置 / する / 島国 / である / 。

明治維新は日本の近代化の始まりとされる。
→ 明治 / 維新 / は / 日本 / の / 近代化 / の / 始まり / と / さ / れる / 。

動画と関係ないけどピアノ上手いですね
→ 動画 / と / 関係 / ない / けど / ピアノ / 上手い / です / ね

鬼滅の刃のコスプレが似合ってますね
→ 鬼滅の刃 / の / コスプレ / が / 似合って / ます / ね

パンピアノが好きです
→ パンピアノ / が / 好きです
```

## Key Findings

1. **MLX FP32 ≈ PyTorch MPS** — same hardware, comparable speed
2. **MLX Q8 = 2.4x faster than MPS** — quantization is the real MLX advantage
3. **Q4 viable** — only 1/35 minor difference (unit compound split)
4. **Bug found during development**: ConvLayer activation was `gelu` not `tanh` (config `conv_act`)
5. **vs GLiNER-Forge INT8 failure**: MLX per-group quantization (group_size=64) preserves accuracy; ONNX per-tensor quantization destroyed it

## Recommendation

| Use Case | Variant | Reason |
|----------|---------|--------|
| Research/Production | Q8 | Perfect accuracy + fast + small |
| Edge/Mobile | Q4 | 53MB, negligible quality loss |
| Debugging | FP32 | Reference |
