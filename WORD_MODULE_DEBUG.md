# Word Module — FIXED ✅

## Root Cause
Label index ordering was wrong. I assumed standard JUMAN ordering but KWJA uses:

```python
# ✓ CORRECT (from kwja.utils.constants)
POS_TAGS = ['特殊','動詞','形容詞','判定詞','助動詞','名詞','指示詞','副詞','助詞','接続詞','連体詞','感動詞','接頭辞','接尾辞']
NE_TAGS = ['O','B-ORGANIZATION','I-ORGANIZATION','B-PERSON','I-PERSON','B-LOCATION','I-LOCATION','B-ARTIFACT','I-ARTIFACT','B-DATE','I-DATE','B-TIME','I-TIME','B-MONEY','I-MONEY','B-PERCENT','I-PERCENT']
```

## Key Architecture Detail
Word module uses **subword → word pooling** (FIRST strategy):
1. Encoder outputs subword-level hidden states (seq_len, 768)
2. `pool_subwords()` picks the FIRST subword's hidden state for each word
3. POS/NE taggers classify at word level

## Verified Facts
- Tokenizer alignment: ✓ (vocab=32000 + 8 special = 32008)
- Weight values: ✓ (max diff=0.0 vs official KWJA v2.4 checkpoint)
- Encoder forward pass: ✓ (matching PyTorch output)
- POS accuracy: ✓ (名詞/動詞/形容詞/助詞/判定詞/接尾辞/特殊 all correct)
- NER: ✓ (PERSON/LOCATION/ORGANIZATION/ARTIFACT/DATE/TIME detected)
- Speed: 35 sent/sec (full char→word pipeline on M-series Mac)

## Known Limitation
- NER B/I boundary confusion (e.g., I-LOCATION instead of B-LOCATION for first token)
- Cause: No CRF Viterbi decoding implemented yet (KWJA uses CRF for NER)
- Fix: Implement CRF layer (optional, POS doesn't need it)

## Paper
- ACL 2023 Demo: https://aclanthology.org/2023.acl-demo.52/
- Ueda et al., "KWJA: A Unified Japanese Analyzer Based on Foundation Models"
