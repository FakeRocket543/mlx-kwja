# Word Module Debug Notes

## Tokenizer Alignment ✓
- Base vocab: 32000 (ku-nlp/deberta-v2-base-japanese)
- + 8 special tokens: [著者][読者][不特定:人][不特定:物][NULL][NA][ROOT][ ]
- = 32008 = weights vocab_size ✓
- Token IDs: 32000-32007 ✓

## Paper
- ACL 2023 Demo: https://aclanthology.org/2023.acl-demo.52/
- Authors: Ueda, Omura, Kodama, Kiyomaru, Murawaki, Kawahara, Kurohashi (京大+早稲田)

## Current Issue
- POS: 助詞/動詞 correct, 名詞→接続詞 wrong
- Tokenizer alignment confirmed OK
- Next: check word.py training data format — may need morpheme boundary markers in input

## File to read
- /path/to/.venv/.../kwja/datamodule/datasets/word.py (training format)
- Hypothesis: word module trained with specific subword-to-morpheme alignment markers

