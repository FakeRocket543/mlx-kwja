"""Convert official KWJA v2.4 .ckpt files to MLX .safetensors + .mlx.json.

Reproduces kevinzhow/kwja-mlx artifacts from Kyoto University checkpoints:
  https://lotus.kuee.kyoto-u.ac.jp/kwja/v2.4/<name>.ckpt

Usage (needs torch + safetensors on a machine that can load .ckpt):
  python scripts/convert_kwja_to_mlx.py --model-size base --module all --out ./converted
  python scripts/convert_kwja_to_mlx.py --model-size base --module char --ckpt path/to.ckpt

Layout per module (mirrors kevinzhow/kwja-mlx):
  <size>/<stem>.mlx.safetensors + <size>/<stem>.mlx.json

Conversion notes (verified byte-identical to kevinzhow artifacts, 2026-10-13):
- char/word: key rename only for Sequential-flattened tagger keys
  'X_tagger.{0,3}.' -> 'X_tagger.layers.{0,3}.' (see _remap_tagger_key).
  Single tensor transform: char conv weight torch Conv1d (O,I,K)
  -> MLX Conv1d (O,K,I), i.e. transpose(0,2,1) (brute-force verified:
  it is the unique permutation of (0,1,2) matching kevinzhow bytes).
- seq2seq: upstream wraps a stock HF T5 in `encoder_decoder.`; strip that
  prefix, then rename to this repo's T5 layout (see _remap_seq2seq_key).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CKPT_BASE = "https://lotus.kuee.kyoto-u.ac.jp/kwja/v2.4"

MODULES = {
    "tiny": {
        "char": "char_deberta-v2-tiny-wwm",
        "seq2seq": "seq2seq_t5-small",
        "word": "word_deberta-v2-tiny",
    },
    "base": {
        "char": "char_deberta-v2-base-wwm",
        "seq2seq": "seq2seq_t5-base",
        "word": "word_deberta-v2-base",
    },
    "large": {
        "char": "char_deberta-v2-large-wwm",
        "seq2seq": "seq2seq_t5-large",
        "word": "word_deberta-v2-large",
    },
}

ENCODER_REPO = {
    "char_deberta-v2-tiny-wwm": "ku-nlp/deberta-v2-tiny-japanese-char-wwm",
    "char_deberta-v2-base-wwm": "ku-nlp/deberta-v2-base-japanese-char-wwm",
    "char_deberta-v2-large-wwm": "ku-nlp/deberta-v2-large-japanese-char-wwm",
    "word_deberta-v2-tiny": "ku-nlp/deberta-v2-tiny-japanese",
    "word_deberta-v2-base": "ku-nlp/deberta-v2-base-japanese",
    "word_deberta-v2-large": "ku-nlp/deberta-v2-large-japanese",
    "seq2seq_t5-small": "retrieva-jp/t5-small-short",
    "seq2seq_t5-base": "retrieva-jp/t5-base-long",
    "seq2seq_t5-large": "retrieva-jp/t5-large-long",
}


def _remap_seq2seq_key(k: str) -> str:
    """HF T5 state_dict key -> kwja_mlx.t5_mlx layout.

    Verified against seq2seq_t5-base.ckpt (284 keys). T5 block layout:
    - encoder layer N: layer.0.SelfAttention (+ln) / layer.1.DenseReluDense (+ln)
    - decoder layer N: layer.0.SelfAttention (+ln) / layer.1.EncDecAttention (+ln)
      / layer.2.DenseReluDense (+ln)
    Target: T5EncoderLayer(attn->ln1, dense->ln2), T5DecoderLayer
    (self->ln1, cross->ln2, dense->ln3).
    """
    import re
    k = k.removeprefix("encoder_decoder.")
    k = k.replace("shared.weight", "wte.weight")
    k = k.replace("encoder.block.", "encoder.layers.")
    k = k.replace("decoder.block.", "decoder.layers.")
    # attention: encoder layer uses plain .attention., decoder keeps .self_attention.
    k = re.sub(r"^(encoder\.layers\.\d+)\.layer\.0\.SelfAttention\.", r"\1.attention.", k)
    k = k.replace(".layer.0.SelfAttention.", ".self_attention.")
    k = k.replace(".layer.1.EncDecAttention.", ".cross_attention.")
    # FFN: layer.{1|2}.DenseReluDense.X -> dense.X
    k = k.replace(".layer.1.DenseReluDense.", ".dense.")
    k = k.replace(".layer.2.DenseReluDense.", ".dense.")
    # layer norms: encoder (0->ln1, 1->ln2), decoder (0->ln1, 1->ln2, 2->ln3)
    m = re.match(r"^(encoder\.layers\.\d+)\.layer\.([01])\.layer_norm\.(.*)$", k)
    if m:
        k = f"{m.group(1)}.ln{int(m.group(2)) + 1}.{m.group(3)}"
    m = re.match(r"^(decoder\.layers\.\d+)\.layer\.([012])\.layer_norm\.(.*)$", k)
    if m:
        k = f"{m.group(1)}.ln{int(m.group(2)) + 1}.{m.group(3)}"
    # relative bias: only block.0 carries it (shared); hoist to top level
    m = re.match(r"^(encoder|decoder)\.layers\.0\.(?:self_)?attention\.relative_attention_bias\.weight$", k)
    if m:
        return f"{m.group(1)}.relative_attention_bias.embeddings.weight"
    k = k.replace(".q.weight", ".query_proj.weight")
    k = k.replace(".k.weight", ".key_proj.weight")
    k = k.replace(".v.weight", ".value_proj.weight")
    k = k.replace(".o.weight", ".out_proj.weight")
    k = k.replace("encoder.final_layer_norm.weight", "encoder.ln.weight")
    k = k.replace("decoder.final_layer_norm.weight", "decoder.ln.weight")
    return k


def _keep_seq2seq_key(k: str) -> bool:
    """Drop tied dupes (embed_tokens == shared) and non-block.0 bias copies.

    Operates on the RAW ckpt key: keep bias only from block.0 (shared across
    layers in T5), drop per-layer embed_tokens (tied to shared).
    """
    if k.endswith(".embed_tokens.weight"):
        return False
    if "relative_attention_bias" in k and ".block.0." not in k:
        return False
    return True


def _remap_tagger_key(k: str) -> str:
    """nn.Sequential-flattened head keys -> kevinzhow dotted layout.

    ckpt stores e.g. 'word_segmentation_tagger.0.weight' (Sequential index);
    kevinzhow artifacts use 'word_segmentation_tagger.layers.0.weight'.
    Also covers dependency_type_parser.{0,3} (same Sequential pattern).
    """
    import re
    k = re.sub(r"^(\w+_tagger)\.([03])\.", r"\1.layers.\2.", k)
    k = re.sub(r"^(dependency_type_parser)\.([03])\.", r"\1.layers.\2.", k)
    return k

def convert_state_dict(stem: str, state_dict: dict) -> dict:
    """Pure key/tensor transform: ckpt state_dict -> mlx safetensors dict.

    Byte-identical to kevinzhow/kwja-mlx (verified 2026-10-13, 214/214 char
    tensors incl. conv). Accepts torch tensors (or anything with .numpy());
    returns {key: ndarray}.
    """
    import numpy as np

    out = {}
    is_seq2seq = stem.startswith("seq2seq")
    for key, tensor in state_dict.items():
        arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
        if is_seq2seq:
            if not key.startswith("encoder_decoder."):
                continue  # training-only buffers (reading_candidate ids etc.)
            if not _keep_seq2seq_key(key):
                continue  # tied dupes (embed_tokens), non-block.0 bias
            out[_remap_seq2seq_key(key)] = arr
            continue
        new_key = _remap_tagger_key(key)
        if new_key == "encoder.encoder.conv.conv.weight" and arr.ndim == 3:
            arr = np.ascontiguousarray(arr.transpose(0, 2, 1))
        out[new_key] = arr
    return out


def convert_ckpt(ckpt_path: str | Path, stem: str, out_dir: str | Path) -> Path:
    """Load .ckpt on CPU, convert, write .mlx.safetensors + .mlx.json."""
    import torch
    from safetensors.numpy import save_file

    ckpt_path, out_dir = Path(ckpt_path), Path(out_dir)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    arrays = convert_state_dict(stem, ckpt["state_dict"])
    out_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = out_dir / f"{stem}.mlx.safetensors"
    save_file(arrays, str(safetensors_path))
    meta = {"encoder_repo": ENCODER_REPO[stem], "source_ckpt": ckpt_path.name}
    hp = ckpt.get("hyper_parameters", {})
    if isinstance(hp, dict):
        for field in ("max_seq_length", "vocab_size", "max_src_length", "max_length"):
            if field in hp:
                meta[field] = hp[field]
    json_path = out_dir / f"{stem}.mlx.json"
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {safetensors_path} ({len(arrays)} tensors) + {json_path}")
    return safetensors_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-size", default="base", choices=["tiny", "base", "large"])
    parser.add_argument("--module", default="all", choices=["all", "char", "seq2seq", "word"])
    parser.add_argument("--ckpt", default=None, help="local .ckpt path (single-module mode)")
    parser.add_argument("--out", default="./converted")
    args = parser.parse_args()

    mods = MODULES[args.model_size]
    targets = [args.module] if args.module != "all" else ["char", "seq2seq", "word"]
    for mod in targets:
        stem = mods[mod]
        ckpt = args.ckpt or f"{CKPT_BASE}/{stem}.ckpt"
        local = ckpt
        if ckpt.startswith("http"):
            import urllib.request
            out_dir = Path(args.out) / args.model_size
            out_dir.mkdir(parents=True, exist_ok=True)
            local = out_dir / f"{stem}.ckpt"
            if not Path(local).exists():
                print(f"downloading {ckpt} ...")
                urllib.request.urlretrieve(ckpt, local)
        convert_ckpt(local, stem, Path(args.out) / args.model_size)


if __name__ == "__main__":
    main()
