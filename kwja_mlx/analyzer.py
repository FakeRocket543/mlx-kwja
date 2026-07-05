"""KWJA MLX Analyzer — Japanese word segmentation using DeBERTa-v2 on Metal GPU."""

import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np


class Analyzer:
    """KWJA-compatible Japanese word segmenter running on MLX Metal GPU.
    
    Uses the DeBERTa-v2 char module from kevinzhow/kwja-mlx weights
    with OpenMed's MLX DeBERTa-v2 implementation.
    """

    def __init__(self, model_size: str = "base", cache_dir: str = "models/cache", verbose: bool = True):
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        from safetensors.numpy import load_file
        from kwja_mlx.deberta_v2_mlx import DebertaV2ForTokenClassification

        t0 = time.time()

        # Download weights
        config_path = hf_hub_download(
            "kevinzhow/kwja-mlx",
            f"{model_size}/char_deberta-v2-{model_size}-wwm.mlx.json",
            cache_dir=cache_dir,
        )
        weights_path = hf_hub_download(
            "kevinzhow/kwja-mlx",
            f"{model_size}/char_deberta-v2-{model_size}-wwm.mlx.safetensors",
            cache_dir=cache_dir,
        )

        # Load config
        with open(config_path) as f:
            self.config = json.load(f)

        # Load tokenizer (char-level)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["encoder_repo"],
            do_word_tokenize=False,
            additional_special_tokens=[" "],
        )

        # Load weights
        weights_np = load_file(weights_path)
        
        # Detect model dimensions from weights
        hidden_size = weights_np["encoder.embeddings.word_embeddings.weight"].shape[1]
        vocab_size = weights_np["encoder.embeddings.word_embeddings.weight"].shape[0]
        
        # Count layers
        num_layers = 0
        for k in weights_np:
            if k.startswith("encoder.encoder.layer."):
                layer_num = int(k.split(".")[3])
                num_layers = max(num_layers, layer_num + 1)
        
        if verbose:
            print(f"KWJA MLX ({model_size}): hidden={hidden_size}, layers={num_layers}, vocab={vocab_size}")

        # Build model config
        model_config = {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": hidden_size // 64,  # standard head_dim=64
            "intermediate_size": hidden_size * 4,
            "vocab_size": vocab_size,
            "max_position_embeddings": 512,
            "relative_attention": True,
            "position_buckets": 256,
            "max_relative_positions": -1,
            "num_labels": 2,  # B/I for word segmentation
        }

        # Build model
        self.model = DebertaV2ForTokenClassification(model_config)

        # Load weights into model
        weight_list = [(k, mx.array(v)) for k, v in weights_np.items() 
                       if "word_segmentation_tagger" in k or "encoder" in k]
        self.model.load_weights(weight_list)
        mx.eval(self.model.parameters())

        if verbose:
            total_params = sum(v.size for v in dict(self.model.parameters()).values())
            print(f"  Loaded: {total_params/1e6:.0f}M params in {time.time()-t0:.1f}s")

    def tokenize(self, text: str) -> list[str]:
        """Segment Japanese text into words."""
        # Encode text to char IDs
        encoded = self.tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        input_ids = mx.array(encoded["input_ids"])

        # Forward pass
        logits = self.model(input_ids)  # (1, seq_len, 2)
        mx.eval(logits)

        # Decode: argmax → B(0)/I(1)
        tags = np.array(mx.argmax(logits[0], axis=-1))

        # Reconstruct words from char-level BIO tags
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        words = []
        current_word = ""

        for i in range(1, len(tokens)):  # skip [CLS]
            token = tokens[i]
            if token in ("[SEP]", "[PAD]", "[CLS]"):
                break
            
            # Get actual character
            char = token
            if not char or char == "[UNK]":
                continue

            tag = tags[i]
            if tag == 0:  # B = word boundary
                if current_word:
                    words.append(current_word)
                current_word = char
            else:  # I = continue word
                current_word += char

        if current_word:
            words.append(current_word)

        return words

    def batch_tokenize(self, texts: list[str], batch_size: int = 32) -> list[list[str]]:
        """Batch tokenize multiple texts."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                results.append(self.tokenize(text))
        return results
