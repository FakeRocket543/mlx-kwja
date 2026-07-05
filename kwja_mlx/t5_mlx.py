"""T5 encoder-decoder implementation in MLX for KWJA seq2seq module."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional


class T5RelativePositionBias(nn.Module):
    """T5-style relative position bias (shared across heads in layer 0)."""

    def __init__(self, num_heads: int = 12, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.embeddings = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: np.ndarray, bidirectional: bool, num_buckets: int = 32, max_distance: int = 128
    ) -> np.ndarray:
        """Compute T5-style bucketed relative position."""
        ret = np.zeros_like(relative_position)
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).astype(np.int32) * num_buckets
            n = np.abs(relative_position)
        else:
            n = np.maximum(-relative_position, 0)

        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            np.log(n.astype(np.float32) / max_exact)
            / np.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        ret += np.where(is_small, n, val_if_large)
        return ret

    def __call__(self, query_len: int, key_len: int, bidirectional: bool = True) -> mx.array:
        """Returns bias of shape (1, num_heads, query_len, key_len)."""
        context_position = np.arange(query_len)[:, None]
        memory_position = np.arange(key_len)[None, :]
        relative_position = memory_position - context_position
        buckets = self._relative_position_bucket(
            relative_position, bidirectional=bidirectional,
            num_buckets=self.num_buckets, max_distance=self.max_distance,
        )
        buckets_mx = mx.array(buckets.astype(np.int32))
        values = self.embeddings(buckets_mx)  # (query, key, heads)
        values = mx.transpose(values, (2, 0, 1))  # (heads, query, key)
        return values[None, :, :, :]  # (1, heads, query, key)


class T5Attention(nn.Module):
    """T5 multi-head attention (no bias in projections)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: Optional[mx.array] = None,
        position_bias: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[tuple] = None,
    ) -> tuple:
        B, L, _ = hidden_states.shape
        is_cross = key_value_states is not None

        q = self.query_proj(hidden_states).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if is_cross and cache is not None:
            # Reuse cached cross-attention K/V
            k, v = cache
        else:
            kv_input = key_value_states if is_cross else hidden_states
            k = self.key_proj(kv_input).reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = self.value_proj(kv_input).reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

            if not is_cross and cache is not None:
                # Self-attention with KV cache
                prev_k, prev_v = cache
                k = mx.concatenate([prev_k, k], axis=2)
                v = mx.concatenate([prev_v, v], axis=2)

        # T5 does NOT scale by 1/sqrt(dk) — this is intentional
        scores = q @ k.transpose(0, 1, 3, 2)

        if position_bias is not None:
            scores = scores + position_bias
        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out), (k, v)


class T5DenseGatedGelu(nn.Module):
    """T5 v1.1 / mT5 style gated FFN: (gelu(x @ Wi_0) * (x @ Wi_1)) @ Wo."""

    def __init__(self, hidden_size: int, ff_size: int):
        super().__init__()
        self.wi_0 = nn.Linear(hidden_size, ff_size, bias=False)
        self.wi_1 = nn.Linear(hidden_size, ff_size, bias=False)
        self.wo = nn.Linear(ff_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.wo(nn.gelu(self.wi_0(x)) * self.wi_1(x))


class T5EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_size: int):
        super().__init__()
        self.attention = T5Attention(hidden_size, num_heads)
        self.ln1 = nn.RMSNorm(hidden_size)
        self.dense = T5DenseGatedGelu(hidden_size, ff_size)
        self.ln2 = nn.RMSNorm(hidden_size)

    def __call__(self, x: mx.array, position_bias: Optional[mx.array] = None, mask: Optional[mx.array] = None):
        # Pre-norm self-attention
        h = self.ln1(x)
        h, _ = self.attention(h, position_bias=position_bias, mask=mask)
        x = x + h
        # Pre-norm FFN
        h = self.ln2(x)
        x = x + self.dense(h)
        return x


class T5DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_size: int):
        super().__init__()
        self.self_attention = T5Attention(hidden_size, num_heads)
        self.ln1 = nn.RMSNorm(hidden_size)
        self.cross_attention = T5Attention(hidden_size, num_heads)
        self.ln2 = nn.RMSNorm(hidden_size)
        self.dense = T5DenseGatedGelu(hidden_size, ff_size)
        self.ln3 = nn.RMSNorm(hidden_size)

    def __call__(
        self,
        x: mx.array,
        encoder_output: mx.array,
        self_attn_bias: Optional[mx.array] = None,
        cross_attn_bias: Optional[mx.array] = None,
        self_attn_mask: Optional[mx.array] = None,
        self_cache: Optional[tuple] = None,
        cross_cache: Optional[tuple] = None,
    ) -> tuple:
        # Self-attention
        h = self.ln1(x)
        h, new_self_cache = self.self_attention(h, position_bias=self_attn_bias, mask=self_attn_mask, cache=self_cache)
        x = x + h
        # Cross-attention
        h = self.ln2(x)
        h, new_cross_cache = self.cross_attention(h, key_value_states=encoder_output, position_bias=cross_attn_bias, cache=cross_cache)
        x = x + h
        # FFN
        h = self.ln3(x)
        x = x + self.dense(h)
        return x, new_self_cache, new_cross_cache


class T5Encoder(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_size: int, num_layers: int):
        super().__init__()
        self.layers = [T5EncoderLayer(hidden_size, num_heads, ff_size) for _ in range(num_layers)]
        self.ln = nn.RMSNorm(hidden_size)
        self.relative_attention_bias = T5RelativePositionBias(num_heads)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        seq_len = x.shape[1]
        position_bias = self.relative_attention_bias(seq_len, seq_len, bidirectional=True)
        for layer in self.layers:
            x = layer(x, position_bias=position_bias, mask=mask)
        return self.ln(x)


class T5Decoder(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_size: int, num_layers: int):
        super().__init__()
        self.layers = [T5DecoderLayer(hidden_size, num_heads, ff_size) for _ in range(num_layers)]
        self.ln = nn.RMSNorm(hidden_size)
        self.relative_attention_bias = T5RelativePositionBias(num_heads)

    def __call__(
        self,
        x: mx.array,
        encoder_output: mx.array,
        self_attn_mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> tuple:
        seq_len = x.shape[1]
        # For autoregressive: position bias covers full decoded length
        if cache is not None and cache[0][0] is not None:
            # Incremental decoding: only compute bias for new position
            past_len = cache[0][0][0].shape[2]  # past K length
            position_bias = self.relative_attention_bias(past_len + seq_len, past_len + seq_len, bidirectional=False)
            # Only take the last row(s) for query
            position_bias = position_bias[:, :, -seq_len:, :]
        else:
            position_bias = self.relative_attention_bias(seq_len, seq_len, bidirectional=False)

        # Cross-attention bias (None — T5 doesn't use relative bias for cross-attention)
        cross_attn_bias = None

        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else (None, None)
            self_cache, cross_cache = layer_cache
            x, new_self, new_cross = layer(
                x, encoder_output,
                self_attn_bias=position_bias,
                cross_attn_bias=cross_attn_bias,
                self_attn_mask=self_attn_mask,
                self_cache=self_cache,
                cross_cache=cross_cache,
            )
            new_cache.append((new_self, new_cross))
        return self.ln(x), new_cache


class T5ForConditionalGeneration(nn.Module):
    """T5 encoder-decoder model for seq2seq generation."""

    def __init__(self, config: dict):
        super().__init__()
        h = config['hidden_size']
        heads = config['num_heads']
        ff = config['ff_size']
        enc_layers = config['num_encoder_layers']
        dec_layers = config['num_decoder_layers']
        vocab = config['vocab_size']

        self.wte = nn.Embedding(vocab, h)
        self.encoder = T5Encoder(h, heads, ff, enc_layers)
        self.decoder = T5Decoder(h, heads, ff, dec_layers)
        self.lm_head = nn.Linear(h, vocab, bias=False)

    def encode(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        x = self.wte(input_ids)
        mask = None
        if attention_mask is not None:
            # Convert (B, L) mask to (B, 1, 1, L) for broadcasting
            mask = (1.0 - attention_mask[:, None, None, :].astype(mx.float32)) * -1e9
        return self.encoder(x, mask=mask)

    def decode_step(
        self,
        decoder_input_ids: mx.array,
        encoder_output: mx.array,
        cache: Optional[list] = None,
    ) -> tuple:
        x = self.wte(decoder_input_ids)
        # Causal mask for self-attention
        seq_len = x.shape[1]
        if cache is not None and cache[0][0] is not None:
            # Incremental: no mask needed (single token)
            self_attn_mask = None
        else:
            # Full decode: causal mask
            causal = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
            self_attn_mask = causal[None, None, :, :]

        h, new_cache = self.decoder(x, encoder_output, self_attn_mask=self_attn_mask, cache=cache)
        logits = self.lm_head(h)
        return logits, new_cache

    def generate(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        max_length: int = 256,
        eos_token_id: int = 1,
        decoder_start_token_id: int = 0,
    ) -> mx.array:
        """Greedy decoding."""
        encoder_output = self.encode(input_ids, attention_mask)
        mx.eval(encoder_output)

        B = input_ids.shape[0]
        decoder_ids = mx.array([[decoder_start_token_id]] * B)
        cache = [(None, None)] * len(self.decoder.layers)
        generated = [decoder_start_token_id]

        for _ in range(max_length):
            logits, cache = self.decode_step(decoder_ids, encoder_output, cache=cache)
            mx.eval(logits)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            token_id = next_token[0].item()
            if token_id == eos_token_id:
                break
            generated.append(token_id)
            decoder_ids = next_token[:, None]

        return generated
