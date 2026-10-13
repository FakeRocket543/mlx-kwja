"""Weight-free tests for scripts/convert_kwja_to_mlx.py (stub tensors only)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "convert_kwja_to_mlx", str(Path(__file__).parent.parent / "scripts" / "convert_kwja_to_mlx.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["convert_kwja_to_mlx"] = _mod
_spec.loader.exec_module(_mod)


class StubTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)
        self.shape = self._arr.shape

    def numpy(self):
        return self._arr


class TestConvertStateDict:
    def test_char_conv_transpose_and_tagger_remap(self):
        w = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        sd = {
            # torch Conv1d (O,I,K) -> MLX (O,K,I)
            "encoder.encoder.conv.conv.weight": StubTensor(w),
            "encoder.embeddings.word_embeddings.weight": StubTensor(np.zeros((5, 8), np.float32)),
            # Sequential-flattened tagger keys gain .layers.
            "word_segmentation_tagger.0.weight": StubTensor(np.zeros((8, 8), np.float32)),
        }
        out = _mod.convert_state_dict("char_deberta-v2-base-wwm", sd)
        assert out["encoder.encoder.conv.conv.weight"].shape == (2, 4, 3)
        assert np.array_equal(out["encoder.encoder.conv.conv.weight"], w.transpose(0, 2, 1))
        assert out["encoder.embeddings.word_embeddings.weight"].shape == (5, 8)
        assert "word_segmentation_tagger.layers.0.weight" in out

    def test_word_keys_verbatim(self):
        sd = {
            "encoder.encoder.layer.0.attention.self.key_proj.weight": StubTensor(np.zeros((4, 4), np.float32)),
            "cohesion_analyzer.classifier": StubTensor(np.zeros((4, 9), np.float32)),
            "crf.transitions": StubTensor(np.zeros((3, 3), np.float32)),
        }
        out = _mod.convert_state_dict("word_deberta-v2-base", sd)
        assert set(out) == set(sd)

    def test_seq2seq_remap(self):
        sd = {
            "encoder_decoder.shared.weight": StubTensor(np.zeros((8, 4), np.float32)),
            "encoder_decoder.encoder.block.0.layer.0.SelfAttention.q.weight": StubTensor(np.zeros((4, 4), np.float32)),
            "encoder_decoder.encoder.final_layer_norm.weight": StubTensor(np.zeros(4, np.float32)),
            "encoder_decoder.lm_head.weight": StubTensor(np.zeros((8, 4), np.float32)),
            "some_training_buffer": StubTensor(np.zeros(2, np.float32)),
        }
        out = _mod.convert_state_dict("seq2seq_t5-base", sd)
        assert "wte.weight" in out
        # encoder attention is plain .attention. (decoder keeps .self_attention.)
        assert "encoder.layers.0.attention.query_proj.weight" in out
        assert "encoder.ln.weight" in out
        assert "lm_head.weight" in out
        assert "some_training_buffer" not in out
