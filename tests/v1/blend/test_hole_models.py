# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch

# First Party
from lmcache_ascend.v1.blend.attention.attention import ZLMCFlashAttnBackend
from lmcache_ascend.v1.blend.models.hole import (
    HoleSynchronizedAttentionBackend,
    infer_hole_model_from_vllm,
)
import lmcache_ascend.v1.blend.models.hole as hole_models


def test_hole_attention_records_completion_after_forward(monkeypatch):
    events = []
    result = torch.tensor([1.0])

    monkeypatch.setattr(
        ZLMCFlashAttnBackend,
        "forward_contiguous",
        lambda *args, **kwargs: events.append("forward") or result,
    )
    connector = type(
        "Connector",
        (),
        {"record_scatter_done": lambda self, layer_id: events.append(layer_id)},
    )()
    backend = HoleSynchronizedAttentionBackend.__new__(HoleSynchronizedAttentionBackend)
    backend.gpu_connector = connector
    backend.layer_id = 3

    actual = backend.forward_contiguous(
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        None,
    )

    assert actual is result
    assert events == ["forward", 3]


def test_infer_hole_model_selects_llama(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        hole_models,
        "LMCLlamaHoleModel",
        lambda model, blender: sentinel,
    )
    model = type("LlamaForCausalLM", (), {})()

    assert infer_hole_model_from_vllm(model, object()) is sentinel


def test_infer_hole_model_selects_qwen_variant(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        hole_models,
        "LMCQwen3HoleModel",
        lambda model, blender: sentinel,
    )
    model = type("CustomQwen3ForCausalLM", (), {})()

    assert infer_hole_model_from_vllm(model, object()) is sentinel
