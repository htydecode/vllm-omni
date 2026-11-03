"""Standalone 120-case tensor regression using the server's actual method source.

This complements, but does not replace, the repository integration test suite.
"""

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from full_decoder_ab import build_methods


@pytest.fixture(scope="module")
def methods():
    value = os.environ.get("TTS_CONTEXT_SOURCE")
    if not value:
        raise RuntimeError(
            "Set TTS_CONTEXT_SOURCE to the actual installed modeling_qwen3_tts_tokenizer_v2.py"
        )
    source = Path(value).read_text(encoding="utf-8")
    if "# TTS_CONTEXT_DIAGNOSTIC_V1" in source:
        raise RuntimeError("Restore the temporary diagnostic patch first")
    tree = ast.parse(source)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "_CONV_CONTEXT_FRAME",
                    "_DOWNSTREAM_CONTEXT_FRAME",
                }:
                    constants[target.id] = ast.literal_eval(node.value)
    if constants.get("_CONV_CONTEXT_FRAME") != 2:
        raise RuntimeError(
            "Unexpected context size; inspect this version before testing"
        )
    return build_methods(source, dict(torch=torch, copy=copy, **constants))


@pytest.mark.parametrize("ref_frames", [0, 1, 72])
@pytest.mark.parametrize("old_frames", [0, 1, 2, 25, 72])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_context_tensor_parity(
    methods, monkeypatch, ref_frames, old_frames, batch, strided, dtype
):
    stride = 2 if strided else 1

    def tensor(frames, offset):
        data = torch.arange(batch * 4 * frames * stride, dtype=torch.float32)
        return (data.reshape(batch, 4, frames * stride) + offset).to(dtype)[
            ..., ::stride
        ]

    ref, old, new = tensor(ref_frames, 0), tensor(old_frames, 100), tensor(3, 200)
    snapshots = [t.clone() for t in (ref, old, new)]
    expected = torch.cat([torch.cat([ref, old], dim=-1)[..., -2:], new], dim=-1)
    actual_cat = torch.cat
    outputs = {}
    for name, fn in methods.items():
        calls = []

        def counted_cat(*args, **kwargs):
            calls.append(1)
            return actual_cat(*args, **kwargs)

        class StopAfterPreConvError(Exception):
            pass

        def pre_conv(value):
            outputs[name] = value.clone()
            raise StopAfterPreConvError

        decoder = SimpleNamespace(
            config=SimpleNamespace(sliding_window=72),
            quantizer=SimpleNamespace(decode=lambda codes: new),
            pre_conv=pre_conv,
        )
        with monkeypatch.context() as patch:
            patch.setattr(torch, "cat", counted_cat)
            with pytest.raises(StopAfterPreConvError):
                fn(decoder, new, old, None, {"ref_hidden": ref}, ref_frames, 3, False)
        assert len(calls) == (1 if name == "candidate" and old_frames >= 2 else 2)
        torch.testing.assert_close(outputs[name], expected, rtol=0, atol=0)
    for value, snapshot in zip((ref, old, new), snapshots):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
