"""library.attention.check_attn_mode_available のテスト（CPU のみで動作する）。"""

import pytest
import torch

from library import attention


def test_torch_and_none_are_always_available():
    attention.check_attn_mode_available("torch")
    attention.check_attn_mode_available(None)
    params = attention.AttentionParams.create_attention_params("torch", False)
    assert params.attn_mode == "torch"


@pytest.mark.parametrize(
    "mode,attr,package",
    [("xformers", "xops", "xformers"), ("flash", "flash_attn_func", "flash-attn"), ("sageattn", "sageattn", "sageattention")],
)
def test_missing_optional_package_raises_clear_error(monkeypatch, mode, attr, package):
    monkeypatch.setattr(attention, attr, None)
    with pytest.raises(RuntimeError) as e:
        attention.check_attn_mode_available(mode)
    assert package in str(e.value)
    assert "torch" in str(e.value)  # 代替の案内を含む


def test_error_is_raised_when_creating_params(monkeypatch):
    monkeypatch.setattr(attention, "xops", None)
    with pytest.raises(RuntimeError):
        attention.AttentionParams.create_attention_params("xformers", False)
    mask = torch.ones(1, 4, dtype=torch.long)
    with pytest.raises(RuntimeError):
        attention.AttentionParams.create_attention_params_from_mask("xformers", False, 16, mask)


def test_available_optional_package_passes(monkeypatch):
    monkeypatch.setattr(attention, "xops", object())
    attention.check_attn_mode_available("xformers")


def test_unknown_mode_is_left_to_attention_function():
    attention.check_attn_mode_available("something_else")  # ここでは例外にしない
