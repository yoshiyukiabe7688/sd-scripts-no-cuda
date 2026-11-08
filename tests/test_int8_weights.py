"""--int8_base（library/int8_weights.py）: 凍結ベース重みだけを int8 + 64 ブロック scale で保持する。CPU のテスト。
基準は fake-int8（同じ量子化値を dequant した weight を持つ通常の Linear / Conv2d）で、出力と入力勾配のビット一致を確認する。
"""

import copy
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from library import int8_weights as i8
from library.int8_weights import (
    BLOCK,
    Int8Conv2d,
    Int8Linear,
    convert_frozen_modules,
    dequantize_weight,
    int8_modules,
    quantize_dequantize,
    quantize_weight,
)


def fake_quantize_modules(model: nn.Module) -> nn.Module:
    """model の Linear / Conv2d の weight を、int8 の往復（quantize_dequantize）に置き換えたコピーを返す。"""
    m = copy.deepcopy(model)
    for mod in m.modules():
        if type(mod) in (nn.Linear, nn.Conv2d):
            mod.weight.data = quantize_dequantize(mod.weight.data)
    return m


# ---- 量子化の規約 ----------------------------------------------------------------------------------------
def test_quantize_layout_and_roundtrip_error():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(48, 200, generator=g)
    q, scale = quantize_weight(w)
    assert q.dtype == torch.int8 and q.shape == (48, 200)
    assert scale.dtype == torch.bfloat16 and scale.shape == (48, (200 + BLOCK - 1) // BLOCK)
    assert q.abs().max().item() <= 127
    back = dequantize_weight(q, scale, w.shape, torch.float32)
    rel = ((back - w).norm() / w.norm()).item()
    assert rel < 0.015  # 64 ブロックの int8 は 1% 台


def test_scale_is_bf16_rounded_max_over_127_and_q_uses_that_scale():
    w = torch.randn(4, 64) * 3
    q, scale = quantize_weight(w)
    expected = (w.abs().amax(dim=1) / 127).to(torch.bfloat16)
    assert torch.equal(scale[:, 0], expected)
    s = scale.float()
    assert torch.equal(q.float(), torch.round(w / s).clamp(-127, 127))


def test_partial_last_block_and_zero_blocks():
    w = torch.randn(3, 36)  # cols=36: 端数ブロック 1 つ
    q, scale = quantize_weight(w)
    assert scale.shape == (3, 1) and q.shape == (3, 36)
    w = torch.randn(2, 130)  # 64 + 64 + 2
    w[0, :64] = 0  # 全 0 のブロック: scale=0, q=0
    q, scale = quantize_weight(w)
    assert scale.shape == (2, 3)
    assert scale[0, 0].item() == 0 and q[0, :64].abs().sum().item() == 0
    back = dequantize_weight(q, scale, w.shape, torch.float32)
    assert torch.isfinite(back).all() and back[0, :64].abs().sum().item() == 0


def test_conv_weight_is_flattened_per_output_channel():
    w = torch.randn(8, 5, 3, 3)
    q, scale = quantize_weight(w)
    assert q.shape == (8, 45) and scale.shape == (8, 1)
    assert torch.equal(dequantize_weight(q, scale, w.shape, torch.float32), quantize_dequantize(w))


# ---- モジュール（fake-int8 とのビット一致）------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("in_f", [64, 36, 200])
def test_int8_linear_matches_fake_bitwise_forward_and_grad(dtype, bias, in_f):
    torch.manual_seed(0)
    lin = nn.Linear(in_f, 24, bias=bias).to(dtype)
    fake = fake_quantize_modules(nn.Sequential(lin))[0]
    int8 = nn.Sequential(copy.deepcopy(lin))
    convert_frozen_modules(int8, log=False)
    assert isinstance(int8[0], Int8Linear) and isinstance(int8[0], nn.Linear)
    x1 = torch.randn(5, 7, in_f).to(dtype).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = fake(x1), int8(x2)
    assert torch.equal(y1, y2)
    g = torch.randn_like(y1)
    y1.backward(g)
    y2.backward(g)
    assert torch.equal(x1.grad, x2.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cfg", [dict(k=3, s=1, p=1), dict(k=3, s=2, p=1), dict(k=1, s=1, p=0), dict(k=3, s=1, p=0, bias=False)])
def test_int8_conv_matches_fake_bitwise_forward_and_grad(dtype, cfg):
    torch.manual_seed(0)
    conv = nn.Conv2d(6, 10, cfg["k"], cfg["s"], cfg["p"], bias=cfg.get("bias", True)).to(dtype)
    fake = fake_quantize_modules(nn.Sequential(conv))[0]
    int8 = nn.Sequential(copy.deepcopy(conv))
    convert_frozen_modules(int8, log=False)
    assert isinstance(int8[0], Int8Conv2d) and isinstance(int8[0], nn.Conv2d)
    x1 = torch.randn(2, 6, 11, 9).to(dtype).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = fake(x1), int8(x2)
    assert torch.equal(y1, y2)
    g = torch.randn_like(y1)
    y1.backward(g)
    y2.backward(g)
    assert torch.equal(x1.grad, x2.grad)


def test_dtype_and_device_conversion_keeps_int8_and_bf16_scale():
    lin = nn.Linear(64, 8).to(torch.bfloat16)
    m = nn.Sequential(lin)
    convert_frozen_modules(m, log=False)
    mod = m[0]
    mod.to(torch.float32)  # .to(dtype): q は int8、scale は bf16 のまま。出力 dtype は追従する
    assert mod.q.dtype == torch.int8 and mod.scale.dtype == torch.bfloat16 and mod.out_dtype == torch.float32
    assert mod(torch.randn(2, 64)).dtype == torch.float32
    mod.to(torch.bfloat16)
    assert mod.out_dtype == torch.bfloat16 and mod.q.dtype == torch.int8


def test_buffers_are_not_in_state_dict_and_weight_is_not_kept():
    m = nn.Sequential(nn.Linear(64, 8))
    convert_frozen_modules(m, log=False)
    assert list(m.state_dict().keys()) == ["0.bias"]
    assert not hasattr(m[0], "weight")
    assert [n for n, _ in m[0].named_parameters()] == ["bias"]


def test_skip_and_include_regex_and_idempotent():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(64, 8)
            self.b = nn.Linear(64, 8)
            self.c = nn.Conv2d(3, 4, 3)
            self.norm = nn.GroupNorm(1, 4)
            self.e = nn.Embedding(5, 8)

    n = Net()
    st = convert_frozen_modules(n, skip="^b$", log=False)
    assert st["layers"] == 2 and isinstance(n.a, Int8Linear) and type(n.b) is nn.Linear and isinstance(n.c, Int8Conv2d)
    assert isinstance(n.norm, nn.GroupNorm) and isinstance(n.e, nn.Embedding)
    assert convert_frozen_modules(n, skip="^b$", log=False)["layers"] == 0  # 変換済みはそのまま
    st2 = convert_frozen_modules(Net(), include="^a$", log=False)
    assert st2["layers"] == 1


def test_memory_is_about_half_of_bf16_plus_scale():
    m = nn.Sequential(nn.Linear(1024, 512), nn.Linear(512, 512), nn.Conv2d(64, 64, 3)).to(torch.bfloat16)
    st = convert_frozen_modules(m, log=False)
    ratio = st["bytes_after"] / st["bytes_before"]
    assert 0.5 < ratio < 0.54  # int8 (1/2) + scale（64 要素に bf16 2 バイト = +3.1%）
    total = sum(x.numel() * x.element_size() for mod in int8_modules(m) for x in (mod.q, mod.scale))
    assert total == st["bytes_after"]


def test_apply_from_args_flags_and_errors():
    m = nn.Sequential(nn.Linear(64, 8))
    assert i8.apply_from_args(SimpleNamespace(int8_base=False), m) is None and type(m[0]) is nn.Linear
    assert i8.apply_from_args(SimpleNamespace(int8_base=True, int8_base_skip=None), None) is None
    with pytest.raises(ValueError):
        i8.apply_from_args(SimpleNamespace(int8_base=True, base_weights=["x"]), m)
    assert i8.apply_from_args(SimpleNamespace(int8_base=True, int8_base_skip=None), m)["layers"] == 1
    assert i8.apply_from_args(SimpleNamespace(int8_base=True, int8_base_skip=None), m) is None  # 冪等


# ---- LoRA との互換（UNet 縮小モデル / Anima 縮小モデル）-----------------------------------------------------
def _lora_grads(model, net, make_input):
    for p in net.parameters():
        p.grad = None
    out = model(*make_input())
    out = out[0] if isinstance(out, tuple) else (out.sample if hasattr(out, "sample") else out)
    loss = out.float().pow(2).mean()
    loss.backward()
    return out.detach(), loss.item(), [p.grad.clone() for p in net.parameters() if p.requires_grad]


def _randomize_lora_up(net, seed=1):
    g = torch.Generator().manual_seed(seed)
    for m in net.unet_loras:
        for p in m.lora_up.parameters():
            p.data = (torch.randn(p.shape, generator=g) * 0.05).to(p.dtype)


class TinyUNet(nn.Module):
    """SD の Transformer2DModel（Conv2d 1x1 の proj_in / proj_out、Linear の注意・FFN）を含む縮小モデル。"""

    def __init__(self, linear_projection):
        super().__init__()
        from library.original_unet import Transformer2DModel

        self.conv_in = nn.Conv2d(4, 32, 3, padding=1)
        self.tr = Transformer2DModel(num_attention_heads=2, attention_head_dim=16, in_channels=32, cross_attention_dim=24,
                                     use_linear_projection=linear_projection)
        self.conv_out = nn.Conv2d(32, 4, 3, padding=1)

    def forward(self, x, ctx):
        h = self.conv_in(x)
        h = self.tr(h, ctx)
        h = h.sample if hasattr(h, "sample") else h
        return self.conv_out(h)


@pytest.mark.parametrize("linear_projection", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lora_on_unet_like_model_matches_fake_int8(linear_projection, dtype):
    from networks import lora

    torch.manual_seed(0)
    base = TinyUNet(linear_projection).to(dtype)
    for p in base.parameters():
        p.requires_grad_(False)
    fake = fake_quantize_modules(base)
    int8 = copy.deepcopy(base)
    stats = convert_frozen_modules(int8, log=False)
    assert stats["layers"] > 4
    outs = []
    for model in (fake, int8):
        torch.manual_seed(2)
        net = lora.LoRANetwork([], model, lora_dim=4, alpha=4)
        net.apply_to(None, model, False, True)
        net.to(dtype)
        _randomize_lora_up(net)
        assert len(net.unet_loras) >= 8
        g = torch.Generator().manual_seed(3)
        inp = (torch.randn(1, 4, 8, 8, generator=g).to(dtype), torch.randn(1, 5, 24, generator=g).to(dtype))
        outs.append(_lora_grads(model, net, lambda: inp))
    assert torch.equal(outs[0][0], outs[1][0])  # 出力
    assert all(torch.equal(a, b) for a, b in zip(outs[0][2], outs[1][2]))  # LoRA 勾配
    assert all(torch.isfinite(g).all() for g in outs[1][2]) and any(g.abs().sum() > 0 for g in outs[1][2])


def _tiny_anima(dtype):
    from library import anima_models

    cfg = dict(
        max_img_h=64, max_img_w=64, max_frames=8, in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
        model_channels=256, num_blocks=2, num_heads=2,
        crossattn_emb_channels=256, pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop", min_fps=1, max_fps=30,
        use_adaln_lora=True, adaln_lora_dim=32,
        rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
        rope_enable_fps_modulation=False, use_llm_adapter=False, attn_mode="torch", split_attn=False,
    )
    m = anima_models.Anima(**cfg).to(dtype)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.mark.parametrize("checkpointing", [False, True])
def test_lora_on_anima_matches_fake_int8_with_and_without_checkpointing(checkpointing):
    from networks import lora_anima

    dtype = torch.float32
    torch.manual_seed(0)
    base = _tiny_anima(dtype)
    fake = fake_quantize_modules(base)
    int8 = copy.deepcopy(base)
    stats = convert_frozen_modules(int8, log=False)
    assert stats["layers"] > 10
    outs = []
    for model in (fake, int8):
        if checkpointing:
            model.enable_gradient_checkpointing()
        model.train()
        torch.manual_seed(2)
        net = lora_anima.LoRANetwork([], model, lora_dim=4, alpha=4)
        net.apply_to([], model, False, True)
        _randomize_lora_up(net)
        assert len(net.unet_loras) >= 10
        g = torch.Generator().manual_seed(3)
        inp = (
            torch.randn(1, 16, 1, 16, 16, generator=g).to(dtype),
            torch.rand(1, generator=g).to(dtype),
            torch.randn(1, 8, 256, generator=g).to(dtype),
        )
        pad = torch.zeros(1, 1, 16, 16, dtype=dtype)

        def run(model=model, net=net, inp=inp, pad=pad):
            return model(*inp, padding_mask=pad)

        for p in net.parameters():
            p.grad = None
        out = run()
        out.float().pow(2).mean().backward()
        outs.append((out.detach(), [p.grad.clone() for p in net.parameters() if p.requires_grad]))
    assert torch.equal(outs[0][0], outs[1][0])
    assert all(torch.equal(a, b) for a, b in zip(outs[0][1], outs[1][1]))
    assert all(torch.isfinite(g).all() for g in outs[1][1])


def test_lora_target_detection_treats_int8_modules_as_their_base_kind():
    from library.int8_weights import lora_base_class_name

    m = nn.Sequential(nn.Linear(64, 8), nn.Conv2d(3, 4, 3))
    assert [lora_base_class_name(x) for x in m] == ["Linear", "Conv2d"]
    convert_frozen_modules(m, log=False)
    assert [lora_base_class_name(x) for x in m] == ["Linear", "Conv2d"]
    assert [x.__class__.__name__ for x in m] == ["Int8Linear", "Int8Conv2d"]


# ---- ブロックスワップとの併用 -------------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)


def test_h2d_swap_selector_includes_int8_q_and_scale_buffers():
    from library import custom_offloading_utils as cou

    plain = _Block()
    assert [(n, a) for m, a in cou.default_swap_tensor_selector(plain) for n in [m.__class__.__name__]] == [("Linear", "weight")]
    blk = _Block()
    convert_frozen_modules(blk, log=False)
    jobs = cou.default_swap_tensor_selector(blk)
    assert [a for _, a in jobs] == ["q", "scale"]
    # q / scale は登録済みの buffer（LoRAStreamOffloader._jobs が buffer として扱える）で、凍結されている
    assert all(a in m._buffers for m, a in jobs)


def test_generic_swap_with_int8_is_an_explicit_error_on_accelerators_and_a_noop_on_cpu():
    from library import custom_offloading_utils as cou

    blocks = [_Block() for _ in range(4)]
    for b in blocks:
        convert_frozen_modules(b, log=False)
    cfg_dev = cou.BlockSwapConfig(device=torch.device("meta"), supports_backward=True, h2d_only=False)
    with pytest.raises(ValueError, match="int8_base"):
        cou.create_offloader("block", blocks, 4, 2, cfg_dev)
    # CPU が計算デバイスなら、スワップは何もしない（既存の挙動）ので、そのまま作れる
    cfg_cpu = cou.BlockSwapConfig(device=torch.device("cpu"), supports_backward=True, h2d_only=False)
    off = cou.create_offloader("block", blocks, 4, 2, cfg_cpu)
    assert off is not None
