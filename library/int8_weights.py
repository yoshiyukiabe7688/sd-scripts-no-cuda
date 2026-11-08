"""凍結したベース重みだけを INT8 で保持する（LoRA 学習のメモリ削減）。`--int8_base` で train_network.py から使う。

格納形式（固定。vk 側の dequant カーネルと共有する規約）:
  q      : int8  [rows, cols]         Linear は weight そのもの、Conv2d は weight を [out, in*kh*kw] に平坦化したもの
  scale  : bf16  [rows, ceil(cols/64)] 各行を 64 要素ずつに区切ったブロックごとの scale（端数ブロックあり）
量子化: s = bf16(max|block| / 127)（max=0 のブロックは s=0, q=0）、q = clamp(round(w / s), -127, 127)（round は偶数丸め）。
        scale は bf16 に丸めた値で量子化するので、dequant で使う値と一致する。
dequant: out = dtype(float(q) * float(scale))（fp32 で掛けてから出力 dtype に丸める。bf16 の scale と 8 ビットの q の積は fp32 で正確）。

対象は nn.Linear / nn.Conv2d の weight のみ（bias・Norm・埋め込み・LoRA は対象外）。元の weight は保持しない（メモリを減らすのが目的）。
Int8Linear / Int8Conv2d は nn.Linear / nn.Conv2d のサブクラス（isinstance と in_features などの属性は保つ）。q と scale は
persistent=False の buffer なので state_dict には入らない。`weight` 属性は**持たない**（実体の dequant を毎回しないため）。

【vk 連携の接点（親が統合する）】
  - dequant の呼び出しは `_dequant()` の 1 か所。q が vk 上で `torch_vulkan.dequant_int8` があればそれを使う。
    想定シグネチャ: dequant_int8(q: int8[rows, cols], scale: bf16[rows, nblocks], dtype: torch.dtype) -> Tensor[rows, cols] (dtype)。
    無ければ純 PyTorch の基準実装 `dequantize_weight`（vk 上の int8 には基準実装は使えない。CPU フォールバックになる）。
  - buffer を vk へ載せる経路は `module.to("vk")`（`_apply`）。int8 の q は dtype 変換の対象外、scale は bf16 のまま保つ。
  - H2D 専用ブロックスワップ（LoRAStreamOffloader）は `default_swap_tensor_selector` が q と scale の buffer を流す
    （library/custom_offloading_utils.py）。通常の交換型スワップ（ModelOffloader）は未対応でエラーにする。
"""
import re
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

BLOCK = 64
_MAXQ = 127


# ---------------------------------------------------------------------------------------------------------------
# 量子化・dequant の基準実装（純 PyTorch）
# ---------------------------------------------------------------------------------------------------------------
def quantize_weight(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """weight を (q: int8 [rows, cols], scale: bf16 [rows, ceil(cols/64)]) に量子化する。"""
    w2 = w.detach().float().reshape(w.shape[0], -1)
    rows, cols = w2.shape
    pad = (-cols) % BLOCK
    if pad:
        w2 = torch.cat([w2, w2.new_zeros(rows, pad)], dim=1)
    wb = w2.reshape(rows, -1, BLOCK)
    scale = (wb.abs().amax(dim=2) / _MAXQ).to(torch.bfloat16)  # [rows, nb]
    s = scale.float().unsqueeze(2)
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.where(s > 0, torch.round(wb / safe), torch.zeros_like(wb)).clamp_(-_MAXQ, _MAXQ)
    q = q.reshape(rows, -1)[:, :cols].to(torch.int8).contiguous()
    return q, scale.contiguous()


def dequantize_weight(q: torch.Tensor, scale: torch.Tensor, shape, dtype: torch.dtype) -> torch.Tensor:
    """基準実装の dequant。q [rows, cols] と scale [rows, nb] から `shape` の weight（dtype）を作る。"""
    rows, cols = q.shape
    nb = scale.shape[1]
    s = scale.float().repeat_interleave(BLOCK, dim=1)[:, :cols]
    return (q.float() * s).to(dtype).reshape(shape)


def _dequant(q: torch.Tensor, scale: torch.Tensor, shape, dtype: torch.dtype) -> torch.Tensor:
    """dequant の唯一の呼び出し点。vk 上で torch_vulkan.dequant_int8 があればそれを使う（親が統合）。"""
    if q.device.type == "vk":
        try:
            import torch_vulkan

            fn = getattr(torch_vulkan, "dequant_int8", None)
        except ImportError:
            fn = None
        if fn is not None:
            return fn(q, scale, dtype).reshape(shape)
    return dequantize_weight(q, scale, shape, dtype)


def quantize_dequantize(w: torch.Tensor) -> torch.Tensor:
    """fake-int8（量子化して元の dtype に戻す）。テストでビット一致の基準にする。"""
    q, scale = quantize_weight(w)
    return dequantize_weight(q, scale, w.shape, w.dtype)


# ---------------------------------------------------------------------------------------------------------------
# 凍結重み用の autograd Function: dequant した weight を保存せず、backward で再 dequant する
# （通常の F.linear は backward 用に weight を保存するので、全層の dequant 結果が同時に生きてメモリ削減が消える）
# ---------------------------------------------------------------------------------------------------------------
class _Int8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q, scale, bias, shape, out_dtype):
        ctx.save_for_backward(q, scale)
        ctx.shape, ctx.out_dtype, ctx.has_bias = shape, out_dtype, bias is not None
        w = _dequant(q, scale, shape, out_dtype)
        return F.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad_out):
        q, scale = ctx.saved_tensors
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            w = _dequant(q, scale, ctx.shape, grad_out.dtype)
            grad_x = grad_out @ w
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = grad_out.reshape(-1, grad_out.shape[-1]).sum(0)
        return grad_x, None, None, grad_bias, None, None


class _Int8ConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q, scale, bias, shape, out_dtype, stride, padding, dilation, groups):
        ctx.save_for_backward(q, scale)
        ctx.shape, ctx.out_dtype, ctx.has_bias = shape, out_dtype, bias is not None
        ctx.conv = (stride, padding, dilation, groups)
        ctx.x_shape = x.shape
        w = _dequant(q, scale, shape, out_dtype)
        return F.conv2d(x, w, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_out):
        q, scale = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.conv
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            w = _dequant(q, scale, ctx.shape, grad_out.dtype)
            grad_x = torch.nn.grad.conv2d_input(ctx.x_shape, w, grad_out, stride, padding, dilation, groups)
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = grad_out.sum(dim=(0, 2, 3))
        return grad_x, None, None, grad_bias, None, None, None, None, None, None


# ---------------------------------------------------------------------------------------------------------------
# モジュール
# ---------------------------------------------------------------------------------------------------------------
class _Int8Mixin:
    """Int8Linear / Int8Conv2d 共通: q・scale の buffer と、dtype/device 変換時の扱い。"""

    # networks/lora*.py が `org_module.__class__.__name__ == "Linear"/"Conv2d"` で種類を判定するための名前（lora_base_class_name を参照）
    lora_base_class_name = ""

    def _init_int8(self, weight: torch.Tensor):
        # 量子化は CPU で行う（weight が vk などにあっても、そこで fp32 の一時テンソルを作らない）。結果は元のデバイスへ置く。
        dev = weight.device
        q, scale = quantize_weight(weight if dev.type == "cpu" else weight.cpu())
        self.out_dtype = weight.dtype
        self._wshape = tuple(weight.shape)
        self.register_buffer("q", q.to(dev), persistent=False)
        self.register_buffer("scale", scale.to(dev), persistent=False)

    def _apply(self, fn, *args, **kwargs):
        # .to(dtype) / .to(device) / .cuda() などの変換。q は int8、scale は bf16 のまま保つ（device だけ追従）。
        out = super()._apply(fn, *args, **kwargs)
        self._buffers["q"] = self._buffers["q"].to(torch.int8) if self._buffers["q"].dtype != torch.int8 else self._buffers["q"]
        if self._buffers["scale"].dtype != torch.bfloat16:
            self._buffers["scale"] = self._buffers["scale"].to(torch.bfloat16)
        # 出力 dtype は、fn が浮動小数のテンソルに行う変換（.to(dtype)）に追従する。
        try:
            self.out_dtype = fn(torch.empty(0, dtype=self.out_dtype)).dtype
        except Exception:
            pass
        return out

    def int8_nbytes(self) -> int:
        return self.q.numel() * self.q.element_size() + self.scale.numel() * self.scale.element_size()


class Int8Linear(_Int8Mixin, nn.Linear):
    lora_base_class_name = "Linear"

    def __init__(self, linear: nn.Linear):
        nn.Module.__init__(self)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self._init_int8(linear.weight.data)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data, requires_grad=linear.bias.requires_grad)
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):  # weight が無いので何もしない
        pass

    def forward(self, x):
        return _Int8LinearFn.apply(x, self.q, self.scale, self.bias, self._wshape, self.out_dtype)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, int8_block={BLOCK}"


class Int8Conv2d(_Int8Mixin, nn.Conv2d):
    lora_base_class_name = "Conv2d"

    def __init__(self, conv: nn.Conv2d):
        nn.Module.__init__(self)
        for name in ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode"):
            setattr(self, name, getattr(conv, name))
        self.transposed = False
        self.output_padding = conv.output_padding
        self._reversed_padding_repeated_twice = conv._reversed_padding_repeated_twice
        self._init_int8(conv.weight.data)
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data, requires_grad=conv.bias.requires_grad)
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        pass

    def forward(self, x):
        if self.padding_mode != "zeros" or isinstance(self.padding, str):
            # 特殊なパディングは基準実装の weight で通常の経路（メモリ削減の対象外）
            w = _dequant(self.q, self.scale, self._wshape, self.out_dtype)
            return self._conv_forward(x, w, self.bias)
        return _Int8ConvFn.apply(
            x, self.q, self.scale, self.bias, self._wshape, self.out_dtype, self.stride, self.padding, self.dilation, self.groups
        )


def lora_base_class_name(module: nn.Module) -> str:
    """networks/lora*.py が org_module の種類（"Linear" / "Conv2d"）を判定するための名前。Int8 モジュールは元の種類を返す。"""
    return getattr(module, "lora_base_class_name", "") or module.__class__.__name__


def int8_modules(model: nn.Module) -> List[nn.Module]:
    return [m for m in model.modules() if isinstance(m, _Int8Mixin)]


# ---------------------------------------------------------------------------------------------------------------
# 変換
# ---------------------------------------------------------------------------------------------------------------
def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def convert_frozen_modules(
    model: nn.Module, include: Optional[str] = None, skip: Optional[str] = None, log: bool = True
) -> dict:
    """model の Linear / Conv2d を Int8Linear / Int8Conv2d に置き換える（凍結した重みの LoRA 学習用）。

    include / skip は、モジュール名（`named_modules()` の名前）に対する正規表現（re.search）。
    戻り値は統計 {"layers", "bytes_before", "bytes_after", "skipped"}。サイズは weight（と scale）のバイト数。
    """
    inc = re.compile(include) if include else None
    skp = re.compile(skip) if skip else None
    targets: List[Tuple[nn.Module, str, nn.Module]] = []
    for pname, parent in model.named_modules():
        for cname, child in parent.named_children():
            full = f"{pname}.{cname}" if pname else cname
            if isinstance(child, _Int8Mixin) or type(child) not in (nn.Linear, nn.Conv2d):
                continue
            if inc and not inc.search(full):
                continue
            if skp and skp.search(full):
                continue
            targets.append((parent, cname, child))
    before = after = 0
    for parent, cname, child in targets:
        before += _nbytes(child.weight)
        new = Int8Linear(child) if type(child) is nn.Linear else Int8Conv2d(child)
        after += new.int8_nbytes()
        setattr(parent, cname, new)
    stats = {"layers": len(targets), "bytes_before": before, "bytes_after": after, "skipped": 0}
    if log:
        mib = 1024 * 1024
        logger.info(
            f"int8 base: converted {len(targets)} Linear/Conv2d layers, weights {before / mib:.0f} MiB -> {after / mib:.0f} MiB"
            f" ({after / max(before, 1) * 100:.1f}%) / 凍結ベース重みを int8 化: {len(targets)} 層、{before / mib:.0f} MiB -> {after / mib:.0f} MiB"
        )
    return stats


def apply_from_args(args, model: Optional[nn.Module]) -> Optional[dict]:
    """`--int8_base` が指定されていれば model の凍結ベース重みを int8 化する（冪等: 変換済みの層は何もしない）。

    train_network.py から、モデル読み込み後・LoRA 作成前に呼ぶ。ブロックスワップを使うモデル（Anima）は、offloader が
    swap 対象のテンソルを生成時に集めるので、enable_block_swap の前（モデル読み込み直後）にも呼ぶ。
    """
    if model is None or not getattr(args, "int8_base", False):
        return None
    if getattr(args, "base_weights", None):
        raise ValueError(
            "--int8_base cannot be combined with --base_weights (merging into int8 weights is not supported)"
            " / --int8_base は --base_weights と併用できません（int8 の重みへの merge は未対応です）"
        )
    if not any(isinstance(m, (nn.Linear, nn.Conv2d)) and not isinstance(m, _Int8Mixin) for m in model.modules()):
        return None  # 変換済み（または対象なし）
    return convert_frozen_modules(model, skip=getattr(args, "int8_base_skip", None))
