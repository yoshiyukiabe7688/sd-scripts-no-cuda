"""unsloth_checkpoint が、順伝播時の autocast 状態を逆伝播の再計算でも再現することを CPU で確認する。

以前は `torch.amp.custom_fwd/custom_bwd(device_type="cuda")` で状態を保存していたため、cuda 以外（cpu / xpu / vk）では
逆伝播の再計算が autocast の外で走り、bf16 の重みに fp32 の入力が渡って dtype 不一致で失敗していた。
"""

import torch
import torch.nn as nn

from library.custom_offloading_utils import unsloth_checkpoint


def _make():
    torch.manual_seed(0)
    lin = nn.Linear(8, 8).to(torch.bfloat16)
    x = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
    return lin, x


def _fn(lin):
    # autocast(bf16) の下でだけ、fp32 入力が bf16 の Linear に渡せる
    return lambda h: lin(h.float())


def test_backward_recompute_runs_under_the_same_autocast():
    lin, x = _make()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = unsloth_checkpoint(_fn(lin), x)
    y.float().sum().backward()  # 以前はここで dtype 不一致の RuntimeError
    assert x.grad is not None and x.grad.dtype == torch.bfloat16

    # 通常の（チェックポイント無し）実行と同じ勾配になる
    lin2, x2 = _make()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y2 = _fn(lin2)(x2)
    y2.float().sum().backward()
    assert torch.equal(x.grad, x2.grad)
    assert torch.equal(lin.weight.grad, lin2.weight.grad)


def test_no_autocast_state_is_not_leaked_into_backward():
    # autocast 無しで順伝播した場合、再計算も autocast 無しで走る（fp32 の入力と重みで一致する）
    torch.manual_seed(0)
    lin = nn.Linear(8, 8)
    x = torch.randn(4, 8, requires_grad=True)
    y = unsloth_checkpoint(lambda h: lin(h), x)
    y.sum().backward()
    assert x.grad is not None and x.grad.dtype == torch.float32
