"""set_weight_data: `.data` 代入が使えないデバイス（PrivateUse1 など）向けのフォールバックのテスト（CPU のみで動く）。"""

import torch
import torch.nn as nn

from library.custom_offloading_utils import set_weight_data, weighs_to_device


class _NoDataSetterParameter(nn.Parameter):
    """`.data = ...` が incompatible tensor type で失敗するデバイスを模擬する Parameter。"""

    @property
    def data(self):
        return torch.Tensor._make_subclass(torch.Tensor, super().data)

    @data.setter
    def data(self, value):
        raise RuntimeError("Attempted to call `variable.set_data(tensor)`, but `variable` and `tensor` have incompatible tensor type.")


def _linear_with_failing_data_setter():
    lin = nn.Linear(4, 3, bias=False)
    lin.weight = _NoDataSetterParameter(lin.weight.detach().clone(), requires_grad=False)
    return lin


def test_set_weight_data_assigns_in_place_when_supported():
    lin = nn.Linear(4, 3, bias=False)
    param = lin.weight
    new = torch.randn(3, 4)
    set_weight_data(lin, new)
    assert lin.weight is param  # Parameter の同一性は保たれる
    assert torch.equal(lin.weight.data, new)


def test_set_weight_data_falls_back_to_replacing_the_parameter():
    lin = _linear_with_failing_data_setter()
    new = torch.randn(3, 4)
    set_weight_data(lin, new)
    assert isinstance(lin.weight, nn.Parameter)
    assert torch.equal(lin.weight.data, new)
    assert lin.weight.requires_grad is False  # 凍結したまま
    assert torch.equal(lin(torch.eye(4)), new.t())  # forward は新しい重みを使う


def test_weighs_to_device_uses_the_fallback_for_every_weight_in_a_block():
    block = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    for m in block:
        m.weight = _NoDataSetterParameter(m.weight.detach().clone(), requires_grad=False)
    expected = [m.weight.detach().clone() for m in block]
    weighs_to_device(block, torch.device("cpu"))
    for m, e in zip(block, expected):
        assert torch.equal(m.weight.data, e)


# ---- to_device_keeping_weights_on_cpu / ホスト側の原本 ----------------------------------------------------------
from library.custom_offloading_utils import to_device_keeping_weights_on_cpu  # noqa: E402


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3, bias=True)
        self.register_buffer("scale", torch.ones(3))


def test_to_device_keeping_weights_on_cpu_moves_everything_except_weights():
    block = _Block()
    w = block.lin.weight
    to_device_keeping_weights_on_cpu(block, torch.device("meta"))  # meta を「別デバイス」の代用にする
    assert block.lin.weight.device.type == "cpu"  # weight は CPU のまま（一度もデバイスに載せない）
    assert block.lin.weight is w
    assert block.lin.bias.device.type == "meta"
    assert block.scale.device.type == "meta"


def test_to_device_keeping_weights_on_cpu_is_a_noop_on_cpu():
    block = _Block()
    before = {k: v.clone() for k, v in block.state_dict().items()}
    to_device_keeping_weights_on_cpu(block, torch.device("cpu"))
    for k, v in block.state_dict().items():
        assert torch.equal(v, before[k])
