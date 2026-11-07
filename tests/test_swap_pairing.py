"""非 CUDA 向けブロックスワップの、サブモジュールのペアリングを確認する。

以前は `zip(a.modules(), b.modules())` で順序対応させていたため、同じクラスでもサブモジュールの構成が違うブロックで
重みを取り違えていた（静かに壊れる）。名前で対応させ、構成が違えば明示的にエラーにする。
"""

import pytest
import torch.nn as nn

from library.custom_offloading_utils import _pair_modules


class Block(nn.Module):
    def __init__(self, with_extra: bool):
        super().__init__()
        self.a = nn.Linear(4, 4)
        if with_extra:
            self.extra = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)


def test_pairs_by_name():
    x, y = Block(True), Block(True)
    pairs = _pair_modules(x, y)
    names_x = dict(x.named_modules())
    names_y = dict(y.named_modules())
    assert len(pairs) == len(names_x)
    for name in names_x:
        assert (names_x[name], names_y[name]) in [(p, q) for p, q in pairs]


def test_structure_mismatch_raises():
    # zip(modules()) だと a↔a, extra↔b, b↔(無し) と取り違える構成
    with pytest.raises(ValueError, match="extra"):
        _pair_modules(Block(True), Block(False))
