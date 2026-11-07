"""library/vram_budget.py（VRAM 予算から勾配チェックポイント・ブロックスワップ数を決める）のテスト。"""

import torch
from safetensors.torch import save_file

import library.vram_budget as vb


def _plan(budget, **kw):
    args = dict(num_blocks=28, block_mib=132.0, non_block_mib=292.0, network_dim=16, pixels=512 * 512)
    args.update(kw)
    return vb.plan(budget, **args)


def test_large_budget_needs_nothing():
    p = _plan(16000)
    assert (p.gradient_checkpointing, p.blocks_to_swap, p.feasible) == (False, 0, True)


def test_medium_budget_uses_checkpointing_only():
    p = _plan(6000)
    assert (p.gradient_checkpointing, p.blocks_to_swap, p.feasible) == (True, 0, True)


def test_small_budget_swaps_blocks_and_fits_the_estimate():
    p = _plan(3072)
    assert p.gradient_checkpointing and p.blocks_to_swap > 0 and p.feasible
    assert p.predicted_mib <= 3072
    # 1 ブロック減らすと予算を超える（最小のスワップ数を選んでいる）
    assert _plan(3072 - 1).blocks_to_swap >= p.blocks_to_swap


def test_swap_count_is_monotonic_in_budget():
    counts = [_plan(b).blocks_to_swap for b in range(1500, 6000, 250)]
    assert counts == sorted(counts, reverse=True)


def test_infeasible_budget_returns_max_swap():
    p = _plan(500)
    assert not p.feasible
    assert p.blocks_to_swap == 26  # num_blocks - 2（enable_block_swap の上限）


def test_resolution_and_rank_increase_the_requirement():
    base = _plan(3072)
    assert _plan(3072, pixels=768 * 768).blocks_to_swap >= base.blocks_to_swap
    assert _plan(3072, network_dim=64).blocks_to_swap >= base.blocks_to_swap


def test_plan_for_dit_file_reads_only_the_header(tmp_path):
    # 4 ブロック。各ブロックは bf16 の 1024x1024 を 2 つ（= 4 MiB）。ブロック外に 1 MiB
    tensors = {}
    for i in range(4):
        tensors[f"net.blocks.{i}.a.weight"] = torch.zeros(1024, 1024, dtype=torch.bfloat16)
        tensors[f"net.blocks.{i}.b.weight"] = torch.zeros(1024, 1024, dtype=torch.bfloat16)
    tensors["net.llm_adapter.w"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
    path = tmp_path / "dit.safetensors"
    save_file(tensors, str(path))

    blocks, other = vb.split_block_sizes(vb.read_safetensors_sizes(str(path), 2))
    assert sorted(blocks) == [0, 1, 2, 3]
    assert blocks[0] == 4 * vb.MiB
    assert other == 1 * vb.MiB

    # fp32 で学習するなら 2 倍
    blocks32, _ = vb.split_block_sizes(vb.read_safetensors_sizes(str(path), 4))
    assert blocks32[0] == 8 * vb.MiB

    p = vb.plan_for_dit_file(str(path), 100000, weight_bytes_per_element=2, network_dim=4, pixels=512 * 512)
    assert p is not None and p.blocks_to_swap == 0 and not p.gradient_checkpointing


def test_plan_for_dit_file_returns_none_for_unreadable_file(tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file")
    assert vb.plan_for_dit_file(str(bad), 3072, weight_bytes_per_element=2, network_dim=16, pixels=512 * 512) is None
