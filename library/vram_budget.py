"""VRAM の予算から、勾配チェックポイントの要否とスワップするブロック数を決める（DiT の LoRA 学習向け）。

`--vram_budget_mb` で指定された予算（または GPU から取得した使用可能量）に学習が収まるように、次の順で最も速い構成を選ぶ。
  1. 何もしない（活性化を全部保持）
  2. 勾配チェックポイントのみ（約 1.4 倍遅い）
  3. 勾配チェックポイント + ブロックスワップ（H2D 専用、スワップ 1 ブロックあたり約 20 ms/step 増）

見積もりの定数は、実 Anima-base（512px, batch 1, bf16, rank 16）を vk（RX 9070 XT）で測った値。
解像度は活性化に比例して増えるものとして補正する。実測との差は slack で吸収する。
"""

import json
import logging
import re
import struct
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MiB = 1 << 20

# ---- 実測値（Anima-base, 512x512 = 1024 トークン, batch 1, bf16）----
ACT_NO_CKPT_MIB = 5100  # 勾配チェックポイント無しの活性化（学習ステップのピーク − ステップ前）
ACT_CKPT_MIB = 320  # 勾配チェックポイントありの活性化（再計算 1 ブロック分の一時メモリ + 各ブロック入力）
LORA_MIB_PER_RANK = 11.0  # LoRA 重み + 勾配 + AdamW の状態（rank 16 で約 177 MiB）
SLACK_MIB = 360  # アロケータの断片化、ステージング、ドライバの管理領域
BASE_PIXELS = 512 * 512


@dataclass
class VramPlan:
    gradient_checkpointing: bool
    blocks_to_swap: int  # 0 ならスワップしない
    predicted_mib: float  # この構成での予想使用量
    budget_mib: float
    feasible: bool  # 予算内に収まる見込みか
    reason: str


def read_safetensors_sizes(path: str, bytes_per_element: float) -> Dict[str, int]:
    """safetensors のヘッダだけを読み、テンソル名 -> 学習時のバイト数（`bytes_per_element` で換算）を返す。重みは読まない。"""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    sizes = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        numel = 1
        for d in info["shape"]:
            numel *= d
        sizes[name] = int(numel * bytes_per_element)
    return sizes


def split_block_sizes(sizes: Dict[str, int], block_pattern: str = r"(?:^|\.)blocks\.(\d+)\.") -> tuple:
    """(ブロック番号 -> バイト数, ブロック以外のバイト数) に分ける。"""
    rx = re.compile(block_pattern)
    blocks: Dict[int, int] = {}
    other = 0
    for name, n in sizes.items():
        m = rx.search(name)
        if m:
            blocks[int(m.group(1))] = blocks.get(int(m.group(1)), 0) + n
        else:
            other += n
    return blocks, other


def plan(
    budget_mib: float,
    *,
    num_blocks: int,
    block_mib: float,
    non_block_mib: float,
    network_dim: int = 16,
    pixels: int = BASE_PIXELS,
    ring_size: int = 2,
    slack_mib: float = SLACK_MIB,
) -> VramPlan:
    """予算に収まる最も速い構成を選ぶ。最大までスワップしても収まらないときは feasible=False で最大スワップの構成を返す。"""
    scale = max(pixels / BASE_PIXELS, 0.25)
    lora = LORA_MIB_PER_RANK * network_dim

    def total(ckpt: bool, swap: int) -> float:
        resident_blocks = num_blocks - swap + (min(ring_size, swap) if swap > 0 else 0)
        act = (ACT_CKPT_MIB if ckpt else ACT_NO_CKPT_MIB) * scale
        return non_block_mib + resident_blocks * block_mib + act + lora + slack_mib

    t = total(False, 0)
    if t <= budget_mib:
        return VramPlan(False, 0, t, budget_mib, True, "全部載せて活性化も保持できる")
    t = total(True, 0)
    if t <= budget_mib:
        return VramPlan(True, 0, t, budget_mib, True, "勾配チェックポイントで収まる")
    max_swap = num_blocks - 2  # enable_block_swap の上限
    for swap in range(1, max_swap + 1):
        t = total(True, swap)
        if t <= budget_mib:
            return VramPlan(True, swap, t, budget_mib, True, f"勾配チェックポイント + {swap} ブロックをスワップ")
    t = total(True, max_swap)
    return VramPlan(True, max_swap, t, budget_mib, False, "最大までスワップしても予算に収まらない見込み")


def plan_for_dit_file(
    path: str,
    budget_mib: float,
    *,
    weight_bytes_per_element: float,
    network_dim: int,
    pixels: int,
    ring_size: int = 2,
) -> Optional[VramPlan]:
    """DiT の safetensors のヘッダからブロック数とサイズを求めて plan() する。読めなければ None。"""
    try:
        blocks, other = split_block_sizes(read_safetensors_sizes(path, weight_bytes_per_element))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not read the DiT header for the VRAM budget plan: {e}")
        return None
    if not blocks:
        return None
    num_blocks = max(blocks) + 1
    block_mib = sum(blocks.values()) / len(blocks) / MiB
    return plan(
        budget_mib,
        num_blocks=num_blocks,
        block_mib=block_mib,
        non_block_mib=other / MiB,
        network_dim=network_dim,
        pixels=pixels,
        ring_size=ring_size,
    )
