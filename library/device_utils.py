import functools
import gc
import importlib
import os
from typing import Optional, Union

import torch


try:
    # intel gpu support for pytorch older than 2.5
    # ipex is not needed after pytorch 2.5
    import intel_extension_for_pytorch as ipex  # noqa
except Exception:
    pass


try:
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    HAS_MPS = False

try:
    HAS_XPU = torch.xpu.is_available()
except Exception:
    HAS_XPU = False


# torch.<name> モジュール（get_rng_state / set_rng_state / manual_seed / seed / empty_cache を持つ）を提供する
# アクセラレータの名前。CUDA 以外のデバイス（将来の vulkan など）は register_accelerator_backend で追加する。
_ACCELERATOR_BACKENDS = ["cuda", "xpu", "mps"]


def register_accelerator_backend(name: str):
    r"""
    `torch.<name>` モジュールを持つデバイスバックエンドを、乱数状態・シード・キャッシュ解放の対象に追加する。
    Add a device backend (`torch.<name>` module) to the targets of the RNG / seed / cache helpers below.
    """
    if name not in _ACCELERATOR_BACKENDS:
        _ACCELERATOR_BACKENDS.append(name)


# 環境変数 SD_SCRIPTS_BACKEND で明示的に選ぶ外部バックエンド（PrivateUse1 デバイス）。名前 -> import するモジュール。
# 例: SD_SCRIPTS_BACKEND=vk  （torch_vulkan をインストール済みの場合）
_EXTERNAL_BACKEND_MODULES = {"vk": "torch_vulkan"}
_external_backend: Optional[str] = None


def init_external_backend(name: Optional[str] = None) -> Optional[str]:
    r"""
    外部バックエンドを import して登録する。name 省略時は環境変数 SD_SCRIPTS_BACKEND を使う（未設定なら何もしない）。
    指定したのに import できない場合は、黙って CPU に落とさずエラーにする。
    Import and register an external device backend. Raises if explicitly requested but unavailable.
    """
    global _external_backend
    name = name or os.environ.get("SD_SCRIPTS_BACKEND")
    if not name:
        return None
    if _external_backend == name:
        return name
    module_name = _EXTERNAL_BACKEND_MODULES.get(name, name)
    try:
        importlib.import_module(module_name)
    except Exception as e:
        raise RuntimeError(
            f"backend '{name}' was requested (SD_SCRIPTS_BACKEND) but module '{module_name}' could not be imported: {e} / "
            f"バックエンド '{name}' が指定されましたが、モジュール '{module_name}' を import できません: {e}"
        ) from e
    register_accelerator_backend(name)
    _external_backend = name
    get_preferred_device.cache_clear()
    return name


def override_accelerator_device(accelerator=None):
    r"""
    accelerate は外部バックエンドを知らず device を cpu と判定するので、外部バックエンド使用時は vk 等に上書きする。
    外部バックエンドが無効なら何もしない。accelerator.device は state.device を返すので、state を書き換える。
    accelerate does not know external backends and reports cpu; override its state device when one is active.
    """
    if _external_backend is None:
        return
    from accelerate.state import AcceleratorState, PartialState

    device = torch.device(_external_backend)
    PartialState().device = device
    AcceleratorState().device = device


def _available_accelerator_modules():
    r"""利用可能なアクセラレータの (名前, torch.<name> モジュール) を列挙する。"""
    for name in _ACCELERATOR_BACKENDS:
        module = getattr(torch, name, None)
        if module is None:
            continue
        try:
            # torch.mps には is_available() が無いので torch.backends.mps で判定する
            available = torch.backends.mps.is_available() if name == "mps" else module.is_available()
            if available:
                yield name, module
        except Exception:
            continue


def get_memory_budget_bytes(device: Union[str, torch.device]) -> Optional[int]:
    r"""
    学習に使ってよい VRAM の量（バイト）。取得できないデバイスでは None。
    - cuda: 現在の空き容量
    - 外部バックエンド（vk）: OS が許す予算（VK_EXT_memory_budget。表示用や他のプロセスの使用分を除く）から、このプロセスの使用分を除いた残り
    Free VRAM usable for training in bytes, or None if the device cannot report it.
    """
    device = torch.device(device)
    try:
        if device.type == "cuda":
            return int(torch.cuda.mem_get_info(device)[0])
        if device.type == _external_backend and hasattr(getattr(torch, device.type), "memory_info"):
            info = getattr(torch, device.type).memory_info()
            return max(int(info["budget"]) - int(info["usage"]), 0)
    except Exception:
        return None
    return None


def set_memory_limit_mb(device: Union[str, torch.device], megabytes: int) -> bool:
    r"""
    小さい VRAM の GPU を模擬するために、使用量の上限（MiB）を設ける。対応するデバイス（vk）のみ。設定できたら True。
    Cap the memory the device may reserve, to emulate a GPU with less VRAM. Supported by the external "vk" backend only.
    """
    device = torch.device(device)
    module = getattr(torch, device.type, None)
    if device.type == _external_backend and module is not None and hasattr(module, "set_memory_limit"):
        module.set_memory_limit(int(megabytes))
        return True
    return False


def get_accelerator_count() -> int:
    r"""
    利用可能なアクセラレータ（cuda / xpu / vk など、`_ACCELERATOR_BACKENDS` のうち使えるもの）の合計デバイス数。CPU のみなら 0。
    Number of visible accelerator devices over all available backends. 0 when only the CPU is available.
    """
    total = 0
    for _, module in _available_accelerator_modules():
        try:
            total += int(module.device_count())
        except Exception:
            total += 1  # device_count を持たないバックエンド（mps など）は 1 台として数える
    return total


def get_ddp_backend() -> str:
    r"""
    分散学習（DDP）のプロセスグループのバックエンド。nccl は Windows 以外の CUDA だけで、それ以外（CPU・xpu・mps・vk など）は gloo。
    Process group backend for DDP: "nccl" only for CUDA on non-Windows, "gloo" otherwise.
    """
    return "nccl" if HAS_CUDA and os.name != "nt" else "gloo"


def empty_cache():
    r"""利用可能な全アクセラレータのキャッシュを解放する（gc.collect は呼ばない）。"""
    for _, module in _available_accelerator_modules():
        try:
            module.empty_cache()
        except Exception:
            pass


def clean_memory():
    gc.collect()
    empty_cache()


def get_rng_states() -> dict:
    r"""
    CPU と、利用可能な全アクセラレータの乱数状態を取得する。サンプル画像生成の前後で学習の乱数列を保つために使う。
    Get RNG states of the CPU and all available accelerators. Pass the result to `set_rng_states` to restore.
    """
    states = {"cpu": torch.get_rng_state()}
    for name, module in _available_accelerator_modules():
        try:
            states[name] = module.get_rng_state()
        except Exception:
            pass
    return states


def set_rng_states(states: dict):
    r"""`get_rng_states` で取得した乱数状態を復元する。取得時に無かった・今は使えないデバイスは無視する。"""
    if "cpu" in states:
        torch.set_rng_state(states["cpu"])
    for name, module in _available_accelerator_modules():
        if name in states:
            try:
                module.set_rng_state(states[name])
            except Exception:
                pass


def manual_seed_all(seed: int):
    r"""CPU と、利用可能な全アクセラレータの乱数シードを固定する。"""
    torch.manual_seed(seed)
    for _, module in _available_accelerator_modules():
        try:
            if hasattr(module, "manual_seed_all"):
                module.manual_seed_all(seed)
            else:
                module.manual_seed(seed)
        except Exception:
            pass


def random_seed():
    r"""CPU と、利用可能な全アクセラレータの乱数シードを非決定的な値にする（`torch.seed()` 相当）。"""
    torch.seed()
    for _, module in _available_accelerator_modules():
        try:
            module.seed()
        except Exception:
            pass


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()
    if device.type == _external_backend:
        getattr(torch, device.type).empty_cache()


def synchronize_device(device: Optional[Union[str, torch.device]]):
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == _external_backend:
        getattr(torch, device.type).synchronize()


@functools.lru_cache(maxsize=None)
def get_preferred_device() -> torch.device:
    r"""
    Do not call this function from training scripts. Use accelerator.device instead.
    """
    if _external_backend is not None:
        # 環境変数で明示された外部バックエンドが最優先
        device = torch.device(_external_backend)
    elif HAS_CUDA:
        device = torch.device("cuda")
    elif HAS_XPU:
        device = torch.device("xpu")
    elif HAS_MPS:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"get_preferred_device() -> {device}")
    return device


def init_ipex():
    """
    Apply IPEX to CUDA hijacks using `library.ipex.ipex_init`.

    This function should run right after importing torch and before doing anything else.

    If xpu is not available, this function does nothing.
    """
    try:
        if HAS_XPU:
            from library.ipex import ipex_init

            is_initialized, error_message = ipex_init()
            if not is_initialized:
                print("failed to initialize ipex:", error_message)
        else:
            return
    except Exception as e:
        print("failed to initialize ipex:", e)


# 環境変数 SD_SCRIPTS_BACKEND が指定されていれば、import 時に外部バックエンドを有効にする（未指定なら何もしない）
init_external_backend()
