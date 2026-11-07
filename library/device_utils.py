import functools
import gc
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


@functools.lru_cache(maxsize=None)
def get_preferred_device() -> torch.device:
    r"""
    Do not call this function from training scripts. Use accelerator.device instead.
    """
    if HAS_CUDA:
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
