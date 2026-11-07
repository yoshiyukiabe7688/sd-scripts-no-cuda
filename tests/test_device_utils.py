"""library.device_utils の乱数状態・シード・キャッシュ解放ヘルパーのテスト（CPU のみで動作する）。"""

import types

import pytest
import torch

from library import device_utils


class FakeDeviceModule:
    """torch.<name> 相当の擬似アクセラレータ。乱数状態はカウンタで表す。"""

    def __init__(self, available=True, with_manual_seed_all=True):
        self._available = available
        self.state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        self.cache_cleared = 0
        self.seeded = None
        self.random_seeded = 0
        if with_manual_seed_all:
            self.manual_seed_all = self._manual_seed_all

    def is_available(self):
        return self._available

    def get_rng_state(self):
        return self.state.clone()

    def set_rng_state(self, state):
        self.state = state.clone()

    def manual_seed(self, seed):
        self.seeded = ("manual_seed", seed)

    def _manual_seed_all(self, seed):
        self.seeded = ("manual_seed_all", seed)

    def seed(self):
        self.random_seeded += 1

    def empty_cache(self):
        self.cache_cleared += 1


@pytest.fixture
def fake_backend(monkeypatch):
    """擬似デバイス 'fakedev' を登録し、終了時に元へ戻す。"""
    module = FakeDeviceModule()
    monkeypatch.setattr(torch, "fakedev", module, raising=False)
    monkeypatch.setattr(device_utils, "_ACCELERATOR_BACKENDS", list(device_utils._ACCELERATOR_BACKENDS))
    device_utils.register_accelerator_backend("fakedev")
    return module


def test_rng_states_contain_cpu_and_restore():
    states = device_utils.get_rng_states()
    assert "cpu" in states
    first = torch.rand(4)
    device_utils.set_rng_states(states)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_manual_seed_all_is_deterministic():
    device_utils.manual_seed_all(123)
    a = torch.rand(8)
    device_utils.manual_seed_all(123)
    b = torch.rand(8)
    assert torch.equal(a, b)


def test_random_seed_changes_sequence():
    device_utils.manual_seed_all(0)
    device_utils.random_seed()
    a = torch.rand(8)
    device_utils.random_seed()
    b = torch.rand(8)
    assert not torch.equal(a, b)


def test_empty_cache_and_clean_memory_do_not_raise_on_cpu():
    device_utils.empty_cache()
    device_utils.clean_memory()


def test_register_backend_is_idempotent(monkeypatch):
    monkeypatch.setattr(device_utils, "_ACCELERATOR_BACKENDS", ["cuda"])
    device_utils.register_accelerator_backend("vulkan")
    device_utils.register_accelerator_backend("vulkan")
    assert device_utils._ACCELERATOR_BACKENDS == ["cuda", "vulkan"]


def test_registered_backend_is_included_in_rng_states(fake_backend):
    states = device_utils.get_rng_states()
    assert "fakedev" in states
    original = states["fakedev"].clone()

    fake_backend.state = torch.tensor([9, 9, 9], dtype=torch.uint8)  # 状態が変わった後に復元できること
    device_utils.set_rng_states(states)
    assert torch.equal(fake_backend.state, original)


def test_registered_backend_receives_seed_calls(fake_backend):
    device_utils.manual_seed_all(7)
    assert fake_backend.seeded == ("manual_seed_all", 7)
    device_utils.random_seed()
    assert fake_backend.random_seeded == 1
    device_utils.empty_cache()
    assert fake_backend.cache_cleared == 1


def test_manual_seed_falls_back_when_manual_seed_all_is_missing(monkeypatch):
    module = FakeDeviceModule(with_manual_seed_all=False)
    monkeypatch.setattr(torch, "fakedev", module, raising=False)
    monkeypatch.setattr(device_utils, "_ACCELERATOR_BACKENDS", ["fakedev"])
    device_utils.manual_seed_all(5)
    assert module.seeded == ("manual_seed", 5)


def test_unavailable_backend_is_ignored(monkeypatch):
    module = FakeDeviceModule(available=False)
    monkeypatch.setattr(torch, "fakedev", module, raising=False)
    monkeypatch.setattr(device_utils, "_ACCELERATOR_BACKENDS", ["fakedev"])
    assert "fakedev" not in device_utils.get_rng_states()
    device_utils.empty_cache()
    assert module.cache_cleared == 0


def test_missing_backend_module_is_ignored(monkeypatch):
    monkeypatch.setattr(device_utils, "_ACCELERATOR_BACKENDS", ["no_such_backend_xyz"])
    assert list(device_utils.get_rng_states()) == ["cpu"]


def test_failing_backend_does_not_break_cpu_state(monkeypatch):
    class Broken(FakeDeviceModule):
        def get_rng_state(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(torch, "fakedev", Broken(), raising=False)
    monkeypatch.setattr(device_utils, "_ACCELERATOR_BACKENDS", ["fakedev"])
    states = device_utils.get_rng_states()
    assert list(states) == ["cpu"]
    device_utils.set_rng_states(states)  # 例外にならない


def test_mps_availability_uses_torch_backends(monkeypatch):
    """torch.mps には is_available() が無いので、torch.backends.mps で判定される。"""
    stub = types.SimpleNamespace(get_rng_state=lambda: torch.tensor([7], dtype=torch.uint8), set_rng_state=lambda s: None)
    monkeypatch.setattr(torch, "mps", stub, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(device_utils, "_ACCELERATOR_BACKENDS", ["mps"])
    assert "mps" in device_utils.get_rng_states()
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert "mps" not in device_utils.get_rng_states()


# ---- 外部バックエンド（SD_SCRIPTS_BACKEND / init_external_backend / override_accelerator_device） ----


@pytest.fixture
def clean_external_backend(monkeypatch):
    """外部バックエンドの状態をテストごとに元へ戻す。"""
    monkeypatch.setattr(device_utils, "_external_backend", None)
    monkeypatch.setattr(device_utils, "_ACCELERATOR_BACKENDS", list(device_utils._ACCELERATOR_BACKENDS))
    monkeypatch.setattr(device_utils, "_EXTERNAL_BACKEND_MODULES", dict(device_utils._EXTERNAL_BACKEND_MODULES))
    device_utils.get_preferred_device.cache_clear()
    yield
    device_utils.get_preferred_device.cache_clear()


def test_init_external_backend_is_noop_without_request(clean_external_backend, monkeypatch):
    monkeypatch.delenv("SD_SCRIPTS_BACKEND", raising=False)
    assert device_utils.init_external_backend() is None
    assert device_utils._external_backend is None


def test_init_external_backend_registers_and_becomes_preferred(clean_external_backend, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "fake_backend_module", types.ModuleType("fake_backend_module"))
    device_utils._EXTERNAL_BACKEND_MODULES["meta"] = "fake_backend_module"
    monkeypatch.setenv("SD_SCRIPTS_BACKEND", "meta")
    assert device_utils.init_external_backend() == "meta"
    assert "meta" in device_utils._ACCELERATOR_BACKENDS
    assert device_utils.get_preferred_device() == torch.device("meta")


def test_init_external_backend_raises_when_requested_but_missing(clean_external_backend):
    device_utils._EXTERNAL_BACKEND_MODULES["fakedev"] = "no_such_module_xyz"
    with pytest.raises(RuntimeError, match="no_such_module_xyz"):
        device_utils.init_external_backend("fakedev")
    assert device_utils._external_backend is None  # CPU に黙って落とさない


def test_override_accelerator_device_is_noop_without_backend(clean_external_backend):
    from accelerate.state import PartialState

    before = PartialState().device
    device_utils.override_accelerator_device()
    assert PartialState().device == before


def test_override_accelerator_device_sets_state_device(clean_external_backend, monkeypatch):
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        acc = Accelerator(cpu=True)
        monkeypatch.setattr(device_utils, "_external_backend", "meta")  # 実在する非 CPU のデバイス型で代用
        device_utils.override_accelerator_device(acc)
        assert acc.device == torch.device("meta")
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


def test_get_ddp_backend_is_gloo_without_cuda(monkeypatch):
    monkeypatch.setattr(device_utils, "HAS_CUDA", False)
    assert device_utils.get_ddp_backend() == "gloo"


def test_get_accelerator_count_sums_registered_backends(monkeypatch):
    import types

    base = device_utils.get_accelerator_count()
    fake = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    monkeypatch.setattr(torch, "fakedev", fake, raising=False)
    monkeypatch.setattr(device_utils, "_ACCELERATOR_BACKENDS", [*device_utils._ACCELERATOR_BACKENDS, "fakedev"])
    assert device_utils.get_accelerator_count() == base + 2
    # device_count を持たないバックエンドは 1 台として数える
    monkeypatch.setattr(torch, "fakedev", types.SimpleNamespace(is_available=lambda: True), raising=False)
    assert device_utils.get_accelerator_count() == base + 1


@pytest.mark.parametrize("value", ["cpu", "CPU", "none", " cpu "])
def test_init_external_backend_treats_cpu_and_none_as_no_backend(clean_external_backend, monkeypatch, value):
    monkeypatch.setenv("SD_SCRIPTS_BACKEND", value)
    assert device_utils.init_external_backend() is None
    assert device_utils._external_backend is None


# ---- cpu_has_native_bf16 / prefer_fp32_for_host_compute ----
def _fake_cpu(monkeypatch, machine="x86_64", avx512_bf16=False, amx=False):
    import platform

    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(torch.cpu, "_is_avx512_bf16_supported", lambda: avx512_bf16, raising=False)
    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: amx, raising=False)


def test_cpu_has_native_bf16_follows_avx512_bf16_or_amx(monkeypatch):
    _fake_cpu(monkeypatch, avx512_bf16=False, amx=False)
    assert device_utils.cpu_has_native_bf16() is False  # AVX2 止まり
    _fake_cpu(monkeypatch, avx512_bf16=True)
    assert device_utils.cpu_has_native_bf16() is True
    _fake_cpu(monkeypatch, amx=True)
    assert device_utils.cpu_has_native_bf16() is True


def test_cpu_has_native_bf16_is_true_for_non_x86_and_when_unknown(monkeypatch):
    _fake_cpu(monkeypatch, machine="arm64")
    assert device_utils.cpu_has_native_bf16() is True  # 判定しない（従来の挙動）
    import platform

    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.delattr(torch.cpu, "_is_avx512_bf16_supported", raising=False)
    assert device_utils.cpu_has_native_bf16() is True  # API が無ければ従来どおり


def test_cpu_has_native_bf16_without_amx_api(monkeypatch):
    # torch 2.8 の実機 (AVX2 のみの CPU) の状況: AMX の判定関数が無くても AVX512_BF16 だけで判定でき、False になること
    _fake_cpu(monkeypatch, avx512_bf16=False)
    monkeypatch.delattr(torch.cpu, "_is_amx_tile_supported", raising=False)
    assert device_utils.cpu_has_native_bf16() is False
    _fake_cpu(monkeypatch, avx512_bf16=True)
    monkeypatch.delattr(torch.cpu, "_is_amx_tile_supported", raising=False)
    assert device_utils.cpu_has_native_bf16() is True


def test_prefer_fp32_for_host_compute(monkeypatch, clean_external_backend):
    _fake_cpu(monkeypatch, avx512_bf16=False)
    assert device_utils.prefer_fp32_for_host_compute("cpu") is True
    assert device_utils.prefer_fp32_for_host_compute(torch.device("cuda")) is False  # CUDA は対象外
    assert device_utils.prefer_fp32_for_host_compute(torch.device("mps")) is False
    monkeypatch.setattr(device_utils, "_external_backend", "meta")  # 外部バックエンド（vk）の代用
    assert device_utils.prefer_fp32_for_host_compute(torch.device("meta")) is True
    _fake_cpu(monkeypatch, avx512_bf16=True)
    assert device_utils.prefer_fp32_for_host_compute("cpu") is False  # bf16 が速い CPU では変えない
