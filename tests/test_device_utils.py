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
