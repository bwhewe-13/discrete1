"""Tests for ML import fallback logic."""

import builtins
import importlib
import sys

import pytest


@pytest.mark.machine_learning
def test_predict_module_exposes_djinn_namespace():
    module = importlib.import_module("discrete1.ml.predict")
    assert hasattr(module, "djinn")
    assert module.djinn is not None


def _block_djinn_import(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "djinn":
            raise ImportError("missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.mark.machine_learning
def test_predict_import_djinn_module_error_message(monkeypatch):
    _block_djinn_import(monkeypatch)
    sys.modules.pop("discrete1.ml.predict", None)

    with pytest.raises(ImportError) as exc:
        importlib.import_module("discrete1.ml.predict")

    assert "discrete1[ml]" in str(exc.value)


@pytest.mark.machine_learning
def test_train_import_djinn_module_error_message(monkeypatch):
    _block_djinn_import(monkeypatch)
    sys.modules.pop("discrete1.ml.train", None)

    with pytest.raises(ImportError) as exc:
        importlib.import_module("discrete1.ml.train")

    assert "discrete1[ml]" in str(exc.value)
