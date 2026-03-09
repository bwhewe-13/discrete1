"""Tests for ML import fallback logic and AutoDJINN backend guards."""

import importlib

import pytest


@pytest.mark.machine_learning
def test_predict_import_djinn_module_uses_primary_path(monkeypatch):
    module = importlib.import_module("discrete1.ml.predict")

    calls = []

    def fake_import(name):
        calls.append(name)
        if name == "djinn.djinn":
            return object()
        raise ImportError("missing")

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    loaded = module._import_djinn_module()

    assert loaded is not None
    assert calls[0] == "djinn.djinn"


@pytest.mark.machine_learning
def test_predict_import_djinn_module_error_message(monkeypatch):
    module = importlib.import_module("discrete1.ml.predict")

    def fake_import(_name):
        raise ImportError("missing")

    monkeypatch.setattr(module.importlib, "import_module", fake_import)

    with pytest.raises(ImportError) as exc:
        module._import_djinn_module()

    assert "discrete1[ml]" in str(exc.value)
    assert "discrete1[tf-ml]" in str(exc.value)


@pytest.mark.machine_learning
def test_train_import_djinn_module_error_message(monkeypatch):
    module = importlib.import_module("discrete1.ml.train")

    def fake_import(_name):
        raise ImportError("missing")

    monkeypatch.setattr(module.importlib, "import_module", fake_import)

    with pytest.raises(ImportError) as exc:
        module._import_djinn_module()

    assert "discrete1[ml]" in str(exc.value)
    assert "discrete1[tf-ml]" in str(exc.value)


@pytest.mark.machine_learning
def test_autodjinn_rejects_invalid_backend():
    module = importlib.import_module("discrete1.ml.predict")

    with pytest.raises(ValueError):
        module.AutoDJINN(
            "enc",
            "dj",
            "dec",
            transformer=lambda x: x,
            detransformer=lambda x: x,
            backend="invalid",
        )
