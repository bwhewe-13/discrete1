"""Tests for machine-learning hyperparameter tuning helpers."""

import numpy as np
import pytest

try:
    import torch
except ImportError as e:
    raise ImportError(
        "ML dependencies are not installed. Install with:\n"
        "   pip install discrete1[ml]"
    ) from e

from discrete1.ml import data as ml_data
from discrete1.ml import tune


class _DummyTrial:
    """Minimal Optuna-like trial object for deterministic unit tests."""

    def __init__(self, values):
        self.values = values
        self.reports = []

    def suggest_categorical(self, name, _choices):
        return self.values[name]

    def suggest_int(self, name, _low, _high):
        return self.values[name]

    def suggest_float(self, name, _low, _high):
        return self.values[name]

    def report(self, metric, epoch):
        self.reports.append((metric, epoch))

    def should_prune(self):
        return False


@pytest.mark.machine_learning
def test_tune_requires_arrays_without_memmap_file():
    with pytest.raises(ValueError) as exc:
        tune.RegressionDeepONet(None, None, None)

    assert "memmap_file" in str(exc.value)


@pytest.mark.machine_learning
def test_tune_infers_input_output_sizes_from_mapped_data(tmp_path):
    n_samples = 20
    flux_dim = 5
    label_dim = 2
    out_dim = 3

    flux = np.random.rand(n_samples, flux_dim).astype(np.float32)
    labels = np.random.rand(n_samples, label_dim).astype(np.float32)
    y = np.random.rand(n_samples, out_dim).astype(np.float32)

    filename = str(tmp_path / "tune_data.memmap")
    ml_data.deeponet_memmap(filename, flux, labels, y, seed=3)

    tuner = tune.RegressionDeepONet(None, None, None, memmap_file=filename)

    assert tuner.flux_input_size == flux_dim
    assert tuner.labels_input_size == label_dim
    assert tuner.output_size == out_dim
    assert tuner._memmap_shape == (n_samples,)


@pytest.mark.machine_learning
def test_tune_clips_effective_batch_size_to_smallest_split(tmp_path):
    n_samples = 20
    flux = np.random.rand(n_samples, 4).astype(np.float32)
    labels = np.random.rand(n_samples, 3).astype(np.float32)
    y = np.random.rand(n_samples, 2).astype(np.float32)

    filename = str(tmp_path / "batch_clip_data.memmap")
    ml_data.deeponet_memmap(filename, flux, labels, y, seed=3)

    tuner = tune.RegressionDeepONet(
        None,
        None,
        None,
        memmap_file=filename,
        train_size=0.6,
        val_size=0.2,
    )

    train_dataset, val_dataset = tuner._create_memmap_datasets(batch_size=10)

    # With n_samples=20 and default 60/20 split, val has 4 samples.
    assert train_dataset.batch_size == 4
    assert val_dataset.batch_size == 4
    assert len(train_dataset) == 3
    assert len(val_dataset) == 1

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        sampler=tune.BatchShuffleSampler(train_dataset),
    )
    batch = next(iter(train_loader))
    assert batch[0].shape == (1, 4, 4)


@pytest.mark.machine_learning
def test_tune_create_datasets_raises_on_empty_split(tmp_path):
    n_samples = 20
    flux = np.random.rand(n_samples, 4).astype(np.float32)
    labels = np.random.rand(n_samples, 3).astype(np.float32)
    y = np.random.rand(n_samples, 2).astype(np.float32)

    filename = str(tmp_path / "empty_split_data.memmap")
    ml_data.deeponet_memmap(filename, flux, labels, y, seed=3)

    tuner = tune.RegressionDeepONet(
        None,
        None,
        None,
        memmap_file=filename,
        train_size=0.0,
        val_size=0.2,
    )

    with pytest.raises(ValueError):
        tuner._create_memmap_datasets(batch_size=8)


@pytest.mark.machine_learning
def test_tune_init_scheduler_returns_none_for_disabled_schedule():
    flux = np.random.rand(16, 4).astype(np.float32)
    labels = np.random.rand(16, 2).astype(np.float32)
    y = np.random.rand(16, 3).astype(np.float32)
    tuner = tune.RegressionDeepONet(flux, labels, y)

    model = tune.DeepONet(
        torch.nn.Linear(4, 8),
        torch.nn.Linear(2, 8),
        latent_features=8,
        out_features=3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trial = _DummyTrial(
        {
            "sched_mode": None,
            "sched_factor": None,
            "sched_patience": None,
            "sched_cooldown": None,
        }
    )

    scheduler = tuner._init_scheduler(trial, optimizer)
    assert scheduler is None


@pytest.mark.machine_learning
def test_tune_run_uses_optuna_study_and_writes_trials_csv(monkeypatch, tmp_path):
    class _FakeTrialsFrame:
        def to_csv(self, path, index=False):
            with open(path, "w", encoding="utf-8") as f:
                f.write("number,value\n0,0.123\n")
            assert index is False

    class _FakeStudy:
        def __init__(self):
            self.best_params = {"batch_size": 32}
            self.optimized = False

        def optimize(self, objective, n_trials, n_jobs):
            self.optimized = True
            assert callable(objective)
            assert n_trials == 2
            assert n_jobs == 1

        def trials_dataframe(self):
            return _FakeTrialsFrame()

    create_kwargs = {}
    finish_called = {"value": False}

    def _fake_create_study(**kwargs):
        create_kwargs.update(kwargs)
        return _FakeStudy()

    def _fake_finish_trial(_study):
        finish_called["value"] = True

    monkeypatch.setattr(tune.optuna, "create_study", _fake_create_study)
    monkeypatch.setattr(tune, "finish_trial", _fake_finish_trial)
    monkeypatch.chdir(tmp_path)

    flux = np.random.rand(12, 3).astype(np.float32)
    labels = np.random.rand(12, 2).astype(np.float32)
    y = np.random.rand(12, 1).astype(np.float32)
    tuner = tune.RegressionDeepONet(flux, labels, y)

    best = tuner.run(n_trials=2, n_jobs=1, output="tune_unit")

    assert best == {"batch_size": 32}
    assert finish_called["value"] is True
    assert create_kwargs["direction"] == "minimize"
    assert create_kwargs["study_name"] is None
    assert create_kwargs["storage"] is None
    assert (tmp_path / "tune_unit_trials.csv").exists()
