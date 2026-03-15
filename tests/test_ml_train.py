"""Tests for machine-learning training helpers and memmap data loading."""

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "ML dependencies are not installed. Install with:\n"
        "   pip install discrete1[ml]"
    ) from e

from discrete1.ml import data as ml_data
from discrete1.ml import train


class _DummyModel(nn.Module):
    """Small model with DeepONet-style forward signature for tests."""

    def __init__(self, flux_dim, labels_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(flux_dim + labels_dim, out_dim)

    def forward(self, flux, labels):
        return self.fc(torch.cat((flux, labels), dim=-1))


@pytest.mark.machine_learning
def test_process_data_chunked_dataloaders(tmp_path):
    n_samples = 20
    flux_dim = 4
    label_dim = 3
    out_dim = 2
    batch_size = 4

    flux = np.random.rand(n_samples, flux_dim).astype(np.float32)
    labels = np.random.rand(n_samples, label_dim).astype(np.float32)
    y = np.random.rand(n_samples, out_dim).astype(np.float32)

    filename = str(tmp_path / "toy_data.memmap")
    ml_data.deeponet_memmap(filename, flux, labels, y, seed=3)

    model = _DummyModel(flux_dim, label_dim, out_dim)
    trainer = train.RegressionDeepONet(
        model,
        flux,
        labels,
        y,
        batch_size=batch_size,
        device="cpu",
        LOUD=False,
    )

    train_dataset, val_dataset, test_dataset = trainer.process_data_memmap(
        filename,
        batch_size=batch_size,
        train_size=0.6,
        val_size=0.2,
        verbose=True,
    )

    assert isinstance(train_dataset, ml_data.DeepONetDataset)
    assert isinstance(val_dataset, ml_data.DeepONetDataset)
    assert isinstance(test_dataset, ml_data.DeepONetDataset)

    train_batch = next(iter(trainer.train_loader))
    val_batch = next(iter(trainer.val_loader))
    test_batch = next(iter(trainer.test_loader))

    # batch_size=1 in DataLoader wraps prebatched dataset entries.
    assert train_batch[0].shape == (1, batch_size, flux_dim)
    assert train_batch[1].shape == (1, batch_size, label_dim)
    assert train_batch[2].shape == (1, batch_size, out_dim)

    assert val_batch[0].shape == (1, batch_size, flux_dim)
    assert test_batch[0].shape == (1, batch_size, flux_dim)


@pytest.mark.machine_learning
def test_process_data_verbose_and_predict_shape():
    n_samples = 20
    flux_dim = 4
    label_dim = 2
    out_dim = 3

    flux = np.random.rand(n_samples, flux_dim).astype(np.float32)
    labels = np.random.rand(n_samples, label_dim).astype(np.float32)
    y = np.random.rand(n_samples, out_dim).astype(np.float32)

    model = _DummyModel(flux_dim, label_dim, out_dim)
    trainer = train.RegressionDeepONet(
        model,
        flux,
        labels,
        y,
        batch_size=5,
        device="cpu",
        LOUD=False,
    )

    train_dataset, val_dataset, test_dataset = trainer.process_data(verbose=True)
    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == n_samples

    pred = trainer.predict(flux[:6], labels[:6], batch_size=2)
    assert pred.shape == (6, out_dim)


@pytest.mark.machine_learning
def test_init_loss_fn_string_and_instance():
    flux = np.random.rand(8, 3).astype(np.float32)
    labels = np.random.rand(8, 2).astype(np.float32)
    y = np.random.rand(8, 1).astype(np.float32)
    model = _DummyModel(3, 2, 1)

    trainer_string = train.RegressionDeepONet(
        model,
        flux,
        labels,
        y,
        loss_fn="L1Loss",
        device="cpu",
        LOUD=False,
    )
    assert isinstance(trainer_string._init_loss_fn(), nn.L1Loss)

    custom_loss = nn.SmoothL1Loss()
    trainer_object = train.RegressionDeepONet(
        model,
        flux,
        labels,
        y,
        loss_fn=custom_loss,
        device="cpu",
        LOUD=False,
    )
    assert trainer_object._init_loss_fn() is custom_loss


@pytest.mark.machine_learning
def test_val_batch_scheduler_step_modes():
    class _SchedulerSpy:
        def __init__(self):
            self.calls = []

        def step(self, *args):
            self.calls.append(args)

    n_samples = 16
    flux = np.random.rand(n_samples, 3).astype(np.float32)
    labels = np.random.rand(n_samples, 2).astype(np.float32)
    y = np.random.rand(n_samples, 1).astype(np.float32)
    model = _DummyModel(3, 2, 1)

    trainer = train.RegressionDeepONet(
        model,
        flux,
        labels,
        y,
        batch_size=4,
        device="cpu",
        LOUD=False,
    )
    trainer.process_data()

    loss_fn = nn.MSELoss()

    trainer.lr_scheduler = "ReduceLROnPlateau"
    scheduler = _SchedulerSpy()
    val_loss = trainer._val_batch(trainer.val_loader, scheduler, loss_fn)
    assert isinstance(val_loss, float)
    assert len(scheduler.calls) == 1
    assert len(scheduler.calls[0]) == 1

    trainer.lr_scheduler = "StepLR"
    scheduler = _SchedulerSpy()
    trainer._val_batch(trainer.val_loader, scheduler, loss_fn)
    assert len(scheduler.calls) == 1
    assert scheduler.calls[0] == ()
