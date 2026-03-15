"""Tests for machine-learning memmap data helpers."""

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


@pytest.mark.machine_learning
def test_deeponet_metadata_roundtrip(tmp_path):
    n_samples = 12
    flux = np.random.rand(n_samples, 4).astype(np.float32)
    labels = np.random.rand(n_samples, 2).astype(np.float32)
    y = np.random.rand(n_samples, 3).astype(np.float32)

    filename = str(tmp_path / "roundtrip_data.memmap")
    ml_data.deeponet_memmap(filename, flux, labels, y, seed=3)

    n_loaded, flux_shape, labels_shape, y_shape, dtype = (
        ml_data.load_deeponet_memmap_metadata(filename)
    )

    assert n_loaded == n_samples
    assert flux_shape == 4
    assert labels_shape == 2
    assert y_shape == 3
    assert dtype.names == ("flux", "labels", "y")


@pytest.mark.machine_learning
def test_deeponet_dataset_len_and_tensor_indexing(tmp_path):
    n_samples = 10
    flux = np.random.rand(n_samples, 3).astype(np.float32)
    labels = np.random.rand(n_samples, 2).astype(np.float32)
    y = np.random.rand(n_samples, 1).astype(np.float32)

    filename = str(tmp_path / "dataset_data.memmap")
    ml_data.deeponet_memmap(filename, flux, labels, y, seed=3)
    n_loaded, _, _, _, dtype = ml_data.load_deeponet_memmap_metadata(filename)

    dataset = ml_data.DeepONetDataset(
        filename,
        dtype,
        (n_loaded,),
        batch_size=3,
        start_idx=0,
        end_idx=10,
    )

    # floor division behavior: 10 samples with batch size 3 -> 3 batches
    assert len(dataset) == 3

    flux_batch, label_batch, y_batch = dataset[torch.tensor(1)]
    assert flux_batch.shape == (3, 3)
    assert label_batch.shape == (3, 2)
    assert y_batch.shape == (3, 1)


@pytest.mark.machine_learning
def test_batch_shuffle_sampler_yields_full_permutation():
    class _DummyDataset:
        def __len__(self):
            return 7

    sampler = ml_data.BatchShuffleSampler(_DummyDataset())
    indices = list(iter(sampler))

    assert len(indices) == 7
    assert sorted(indices) == list(range(7))
