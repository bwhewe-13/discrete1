"""Data helpers for DeepONet training and tuning.

This module centralizes utilities for writing and loading memmapped DeepONet
regression datasets so both training and hyperparameter tuning can share a
single implementation.
"""

import pickle

import numpy as np
import torch
from sklearn import model_selection
from torch.utils.data import Dataset


def deeponet_memmap(filename, flux, labels, y, seed=3):
    """Save flux, labels, and y data for DeepONets into a memmap file.

    Parameters
    ----------
    filename : str
        Name of the memmap file to save to.
    flux : numpy.ndarray
        Flux data of size (n_samples, n_groups).
    labels : numpy.ndarray
        Label data of size (n_samples, n_labels).
    y : numpy.ndarray
        Target data of size (n_samples, n_groups).
    seed : int, default 3
        Random seed for reproducible splits.

    Notes
    -----
    A companion pickle file is written with shape metadata needed to rebuild
    dataset dtypes later.
    """
    samples, flux_shape = flux.shape
    labels_shape = labels.shape[1]
    y_shape = y.shape[1]

    dtype = np.dtype(
        [
            ("flux", np.float32, (flux_shape,)),
            ("labels", np.float32, (labels_shape,)),
            ("y", np.float32, (y_shape,)),
        ]
    )

    # Shuffle while preserving train/val/test ordering used by memmap loaders.
    idx = np.arange(samples, dtype=int)
    train_idx, test_idx = model_selection.train_test_split(
        idx, test_size=0.2, random_state=seed
    )
    train_idx, val_idx = model_selection.train_test_split(
        train_idx, test_size=0.25, random_state=seed
    )
    idx = np.concatenate((train_idx, val_idx, test_idx))

    data = np.memmap(filename, dtype=dtype, mode="w+", shape=(samples,))
    data["flux"][:] = flux[idx].astype(np.float32)
    data["labels"][:] = labels[idx].astype(np.float32)
    data["y"][:] = y[idx].astype(np.float32)

    data.flush()
    print(f"File saved as: {filename}")

    pickle_file = filename.replace("memmap", "pickle")
    with open(pickle_file, "wb") as file:
        pickle.dump((samples, flux_shape, labels_shape, y_shape), file)


def load_deeponet_memmap_metadata(filename):
    """Load shape and dtype metadata for a DeepONet memmap file.

    Parameters
    ----------
    filename : str
        Name/path of the memmap file.

    Returns
    -------
    tuple
        ``(n_samples, flux_shape, labels_shape, y_shape, dtype)``.
    """
    pickle_file = filename.replace("memmap", "pickle")
    with open(pickle_file, "rb") as file:
        n_samples, flux_shape, labels_shape, y_shape = pickle.load(file)

    dtype = np.dtype(
        [
            ("flux", np.float32, (flux_shape,)),
            ("labels", np.float32, (labels_shape,)),
            ("y", np.float32, (y_shape,)),
        ]
    )
    return n_samples, flux_shape, labels_shape, y_shape, dtype


class DeepONetDataset(Dataset):
    """Custom PyTorch Dataset for Deep Operator Networks using numpy.memmap."""

    def __init__(self, path, dtype, shape, batch_size=1, start_idx=0, end_idx=None):
        """Initialize a chunked dataset view over a DeepONet memmap file.

        Parameters
        ----------
        path : str
            Path to the memmap file created by :func:`deeponet_memmap`.
        dtype : numpy.dtype
            Structured dtype containing ``flux``, ``labels``, and ``y`` fields.
        shape : tuple of int
            Shape passed to ``numpy.memmap`` (typically ``(n_samples,)``).
        batch_size : int, default 1
            Number of samples grouped into each returned dataset item.
        start_idx : int, default 0
            Inclusive start sample index for this dataset partition.
        end_idx : int or None, default None
            Exclusive end sample index for this dataset partition. If ``None``,
            uses ``shape[0]``.

        Notes
        -----
        The dataset length is computed with floor division, so any trailing
        samples that do not fill a complete batch are not yielded.
        """
        super().__init__()
        self.path = path
        self.dtype = dtype
        self.shape = shape
        self.batch_size = batch_size
        self.start_idx = start_idx
        self.end_idx = end_idx or shape[0]

        self._data = None

        self.num_samples = self.end_idx - self.start_idx
        self.num_batches = self.num_samples // batch_size

    def _ensure_open(self):
        """Ensure the memmap file is opened lazily per worker process."""
        if self._data is None:
            self._data = np.memmap(
                self.path, shape=self.shape, dtype=self.dtype, mode="r"
            )

    def __len__(self):
        """Return the number of chunked batches available in this partition.

        Returns
        -------
        int
            Number of dataset items (batches) in this partition.
        """
        return self.num_batches

    def __getitem__(self, batch_idx):
        """Fetch one pre-chunked batch from the memmap-backed dataset.

        Parameters
        ----------
        batch_idx : int or torch.Tensor
            Zero-based batch index within this dataset partition.

        Returns
        -------
        tuple of torch.Tensor
            ``(flux, labels, y)`` tensors with leading dimension ``batch_size``.
        """
        self._ensure_open()

        if torch.is_tensor(batch_idx):
            batch_idx = batch_idx.item()

        start = self.start_idx + batch_idx * self.batch_size
        end = start + self.batch_size
        batch = self._data[start:end]

        flux = np.array(batch["flux"], dtype=np.float32, copy=True)
        labels = np.array(batch["labels"], dtype=np.float32, copy=True)
        y = np.array(batch["y"], dtype=np.float32, copy=True)

        return torch.from_numpy(flux), torch.from_numpy(labels), torch.from_numpy(y)


class BatchShuffleSampler(torch.utils.data.Sampler):
    """Shuffles memmap batches for training."""

    def __init__(self, dataset):
        """Initialize a sampler that randomly permutes dataset batch indices.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset whose index range is sampled in random order.
        """
        self.dataset = dataset

    def __iter__(self):
        """Yield one random permutation of dataset indices.

        Returns
        -------
        iterator of int
            Iterator over shuffled dataset indices.
        """
        indices = torch.randperm(len(self.dataset)).tolist()
        return iter(indices)

    def __len__(self):
        """Return the number of indices emitted by this sampler.

        Returns
        -------
        int
            Number of elements in the associated dataset.
        """
        return len(self.dataset)
