"""Training utilities for machine learning models in discrete1.

This module exposes two primary entry points:

- djinn_regression: Trains DJINN regression ensembles over combinations of
  tree counts and depths, saving model checkpoints and evaluation metrics.
- RegressionNeuralNetwork: A lightweight training harness for PyTorch networks
  with train/validation/test splits, progress display, checkpointing, and
  batched inference.

Notes
-----
This module relies on optional ML dependencies (``torch``, ``sklearn`` and
``djinn``). Install them with::

    pip install discrete1[ml]
"""

import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn import model_selection
from torch.utils.data import DataLoader, TensorDataset

from discrete1.ml.tools import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    from djinn import djinn  # noqa: E402
except ImportError as e:
    raise ImportError(
        "ML dependencies are not installed. Install with:\n"
        "   pip install discrete1[ml]"
    ) from e


def djinn_regression(X, y, path, trees, depth, **kwargs):
    """Train DJINN models across tree count and depth grids.

    For each combination of number of trees and maximum depth, this function
    tunes DJINN hyperparameters, trains a model, evaluates it on a held-out
    test set, and saves both the model checkpoint and error metrics.

    Parameters
    ----------
    X : numpy.ndarray
        Input features of shape (n_samples, n_features).
    y : numpy.ndarray
        Target values of shape (n_samples,) or (n_samples, n_targets).
    path : str
        Directory path to save models and metrics. Filenames are derived from
        the grid configuration.
    trees : sequence of int
        Candidate numbers of trees (i.e., neural nets in the ensemble).
    depth : sequence of int
        Candidate maximum tree depths.
    test_size : float, optional
        Fraction of data reserved for testing. Default is 0.2.
    seed : int, optional
        Random seed for data splitting. Default is 3.
    dropout_keep : float, optional
        Dropout keep probability. Typical value is 1.0 for non-Bayesian
        models. Default is 1.0.
    LOUD : int, optional
        Verbosity level, where 0 is silent. Default is 0.

    Notes
    -----
    For each tree count and depth combination, this routine:
    - Optimizes batch size, learning rate, and epochs.
    - Trains the model and saves a checkpoint.
    - Computes MAE, MSE, EVS, and R2 metrics and saves them to disk.

    Returns
    -------
    None
        Results are saved to disk.
    """
    # number of trees = number of neural nets in ensemble
    # max depth of tree = optimize this for each data set
    # dropout typically set to 1 for non-Bayesian models

    test_size = kwargs.get("test_size", 0.2)
    seed = kwargs.get("seed", 3)
    dropout_keep = kwargs.get("dropout_keep", 1.0)
    LOUD = kwargs.get("LOUD", 0)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, seed=seed
    )

    for ntrees, maxdepth in itertools.product(trees, depth):
        if LOUD:
            print(
                "\nNumber of Trees: {}\tMax Depth: {}\n{}".format(
                    ntrees, maxdepth, "=" * 40
                )
            )

        fntrees = str(ntrees).zfill(3)
        fmaxdepth = str(maxdepth).zfill(3)
        modelname = f"model_{fntrees}{fmaxdepth}"

        # initialize the model
        model = djinn.DJINN_Regressor(ntrees, maxdepth, dropout_keep)

        # find optimal settings
        optimal = model.get_hyperparameters(X_train, y_train)
        batchsize = optimal["batch_size"]
        learnrate = optimal["learn_rate"]
        epochs = np.min((300, optimal["epochs"]))
        # epochs = optimal['epochs']

        # train the model with these settings
        model.train(
            X_train,
            y_train,
            epochs=epochs,
            learn_rate=learnrate,
            batch_size=batchsize,
            display_step=0,
            save_files=True,
            model_path=path,
            file_name=modelname,
            save_model=True,
            model_name=modelname,
        )

        # Estimate
        y_estimate = model.predict(X_test)

        # evaluate results
        error_dict = {
            "MAE": mean_absolute_error(y_test, y_estimate),
            "MSE": mean_squared_error(y_test, y_estimate),
            "EVS": explained_variance_score(y_test, y_estimate),
            "R2": r2_score(y_test, y_estimate),
        }
        np.savez(path + "error_" + modelname, **error_dict)

        # close model
        model.close_model()


class RegressionNeuralNetwork:
    """Simple training harness for regression-style PyTorch networks.

    This class wraps common training utilities for regression-style neural
    networks, including dataset preparation, training with validation, testing,
    checkpointing, and batched inference.

    Notes
    -----
    Only public methods are intended for external use. Helper methods prefixed
    with an underscore are internal implementation details.
    """

    def __init__(self, network, X, y, **kwargs):
        """Initialize the trainer with a network and data.

        Parameters
        ----------
        network : torch.nn.Module
            The PyTorch model to train. It should accept input tensors matching
            the shape of ``X`` and output tensors compatible with ``y``.
        X : numpy.ndarray
            Input features of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples, n_targets).
        **kwargs
            Additional keyword arguments.

        Other Parameters
        ----------------
        n_epochs : int, default 100
            Number of training epochs.
        batch_size : int, default 32
            Mini-batch size for training and evaluation.
        learning_rate : float, default 1e-3
            Optimizer learning rate.
        weight_decay : float, default 0.0
            L2 weight decay (regularization) for the optimizer.
        loss_function : str or torch.nn.modules.loss._Loss, default nn.MSELoss()
            Either an instantiated loss object or the name of a loss class
            in ``torch.nn`` (e.g., "MSELoss").
        optimizer : str, default "Adam"
            Name of an optimizer in ``torch.optim`` to use.
        device : str, default "cpu"
            Device specifier for computation (e.g., "cpu", "cuda").
        LOUD : bool, default True
            If True, shows a progress bar during training.
        """
        self.network = network
        self.X = X
        self.y = y
        self.n_epochs = kwargs.get("n_epochs", 100)
        self.batch_size = kwargs.get("batch_size", 32)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.loss_function = kwargs.get("loss_function", nn.MSELoss())
        self.optimizer_name = kwargs.get("optimizer", "Adam")
        self.device = torch.device(kwargs.get("device", "cpu"))
        self.LOUD = kwargs.get("LOUD", True)

    def process_data(self, verbose=False, seed=3):
        """Split arrays into train/val/test sets and build data loaders.

        Parameters
        ----------
        verbose : bool, default False
            If True, also return the created ``TensorDataset`` objects.
        seed : int, default 3
            Random seed for reproducible splits.

        Returns
        -------
        tuple of torch.utils.data.TensorDataset or None
            If ``verbose`` is True, returns
            ``(train_dataset, val_dataset, test_dataset)``. Otherwise, returns
            None.

        Notes
        -----
        DataLoader instances are stored on the instance as
        ``train_loader``, ``val_loader``, and ``test_loader``.
        """
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            self.X, self.y, test_size=0.2, seed=seed
        )

        X_train, X_val, y_train, y_val = model_selection.train_test_split(
            X_train, y_train, test_size=0.25, seed=seed
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        if verbose:
            return train_dataset, val_dataset, test_dataset

    def _training_optimizer_loss_fn(self):
        """Construct optimizer and loss criterion for training.

        Returns
        -------
        tuple
            A pair ``(optimizer, loss_criterion)`` where ``optimizer`` is a
            ``torch.optim.Optimizer`` instance configured from the class
            attributes, and ``loss_criterion`` is a ``torch.nn.modules.loss._Loss``
            instance resolved from ``loss_function``.
        """
        optimizer = getattr(optim, self.optimizer_name)(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if isinstance(self.loss_function, str):
            loss_criterion = getattr(nn, self.loss_function)()
        else:
            loss_criterion = self.loss_function
        return optimizer, loss_criterion

    def _batch_to_device(self, X_batch, y_batch):
        """Move a mini-batch to the configured computation device.

        Parameters
        ----------
        X_batch : torch.Tensor
            Input batch tensor.
        y_batch : torch.Tensor
            Target batch tensor.

        Returns
        -------
        tuple of torch.Tensor
            The pair ``(X_batch_device, y_batch_device)`` located on
            ``self.device``.
        """
        return X_batch.to(self.device), y_batch.to(self.device)

    def _update_progress_bar(self, epoch):
        """Create a progress bar over the training data loader.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index used in the progress bar description.

        Returns
        -------
        tqdm.tqdm
            Progress bar wrapped around ``enumerate(self.train_loader)``.
        """
        return tqdm.tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch [{epoch+1}/{self.n_epochs}]",
            unit="batch",
            mininterval=0,
            disable=not self.LOUD,
        )

    def _train_batch(self, progress_bar, optimizer, loss_fn):
        """Execute one training pass over the training loader.

        Parameters
        ----------
        progress_bar : Iterable
            Iterable yielding ``(batch_index, (X_batch, y_batch))``.
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters.
        loss_fn : callable
            Loss function mapping ``(outputs, targets)`` to a scalar loss.

        Returns
        -------
        float
            Sum of per-sample losses across the epoch (not averaged).
        """
        train_loss = 0.0

        for _, (X_batch, y_batch) in progress_bar:
            X_batch, y_batch = self._batch_to_device(X_batch, y_batch)

            optimizer.zero_grad()
            outputs = self.network(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            progress_bar.set_postfix({"loss": train_loss})

        return train_loss

    def _val_batch(self, data_loader, loss_fn):
        """Evaluate the model on a data loader without gradient updates.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Data loader to iterate over for evaluation.
        loss_fn : callable
            Loss function mapping ``(outputs, targets)`` to a scalar loss.

        Returns
        -------
        float
            Sum of per-sample losses across the loader (not averaged).
        """
        test_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = self._batch_to_device(X_batch, y_batch)
                outputs = self.network(X_batch)
                loss = loss_fn(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)

        return test_loss

    def train(self, verbose=True):
        """Train the network with validation and test evaluation.

        Performs epoch-wise training using the configured optimizer and loss
        function, tracks training/validation loss, selects the best model
        weights by validation loss, evaluates on the test set, and restores
        the best weights.

        Parameters
        ----------
        verbose : bool, default True
            If True, returns a dictionary of tracked metrics.

        Returns
        -------
        dict or None
            If ``verbose`` is True, returns a dictionary with:
            - ``train_loss`` (list of float): Average training loss per epoch.
            - ``val_loss`` (list of float): Average validation loss per epoch.
            - ``test_loss`` (float): Average test loss computed with the best
              model.
            Otherwise, returns None.

        Notes
        -----
        Metrics are also stored on the instance as ``metric_data``.
        """
        optimizer, loss_criterion = self._training_optimizer_loss_fn()

        best_error = np.inf
        best_model_weights = None
        train_loss_history = []
        val_loss_history = []

        for epoch in range(self.n_epochs):
            progress_bar = self._update_progress_bar(epoch)

            # Training
            self.network.train()
            train_loss = self._train_batch(progress_bar, optimizer, loss_criterion)
            train_loss_history.append(train_loss / len(self.train_loader.dataset))

            # Validation
            self.network.eval()
            val_loss = self._val_batch(self.val_loader, loss_criterion)
            val_loss_history.append(val_loss / len(self.val_loader.dataset))
            if val_loss < best_error:
                best_error = val_loss
                best_model_weights = copy.deepcopy(self.network.state_dict())

        # Testing
        with torch.no_grad():
            self.network.eval()
            test_loss = self._val_batch(self.test_loader, loss_criterion)

        # Return best model
        self.network.load_state_dict(best_model_weights)
        self.metric_data = {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "test_loss": test_loss / len(self.test_loader.dataset),
        }
        if verbose:
            return self.metric_data

    def save_model(self, model_name):
        """Save the trained model parameters and loss metrics to disk.

        Parameters
        ----------
        model_name : str
            Base filename (without extension). Saves ``f"{model_name}.pt"`` for
            model weights and ``f"{model_name}_loss_metrics.npz"`` for metrics
            captured in ``metric_data``.
        """
        torch.save(self.network.state_dict(), f"{model_name}.pt")
        np.savez(f"{model_name}_loss_metrics.npz", **self.metric_data)

    def load_model(self, model_name):
        """Load model parameters from disk into the network.

        Parameters
        ----------
        model_name : str
            Base filename used when saving (without extension). Reads
            ``f"{model_name}.pt"``.
        """
        self.network.load_state_dict(torch.load(f"{model_name}.pt"))

    def predict(self, X_new, batch_size=1):
        """Run batched inference and return predictions.

        Parameters
        ----------
        X_new : numpy.ndarray
            Input features of shape (n_samples, n_features).
        batch_size : int, default 1
            Batch size for inference.

        Returns
        -------
        numpy.ndarray
            Stacked predictions of shape (n_samples, n_targets).
        """
        test_dataset = TensorDataset(torch.tensor(X_new, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            self.network.eval()
            for X_batch in test_loader:
                outputs = self.network(X_batch[0].to(self.device))
                predictions.append(outputs.cpu().numpy())

        return np.vstack(predictions)
