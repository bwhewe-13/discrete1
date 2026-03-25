"""Training utilities for machine learning models in discrete1.

This module exposes two primary entry points:

- djinn_regression: Trains DJINN regression ensembles over combinations of
  tree counts and depths, saving model checkpoints and evaluation metrics.
- RegressionDeepONet: A lightweight training harness for PyTorch models
  with train/validation/test splits, progress display, checkpointing, and
  batched inference.

Notes
-----
This module relies on optional ML dependencies (``torch``, ``sklearn`` and
``djinn``). Install them with:

    pip install discrete1[ml]

"""

import copy
import itertools
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn import model_selection
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

try:
    from djinn import djinn  # pyright: ignore[reportMissingImports]
except ImportError as e:
    raise ImportError(
        "DJINN dependencies are not installed. Install with one of:\n"
        "   pip install discrete1[ml]\n"
        "   pip install discrete1[tf-ml]"
    ) from e

from discrete1.ml.data import (
    BatchShuffleSampler,
    DeepONetDataset,
    load_deeponet_memmap_metadata,
)
from discrete1.ml.tools import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from discrete1.utils.decorators import break_loop, grange


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


class RegressionDeepONet:
    """Simple training harness for regression-style PyTorch models.

    This class wraps common training utilities for regression-style neural
    models, including dataset preparation, training with validation, testing,
    checkpointing, and batched inference.

    Notes
    -----
    Only public methods are intended for external use. Helper methods prefixed
    with an underscore are internal implementation details.
    """

    __valid_kwargs = {
        "n_epochs",
        "batch_size",
        "loss_fn",
        "optimizer",
        "lr",
        "weight_decay",
        "lr_scheduler",
        "lr_scheduler_params",
        "checkpoint",
        "device",
        "LOUD",
        "tensorboard",
    }

    def __init__(self, model, flux, labels, y, **kwargs):
        """Initialize the trainer with a PyTorch model and datasets.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to train. Its ``forward`` should accept two
            inputs ``(flux, labels)`` and return predictions compatible with ``y``.
        flux : numpy.ndarray, shape (n_samples, n_flux_features)
            Branch input features.
        labels : numpy.ndarray, shape (n_samples, n_label_features)
            Trunk input features.
        y : numpy.ndarray, shape (n_samples, n_targets)
            Regression targets.
        **kwargs : dict
        n_epochs : int, default 100
            Number of training epochs.
        batch_size : int, default 32
            Mini-batch size for training and evaluation.
        loss_fn : str or torch.nn.modules.loss._Loss, default "MSELoss"
            Either an instantiated loss object or the name of a loss class
            in ``torch.nn`` (e.g., "MSELoss").
        optimizer : str, default "Adam"
            Name of an optimizer in ``torch.optim`` to use.
        lr : float, default 1e-3
            Optimizer learning rate.
        weight_decay : float, default 0.0
            L2 weight decay (regularization) for the optimizer.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler or None, optional
            Optional learning rate scheduler. Not applied by default.
        lr_scheduler_params : dict
            Parameters for learning rate scheduler.
        checkpoint : str, default None
            If not None, saves checkpoint data at end of n_epochs
        device : str, default "cpu"
            Device specifier for computation (e.g., "cpu", "cuda").
        LOUD : bool, default True
            If True, shows a progress bar during training.
        tensorboard : str, default "run-01"
            TensorBoard log directory passed to ``SummaryWriter``.
        """
        self.model = model
        self.flux = flux
        self.labels = labels
        self.y = y

        # Raise a UserWarning for unexpected arguments
        unexpected_kwargs = set(kwargs.keys()) - RegressionDeepONet.__valid_kwargs
        for kwarg in unexpected_kwargs:
            warning_string = f"Unexpected keyword arguments: {kwarg}"
            warnings.warn(warning_string, UserWarning, stacklevel=2)

        self.n_epochs = kwargs.get("n_epochs", 100)
        self.batch_size = kwargs.get("batch_size", 32)
        self.loss_fn = kwargs.get("loss_fn", "MSELoss")
        self.optimizer_name = kwargs.get("optimizer", "Adam")
        self.lr = kwargs.get("lr", 0.001)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.lr_scheduler_params = kwargs.get("lr_scheduler_params", {})
        self.ckpt_name = kwargs.get("checkpoint", None)
        self.device = torch.device(kwargs.get("device", "cpu"))
        self.LOUD = kwargs.get("LOUD", True)
        self.writer = kwargs.get("tensorboard", None)
        if self.writer is not None:
            self.writer = SummaryWriter(log_dir=self.writer)

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
        The datasets pack triples ``(flux, labels, y)``. DataLoader instances
        are stored on the instance as ``train_loader``, ``val_loader``, and
        ``test_loader``.
        """
        flux_train, flux_test, labels_train, labels_test, y_train, y_test = (
            model_selection.train_test_split(
                self.flux, self.labels, self.y, test_size=0.2, random_state=seed
            )
        )

        flux_train, flux_val, labels_train, labels_val, y_train, y_val = (
            model_selection.train_test_split(
                flux_train, labels_train, y_train, test_size=0.25, random_state=seed
            )
        )

        train_dataset = TensorDataset(
            torch.tensor(flux_train, dtype=torch.float32),
            torch.tensor(labels_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(flux_val, dtype=torch.float32),
            torch.tensor(labels_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        test_dataset = TensorDataset(
            torch.tensor(flux_test, dtype=torch.float32),
            torch.tensor(labels_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        if verbose:
            return train_dataset, val_dataset, test_dataset

    def process_data_memmap(
        self,
        filename,
        batch_size=1,
        train_size=0.6,
        val_size=0.2,
        verbose=False,
    ):
        """Split arrays into train/val/test sets and build data loaders.

        This uses the ``DeepONetDataset`` Dataset and memmap files to work with
        large datasets when training with RegressionDeepONet.

        Parameters
        ----------
        filename : str
            Name of the memmap file. File stored using the
            ``data.deeponet_memmap`` function.
        batch_size : int, default 1
            batch_size used for memmap batch chunking. Replaces the batch_size
            in the __init__ call.
        train_size : float, default 0.6
            Fraction of the data used for training data.
        val_size : float, default 0.2
            Fraction of the data used for validation data.
        verbose : bool, default False
            If True, also return the created ``TensorDataset`` objects.

        Returns
        -------
        tuple of torch.utils.data.TensorDataset or None
            If ``verbose`` is True, returns
            ``(train_dataset, val_dataset, test_dataset)``. Otherwise, returns
            None.

        Notes
        -----
        The datasets pack triples ``(flux, labels, y)``. DataLoader instances
        are stored on the instance as ``train_loader``, ``val_loader``, and
        ``test_loader``. There is a pickle file with the same name as
        ``filename`` that loads the appropriate shape for the custom dtype.
        """
        n_samples, flux_shape, labels_shape, y_shape, dtype = (
            load_deeponet_memmap_metadata(filename)
        )

        # Sizes of train/test/validation
        train_end = int(train_size * n_samples)
        val_end = int((train_size + val_size) * n_samples)

        # Custom datasets
        train_dataset = DeepONetDataset(
            filename,
            dtype,
            (n_samples,),
            batch_size=batch_size,
            start_idx=0,
            end_idx=train_end,
        )
        val_dataset = DeepONetDataset(
            filename,
            dtype,
            (n_samples,),
            batch_size=batch_size,
            start_idx=train_end,
            end_idx=val_end,
        )
        test_dataset = DeepONetDataset(
            filename,
            dtype,
            (n_samples,),
            batch_size=batch_size,
            start_idx=val_end,
            end_idx=n_samples,
        )

        # Convert to DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            sampler=BatchShuffleSampler(train_dataset),
        )
        self.val_loader = DataLoader(val_dataset, batch_size=1)
        self.test_loader = DataLoader(test_dataset, batch_size=1)
        if verbose:
            return train_dataset, val_dataset, test_dataset

    def _init_optimizer(self):
        """Construct optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer instance.
        """
        optimizer = getattr(optim, self.optimizer_name)(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def _init_lr_scheduler(self, optimizer):
        """Construct learning rate scheduler for training.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Configured optimizer instance.

        Returns
        -------
        torch.optim.lr_scheduler._LRScheduler
            Configured learning rate scheduler instance.
        """
        scheduler = None
        if self.lr_scheduler is not None:
            scheduler = getattr(optim.lr_scheduler, self.lr_scheduler)(
                optimizer, **self.lr_scheduler_params
            )
        return scheduler

    def _init_loss_fn(self):
        """Construct loss function for training.

        Returns
        -------
        torch.nn.modules.loss._Loss
            Configured loss function instance.
        """
        if isinstance(self.loss_fn, str):
            loss_criterion = getattr(nn, self.loss_fn)()
        else:
            loss_criterion = self.loss_fn
        return loss_criterion

    def _batch_to_device(self, flux_batch, labels_batch, y_batch):
        """Move a mini-batch to the configured computation device.

        Parameters
        ----------
        flux_batch : torch.Tensor
            Batch of branch inputs.
        labels_batch : torch.Tensor
            Batch of trunk inputs.
        y_batch : torch.Tensor
            Batch of regression targets.

        Returns
        -------
        tuple of torch.Tensor
            The trio ``(flux_device, labels_device, y_device)`` located on
            ``self.device``.
        """
        return (
            flux_batch.to(self.device),
            labels_batch.to(self.device),
            y_batch.to(self.device),
        )

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
            ncols=0,
        )

    def _train_batch(self, progress_bar, optimizer, loss_fn):
        """Execute one training pass over the training loader.

        Parameters
        ----------
        progress_bar : Iterable
            Iterable yielding ``(batch_index, (flux_batch, labels_batch, y_batch))``.
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters.
        loss_fn : callable
            Loss function mapping ``(outputs, targets)`` to a scalar loss.

        Returns
        -------
        float
            Average training loss across the epoch.
        """
        train_loss = 0.0

        for _, (flux_batch, labels_batch, y_batch) in progress_bar:
            flux_batch, labels_batch, y_batch = self._batch_to_device(
                flux_batch, labels_batch, y_batch
            )

            optimizer.zero_grad()
            outputs = self.model(flux_batch, labels_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * flux_batch.size(0)
            avg_loss = train_loss / len(self.train_loader.dataset)
            progress_bar.set_postfix({"loss": avg_loss})

        return train_loss / len(self.train_loader.dataset)

    def _val_batch(self, data_loader, scheduler, loss_fn):
        """Evaluate the model on a data loader without gradient updates.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Data loader to iterate over for evaluation.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler.
        loss_fn : callable
            Loss function mapping ``(outputs, targets)`` to a scalar loss.

        Returns
        -------
        float
            Average loss over the provided data loader.
        """
        val_loss = 0.0

        with torch.no_grad():
            for flux_batch, labels_batch, y_batch in data_loader:

                flux_batch, labels_batch, y_batch = self._batch_to_device(
                    flux_batch, labels_batch, y_batch
                )
                outputs = self.model(flux_batch, labels_batch)
                loss = loss_fn(outputs, y_batch)
                val_loss += loss.item() * flux_batch.size(0)

        val_loss /= len(data_loader.dataset)
        if scheduler is not None:
            if self.lr_scheduler == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        return val_loss

    def train_one_epoch(self, epoch, loss_fn, optimizer, scheduler):
        """Train and validate one epoch.

        Parameters
        ----------
        epoch : int
            The current epoch to train.
        loss_fn : callable
            Loss function mapping ``(outputs, targets)`` to a scalar loss.
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler.

        Returns
        -------
        tuple of float, float
            Training and validation loss.
        """
        progress_bar = self._update_progress_bar(epoch)

        # Training
        self.model.train()
        train_loss = self._train_batch(progress_bar, optimizer, loss_fn)

        # Validation
        self.model.eval()
        val_loss = self._val_batch(self.val_loader, scheduler, loss_fn)

        # Tensorboard logging
        if self.writer is not None:
            self.writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
            current_lr = optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Learning Rate", current_lr, epoch)

        return train_loss, val_loss

    @break_loop
    def train(wrapper, self):
        """Train the model with validation and test evaluation.

        Performs epoch-wise training using the configured optimizer and loss
        function, tracks training/validation loss, selects the best model
        weights by validation loss, evaluates on the test set, and restores
        the best weights.

        Notes
        -----
        Metrics are also stored on the instance as ``metric_data``.
        """
        loss_criterion = self._init_loss_fn()
        optimizer = self._init_optimizer()
        scheduler = self._init_lr_scheduler(optimizer)

        best_val_loss = np.inf
        best_model_weights = None
        train_loss_history = []
        val_loss_history = []

        for epoch in grange(self.n_epochs, stop_event=wrapper.stop_event):
            train_loss, val_loss = self.train_one_epoch(
                epoch, loss_criterion, optimizer, scheduler
            )
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())

        # Testing
        with torch.no_grad():
            self.model.eval()
            test_loss = self._val_batch(self.test_loader, None, loss_criterion)

        # Load best model
        self.model.load_state_dict(best_model_weights)

        # Save checkpoint
        if self.ckpt_name is not None:
            self.save_checkpoint(epoch, best_val_loss, optimizer, scheduler)

        # History
        self.metric_data = {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "test_loss": test_loss,
        }

    def save_checkpoint(self, epoch, val_loss, optimizer, scheduler):
        """Save checkpoint data after epoch to ``self.ckpt_name``.

        Parameters
        ----------
        epoch : int
            The current epoch to train.
        val_loss : float
            Best validation loss from training.
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler.
        """
        ckpt = {
            "epoch": epoch + 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_val_loss": val_loss,
        }
        torch.save(ckpt, self.ckpt_name)

    def load_checkpoint(self, optimizer, scheduler):
        """Load checkpoint data to continue training from ``self.ckpt_name``.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler.

        Returns
        -------
        tuple of int, float
            Starting epoch and best validation loss.
        """
        ckpt = torch.load(self.ckpt_name, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        if scheduler is not None and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])

        return ckpt["epoch"], ckpt["best_val_loss"]

    @break_loop
    def continue_training(wrapper, self):
        """Continue train the model with validation and test evaluation.

        Performs epoch-wise training using the configured optimizer and loss
        function, tracks training/validation loss, selects the best model
        weights by validation loss, evaluates on the test set, and restores
        the best weights.

        Notes
        -----
        Metrics are also stored on the instance as ``metric_data``.
        """
        loss_criterion = self._init_loss_fn()
        optimizer = self._init_optimizer()
        scheduler = self._init_lr_scheduler(optimizer)

        epoch_start, best_val_loss = self.load_checkpoint(optimizer, scheduler)
        self.n_epochs += epoch_start

        best_model_weights = None
        train_loss_history = []
        val_loss_history = []

        for epoch in grange(epoch_start, self.n_epochs, stop_event=wrapper.stop_event):
            train_loss, val_loss = self.train_one_epoch(
                epoch, loss_criterion, optimizer, scheduler
            )
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())

        # Testing
        with torch.no_grad():
            self.model.eval()
            test_loss = self._val_batch(self.test_loader, None, loss_criterion)

        # Return best model
        self.model.load_state_dict(best_model_weights)

        if self.ckpt_name is not None:
            self.save_checkpoint(epoch, best_val_loss, optimizer, scheduler)

        # History
        self.metric_data = {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "test_loss": test_loss,
        }

    def save_model(self, model_name):
        """Save the trained model parameters and loss metrics to disk.

        Parameters
        ----------
        model_name : str
            Base filename (without extension). Saves ``f"{model_name}.pt"`` for
            model weights and ``f"{model_name}_loss_metrics.npz"`` for metrics
            captured in ``metric_data``.
        """
        torch.save(self.model.state_dict(), f"{model_name}.pt")
        np.savez(f"{model_name}-loss-metrics.npz", **self.metric_data)

    def load_model(self, model_name):
        """Load model parameters from disk into the model.

        Parameters
        ----------
        model_name : str
            Base filename used when saving (without extension). Reads
            ``f"{model_name}.pt"``.
        """
        self.model.load_state_dict(torch.load(f"{model_name}.pt", weights_only=True))

    def predict(self, flux, labels, batch_size=1):
        """Run batched inference and return predictions.

        Parameters
        ----------
        X_new : numpy.ndarray
            Input features of shape (n_samples, n_features). This helper
            assumes the wrapped model accepts a single tensor input. If your
            model requires multiple inputs (e.g., ``flux`` and ``labels``),
            implement a custom prediction routine consistent with ``forward``.
        batch_size : int, default 1
            Batch size for inference.

        Returns
        -------
        numpy.ndarray
            Stacked predictions of shape (n_samples, n_targets).
        """
        test_dataset = TensorDataset(
            torch.tensor(flux, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            self.model.eval()
            for flux_batch, labels_batch in test_loader:
                outputs = self.model(
                    flux_batch.to(self.device), labels_batch.to(self.device)
                )
                predictions.append(outputs.cpu().numpy())

        return np.vstack(predictions)
