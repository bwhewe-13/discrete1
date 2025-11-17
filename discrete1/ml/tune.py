"""Hyperparameter tuning utilities for DeepONet regressors.

This module provides an Optuna-based tuner for DeepONet-style models
that combine separate "branch" and "trunk" subnetworks to predict
vector-valued targets from flux and label inputs. It constructs search
spaces for network architecture and training hyperparameters, executes a
study, and exports trial histories.

Notes
-----
Optional ML dependencies are required. Install with::

    pip install discrete1[ml]

The tuner uses PyTorch for model definition/training and Optuna for
hyperparameter optimization.
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import model_selection
from torch.utils.data import DataLoader, TensorDataset


class RegressionDeepONet:
    """Optuna tuner for DeepONet regression.

    Given input features ``flux`` and ``labels`` and targets ``y``, this
    class defines a search space over fully connected subnetworks used by a
    DeepONet model and over training hyperparameters. It then runs an Optuna
    study to minimize a validation loss metric.

    Parameters
    ----------
    flux : numpy.ndarray, shape (n_samples, n_flux_features)
        Feature matrix for the branch network input (e.g., flux).
    labels : numpy.ndarray, shape (n_samples, n_label_features)
        Feature matrix for the trunk network input (e.g., material labels).
    y : numpy.ndarray, shape (n_samples, n_outputs)
        Target array for supervised regression.
    **kwargs
        Optional keyword arguments. Recognized keys include:
        - ``device`` (str): computation device ("cpu" or "cuda"). Default "cpu".
        - ``seed`` (int): random seed for train/val split. Default 3.
        - ``test_size`` (float): validation fraction. Default 0.2.
        - Search-space overrides, see :meth:`_init_search_parameters`.

    Examples
    --------
    A minimal end-to-end run with random data:

    >>> import numpy as np
    >>> from discrete1.ml.tune import RegressionDeepONet as DeepO
    >>> n, g_flux, g_labels, n_out = 256, 8, 2, 16
    >>> X_flux = np.random.rand(n, g_flux).astype(np.float32)
    >>> X_labels = np.random.rand(n, g_labels).astype(np.float32)
    >>> y = np.random.rand(n, n_out).astype(np.float32)
    >>> tuner = DeepO(X_flux, X_labels, y, device="cpu", seed=3, test_size=0.2)
    >>> best = tuner.run(n_trials=5, study_name=None, output="demo_study")
    >>> isinstance(best, dict)
    True
    """

    def __init__(self, flux, labels, y, **kwargs):
        """Initialize the tuner with datasets and defaults.

        Parameters
        ----------
        flux : numpy.ndarray, shape (n_samples, n_flux_features)
            Branch input features (e.g., flux values).
        labels : numpy.ndarray, shape (n_samples, n_label_features)
            Trunk input features (e.g., material labels or coordinates).
        y : numpy.ndarray, shape (n_samples, n_outputs)
            Target array for supervised regression.
        **kwargs
            Optional arguments:
            - ``device`` (str): computation device ("cpu" or "cuda").
              Default "cpu".
            - ``seed`` (int): random seed used for train/val splitting.
              Default 3.
            - ``test_size`` (float): validation fraction in the split.
              Default 0.2.
            - Search space overrides used by :meth:`_init_search_parameters`:
              ``max_fc_layers``, ``nodes``, ``fc_activations``, ``dropout``,
              ``n_epochs``, ``batch_size``, ``optimizer``, ``learning_rate``,
              ``weight_decay``, ``loss_functions``.

        Notes
        -----
        This constructor computes input/output sizes, sets ``self.device``,
        and calls :meth:`_process_data` and :meth:`_init_search_parameters` to
        prepare datasets and define the hyperparameter search space.
        """
        self.flux = flux
        self.labels = labels
        self.y = y
        self.flux_input_size = flux.shape[1]
        self.labels_input_size = labels.shape[1]
        self.output_size = y.shape[1]
        self.device = torch.device(kwargs.get("device", "cpu"))
        self._process_data(**kwargs)
        self._init_search_parameters(**kwargs)

    def _process_data(self, **kwargs):
        """Create train/validation datasets.

        Parameters
        ----------
        **kwargs
            May include ``seed`` (int) and ``test_size`` (float).

        Returns
        -------
        None
            Sets ``self.train_dataset`` and ``self.val_dataset`` as
            ``TensorDataset`` instances on CPU tensors.
        """
        seed = kwargs.get("seed", 3)
        test_size = kwargs.get("test_size", 0.2)
        flux_train, flux_val, labels_train, labels_val, y_train, y_val = (
            model_selection.train_test_split(
                self.flux,
                self.labels,
                self.y,
                test_size=test_size,
                seed=seed,
            )
        )

        self.train_dataset = TensorDataset(
            torch.tensor(flux_train, dtype=torch.float32),
            torch.tensor(labels_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        self.val_dataset = TensorDataset(
            torch.tensor(flux_val, dtype=torch.float32),
            torch.tensor(labels_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

    def _init_search_parameters(self, **kwargs):
        """Initialize the hyperparameter search space.

        Parameters
        ----------
        **kwargs
            Optional overrides for search lists. Recognized keys:
            ``max_fc_layers``, ``nodes``, ``fc_activations``, ``dropout``,
            ``n_epochs``, ``batch_size``, ``optimizer``, ``learning_rate``,
            ``weight_decay``, ``loss_functions``.

        Returns
        -------
        None
            Populates ``self.search_space`` with lists of candidate values.
        """
        # NN Construction Parameters
        max_layers = kwargs.get("max_fc_layers", 5)
        nodes = kwargs.get("nodes", [64, 128, 256])
        activations = kwargs.get("fc_activations", ["ReLU", "GELU", "Tanh"])
        dropout = kwargs.get("dropout", [0.0, 0.1, 0.2])

        # Training Parameters
        n_epochs = kwargs.get("n_epochs", [50, 100, 200])
        batch_size = kwargs.get("batch_size", [32, 64, 128])
        optimizer = kwargs.get("optimizer", ["Adam", "SGD"])
        learning_rate = kwargs.get("learning_rate", [1e-4, 1e-3, 1e-2])
        weight_decay = kwargs.get("weight_decay", [0.0, 1e-5, 1e-4])
        loss_functions = kwargs.get(
            "loss_functions", ["MSELoss", "L1Loss", "SmoothL1Loss"]
        )

        # Store parameters
        self.search_space = {
            "max_layers": max_layers,
            "nodes": nodes,
            "activations": activations,
            "dropout": dropout,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss_functions": loss_functions,
        }

    def _init_nn_model(self, trial, in_dim, onet_out_dim, label):
        """Build a feed-forward subnetwork defined by a trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial object used to suggest architecture choices.
        in_dim : int
            Input feature dimension for the first layer.
        onet_out_dim : int
            Output dimension for the subnetwork (latent features).
        label : str
            Prefix used to name per-layer trial parameters (e.g., "b" or "t").

        Returns
        -------
        torch.nn.Sequential
            A sequential model on the configured device.
        """
        # Suggest NN architecture
        num_layers = trial.suggest_int("num_layers", 1, self.search_space["max_layers"])
        layers = []

        for i in range(num_layers):
            # Add FC Layer
            out_dim = trial.suggest_categorical(
                f"{label}_layer_{i+1}_nodes", self.search_space["nodes"]
            )
            layers.append(nn.Linear(in_dim, out_dim))

            # Add Activation Function
            activation_name = trial.suggest_categorical(
                f"{label}_layer_{i+1}_activation", self.search_space["activations"]
            )
            layers.append(getattr(nn, activation_name)())

            # Add Dropout Layer
            dropout_rate = trial.suggest_categorical(
                f"{label}_layer_{i+1}_dropout", self.search_space["dropout"]
            )
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

            in_dim = out_dim

        layers.append(nn.Linear(in_dim, onet_out_dim))
        model = nn.Sequential(*layers).to(self.device)
        return model

    def _init_onet_model(self, trial):
        """Construct a DeepONet from branch and trunk subnetworks.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial object that selects the latent feature size.

        Returns
        -------
        DeepONet
            A DeepONet model moved to the configured device.
        """
        onet_out_dim = trial.suggest_categorical(
            "onet_latent_features", self.search_space["nodes"]
        )
        branch_model = self._init_nn_model(
            trial, self.flux_input_size, onet_out_dim, "b"
        )
        trunk_model = self._init_nn_model(
            trial, self.labels_input_size, onet_out_dim, "t"
        )
        model = DeepONet(branch_model, trunk_model, onet_out_dim, self.output_size)
        model = model.to(self.device)
        return model

    def _init_optimizer(self, trial, model):
        """Instantiate an optimizer from the search space.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial object that selects optimizer hyperparameters.
        model : torch.nn.Module
            Model whose parameters will be optimized.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer instance.
        """
        optimizer_name = trial.suggest_categorical(
            "optimizer", self.search_space["optimizer"]
        )
        learning_rate = trial.suggest_categorical(
            "learning_rate", self.search_space["learning_rate"]
        )
        weight_decay = trial.suggest_categorical(
            "weight_decay", self.search_space["weight_decay"]
        )
        optimizer = getattr(optim, optimizer_name)(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        return optimizer

    def _init_loss_function(self, trial):
        """Create the loss criterion from the search space.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial object that selects the loss function name.

        Returns
        -------
        torch.nn.modules.loss._Loss
            Instantiated loss criterion.
        """
        loss_function = trial.suggest_categorical(
            "loss_function", self.search_space["loss_functions"]
        )
        loss_criterion = getattr(nn, loss_function)()
        return loss_criterion

    def _batch_to_device(self, flux_batch, labels_batch, y_batch):
        """Move a training mini-batch to the configured device.

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
            ``(flux_device, labels_device, y_device)`` tensors on ``self.device``.
        """
        return (
            flux_batch.to(self.device),
            labels_batch.to(self.device),
            y_batch.to(self.device),
        )

    def _train_batch(self, model, optimizer, loss_criterion, train_loader):
        """Run one training epoch over ``train_loader``.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        optimizer : torch.optim.Optimizer
            Optimizer for parameter updates.
        loss_criterion : callable
            Loss function mapping ``(outputs, targets)`` to a scalar loss.
        train_loader : torch.utils.data.DataLoader
            DataLoader yielding training batches.

        Returns
        -------
        None
        """
        for flux_batch, labels_batch, y_batch in train_loader:
            flux_batch, labels_batch, y_batch = self._batch_to_device(
                flux_batch, labels_batch, y_batch
            )
            optimizer.zero_grad()
            outputs = model(flux_batch, labels_batch)
            loss = loss_criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    def _validate(self, model, loss_criterion, val_loader):
        """Compute average validation loss across ``val_loader``.

        Parameters
        ----------
        model : torch.nn.Module
            Model to evaluate.
        loss_criterion : callable
            Loss function mapping ``(outputs, targets)`` to a scalar loss.
        val_loader : torch.utils.data.DataLoader
            DataLoader yielding validation batches.

        Returns
        -------
        float
            Mean loss over the validation dataset.
        """
        val_loss = 0.0
        with torch.no_grad():
            for flux_batch, labels_batch, y_batch in val_loader:
                flux_batch, labels_batch, y_batch = self._batch_to_device(
                    flux_batch, labels_batch, y_batch
                )
                outputs = model(flux_batch, labels_batch)
                loss = loss_criterion(outputs, y_batch)
                val_loss += loss.item() * flux_batch.size(0)
        return val_loss / len(val_loader.dataset)

    def objective(self, trial):
        """Optuna objective function minimizing validation loss.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial instance providing hyperparameter suggestions.

        Returns
        -------
        float
            Final validation metric for the last epoch.
        """
        # Generate the model
        model = self._init_onet_model(trial)
        optimizer = self._init_optimizer(trial, model)
        loss_criterion = self._init_loss_function(trial)
        epochs = trial.suggest_categorical("n_epochs", self.search_space["n_epochs"])
        batch_size = trial.suggest_categorical(
            "batch_size", self.search_space["batch_size"]
        )

        # Create train and validation dataloaders
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        # Training loop
        for epoch in range(epochs):
            model.train()
            self._train_batch(model, optimizer, loss_criterion, train_loader)

            # Validation
            model.eval()
            metric = self._validate(model, loss_criterion, val_loader)
            print(epoch, metric, end="\r")
            trial.report(metric, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return metric

    def run(self, **kwargs):
        """Run a tuning study and return the best parameters.

        Parameters
        ----------
        **kwargs
            Optional arguments:
            - ``n_trials`` (int): number of trials. Default 50.
            - ``study_name`` (str or None): Named study (uses SQLite file
              if provided). Default None.
            - ``output`` (str): Prefix for trial CSV output. Default
              "hyperparameter_search".

        Returns
        -------
        dict
            Best parameter set found by the study.
        """
        n_trials = kwargs.get("n_trials", 50)
        study_name = kwargs.get("study_name", None)
        storage_name = None if study_name is None else f"sqlite:///{study_name}.db"
        output = kwargs.get("output", "hyperparameter_search")

        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
        )
        study.optimize(self.objective, n_trials=n_trials)
        study.trials_dataframe().to_csv(f"{output}_trials.csv", index=False)
        return study.best_params


################################################################################
# Helper Classes
################################################################################
class DeepONet(nn.Module):
    """Two-branch DeepONet for regression.

    DeepONet combines a branch network that processes input functions
    (here, flux) and a trunk network that processes coordinates/labels.
    Their elementwise product is projected to the output dimension by a
    final linear layer without bias.
    """

    def __init__(self, branch, trunk, latent_features, out_features):
        """Initialize a DeepONet model.

        Parameters
        ----------
        branch : torch.nn.Module
            Subnetwork mapping branch inputs to ``latent_features``.
        trunk : torch.nn.Module
            Subnetwork mapping trunk inputs to ``latent_features``.
        latent_features : int
            Size of the latent space shared between branch and trunk.
        out_features : int
            Output dimensionality of the prediction.
        """
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.fc = nn.Linear(latent_features, out_features, bias=False)

    def forward(self, u, y):
        """Forward pass.

        Parameters
        ----------
        u : torch.Tensor, shape (batch, n_flux_features)
            Batch of branch inputs (e.g., flux features).
        y : torch.Tensor, shape (batch, n_label_features)
            Batch of trunk inputs (e.g., labels or coordinates).

        Returns
        -------
        torch.Tensor, shape (batch, out_features)
            Network predictions.
        """
        return self.fc(self.trunk(y) * self.branch(u))
