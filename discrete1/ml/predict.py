"""Module for training and using DJINN machine learning models."""

import itertools
import os

import numpy as np

from discrete1.ml.tools import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf  # noqa: E402
    from djinn import djinn  # noqa: E402
except ImportError as e:
    raise ImportError(
        "ML dependencies are not installed. Install with:\n"
        "   pip install discrete1[ml]"
    ) from e


def train_model(
    x_train, x_test, y_train, y_test, path, trees, depth, dropout_keep=1.0, display=0
):
    """Train DJINN models with different tree configurations.

    Trains multiple DJINN models exploring combinations of tree counts
    and depths. Each model is trained with optimized hyperparameters
    and evaluated on test data. Results and model checkpoints are saved.

    Parameters
    ----------
    x_train : numpy.ndarray
        Training input data.
    x_test : numpy.ndarray
        Test input data.
    y_train : numpy.ndarray
        Training target values.
    y_test : numpy.ndarray
        Test target values.
    path : str
        Path to save models and metrics.
    trees : sequence
        Number of trees (neural nets) to try.
    depth : sequence
        Maximum tree depths to try.
    dropout_keep : float, optional
        Dropout keep probability, default 1.0.
    display : int, optional
        Verbosity level (0=silent).

    Notes
    -----
    For each tree count and depth combination:
    - Optimizes batch size, learning rate, epochs
    - Trains model and saves checkpoint
    - Computes and saves error metrics (MAE, MSE, EVS, R2)
    """
    # number of trees = number of neural nets in ensemble
    # max depth of tree = optimize this for each data set
    # dropout typically set to 1 for non-Bayesian models

    for ntrees, maxdepth in itertools.product(trees, depth):
        if display:
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
        optimal = model.get_hyperparameters(x_train, y_train)
        batchsize = optimal["batch_size"]
        learnrate = optimal["learn_rate"]
        epochs = np.min((300, optimal["epochs"]))
        # epochs = optimal['epochs']

        # train the model with these settings
        model.train(
            x_train,
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
        y_estimate = model.predict(x_test)

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


def load_djinn_models(model_path):
    """Load one or more trained DJINN models.

    Parameters
    ----------
    model_path : str or sequence
        Single model path as string, or sequence of model paths.
        Use 0 in sequence for placeholder/dummy model.

    Returns
    -------
    djinn.DJINN_Regressor or list
        Loaded model if input is string, or list of models for sequence input.

    Notes
    -----
    When passing a sequence, elements that are 0 will result in 0 being
    included in the returned list instead of a loaded model.
    """
    print("Loading DJINN Models...\n{}".format("=" * 30))
    if isinstance(model_path, str):
        return djinn.load(model_name=model_path)
    models = []
    for path in model_path:
        if path == 0:
            models.append(0)
        else:
            models.append(djinn.load(model_name=path))
    print("Loading Complete\n{}\n".format("=" * 30))
    return models


class AutoDJINN:
    """Autoencoder-DJINN composite model for reduced-order prediction.

    This class combines an autoencoder (encoder-decoder) with a DJINN model
    to perform prediction in a reduced latent space. The workflow is:
    1. Encode high-dimensional input to latent representation
    2. Apply DJINN model in latent space
    3. Decode back to original dimensionality

    Parameters
    ----------
    file_encoder : str
        Path to saved encoder model.
    file_djinn : str
        Path to saved DJINN model.
    file_decoder : str
        Path to saved decoder model.
    transformer : callable
        Function to transform input data.
    detransformer : callable
        Function to inverse transform output data.
    groups : int
        Number of energy groups.
    optimizer : str, optional
        Optimizer name for Keras models.
    loss : str, optional
        Loss function name for Keras models.
    """

    def __init__(
        self,
        file_encoder,
        file_djinn,
        file_decoder,
        transformer,
        detransformer,
        groups,
        optimizer="adam",
        loss="mse",
    ):
        """Initialize AutoDJINN instance.

        Loads encoder, DJINN, and decoder models from files and compiles
        the Keras models with specified optimizer and loss. See class
        docstring for parameter descriptions.
        """

        self.encoder = tf.keras.models.load_model(file_encoder)
        self.encoder.compile(optimizer=optimizer, loss=loss)

        self.model_djinn = djinn.load(model_name=file_djinn)

        self.decoder = tf.keras.models.load_model(file_decoder)
        self.decoder.compile(optimizer=optimizer, loss=loss)

        self.transformer = transformer
        self.detransformer = detransformer
        self.groups = groups

    def predict(self, flux):
        """Make predictions using the composite model.

        Handles both labeled and unlabeled input formats through the
        full autoencoder-DJINN pipeline.

        Parameters
        ----------
        flux : numpy.ndarray
            Input flux array. If unlabeled, shape should be
            (samples, groups). If labeled, shape should be
            (samples, groups+1) with labels in first column.

        Returns
        -------
        numpy.ndarray
            Predicted values in original space after full pipeline
            transform.
        """
        # Non-labeled model
        if flux.shape[1] == self.groups:
            encoded = self.encoder.predict(self.transformer(flux), verbose=0)
            latent = self.model_djinn.predict(encoded)
            return self.detransformer(self.decoder.predict(latent, verbose=0))
        # Labeled model
        else:
            encoded = self.encoder.predict(self.transformer(flux[:, 1:]), verbose=0)
            latent = self.model_djinn.predict(np.hstack((flux[:, 0:1], encoded)))
            return self.detransformer(self.decoder.predict(latent, verbose=0))
