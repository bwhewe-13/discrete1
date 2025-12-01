"""Prediction and inference utilities for machine learning models.

This module provides wrappers and utilities for various machine learning
models used in reduced-order modeling and operator learning for neutron
transport problems. The module includes:

- ``load_djinn_models``: Utility to load trained DJINN models
- ``DJINN``: Wrapper for DJINN regression models with optional transformations
- ``AutoDJINN``: Composite autoencoder-DJINN model for latent space prediction
- ``DeepONet``: Deep operator network for learning mappings between function spaces

These classes provide consistent interfaces for making predictions with
pre-trained models, handling input/output transformations, and managing
labeled/unlabeled prediction scenarios.

Notes
-----
Optional ML dependencies are required. Install with::

    pip install discrete1[ml]

TensorFlow/Keras is used for the autoencoder components, ``djinn`` is
used for the DJINN regressors, and custom DeepONet implementations are
provided through ``discrete1.ml.train``.

Examples
--------
Load and use a DJINN model:

>>> from discrete1.ml.predict import DJINN
>>> import numpy as np
>>> model = DJINN('path/to/model', transformer=np.cbrt, detransformer=lambda x: x**3)
>>> predictions = model.predict(flux_data)

Use an AutoDJINN composite model:

>>> from discrete1.ml.predict import AutoDJINN
>>> auto_model = AutoDJINN(
...     'encoder.h5', 'djinn_model', 'decoder.h5',
...     transformer=lambda x: x/np.max(x),
...     detransformer=lambda x: x*scale_factor
... )
>>> predictions = auto_model.predict(flux_data)
"""

import os

import numpy as np

from discrete1.ml import train

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf  # noqa: E402
    from djinn import djinn  # noqa: E402
except ImportError as e:
    raise ImportError(
        "ML dependencies are not installed. Install with:\n"
        "   pip install discrete1[ml]"
    ) from e


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


class DJINN:
    """DJINN regression model wrapper for reduced-order prediction.

    This class provides a convenient interface for loading and using trained
    DJINN (Deep Joint Inverse-Neuron Network) models with optional input/output
    transformations. DJINN models are used for regression tasks in reduced-order
    modeling of neutron transport problems.

    Parameters
    ----------
    file : str
        Path to saved DJINN model checkpoint.
    transformer : callable, optional
        Function to transform/normalize input data before prediction.
        If None, no transformation is applied.
    detransformer : callable, optional
        Function to inverse transform/denormalize output predictions.
        If None, no inverse transformation is applied.

    Attributes
    ----------
    model : djinn.DJINN_Regressor
        The underlying DJINN regression model.
    transformer : callable or None
        Input transformation function.
    detransformer : callable or None
        Output inverse transformation function.

    Notes
    -----
    The transformer and detransformer are useful for data normalization.
    Common transformations include:

    - Cube root: ``transformer = np.cbrt``, ``detransformer = lambda x: x**3``
    - Min-max scaling: ``transformer = lambda x: (x - x_min) / (x_max - x_min)``
    - Standardization: ``transformer = lambda x: (x - mean) / std``

    Examples
    --------
    Create a DJINN model with cube root transformation:

    >>> model = DJINN('model.pkl', transformer=np.cbrt, detransformer=lambda x: x**3)
    >>> predictions = model.predict(flux_data)

    Use with labeled data:

    >>> label = np.array([[param1, param2]])
    >>> predictions = model.predict(flux_data, label=label)
    """

    def __init__(self, file, transformer=None, detransformer=None):
        """Initialize DJINN instance.

        Loads a trained DJINN model from the specified file path and
        sets up optional input/output transformation functions.

        Parameters
        ----------
        file : str
            Path to saved DJINN model checkpoint.
        transformer : callable, optional
            Function to transform/normalize input data before prediction.
            If None, no transformation is applied.
        detransformer : callable, optional
            Function to inverse transform/denormalize output predictions.
            If None, no inverse transformation is applied.
        """
        self.model = djinn.load(model_name=file)
        self.transformer = transformer
        self.detransformer = detransformer
        # transformer = lambda x: np.cbrt(x)
        # detransformer = lambda x: x**(3)

    def predict(self, flux, label=None):
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
        # If there is a transformer, apply it
        if self.transformer is not None:
            flux = self.transformer(flux.copy())
        # Labeled model
        if label is not None:
            labels = np.tile(label, (flux.shape[0], 1))
            pred_y = self.model.predict(np.hstack((labels, flux)))
        # Non-labeled model
        else:
            pred_y = self.model.predict(flux)
        # If there is a detransformer, apply it
        if self.detransformer is not None:
            pred_y = self.detransformer(pred_y)
        return pred_y


class AutoDJINN:
    """Autoencoder-DJINN composite model for reduced-order prediction.

    This class combines an autoencoder (encoder-decoder) with a DJINN model
    to perform prediction in a reduced latent space. The workflow is:

    1. Transform input data (optional normalization)
    2. Encode high-dimensional input to low-dimensional latent representation
    3. Apply DJINN regression model in latent space
    4. Decode latent predictions back to original dimensionality
    5. Inverse transform output (optional denormalization)

    This architecture is particularly effective for high-dimensional problems
    where direct regression would be computationally expensive. The autoencoder
    compresses the data while preserving essential features, and DJINN performs
    regression in the compressed space.

    Parameters
    ----------
    file_encoder : str
        Path to saved Keras encoder model (.h5 file).
    file_djinn : str
        Path to saved DJINN model checkpoint.
    file_decoder : str
        Path to saved Keras decoder model (.h5 file).
    transformer : callable
        Function to transform/normalize input data before encoding.
    detransformer : callable
        Function to inverse transform/denormalize output after decoding.
    optimizer : str, optional
        Optimizer name for compiling Keras encoder/decoder models.
        Default is 'adam'.
    loss : str, optional
        Loss function name for compiling Keras encoder/decoder models.
        Default is 'mse' (mean squared error).

    Attributes
    ----------
    encoder : tensorflow.keras.Model
        Loaded and compiled encoder network.
    model_djinn : djinn.DJINN_Regressor
        Loaded DJINN regression model.
    decoder : tensorflow.keras.Model
        Loaded and compiled decoder network.
    transformer : callable
        Input transformation function.
    detransformer : callable
        Output inverse transformation function.

    Notes
    -----
    The encoder and decoder are TensorFlow/Keras models that must be
    pre-trained and compatible with each other (encoder output dimension
    must match DJINN input dimension, DJINN output must match decoder input).

    Examples
    --------
    Create an AutoDJINN model:

    >>> model = AutoDJINN(
    ...     'encoder.h5', 'djinn_latent', 'decoder.h5',
    ...     transformer=lambda x: x / x.max(),
    ...     detransformer=lambda x: x * scale,
    ...     optimizer='adam', loss='mse'
    ... )
    >>> predictions = model.predict(flux_data)
    """

    def __init__(
        self,
        file_encoder,
        file_djinn,
        file_decoder,
        transformer,
        detransformer,
        optimizer="adam",
        loss="mse",
    ):
        """Initialize AutoDJINN instance.

        Loads encoder, DJINN, and decoder models from files and compiles
        the Keras encoder/decoder models with the specified optimizer and
        loss function.

        Parameters
        ----------
        file_encoder : str
            Path to saved Keras encoder model (.h5 file).
        file_djinn : str
            Path to saved DJINN model checkpoint.
        file_decoder : str
            Path to saved Keras decoder model (.h5 file).
        transformer : callable
            Function to transform/normalize input data.
        detransformer : callable
            Function to inverse transform/denormalize output data.
        optimizer : str, optional
            Optimizer name for Keras models (default: 'adam').
        loss : str, optional
            Loss function name for Keras models (default: 'mse').
        """
        self.encoder = tf.keras.models.load_model(file_encoder)
        self.encoder.compile(optimizer=optimizer, loss=loss)

        self.model_djinn = djinn.load(model_name=file_djinn)

        self.decoder = tf.keras.models.load_model(file_decoder)
        self.decoder.compile(optimizer=optimizer, loss=loss)

        self.transformer = transformer
        self.detransformer = detransformer

    def predict(self, flux, label=None):
        """Make predictions using the AutoDJINN composite model.

        Performs the complete prediction pipeline: transformation, encoding,
        DJINN regression in latent space, decoding, and inverse transformation.
        Supports both labeled and unlabeled prediction modes.

        Parameters
        ----------
        flux : numpy.ndarray
            Input flux array of shape (samples, features). High-dimensional
            flux values to be compressed and predicted.
        label : numpy.ndarray, optional
            Parameter/label array of shape (1, n_params) that will be tiled
            and concatenated with encoded features. Used for parametric
            predictions where additional parameters (e.g., cross sections)
            affect the output. If None, performs unlabeled prediction.

        Returns
        -------
        numpy.ndarray
            Predicted values in original space after the full pipeline.
            Shape is (samples, output_features) matching the decoder output.

        Notes
        -----
        The prediction pipeline for labeled data:

        1. Transform flux using transformer
        2. Concatenate labels with transformed flux
        3. Encode to latent space
        4. Concatenate labels with latent representation
        5. Apply DJINN model in latent space
        6. Decode latent predictions to original dimensionality
        7. Apply detransformer to final output

        For unlabeled data, steps 2 and 4 are skipped.
        """
        # Apply transformer
        flux = self.transformer(flux.copy())

        # Labeled model
        if label is not None:
            labels = np.tile(label, (flux.shape[0], 1))
            encoded = self.encoder.predict(np.hstack((labels, flux)), verbose=0)
            latent = self.model_djinn.predict(np.hstack((labels, encoded)))
            decoded = self.decoder.predict(latent, verbose=0)
        # Non-labeled model
        else:
            encoded = self.encoder.predict(flux, verbose=0)
            latent = self.model_djinn.predict(encoded)
            decoded = self.decoder.predict(latent, verbose=0)

        # Apply detransformer
        return self.detransformer(decoded)


class DeepONet:
    """Deep Operator Network composite model for reduced-order prediction.

    This class provides a wrapper around a trained DeepONet regression model
    for operator learning. DeepONet learns mappings between function spaces
    and can predict output functions given input functions and parameters.

    Parameters
    ----------
    network : dict or object
        Network architecture specification for the DeepONet model.
    file : str
        Path to saved DeepONet model checkpoint.
    transformer : callable
        Function to transform/normalize input data before prediction.
    detransformer : callable
        Function to inverse transform/denormalize output predictions.

    Attributes
    ----------
    model : RegressionDeepONet
        The underlying DeepONet regression model.
    transformer : callable
        Input transformation function.
    detransformer : callable
        Output inverse transformation function.

    Notes
    -----
    The transformer and detransformer are typically used to normalize
    inputs and denormalize outputs. For example, cube root transformations
    can be used: ``transformer = lambda x: np.cbrt(x)`` and
    ``detransformer = lambda x: x**3``.
    """

    def __init__(self, network, file, transformer, detransformer):
        """Initialize DeepONet instance.

        Creates a RegressionDeepONet model with the specified network
        architecture and loads pre-trained weights from a checkpoint file.

        Parameters
        ----------
        network : dict or object
            Network architecture specification for the DeepONet model.
        file : str
            Path to saved DeepONet model checkpoint.
        transformer : callable
            Function to transform/normalize input data.
        detransformer : callable
            Function to inverse transform/denormalize output data.
        """
        self.model = train.RegressionDeepONet(network, None, None, None)
        self.model.load_model(file)
        self.transformer = transformer
        self.detransformer = detransformer
        # transformer = lambda x: np.cbrt(x)
        # detransformer = lambda x: x**(3)

    def predict(self, flux, label):
        """Make predictions using the DeepONet model.

        Applies the trained DeepONet model to predict output functions given
        input flux data and label parameters. The prediction pipeline includes
        input transformation, model inference, and output detransformation.

        Parameters
        ----------
        flux : numpy.ndarray
            Input flux array of shape (samples, features) representing the
            input function values at discrete points.
        label : numpy.ndarray
            Parameter/label array that will be tiled to match the number of
            samples. Represents operator parameters or boundary conditions.

        Returns
        -------
        numpy.ndarray
            Predicted output values in original space after applying the
            detransformation. Shape matches the model's output dimensionality.

        Notes
        -----
        The label parameter is automatically tiled to create one copy per
        sample in the flux input, allowing batch predictions with consistent
        parameters.
        """
        # Non-labeled model
        labels = np.tile(label, (flux.shape[0], 1))
        scaled_flux = self.transformer(flux)
        pred_y = self.model.predict(
            scaled_flux, labels, batch_size=scaled_flux.shape[0]
        )
        return self.detransformer(pred_y)
