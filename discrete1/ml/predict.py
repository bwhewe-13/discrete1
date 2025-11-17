"""Training and inference utilities for DJINN-based models.

This module contains helpers to load DJINN models and a composite
``AutoDJINN`` predictor that combines an autoencoder with a DJINN model
to perform reduced-order predictions.

Notes
-----
Optional ML dependencies are required. Install with::

    pip install discrete1[ml]

TensorFlow/Keras is used for the autoencoder components and ``djinn`` is
used for the DJINN regressors.
"""

import os

import numpy as np

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
