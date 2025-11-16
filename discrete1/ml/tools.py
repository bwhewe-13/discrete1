"""Machine learning utilities for DJINN model training and inference.

This module provides utilities for data preprocessing, model training,
and inference with DJINN (Deep Jointly-Informed Neural Networks) models.
It includes functions for cleaning training data, normalizing inputs/outputs,
computing error metrics, and managing model lifecycle.

The module supports both fission and scatter rate prediction models,
with utilities for data splitting, normalization, and error analysis.
Key components are organized into data preparation, model training,
and inference sections.
"""

from glob import glob

import numpy as np


def _combine_flux_reaction(flux, xs_matrix, medium_map, labels):
    # Flux parameters
    iterations, cells_x, groups = flux.shape
    # Initialize training data
    data = np.zeros((2, iterations, cells_x, groups + 1))
    # Iterate over iterations and spatial cells
    for cc in range(iterations):
        for ii in range(cells_x):
            mat = medium_map[ii]
            # Add labels
            data[:, cc, ii, 0] = labels[ii]
            # Add flux (x variable)
            data[0, cc, ii, 1:] = flux[cc, ii].copy()
            # Add reaction rate (y variable)
            data[1, cc, ii, 1:] = flux[cc, ii] @ xs_matrix[mat].T
    # Collapse iteration and spatial dimensions
    data = data.reshape(2, iterations * cells_x, groups + 1)
    # Remove zero values
    idx = np.argwhere(np.sum(data[..., 1:], axis=(0, 2)) != 0)
    data = data[:, idx.flatten(), :].copy()
    return data


def _split_by_material(training_data, path, xs, splits):
    # Splits is list([string name, [float labels]])
    # i.e. [["hdpe", [15.04]], ["uh3", [0.0, 0.15]]]

    # Initialize counter
    counts = 0
    # Iterate over splits
    for name, label in splits:
        # Identify location of labels
        idx = np.argwhere(np.isin(training_data[0, :, 0], label)).flatten()
        # Continue for non-existant labels
        if len(idx) == 0:
            continue
        # Separate data
        split_data = training_data[:, idx].copy()
        # Keeping track of data points
        counts += split_data.shape[1]
        # Save data
        np.save(path + f"{xs}_{name}_training_data", split_data)
    # Making sure no data was lost
    assert counts == training_data.shape[1], "Need to equal"


def clean_data_fission(path, labels, splits=None):
    """Clean training data for fission rate prediction.

    Takes the flux before the fission rates are calculated (x data),
    calculates the reaction rates (y data), and adds a label for the
    enrichment level (G+1). Also removes non-fissioning materials.
    Arguments:
        path (str): location of all files named in djinn1d.collections()
        labels (float [materials]): labels for each of the materials
        splits (list [name, [labels]]): splitting training data into
                                    fissible and non-fissible materials
    Returns:
        Processed data saved to path
    """
    # Load the data
    flux = np.load(path + "flux_fission_model.npy")
    xs_fission = np.load(path + "fission_cross_sections.npy")
    medium_map = np.load(path + "medium_map.npy")
    training_data = _combine_flux_reaction(flux, xs_fission, medium_map, labels)

    if splits is not None:
        _split_by_material(training_data, path, "fission", splits)
    else:
        np.save(path + "fission_training_data", training_data)
    # return training_data


def clean_data_scatter(path, labels, splits=None):
    """Clean training data for scatter rate prediction.

    Takes the flux before the scattering rates are calculated (x data),
    calculates the reaction rates (y data), and adds a label for the
    enrichment level (G+1).
    Arguments:
        path (str): location of all files named in djinn1d.collections()
        labels (float [materials]): labels for each of the materials
        splits (list [name, [labels]]): splitting training data into
                                    fissible and non-fissible materials
    Returns:
        Processed data saved to path
    """
    # Load the data
    files = np.sort(glob(path + "flux_scatter_model*.npy"))
    xs_scatter = np.load(path + "scatter_cross_sections.npy")
    medium_map = np.load(path + "medium_map.npy")
    training_data = np.empty((2, 0, xs_scatter.shape[1] + 1))
    for file in files:
        flux = np.load(file)
        single_iteration = _combine_flux_reaction(flux, xs_scatter, medium_map, labels)
        training_data = np.hstack((training_data, single_iteration))
    if splits is not None:
        _split_by_material(training_data, path, "scatter", splits)
    else:
        np.save(path + "scatter_training_data", training_data)
    # return training_data


def min_max_normalization(data, verbose=False):
    """Normalize data to [0,1] range using min-max scaling.

    Scales each feature to the [0,1] interval by subtracting the minimum
    and dividing by the range. Handles NaN and infinite values safely.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array to normalize.
    verbose : bool, optional
        If True, return normalization bounds with result.

    Returns
    -------
    numpy.ndarray or tuple
        Normalized data array if verbose=False.
        Tuple of (normalized data, max values, min values) if verbose=True.
    """
    # Find maximum and minimum values
    data = np.nan_to_num(data, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    high = np.max(data, axis=1)
    low = np.min(data, axis=1)
    # Normalize between 0 and 1
    ndata = (data - low[:, None]) / (high - low)[:, None]
    # Remove undesirables
    np.nan_to_num(ndata, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # Return high and low values
    if verbose:
        return ndata, high, low
    return ndata


def root_normalization(data, root):
    """Apply root normalization to input data.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    root : float
        Root power to apply.

    Returns
    -------
    numpy.ndarray
        Root-normalized data.
    """
    return data**root


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error (MAE) between predictions and truth.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    numpy.ndarray
        MAE values averaged over last axis.
    """
    return np.mean(np.fabs(y_true - y_pred), axis=-1)


def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error (MSE) between predictions and truth.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    numpy.ndarray
        MSE values averaged over last axis.
    """
    return np.mean((y_true - y_pred) ** 2, axis=-1)


def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    numpy.ndarray
        RMSE values.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def explained_variance_score(y_true, y_pred):
    """Calculate explained variance regression score.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    numpy.ndarray
        Explained variance scores, handling NaN/inf cases.
    """
    evs = 1 - np.var(y_true - y_pred, axis=-1) / np.var(y_true, axis=-1)
    np.nan_to_num(evs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return evs


def r2_score(y_true, y_pred):
    """Calculate R2 score.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    numpy.ndarray
        R2 scores, handling NaN/inf cases.
    """
    numerator = np.sum((y_true - y_pred) ** 2, axis=-1)
    denominator = np.sum((y_true - np.mean(y_true, axis=1)[:, None]) ** 2, axis=-1)
    r2 = 1 - numerator / denominator
    np.nan_to_num(r2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return r2


def update_cross_sections(xs_matrix, model_idx):
    """Update cross sections for DJINN model compatibility.

    Sums cross sections over specified axes while preserving array shape
    for materials that are handled by DJINN models.

    Parameters
    ----------
    xs_matrix : numpy.ndarray
        Cross section matrix to update.
    model_idx : sequence
        Material indices handled by DJINN models.

    Returns
    -------
    numpy.ndarray
        Updated cross section matrix with same shape as input.
    """
    # Summing the cross sections over a specific axis while keeping the
    # same shape for DJINN models
    updated_xs = np.zeros(xs_matrix.shape)
    for mat in range(xs_matrix.shape[0]):
        if mat in model_idx:
            updated_xs[mat, :, 0] = np.sum(xs_matrix[mat], axis=0)
        else:
            updated_xs[mat] = xs_matrix[mat].copy()
    return updated_xs
    return updated_xs
