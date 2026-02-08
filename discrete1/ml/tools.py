"""Utilities for DJINN model training and inference.

This module provides utilities for data preprocessing, model training,
and inference with DJINN (Deep Jointly-Informed Neural Networks) models.
It includes functions for cleaning training data, normalizing inputs/outputs,
computing error metrics, and managing model lifecycle.

Notes
-----
The utilities support both fission and scatter rate prediction models,
with helpers for data splitting, normalization, and error analysis.
"""

import os
import string
from glob import glob

import numpy as np


def combine_flux_reaction(flux, xs_matrix, medium_map, labels):
    """Build labeled input/target pairs from flux and cross sections.

    Combines group-wise flux with material-specific cross sections to
    compute reaction rates and attaches a per-cell label. The output packs
    inputs (flux) and targets (reaction rates) into a single array for
    convenient downstream saving/splitting.

    Parameters
    ----------
    flux : numpy.ndarray, shape (iterations, cells_x, groups)
        Multigroup flux values per iteration and spatial cell.
    xs_matrix : numpy.ndarray, shape (n_materials, groups, groups)
        Material-dependent cross-section matrices. For a material index
        ``m``, ``xs_matrix[m]`` is multiplied by the flux vector to produce
        reaction rates.
    medium_map : numpy.ndarray, shape (cells_x,)
        Integer mapping from spatial cell index to material index used to
        look up rows in ``xs_matrix``.
    labels : array-like of float, length ``cells_x``
        Per-cell labels (e.g., enrichments) to store in column 0.

    Returns
    -------
    numpy.ndarray, shape (2, iterations*cells_x, groups+1)
        Stacked data where the first axis selects input vs target:
        - index 0: ``[label, flux_g1, ..., flux_gG]``
        - index 1: ``[label, rate_g1, ..., rate_gG]``

    Notes
    -----
    Rows with all-zero flux and rate are removed.
    """
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
    """Write per-material splits of the training array to disk.

    Parameters
    ----------
    training_data : numpy.ndarray, shape (2, N, G+1)
        Combined inputs/targets array produced by
        :func:`combine_flux_reaction`. Column 0 contains labels used for
        splitting.
    path : str
        Output directory (prefix) where files are written.
    xs : str
        Cross-section type identifier used in filenames, e.g., ``"fission"``
        or ``"scatter"``.
    splits : list of tuple[str, list[float]]
        Sequence of pairs ``(name, labels)``. For each pair, a file named
        ``{xs}_{name}_training_data.npy`` is written with rows whose label is
        contained in ``labels``.

    Returns
    -------
    None
        Creates one ``.npy`` file per split and asserts that all rows are
        accounted for across outputs.

    Examples
    --------
    >>> splits = [["hdpe", [15.04]], ["uh3", [0.0, 0.15]]]
    >>> _split_by_material(training_data, "/tmp/", "fission", splits)
    """
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
    """Prepare training data for fission rate prediction.

    Uses precomputed flux as inputs (``X``) to compute fission reaction
    rates as targets (``y``), and adds a material enrichment label as an
    extra feature (``G+1``). Optionally splits and saves data by material
    group.

    Parameters
    ----------
    path : str
        Directory containing files produced by ``djinn1d.collections()``.
        Expected filenames include ``flux_fission_model.npy``,
        ``fission_cross_sections.npy`` and ``medium_map.npy``.
    labels : array-like of float
        Material labels (e.g., enrichments) aligned with ``medium_map``.
    splits : list of (str, list of float), optional
        Optional split specification such as
        ``[["hdpe", [15.04]], ["uh3", [0.0, 0.15]]]``. When provided,
        separate datasets are written per split name containing rows whose
        label matches one of the listed values.

    Returns
    -------
    None
        Saves processed arrays to ``path``. If ``splits`` is ``None``, a
        single file ``fission_training_data.npy`` is written. Otherwise,
        files of the form ``fission_<name>_training_data.npy`` are written.
    """
    # Load the data
    flux = np.load(path + "flux_fission_model.npy")
    xs_fission = np.load(path + "fission_cross_sections.npy")
    medium_map = np.load(path + "medium_map.npy")
    training_data = combine_flux_reaction(flux, xs_fission, medium_map, labels)

    if splits is not None:
        _split_by_material(training_data, path, "fission", splits)
    else:
        np.save(path + "fission_training_data", training_data)
    # return training_data


def clean_data_scatter(path, labels, splits=None):
    """Prepare training data for scatter rate prediction.

    Reads one or more flux arrays and computes scattering reaction rates
    as targets (``y``), adding a material enrichment label as an extra
    feature (``G+1``). Optionally splits and saves data by material group.

    Parameters
    ----------
    path : str
        Directory containing files produced by ``djinn1d.collections()``.
        Expected filenames include ``scatter_cross_sections.npy``,
        ``medium_map.npy`` and one or more files matching
        ``flux_scatter_model*.npy``.
    labels : array-like of float
        Material labels (e.g., enrichments) aligned with ``medium_map``.
    splits : list of (str, list of float), optional
        Optional split specification such as
        ``[["hdpe", [15.04]], ["uh3", [0.0, 0.15]]]``. When provided,
        separate datasets are written per split name containing rows whose
        label matches one of the listed values.

    Returns
    -------
    None
        Saves processed arrays to ``path``. If ``splits`` is ``None``, a
        single file ``scatter_training_data.npy`` is written. Otherwise,
        files of the form ``scatter_<name>_training_data.npy`` are written.
    """
    # Load the data
    files = np.sort(glob(path + "flux_scatter_model*.npy"))
    xs_scatter = np.load(path + "scatter_cross_sections.npy")
    medium_map = np.load(path + "medium_map.npy")
    training_data = np.empty((2, 0, xs_scatter.shape[1] + 1))
    for file in files:
        flux = np.load(file)
        single_iteration = combine_flux_reaction(flux, xs_scatter, medium_map, labels)
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


def _alphabet_generator():
    alphabet = string.ascii_lowercase
    idx1 = 0
    idx2 = 1
    tmp = ""
    letter = 0
    while True:
        if idx2 == len(alphabet) + 1:
            idx1 = 0
            idx2 = 1
            tmp = alphabet[letter]
            letter += 1
        yield f"{tmp}{alphabet[idx1:idx2]}"
        idx1 += 1
        idx2 += 1


def _number_generator(fill):
    idx = 1
    while True:
        fidx = str(idx).zfill(fill)
        idx += 1
        yield fidx


def update_file_name(file_name, fmt, label="alphabet", fill=3):
    """Update file names for continued training of saved model.

    Sums cross sections over specified axes while preserving array shape
    for materials that are handled by DJINN models.

    Parameters
    ----------
    file_name : str
        Name of the original file.
    fmt : str
        File extension type.
    label : str, optional
        Selection between alphabet and number versioning.
        default is 'alphabet'
    fill : int, optional
        Number of zero padding for label='number'.
        default is 3

    Returns
    -------
    str
        New file name to prevent overwriting files.
    """
    if label == "alphabet":
        generator = _alphabet_generator()
    elif label == "number":
        generator = _number_generator(fill)
    else:
        print("Label version type not available")
        return

    tmp_name = file_name
    version_not_found = os.path.exists(f"{tmp_name}.{fmt}")

    while version_not_found:
        label = next(generator)
        tmp_name = f"{file_name}{label}"
        version_not_found = os.path.exists(f"{tmp_name}.{fmt}")
    return f"{tmp_name}.{fmt}"
