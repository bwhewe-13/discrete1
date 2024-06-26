
# This is for cleaning, running and transforming DJINN models

import numpy as np
from glob import glob
import itertools
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from djinn import djinn


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
            data[:,cc,ii,0] = labels[ii]
            # Add flux (x variable)
            data[0,cc,ii,1:] = flux[cc,ii].copy()
            # Add reaction rate (y variable)
            data[1,cc,ii,1:] = flux[cc,ii] @ xs_matrix[mat].T
    # Collapse iteration and spatial dimensions
    data = data.reshape(2, iterations * cells_x, groups + 1)
    # Remove zero values
    idx = np.argwhere(np.sum(data[...,1:], axis=(0,2)) != 0)
    data = data[:,idx.flatten(),:].copy()
    return data


def _split_by_material(training_data, path, xs, splits):
    # Splits is list([string name, [float labels]])
    # i.e. [["hdpe", [15.04]], ["uh3", [0.0, 0.15]]]

    # Initialize counter
    counts = 0
    # Iterate over splits
    for name, label in splits:
        # Identify location of labels
        idx = np.argwhere(np.isin(training_data[0,:,0], label)).flatten()
        # Continue for non-existant labels
        if len(idx) == 0:
            continue
        # Separate data
        split_data = training_data[:,idx].copy()
        # Keeping track of data points
        counts += split_data.shape[1]
        # Save data
        np.save(path + f"{xs}_{name}_training_data", split_data)
    # Making sure no data was lost
    assert counts == training_data.shape[1], "Need to equal"


def clean_data_fission(path, labels, splits=None):
    """ Takes the flux before the fission rates are calculated (x data),
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
    """ Takes the flux before the scattering rates are calculated (x data),
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
        single_iteration = _combine_flux_reaction(flux, xs_scatter, \
                                                  medium_map, labels)
        training_data = np.hstack((training_data, single_iteration))
    if splits is not None:
        _split_by_material(training_data, path, "scatter", splits)
    else:
        np.save(path + "scatter_training_data", training_data)
    # return training_data


def min_max_normalization(data, verbose=False):
    # Find maximum and minimum values
    high = np.max(data, axis=1)
    np.nan_to_num(high, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    low = np.min(data, axis=1)
    np.nan_to_num(low, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize between 0 and 1
    ndata = (data - low[:,None])/(high - low)[:,None]
    # Remove undesirables
    np.nan_to_num(ndata, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # Return high and low values
    if verbose:
        return ndata, high, low
    return ndata


def root_normalization(data, root):
    return data ** root


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.fabs(y_true - y_pred), axis=-1)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2, axis=-1)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def explained_variance_score(y_true, y_pred):
    evs = 1 - np.var(y_true - y_pred, axis=-1) / np.var(y_true, axis=-1)
    np.nan_to_num(evs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return evs


def r2_score(y_true, y_pred):
    numerator = np.sum((y_true - y_pred)**2, axis=-1)
    denominator = np.sum((y_true - np.mean(y_true, axis=1)[:,None])**2, axis=-1)
    r2 = 1 - numerator / denominator
    np.nan_to_num(r2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return r2


def train_model(x_train, x_test, y_train, y_test, path, trees, depth, \
        dropout_keep=1.0, display=0):
    # number of trees = number of neural nets in ensemble
    # max depth of tree = optimize this for each data set
    # dropout typically set to 1 for non-Bayesian models

    for ntrees, maxdepth in itertools.product(trees, depth):
        if display:
            print("\nNumber of Trees: {}\tMax Depth: {}\n{}".format(ntrees, \
                    maxdepth, "="*40))

        fntrees = str(ntrees).zfill(3)
        fmaxdepth = str(maxdepth).zfill(3)
        modelname = f"model_{fntrees}{fmaxdepth}"

        # initialize the model
        model = djinn.DJINN_Regressor(ntrees, maxdepth, dropout_keep)

        # find optimal settings
        optimal = model.get_hyperparameters(x_train, y_train)
        batchsize = optimal['batch_size']
        learnrate = optimal['learn_rate']
        epochs = np.min((300, optimal['epochs']))
        # epochs = optimal['epochs']

        # train the model with these settings
        model.train(x_train,y_train, epochs=epochs, learn_rate=learnrate, \
                    batch_size=batchsize, display_step=0, save_files=True, \
                    model_path=path, file_name=modelname, \
                    save_model=True, model_name=modelname)

        # Estimate
        y_estimate = model.predict(x_test)

        # evaluate results
        error_dict = {"MAE": mean_absolute_error(y_test, y_estimate),
                      "MSE": mean_squared_error(y_test, y_estimate),
                      "EVS": explained_variance_score(y_test, y_estimate),
                      "R2": r2_score(y_test, y_estimate)}
        np.savez(path + "error_" + modelname, **error_dict)

        # close model
        model.close_model()


def load_djinn_models(model_path):
    print("Loading DJINN Models...\n{}".format("="*30))
    if isinstance(model_path, str):
        return djinn.load(model_name=model_path)
    models = []
    for path in model_path:
        if path == 0:
            models.append(0)
        else:
            models.append(djinn.load(model_name=path))
    print("Loading Complete\n{}\n".format("="*30))
    return models


def update_cross_sections(xs_matrix, model_idx):
    # Summing the cross sections over a specific axis while keeping the 
    # same shape for DJINN models
    updated_xs = np.zeros(xs_matrix.shape)
    for mat in range(xs_matrix.shape[0]):
        if mat in model_idx:
            updated_xs[mat,:,0] = np.sum(xs_matrix[mat], axis=0)
        else:
            updated_xs[mat] = xs_matrix[mat].copy()
    return updated_xs


class AutoDJINN:

    def __init__(self, file_encoder, file_djinn, file_decoder, transformer, \
            detransformer, groups, optimizer="adam", loss="mse"):

        self.encoder = tf.keras.models.load_model(file_encoder)
        self.encoder.compile(optimizer=optimizer, loss=loss)

        self.model_djinn = djinn.load(model_name=file_djinn)

        self.decoder = tf.keras.models.load_model(file_decoder)
        self.decoder.compile(optimizer=optimizer, loss=loss)
        
        self.transformer = transformer
        self.detransformer = detransformer
        self.groups = groups

    def predict(self, flux):
        # Non-labeled model
        if (flux.shape[1] == self.groups):        
            encoded = self.encoder.predict(self.transformer(flux), verbose=0)
            latent = self.model_djinn.predict(encoded)
            return self.detransformer(self.decoder.predict(latent, verbose=0))
        # Labeled model
        else:
            encoded = self.encoder.predict(self.transformer(flux[:,1:]), verbose=0)
            latent = self.model_djinn.predict(np.hstack((flux[:,0:1], encoded)))
            return self.detransformer(self.decoder.predict(latent, verbose=0))