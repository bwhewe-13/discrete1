""" Tools for autoencoders and neural networks """
import numpy as np
import glob
import re

def phi_normalize(matrix,maxi,mini):
    data = matrix.copy()
    if np.argwhere(maxi == 0).shape[0] > 0:
        ind = np.argwhere(maxi != 0).flatten()
        norm = np.zeros((data.shape))
        norm[ind] = (data[ind] - mini[ind][:,None])/(maxi[ind]-mini[ind])[:,None]
    else:
        norm = (data - mini[:,None])/(maxi-mini)[:,None]
    return norm

def phi_normalize_single(matrix,maxi,mini):
    data = matrix.copy()
    if maxi == 0:
        norm = data
    else:
        norm = (data-mini)/(maxi-mini)
    return norm

def randomize(address,number,index=False):
    """ To be used with autoencoder """
    if index:
        index = []
    dataset = []
    for add in address:
        temp = np.load(add)
        dim = np.arange(len(temp))
        np.random.shuffle(dim)
        dataset.append(temp[dim][:number])
        index.append(dim[:number])
        del temp,dim
    if index:
        return np.concatenate((dataset)),index
    return np.concatenate((dataset))

def retrieve_randomize(address,number,index):
    """ To be used with autoencoder """  
    dataset = []
    for ind,add in zip(index,address):
        temp = np.load(add)
        dataset.append(temp[ind])
        del temp
    return np.concatenate((dataset))

def normalize(matrix,verbose=False,angle=False):
    data = matrix.copy()
    maxi = np.max(data,axis=1)
    mini = np.min(data,axis=1)
    # if angle:
    #     mini += 1e-20
    if np.argwhere(maxi == 0).shape[0] > 0:
        ind = np.argwhere(maxi != 0).flatten()
        norm = np.zeros((data.shape))
        norm[ind] = (data[ind] - mini[ind][:,None])/(maxi[ind]-mini[ind])[:,None]
    else:
        norm = (data - mini[:,None])/(maxi-mini)[:,None]
    norm[np.isnan(norm)] = 0
    mini[np.isnan(mini)] = 0; mini[np.isnan(mini)] = 0
    if verbose:
        return norm,maxi,mini
    return norm

def scale_back(scale,data):
    if np.argwhere(scale == 0).shape[0] > 0:
        ind = np.argwhere(scale != 0).flatten()
        scale_data = np.zeros((data.shape))
        scale_data[ind] = (scale[ind]/np.sum(data[ind],axis=1))[:,None] * data[ind]
        return scale_data
    return (scale/np.sum(data,axis=1))[:,None]*data


def normalize_single(data,verbose=False,angle=False):
    maxi = np.max(data); mini = np.min(data)
    # if angle:
    #     mini += 1e-20
    if maxi == 0:
        norm = data
    else:
        norm = (data-mini)/(maxi-mini)
    if verbose:
        return norm,maxi,mini
    return norm

def unnormalize(data,maxi,mini):
    if np.argwhere(maxi == 0).shape[0] > 0:
        ind = np.argwhere(maxi != 0).flatten()
        unnorm = np.zeros((data.shape))
        unnorm[ind] = data[ind]*(maxi-mini)[ind][:,None]+mini[ind][:,None]
    else:
        unnorm = data*(maxi-mini)[:,None]+mini[:,None]
    return unnorm

# def unnormalize_single(data,maxi,mini):
#     return data*(maxi-mini)+mini

def unnormalize_single(data,maxi,mini):
    if maxi <= 1e-10:
        return data
    return data*(maxi-mini)+mini

def sizer(dim,original=87**2,half=True):
    ''' Calculates the trainable parametes for autoencoder
    Arguments:
        dim: list of dimension size of each encoding layer
        original: original data size (default is 87**2)
        half: if to take half of the parameters
    Returns:
        integer for number of trainable parameters
    '''
    if original:
        dim.insert(0,original)
    if len(dim) == 2:
        total = sum(dim)
    else:
        total = sum(dim)*2-dim[0]-dim[-1]
    for ii in range(len(dim)-1):
        total += (dim[ii]*dim[ii+1]*2)
    if half:
        return 0.5*total
    return int(total)

def djinn_metric(loc,metric='MSE',score=False,clean=False):
    ''' Gets the Best Metric for DJINN Models - Rough Comparison
    Arguments:
        loc: string location of DJINN model errors
            i.e. y_djinn_errors/model_ntrees*
    Returns None   '''
    address = np.sort(glob.glob(loc))
    error_data = []
    # gets list of all data errors
    for ii in address:
        temp = []
        with open(ii,'r') as f:
            for ii in f.readlines():
                broken = ii.split()
                temp.append(np.array([broken[1],broken[5],broken[-1]],dtype=float))
        error_data.append(np.array(temp))
    # Calculate the least error from MSE, MAE, EVS
    tots = []
    for ii in range(len(address)):
        if metric == 'MSE':
            tots.append(np.sum(np.fabs(np.subtract(error_data[ii],[0,0,1])),axis=0)[0])
        elif metric == 'MAE':
            tots.append(np.sum(np.fabs(np.subtract(error_data[ii],[0,0,1])),axis=0)[1])
        elif metric == 'EVS':
            tots.append(np.sum(np.fabs(np.subtract(error_data[ii],[0,0,1])),axis=0)[2]) #explained variance score
        else:
            print('Combination of MSE, MAE, EVS')
            tots.append(np.sum(np.fabs(np.subtract(error_data[ii],[0,0,1]))))
    tots = np.array(tots)
    if score:
        print(tots[np.argmin(tots)])
    if clean:
        return ''.join(re.findall('\d{3}',address[np.argmin(tots)]))
    return address[np.argmin(tots)]