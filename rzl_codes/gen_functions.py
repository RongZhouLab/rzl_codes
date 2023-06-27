import numpy as np
from inspect import signature
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt

def un_interleave(data, slice_axis):
    """
    Un-interleaves data in the defined slice dimension.
        - data: numpy array to be un-interleaved
        - slice_dim: dimension of data to be un-interleaved
    """
    n_slices = data.shape[slice_axis]
    slice_range = [*range(1, n_slices+1)]
    slice_map = [((sl-1)//2) + ((sl-1)%2)*(n_slices+1)//2 for sl in slice_range]

    data = np.moveaxis(data, slice_axis, 0)
    data_new = np.zeros_like(data)
    for isl, sl_idx in enumerate(slice_map):
        data_new[isl] = data[sl_idx]

    data_new = np.moveaxis(data_new, 0, slice_axis)

    return data_new


def read_list(filename, delim):
    """
    Imports list from text file filename where elements are delimited by delim
    Returns list.
    """

    file = open(filename, 'r')
    out_list = file.read().split(delim)

    return out_list


def write_list(l, filename, delim):
    """
    Writes list to text file filename where elements are delimited by delim
    Returns None.
    """

    file = open(filename, 'w')
    for i in l[:-1]:
        file.write(f'{i}{delim}')
    file.write(f'{l[-1]}')
    file.close()

    return


def ls_pixelwise_fit(x, arr, func, axis, mask=None, bounds=(-np.inf, np.inf), p0=None, error=np.nan, verbose=False):
    """
    Conducts least squares pixel-wise fit of an array.
        - x: 1-d numpy array of x-values to fit function to
        - arr: n-d numpy array containing data to fit to function. Dimension "axis" must be of same size as x
        - func: function to fit data to
        - axis: axis along "arr" which must be fit to function "func"
        - bounds, p0: 2-tuples, same as arguments of scipy.optimize.curve_fit
        - error: default values in cases where fit diverges
    Returns:
        - fits: numpy array containing fits. First dimension is size n_params
    """
    # Flatten to make for loop easier
    flat_arr = np.moveaxis(arr, axis, 0).reshape((arr.shape[axis], -1))

    if mask is None:
        mask = np.ones(flat_arr.shape[1], dtype='bool')
    elif mask.size != flat_arr[1].size:
        print('Mask shape incorrect, will fit all pixels')
        mask = np.ones(flat_arr.shape[1], dtype='bool')
    mask = np.reshape(mask, (-1))

    # Obtain dimensions for fits; signature(func).parameters returns an ordered dict of the arguments of function "func"
    nparams = len(signature(func).parameters) - 1

    # Fit data
    fits = np.zeros((nparams, flat_arr.shape[-1]))
    for i in range(flat_arr.shape[-1]):
        if verbose and (i*10//flat_arr.shape[-1] > (i-1)*10//flat_arr.shape[-1]):
            print(f'Fitting {(i*100//flat_arr.shape[-1])}% complete')
        if mask[i]:
            try:
                popt, pcov = curve_fit(f=func, xdata=x, ydata=flat_arr[:, i],
                                       bounds=bounds,
                                       p0=p0,
                                       )
                fits[:, i] = popt
            except RuntimeError:
                fits[:, i] = error
        else:
            fits[:, i] = error

    # Reshape fits to correct shape
    out_shape = list(arr.shape)
    del out_shape[axis]
    out_shape.insert(0, nparams)
    fits = fits.reshape(out_shape)
    
    if verbose:
        print('Fitting completed')

    return fits

