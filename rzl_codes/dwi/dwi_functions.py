import os
import sys
import numpy as np
import math
import cmath
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import rzl_codes.gen_functions as gf
import rzl_codes.radial_functions_functional as rff
# import radial_functions


def recon_radial_dwi(raw_data_path, raw_data_shape, matrix_size,
                     stored_shape=None, stored_padding='fid', offset=0, raw_data_dtype=np.int32,
                     raw_data_dims=('views', 'b_values', 'slices', 'readout', 'imaginary'),
                     interleaved=False, golden_angle=False,
                     ref_slice=None,
                     return_complex=False, normalize=True,
                     ):
    
    """
    Reconstructs radially a radially acquired diffusion-weighted image using the radial_functions class. Requires:
        - raw_data_path
        - xres_ro: number of points stored per view
        - views
        - n_b_values: integer, number of b-values
        - matrix_size: output in-slice resolution (ie. [xres, yres])

        Optionally:
        - slices: defaults to 1
        - ref_slice:
        - interleaved: is slice dimension interleaved? defaults to False
        - golden_angle: defaults to False
        - offset: offset for heading from raw data file
        - raw_data_dtype
    """
    view_dim = raw_data_dims.index('views')
    b_value_dim = raw_data_dims.index('b_values')
    slices_dim = raw_data_dims.index('slices')
    readout_dim = raw_data_dims.index('readout')
    imag_dim = raw_data_dims.index('imaginary')

    n_views = raw_data_shape[view_dim]
    n_b_values = raw_data_shape[b_value_dim]
    n_slices = raw_data_shape[slices_dim]
    xres_ro = raw_data_shape[readout_dim]

    raw_data = read_raw_dwi_data(path=raw_data_path, data_shape=raw_data_shape,
                                 stored_shape=stored_shape, stored_padding=stored_padding,
                                 offset=offset, dtype=raw_data_dtype, dims=raw_data_dims
                                 )
    if interleaved:
        raw_data = gf.un_interleave(raw_data, slice_axis=1)

    if ref_slice is None:
        ref_slice = n_slices//2 * n_b_values
    raw_data = raw_data.reshape(raw_data.shape[0], -1, raw_data.shape[-1])
    dwi = rff.reconstruct_image(raw_data, matrix_size,
                                golden_angle=golden_angle,
                                zerofill_factor=8, ref_slice=ref_slice,
                                return_complex=return_complex)
    dwi = dwi.reshape((n_slices, n_b_values, matrix_size[0], matrix_size[1]))
    if normalize:
        dwi = dwi / np.amax(dwi)

    return dwi

def read_raw_dwi_data(path, data_shape,
                      stored_shape=None, stored_padding='fid', offset=0, dtype=np.int32,
                      dims=('views', 'b_values', 'slices', 'readout', 'imaginary')):

    data_shape = np.array(data_shape, dtype='int')
    if stored_shape is None:
        stored_shape = data_shape
        stored_diff = np.zeros_like(data_shape, dtype='int')
    else:
        stored_shape = np.array(stored_shape, dtype='int')
        stored_diff = stored_shape - data_shape
        if np.any(stored_diff < 0):
            sys.exit("Size of each dimension in stored_shape must be >= to size in data_shape.")

    view_dim = dims.index('views')
    slice_dim = dims.index('slices')
    readout_dim = dims.index('readout')
    imag_dim = dims.index('imaginary')
    b_value_dim = dims.index('b_values')

    raw_data = np.fromfile(path, count=(np.prod(stored_shape)), offset=offset, dtype=dtype)
    raw_data = np.reshape(raw_data, stored_shape)

    if np.any(stored_diff > 0):
        if stored_padding == 'fid':
            # By ChatGPT, creates a slice of shape data_shape
            data_slice = np.s_[tuple(slice(0, dim_end) for dim_end in data_shape)]
            raw_data = raw_data[data_slice]
        elif stored_padding == 'echo':
            print('Not sure if echo unpadding is working; check before use')
            data_slice = np.s_[tuple(slice(store_size//2 - dim_size//2, store_size//2 - dim_size//2 + dim_size) for (store_size, dim_size) in zip(stored_shape, data_shape))]
            raw_data = raw_data[data_slice]

    raw_data = np.transpose(raw_data, (view_dim, slice_dim, b_value_dim, readout_dim, imag_dim))
    raw_data = raw_data[:, :, :, :, 0] + 1j * raw_data[:, :, :, :, 1]

    return raw_data


def recon_dw_sinogram(raw_data_path, xres_ro, views, n_b_values, xres, slices=None, ref_slice=None, interleaved=False,
                     return_complex=False, golden_angle=False, raw_data_dims=('views', 'bvalues', 'slices', 'readout', 'imaginary'),
                     normalize=False, offset=0, raw_data_dtype=np.int32, output_precision='double'):

    if slices is None:
        slices = 1

    total_slices = n_b_values * slices
    rf = radial_functions(xres_ro, views, slices=total_slices, golden_angle=golden_angle)  # Instantiate radial_functions
    raw_data_dims = ('views', 'slices', 'readout', 'imaginary')  # HAVE TO EDIT THIS TO GENERALIZE
    raw_data = rf.read_raw_data(raw_data_path, raw_data_dims=raw_data_dims, offset=offset, raw_data_dtype=raw_data_dtype)
    raw_data = np.reshape(raw_data, (views, n_b_values, slices, xres_ro))
    if interleaved:
        raw_data = gf.un_interleave(raw_data, 2)
    raw_data = np.reshape(raw_data, (views, total_slices, xres_ro))

    if ref_slice is None:
        ref_slice = (slices//2 * n_b_values) + 1
    sinogram = rf.reconstruct_sinogram(raw_data, xres, return_complex=return_complex, ref_slice=ref_slice)
    sinogram = np.reshape(sinogram, (n_b_values, slices, views, xres))
    sinogram = np.transpose(sinogram, (1, 0, 2, 3))  # Into slices, bvalues, views, xres
    if normalize:
        sinogram = sinogram/np.amax(np.absolute(sinogram))

    # Change precision
    if output_precision == 'double':
        if return_complex:
            sinogram = sinogram.astype(np.cdouble)
        else:
            sinogram = sinogram.astype(np.double)
    if output_precision == 'single':
        if return_complex:
            sinogram = sinogram.astype(np.csingle)
        else:
            sinogram = sinogram.astype(np.single)
    if output_precision == 'half':
        if return_complex:
            print("No half precision complex NumPy datatype, using single precision complex instead")
        else:
            sinogram = sinogram.astype(np.half)

    return sinogram

def dw_image_from_sinogram(sinogram, matrix_size, return_complex=False):  # HAVE TO GENERALIZE (INCORPERATE RAW_DATA_DIMS, ETC.)

    n_b_values = sinogram.shape[1]
    views = sinogram.shape[-2]
    xres_ro = sinogram.shape[-1]

    sinogram = sinogram.reshape((-1, views, xres_ro))
    total_slices = sinogram.shape[0]

    rf = radial_functions(xres_ro, views, slices=total_slices)
    dwi = rf.image_from_sinogram(sinogram, matrix_size, return_complex=return_complex)

    dwi = dwi.reshape((-1, n_b_values, matrix_size[0], matrix_size[1]))
    
    return dwi


def get_dwi_metrics(dwi, b_values, axis=1, fitting_scheme='2param', mask=None, bounds=(-np.inf, np.inf), p0=None, error=np.nan, verbose=False):
    """
    Conducts fitting of DW images to obtain metrics such as ADC, KI, etc. Requires:
        - dwi: DWIs where b-values are in dimension "axis"
        - axis: index of b-value dimension
        - bvalues: b-value array
        - fitting_scheme: method to fit DWIs to obtain metrics. Choice from ['2param', '3param', 'biexponential',]
    """
    # Normalize dwi
    dwi = dwi / np.amax(dwi)

    if fitting_scheme == '2param':
        def fitting_func(x, a, b):
            return a * np.exp(-b * x)
        bounds = ([0.001, 0], [1, 5e-3])
        # p0 = (0.5, 2e-3)
        fits_index = ['s0', 'adc']

    elif fitting_scheme == '3param':
        def fitting_func(x, a, b, c):
            return a * np.exp(-b * x) + c
        bounds = ([0.001, 0, 0], [1, 5e-3, 1])
        # p0 = (0.5, 2e-3, 0.1)
        fits_index = ['s0', 'adc', 'bl']

    fits = gf.ls_pixelwise_fit(x=b_values,
                               arr=dwi,
                               axis=axis,
                               mask=mask,
                               func=fitting_func,
                               bounds=bounds,
                               p0=p0,
                               error=error,
                               verbose=verbose
                               )

    return fits, fits_index


def get_diffusion_tensor(dti, b_vectors, b_value, mask=None, error=np.nan):
    """
    Conducts fitting of DW images in different gradient directions to obtain the diffusion tensor. Adapted from
    http://www.diffusion-imaging.com/2014/04/from-diffusion-weighted-images-to.html by Do Tromp. Requires:
    - dwi: DWIs where b-values are second dimension (ie. index 1)
    - b_vectors: b-value arrays
    - b_value: integer, b-value for all gradient directions, must be the same in every direction
    """
    n_slices = dti.shape[0]
    xres = dti.shape[-2]
    yres = dti.shape[-1]

    flat_dti = np.moveaxis(dti, 1, 0).reshape((len(b_vectors), -1))  # Flatten array

    if mask is None:
        mask = np.ones(flat_dti.shape[-1], dtype='bool')
    elif mask.size != flat_dti[-1].size:
        print('Mask shape incorrect, will fit all pixels')
        mask = np.ones(flat_dti.shape[-1], dtype='bool')
    mask = np.reshape(mask, (-1))

    # Create matrix "H"
    # THIS ONLY WORKS FOR 6 DIRECTIONS
    if len(b_vectors) != 7:
        print('THE DTI FUNCTION WILL NOT WORK, USING A SPECIFIC CODE FOR 6 GRADIENT DIRECTIONS PLUS S0')

    H1 = b_vectors[1:, :]**2  # Left half of the H matrix
    H2 = 2 * np.array([b_vectors[1:, 0]*b_vectors[1:, 1],
                       b_vectors[1:, 0]*b_vectors[1:, 2],
                       b_vectors[1:, 1]*b_vectors[1:, 2]]).T  # Right half of the H matrix
    H = np.concatenate([H1, H2], axis=1)

    diffusion_tensors = np.zeros((3, 3, flat_dti.shape[-1]))
    for i in range(flat_dti.shape[-1]):
        if mask[i]:
            Y = np.log(flat_dti[0, i] / flat_dti[1:, i]) / b_value
            d = np.matmul(np.linalg.inv(H), Y)  # 6 value vector, Dxx, Dyy, Dzz, Dxy, Dxz, Dzy
            diffusion_tensors[:, :, i] = [[d[0], d[3], d[4]],
                                          [d[3], d[1], d[5]],
                                          [d[4], d[5], d[2]]]
        else:
            diffusion_tensors[:, :, i] = error
    diffusion_tensors = np.reshape(diffusion_tensors, (3, 3, n_slices, xres, yres))

    return diffusion_tensors


def get_dti_metrics(tensors):

    n_slices = tensors.shape[-3]
    xres = tensors.shape[-2]
    yres = tensors.shape[-1]

    tensors = np.reshape(tensors, (3, 3, -1))
    dti_metrics = np.zeros((4, tensors.shape[-1]))  # Will be FA, MD, AD, RD
    for i in range(tensors.shape[-1]):
        if not np.any(np.isnan(tensors[:, :, i])):
            eig_vals, eig_vects = np.linalg.eig(tensors[:, :, i])
            dti_metrics[0, i] = np.sqrt((eig_vals[0]-eig_vals[1])**2 +
                                        (eig_vals[0]-eig_vals[2])**2 +
                                        (eig_vals[1]-eig_vals[2])**2) / np.linalg.norm(eig_vals) * np.sqrt(1/2)  # FA
            dti_metrics[1, i] = np.sum(eig_vals) / 3  # MD
            dti_metrics[2, i] = np.amax(eig_vals)  # AD
            dti_metrics[3, i] = (np.sum(eig_vals) - np.amax(eig_vals))/2  # RD

        else:
            dti_metrics[:, i] = np.nan

    dti_metrics = np.reshape(dti_metrics, (4, n_slices, xres, yres))

    return dti_metrics

def get_dwi_noise(dwi):
    """
    Get mean noise value for a standard mouse scan
    """
    xres = dwi.shape[2]
    yres = dwi.shape[3]
    sq_size_x = xres//10
    sq_size_y = yres//10

    noise = np.mean(dwi[:, :, -sq_size_x:, yres//2-sq_size_y//2 : yres//2+sq_size_y//2])

    return noise
