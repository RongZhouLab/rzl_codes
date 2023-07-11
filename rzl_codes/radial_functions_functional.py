import os
import sys
import numpy as np
import math
import cmath
from scipy import signal
from scipy.spatial import Voronoi, ConvexHull
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

"""
Python functions for processing + manipulation of radially acquired data. Should be as generally 
applicable as possible.
"""
# TODO: 2D/3D functionality is confusing and might be problematic in the future, esp. with the 'n_slices' variable

def read_raw_data(path, data_shape, stored_shape=None, dims=('views', 'slices', 'readout', 'imaginary'),
                  stored_padding='fid', offset=0, dtype=np.int32):
    """
    Opens 3D radially-acquired k-space data. Outputs in (views, slices, readout) dimension order.
    Inputs:
        - path: path to raw data file
        - data_shape: list or np.array shape (4,), true shape of k-space data
        _______________________________________________________________________________________________________________
        - stored_shape: shape of data in file if saved in shape different from data_shape. defaults to data_shape.
        - dims: order of dimensions in data_shape in order to transpose to code standard. defaults to Bruker default.
        - stored_padding: if stored_shape > data_shape, method by which file pads additional points
        - offset: header size
        - dtype: data type in file

    Outputs:
        - raw_data: complex numpy array shape [views, slices, readout].
    """

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

    raw_data = np.transpose(raw_data, (view_dim, slice_dim, readout_dim, imag_dim))
    raw_data = raw_data[:, :, :, 0] + 1j * raw_data[:, :, :, 1]

    return raw_data


def polar_coords_to_cartesian(polar_coords, angle_units='rad'):
    """
    Returns cartesian_coords, a 2D numpy array shape (total points, 2) where each row [x, y] are the cartesian
    coordinates for the inputted radial coordinates.
    Inputs:
        - polar_coords: 2D numpy array shape (total points, 2) with polar coordinates (radius, theta)
        - angle_units: units for polar coordinates. Either 'rad' for radians or 'deg' for degrees
    Outputs:
        -
    """
    if angle_units == 'rad':
        pass
    elif angle_units == 'deg':
        polar_coords[:, 0] = polar_coords[:, 0] * np.pi / 180
    else:
        sys.exit('angle_units argument must be either "rad" or "deg".')

    cartesian_coords = np.array([polar_coords[:, 0] * np.cos(polar_coords[:, 1]),  # x
                                 polar_coords[:, 0] * np.sin(polar_coords[:, 1])]).T  # y

    return cartesian_coords


def get_voronoi_weights(n_views, xres_ro, n_slices=None, angles=None, golden_angle=False, zerofill_factor=1):
    """
    Returns voronoi weights for radially acquired data for density compensation.
    Inputs:
        - n_views: integer, number of views
        - xres_ro: integer, number of readout points per view
        - n_slices: integer, number of slices. If None, returns 2D data. Default is None.
        - angles: list or np.array containing angles of each view if not sampled using sequential or golden angles
        - golden_angle: boolean, whether golden angle sampling is used. Overrides "angles"
        - zerofill_factor: integer, factor by which readout dimension will increase. Default is 1, or no zerofilling.
    Outputs:
        - weights: np.array shape [views, slices, readout] with density compensation weights.
                    If n_slices = None (default), 2D array shape [views, readout].
    """
    if angles is None:
        angles = get_angle_array(n_views, golden_angle=golden_angle)

    xres_ro = int(xres_ro * zerofill_factor)
    x = np.arange(xres_ro) - xres_ro // 2

    # Create an numpy array shape (points*views, 2) with polar coordinates for sampling scheme
    rad_coords = np.array(np.meshgrid(x, angles)).reshape((2, -1)).T
    cart_coords = polar_coords_to_cartesian(rad_coords, angle_units='rad')

    # Find voronoi weights
    voronoi = Voronoi(cart_coords)
    weights = np.zeros(voronoi.npoints)
    for i, reg_num in enumerate(voronoi.point_region):
        indices = voronoi.regions[reg_num]
        if -1 in indices:  # some regions can't be opened
            weights[i] = np.inf
        else:
            weights[i] = ConvexHull(voronoi.vertices[indices]).volume
    weights = np.reshape(weights, (n_views, xres_ro))

    # need to divide center by number of views (overlapping points) and set the edge weights
    weights[:, (xres_ro // 2)] = weights[:, (xres_ro // 2)] / n_views
    weights[:, 0] = weights[:, 1] + (weights[:, 1] - weights[:, 2])  # use slope to determine end weights
    weights[:, (xres_ro - 1)] = weights[:, (xres_ro - 2)] + (weights[:, (xres_ro - 2)] - weights[:, (xres_ro - 3)])

    if n_slices is not None:
        weights = weights[:, np.newaxis, :].repeat(n_slices, axis=1)  # Convert to same shape as 3D data

    return weights


def get_angle_array(n_views, max_angle=2*np.pi, golden_angle=False):
    """
    Returns a 1D numpy array with the angles, in radians, for views in radial acquisition.
    Inputs:
        n_views: integer, total number of views
        max_angle: total rotation, in radians, if acquired sequentially
        golden_angle: boolean
    Outputs:
        angles: 1D numpy array length n_views where angles[i] is angle in radians of the i'th view
    """
    if golden_angle:
        angles = np.arange(n_views) * np.pi * (math.sqrt(5) - 1) / 2
    else:
        angles = np.arange(n_views)/n_views * max_angle

    return angles


def get_frequency_offset(raw_data, angles=None, golden_angle=False, zerofill_factor=8):
    """
    Finds mean frequency offset using all provided views.
    Inputs:
        raw_data: 2D or 3D numpy array with k-space data in radial coordinates
        angles: angle in radians of each view raw_data[view, :]. If None, uses get_angle_array() function.
        golden_angle: If angles is not defined or is None, golden_angle is passed to get_angle_array() to define angles
        zerofill_factor: Factor by which k-space is extended in order to increase spatial resolution for analysis.
                         NOTE: Different from zero-filling used to extend FOV.
    Outputs:
        offreson: float, off-resonance frequency in readout points.
    """
    n_views = raw_data.shape[0]
    xres_ro = raw_data.shape[-1]
    if len(raw_data.shape) < 3:
        raw_data = raw_data[:, np.newaxis, :]  # Convert to 3D as per code standard

    if angles is None:
        angles = get_angle_array(n_views, golden_angle=golden_angle)

    # Zerofill in k-space
    raw_data = zerofill(raw_data, zerofill_factor=zerofill_factor, axis=-1)

    angles0_idxs = []  # Forward views
    angles180_idxs = []  # Reverse views
    n_view_corrections = n_views // 2 - 5  # must be less than 1/2 total views
    for i_view in range(n_view_corrections):
        angle0 = angles[i_view]
        angle180_diff = angles - np.pi - angle0
        if np.amin(abs(angle180_diff)) < (3*np.pi/n_views):  # Only add to list if difference is smaller than threshold of 1.5x the angle if equally sampled
            angle_180_idx = np.argmin(abs(angle180_diff))
            angles0_idxs.append(i_view)
            angles180_idxs.append(angle_180_idx)

    profiles_0 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(raw_data[angles0_idxs], axes=-1), axis=-1), axes=-1)
    profiles_180 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(raw_data[angles180_idxs], axes=-1), axis=-1), axes=-1)

    sum_profiles_0 = np.sum(abs(profiles_0), axis=(0, 1))
    sum_rev_profiles_180 = np.flipud(np.sum(abs(profiles_180), axis=(0, 1)))

    offsets_to_test = np.arange(-25*zerofill_factor, 25*zerofill_factor)
    profile_diffs = []
    for i_offset in offsets_to_test:
        profile_diffs.append(
            np.sum(abs(sum_profiles_0 - np.roll(sum_rev_profiles_180, i_offset)))
        )

    best_offset_idx = np.argmin(np.array(profile_diffs))
    best_offset = offsets_to_test[best_offset_idx]

    # To match with HK's code: I think the idea is dividing by two because all views have to be corrected the same
    # amount, so you should correct opposing views by half the total offset each. The negative is because of the
    # difference between the np.roll function and the direction of the correction.
    offreson = -best_offset / (2*zerofill_factor)

    return offreson


def get_kspace_centers(raw_data):
    """
    Find index for peak intensity of each view, corresponding to its center of k-space.
    Input:
        raw_data: N-D array of k-space data where last dimension is the readout dimension
    Output:
        (N-1)-D array with index of peak signal in k-space for each view
    """
    return np.argmax(raw_data, axis=-1)


def grid_to_cartesian(raw_data, image_matrix_size, angles=None, golden_angle=False, M=501, L=4):
    """
    Applies re-gridding of radially acquired points to a cartesian grid. Assumes that all the slices inputted
    require the same gridding scheme.
    Inputs:
        - raw_data: 2D or 3D numpy array with radial k-space data with dimensions [views, (slices), readout]
        - image_matrix_size: list length 2 with [xres, yres] for k-space to be re-gridded onto
        - angles: 1D numpy array with angles, in radians, for each view. If None, is compuuted by get_angle_array()
        - golden_angle: boolean, fed to get_angle_array() if angles is None
        - M, L: Kaiser-Bessel kernel parameters
    Outputs:
        - kgrid: 2D or 3D numpy array shape [(slices), yres, xres]
    """
    xres, yres = image_matrix_size

    if len(raw_data.shape) < 3:
        raw_data = raw_data[:, np.newaxis, :]
        n_slices = None
    else:
        n_slices = raw_data.shape[1]  # Value not used

    n_views, _, xres_ro = raw_data.shape

    if angles is None:
        angles = get_angle_array(n_views=n_views, golden_angle=golden_angle)
    voronoi_weights = get_voronoi_weights(n_views=raw_data.shape[0],
                                          xres_ro=raw_data.shape[-1],
                                          n_slices=raw_data.shape[1],
                                          angles=angles)
    raw_data = raw_data * voronoi_weights

    # Get Kaiser-Bessel kernel
    kb_kernel = signal.kaiser(M, np.pi * L / 2)

    # Gridding
    kgrid = np.zeros((raw_data.shape[1], xres, yres), dtype=complex)
    for i_view, angle in enumerate(angles):
        for i_x_ro in range(xres):
            # x, y positions on cartesian grid centered at xres/2, yres/2
            x = (i_x_ro - xres / 2) * math.cos(angle) + xres / 2
            y = (i_x_ro - xres / 2) * math.sin(angle) + xres / 2

            x1 = math.ceil(x - L / 2)  # Range (x1->x2, y1->y2) of points by convolution
            x2 = math.floor(x + L / 2)
            y1 = math.ceil(y - L / 2)
            y2 = math.floor(y + L / 2)

            if x1 < 0:
                x1 = 0
            if x2 > xres - 1:
                x2 = xres - 1
            if y1 < 0:
                y1 = 0
            if y2 > yres - 1:
                y2 = yres - 1

            yy = y1  # Start convolution
            while yy <= y2:
                xx = x1
                ay = round(abs(y - yy) * M / L + M / 2)

                while xx <= x2:
                    ax = round(abs(x - xx) * M / L + M / 2)
                    if ay > M - 1:
                        ay = M - 1
                    if ax > M - 1:
                        ax = M - 1
                    # j = row (view) number, i = column number
                    kgrid[:, yy, xx] += raw_data[i_view, :, i_x_ro] * kb_kernel[ax] * kb_kernel[ay]
                    xx += 1
                yy += 1

    if n_slices is None:
        kgrid = np.squeeze(kgrid)

    return kgrid


def get_gridding_matrix(n_views, xres_ro, image_matrix_size,
                        angles=None, golden_angle=False, M=501, L=4, return_sparse=False):
    """
    Returns gridding matrix shape [views, readout, yres, xres] where the array [view, x_ro, :, :] indicates how
    much the readout point in radial k-space [view, x_ro] contributes to each point in cartesian k-space [y, x],
    including density compensation and convolution.

    Inputs:
        - n_views: integer, number of views
        - xres_ro: integer, number of readout points
        - image_matrix_size: list length 2 with [xres, yres] for k-space to be re-gridded onto
        - angles: 1D numpy array with angles, in radians, for each view. If None, is computed by get_angle_array()
        - golden_angle: boolean, fed to get_angle_array() if angles is None
        - M, L: Kaiser-Bessel kernel parameters
    Outputs:
        - gridding_matrix: 4D numpy array shape [views, xres_ro, yres, xres] to be used with tensordot function to
                           directly obtain cartesian k-space from radial k-space.
    """
    # TODO: Seems like an instance of "grid_to_cartesian", any way to consolidate both?
    xres, yres = image_matrix_size

    if angles is None:
        angles = get_angle_array(n_views=n_views, golden_angle=golden_angle)
    voronoi_weights = get_voronoi_weights(n_views=n_views,
                                          xres_ro=xres_ro,
                                          n_slices=None,
                                          angles=angles)

    # Get Kaiser-Bessel kernel
    kb_kernel = signal.kaiser(M, np.pi * L / 2)

    gridding_matrix = np.zeros((n_views, xres_ro, yres, xres))
    # Gridding
    for i_view, angle in enumerate(angles):
        for i_x_ro in range(xres_ro):
            # x, y positions on cartesian grid centered at xres/2, yres/2
            x = (i_x_ro - xres / 2) * math.cos(angle) + xres / 2
            y = (i_x_ro - xres / 2) * math.sin(angle) + xres / 2

            x1 = math.ceil(x - L / 2)  # Range (x1->x2, y1->y2) of points by convolution
            x2 = math.floor(x + L / 2)
            y1 = math.ceil(y - L / 2)
            y2 = math.floor(y + L / 2)

            if x1 < 0:
                x1 = 0
            if x2 > xres - 1:
                x2 = xres - 1
            if y1 < 0:
                y1 = 0
            if y2 > yres - 1:
                y2 = yres - 1

            yy = y1  # Start convolution
            while yy <= y2:
                xx = x1
                ay = round(abs(y - yy) * M / L + M / 2)

                ele = 0
                while xx <= x2:
                    ax = round(abs(x - xx) * M / L + M / 2)
                    if ay > M - 1:
                        ay = M - 1
                    if ax > M - 1:
                        ax = M - 1

                    # j = row (view) number, i = column number
                    # if return_sparse:
                    #     # if [i_view, i_x_ro, yy, xx] in gridding_matrix[0]:
                    #     #     idx = gridding_matrix[0].index([i_view, i_x_ro, yy, xx])
                    #     #     gridding_matrix[1][idx] += np.squeeze(voronoi_weights)[i_view, i_x_ro] * \
                    #     #                                        kb_kernel[ax] * kb_kernel[ay]
                    #     # else:
                    #     #     gridding_matrix[0].append([i_view, i_x_ro, yy, xx])
                    #     #     gridding_matrix[1].append(np.squeeze(voronoi_weights)[i_view, i_x_ro] * \
                    #     #                                        kb_kernel[ax] * kb_kernel[ay])
                    # else:
                    gridding_matrix[i_view, i_x_ro, yy, xx] += np.squeeze(voronoi_weights)[i_view, i_x_ro] * \
                                                               kb_kernel[ax] * kb_kernel[ay]
                    xx += 1
                yy += 1

    if return_sparse:
        sparse_idxs = np.transpose(np.nonzero(gridding_matrix))
        sparse_vals = np.zeros((sparse_idxs.shape[0]))
        for i in range(sparse_idxs.shape[0]):
            sparse_vals[i] = gridding_matrix[tuple(sparse_idxs[i])]

        output = (sparse_idxs, sparse_vals)

    else:
        output = gridding_matrix

    return output


def zerofill_fov(raw_data, zerofill_factor, axis=-1):
    """
    Zerofills radial k-space data by artificially extending the field of view in image space and FFTing back to k-space.
    Inputs:
        - raw_data: N-D numpy array with radial k-space data
        - zerofill_factor: factor by which readout dimension will be extended by. raw_data.shape[axis] * zerofill_factor
                           must be an integer
        - axis: readout dimension
    Output:
        - raw_data: N-D numpy array with zerofilled k-space data
    """
    xres_ro = raw_data.shape[axis]

    old_shape = raw_data.shape
    new_shape = list(old_shape)
    new_shape[axis] = int(new_shape[axis] * zerofill_factor)

    raw_data_zf = np.zeros(new_shape, dtype=complex)

    # Define numpy slice limits for each of the dimensions
    dim_lims = [[0, i] for i in old_shape]
    dim_lims[axis] = [new_shape[axis] // 2 - xres_ro // 2,
                      new_shape[axis] // 2 - xres_ro // 2 + xres_ro]
    data_slice = np.s_[tuple(slice(dim_lim[0], dim_lim[1]) for dim_lim in dim_lims)]

    # Transform into projection space, adds zeros in edges, transforms back to k-space
    tmp = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(raw_data, axes=axis), axis=axis), axes=axis)
    raw_data_zf[data_slice] = tmp
    raw_data_zf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(raw_data_zf, axes=axis), axis=axis), axes=axis)

    return raw_data_zf


def zerofill(raw_data, zerofill_factor, axis=-1):
    """
    Zerofills k-space in the readout dimension by zerofill_factor, increasing the spatial resolution in image space.
    Inputs:
        - raw_data: N-D numpy array with k-space data
        - zerofill_factor: factor by which readout dimension will be extended by. raw_data.shape[axis] * zerofill_factor
                           must be an integer
        - axis: readout dimension
    Output:
        - raw_data: N-D numpy array with zerofilled k-space data
    """
    xres_ro = raw_data.shape[axis]

    old_shape = raw_data.shape
    new_shape = list(old_shape)
    new_shape[axis] = new_shape[axis] * zerofill_factor

    # Define numpy slice limits for each of the dimensions
    dim_lims = [[0, i] for i in old_shape]
    dim_lims[axis] = [new_shape[axis] // 2 - xres_ro // 2, new_shape[axis] // 2 + xres_ro // 2]
    data_slice = np.s_[tuple(slice(dim_lim[0], dim_lim[1]) for dim_lim in dim_lims)]

    raw_data_zf = np.zeros(new_shape, dtype=complex)
    raw_data_zf[data_slice] = raw_data

    return raw_data_zf

def deapodization_filter(matrix_size, L=4):
    """
    Finds deapodizing filter from L and k-space resolutions.
    Inputs:
        - matrix_size: list [xres, yres] with dimensions of cartesian space that radial data was gridded onto
        - L: convolution filter parameter
    Output:
        - d_filter: deapodization filter as a numpy array shape [xres, yres]
    """
    xres, yres = matrix_size
    beta = np.pi * L / 2

    filterx = np.arange(1.0 * xres)
    for i in range(0, xres):
        d2 = (np.pi * L * (i - xres / 2) / xres) ** 2 - beta ** 2
        if d2 > 0.0:
            d2 = math.sqrt(d2)
            filterx[i] = math.sin(d2) / d2
        elif d2 < 0.0:
            d2 = complex(0, math.sqrt(-d2))
            filterx[i] = abs(cmath.sin(d2) / d2)
        else:
            filterx[i] = 1

    filtery = np.arange(1.0 * yres)
    for i in range(0, yres):
        d2 = (np.pi * L * (i - yres / 2) / yres) ** 2 - beta ** 2
        if d2 > 0.0:
            d2 = math.sqrt(d2)
            filtery[i] = math.sin(d2) / d2
        elif d2 < 0.0:
            d2 = complex(0, math.sqrt(-d2))
            filtery[i] = abs(cmath.sin(d2) / d2)
        else:
            filtery[i] = 1

    d_filter = np.zeros((yres, xres))
    for i in range(0, xres):
        for j in range(0, yres):
            d_filter[j][i] = filterx[i] * filtery[j]

    return d_filter


def do_kspace_corrections(raw_data, quick_correct=True, ref_slice=None):
    """
    Shifts views to the center of k-space, applies offset frequency correction, applies phase normalization.
    If quick_correct is False:
        - views are shifted to the center of k-space individually
        - offset frequency is computed from half of the total number of views (a single value is still used for
          every view)
    If quick_correct is True: (default, about x quicker, better for low SNR images)
        - average k-space peak location is computed from a single slice (ref_slice)
        - offset frequency is computed from a single slice (ref_slice)
        - both are used for corrections of every view
        - if ref_slice not defined, defaults to center slice

    Inputs:
        - raw_data: 2D or 3D numpy array with radially acquired k-space data
        - quick_correct: boolean, see above
        - ref_slice: boolean, see above (only used if quick_correct is set to True)
    Outputs:
        - raw_data: corrected radial k-space data with same shape as input
    """

    n_views = raw_data.shape[0]
    xres_ro = raw_data.shape[-1]

    if len(raw_data.shape) < 3:
        raw_data = raw_data[:, np.newaxis, :]
        n_slices = None  # Confusing variable name in this function, but using for consistency
    else:
        n_slices = raw_data.shape[1]  # Actual value not used

    if quick_correct:
        if ref_slice is None:  # use center slice (not ideal if actually 4D image, like DWI)
            ref_slice = raw_data.shape[1] // 2
        ref_raw_data = np.expand_dims(raw_data[:, ref_slice], axis=1)  # Preserve slice dimension
        frequency_offset = get_frequency_offset(ref_raw_data)
        kspace_peak_idxs = np.ones((raw_data.shape[:2])) * np.mean(
            get_kspace_centers(ref_raw_data))  # Same for all views
    else:
        frequency_offset = get_frequency_offset(raw_data)
        kspace_peak_idxs = get_kspace_centers(raw_data)  # Size [views, slices]

    # Apply corrections
    N_arr = (np.arange(xres_ro) - xres_ro / 2) * 2*np.pi/xres_ro

    # Move views to k-space center
    linphases = np.expand_dims((kspace_peak_idxs - xres_ro / 2), axis=2) * np.expand_dims(N_arr, axis=(0, 1))
    raw_data = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(raw_data, axes=2), axis=2), axes=2)
    raw_data = raw_data * (np.cos(linphases) + 1j * np.sin(linphases))
    raw_data = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(raw_data, axes=2), axis=2), axes=2)

    # Phase normalize
    datphase = np.angle(raw_data[:, :, xres_ro // 2])
    raw_data = raw_data * np.expand_dims(np.cos(datphase) - 1j * np.sin(datphase), axis=2)

    # Correct for offset frequency
    linphases = np.ones_like(raw_data, dtype=np.float64) * frequency_offset * np.expand_dims(N_arr, axis=(0, 1))
    raw_data = raw_data * (np.cos(linphases) + 1j * np.sin(linphases))

    if n_slices is None:
        raw_data = np.squeeze(raw_data)

    return raw_data


def reconstruct_image(raw_data, matrix_size,
                      angles=None, golden_angle=False,  # Acquisition parameters
                      zerofill_factor=1, do_corrections=True, quick_correct=True, ref_slice=None,  # Correction params
                      gridding_matrix=None, M=501, L=4,  # Reconstruction parameters
                      return_complex=False, return_cropped=True,  # Return parameters
                      ):
    """
    Reconstructs 2D or 3D image from radially acquired k-space data. Corrects k-space data prior to reconstruction by
    shifting views to the center of k-space, phase normalizing, and correcting for offset frequency of opposite views.
    Inputs:
        - raw_data: 2D or 3D numpy array with radially acquired k-space data
        - matrix_size: [xres, yres] of reconstructed image
        - angles: 1D numpy array with angles of each view in radians. If None, assigned using get_angle_array()
        - golden_angle: boolean, fed to get_angle_array() if "angles" is None
        - zerofill_factor: factor by which FOV is increased (1: no zerofilling, 2: doubling, etc.)
        - quick_correct: see do_kspace_corrections() documentation
        - ref_slice: integer, reference slice used to compute k-space corrections for all slices.
                     see do_kspace_corrections() documentation for specifics.
        - M, L: Kaiser-Bessel convolution kernel parameters
        - return_complex: boolean, whether the complex image is returned
        - return_cropped: boolean. If k-space is zerofilled prior to reconstruction to increase FOV, whether image is
                          cropped to original FOV prior to output
    Outputs:
        - image: 2D or 3D numpy array shape [(slices), yres, xres]. If zerofill_factor > 1 and return_cropped = False,
                 shape is [(slices), yres*zerofill_factor, xres*zerofill_factor].
    """

    xres, yres = matrix_size

    if len(raw_data.shape) < 3:
        raw_data = raw_data[:, np.newaxis, :]
        n_slices = None
    else:
        n_slices = raw_data.shape[1]  # Value not used

    # k-space corrections and gridding to cartesian k-space
    if do_corrections:
        raw_data = do_kspace_corrections(raw_data, quick_correct=quick_correct, ref_slice=ref_slice)

    zf_bool = False
    if zerofill_factor > 1:
        zf_bool = True
        raw_data = zerofill_fov(raw_data, zerofill_factor)
        matrix_size = [int(i*zerofill_factor) for i in matrix_size]

    if gridding_matrix is not None:
        if gridding_matrix.shape[-4:] != (raw_data.shape[0], raw_data.shape[-1], matrix_size[0], matrix_size[1]):
            print(f'Gridding matrix shape invalid for k-space shape: expecting ({raw_data.shape[0]}, {raw_data.shape[-1]}, {matrix_size[0]}, {matrix_size[1]}). Got {gridding_matrix.shape}.')
            gridding_matrix = None
        else:
            kgrid = np.tensordot(raw_data, gridding_matrix, axes=((0, -1), (0, 1))).reshape((-1, matrix_size[0], matrix_size[1]))

    if gridding_matrix is None:
        kgrid = grid_to_cartesian(raw_data, matrix_size,
                              angles=angles, golden_angle=golden_angle, M=M, L=L)  # Returns in [slices, xres, yres]

    image = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(kgrid, axes=(1, 2))), axes=(1, 2))

    # If zerofilled and should return cropped image, do cropping
    if zf_bool:
        if return_cropped:
            image = image[:,
                            (image.shape[1]//2 - xres//2): (image.shape[1]//2 - xres//2 + xres),
                            (image.shape[2]//2 - yres//2): (image.shape[2]//2 - yres//2 + yres)]

    if not return_complex:
        image = np.absolute(image)

    if n_slices is None:
        image = np.squeeze(image)

    return image


def reconstruct_sinogram(raw_data, do_corrections=True, quick_correct=True, ref_slice=None, return_complex=False):
    """
    Reconstructs sinogram using offset frequency corrections and k-space center data from reference slice only.
    If no reference slice ref_slice is defined, uses center slice.
    Inputs:
        - raw_data: 2D or 3D numpy array with radial k-space data
        - return_complex: boolean, whether output is in complex space
        - quick_correct: boolean, whether only ref_slice is used for k-space correction
        - ref_slice: integer, if quick_correct = True, which slice is used for k-space correction
    Outputs:
        - sinogram: 2D or 3D numpy array shape [(slices), views, readout]
    """
    if len(raw_data.shape) < 3:
        raw_data = raw_data[:, np.newaxis, :]
        n_slices = None
    else:
        n_slices = raw_data.shape[1]

    if do_corrections:
        raw_data = do_kspace_corrections(raw_data,
                                         quick_correct=quick_correct,
                                         ref_slice=ref_slice)

    sinogram = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(raw_data, axes=-1), axis=-1), axes=-1)
    sinogram = np.transpose(sinogram, (1, 0, 2))  # Into [slices, views, xres]
    if not return_complex:
        sinogram = np.absolute(sinogram)

    if n_slices is None:
        sinogram = np.squeeze(sinogram)

    return sinogram


def image_from_sinogram(sinogram, matrix_size,
                        force_kspace_corrections=False, zerofill_factor=1, quick_correct=True, ref_slice=None,
                        gridding_matrix=None, M=501, L=4,  # Reconstruction parameters
                        return_complex=False, return_cropped=True  # Return parameters
                        ):
    """
    Reconstructs 2D or 3D image from sinogram.
    Inputs:
        - raw_data: 2D or 3D numpy array with radially acquired k-space data
        - matrix_size: [xres, yres] of reconstructed image
        - angles: 1D numpy array with angles of each view in radians. If None, assigned using get_angle_array()
        - golden_angle: boolean, fed to get_angle_array() if "angles" is None
        - zerofill_factor: factor by which FOV is increased (1: no zerofilling, 2: doubling, etc.)
        - quick_correct: see do_kspace_corrections() documentation
        - ref_slice: integer, reference slice used to compute k-space corrections for all slices.
                     see do_kspace_corrections() documentation for specifics.
        - gridding_matrix: gridding numpy array in the format outputted from get_gridding_array(). If none is provided
                           or shape is invalid, it is not used.
        - M, L: Kaiser-Bessel convolution kernel parameters
        - return_complex: boolean, whether the complex image is returned
        - return_cropped: boolean. If k-space is zerofilled prior to reconstruction to increase FOV, whether image is
                          cropped to original FOV prior to output
    Outputs:
        - image: 2D or 3D numpy array shape [(slices), yres, xres]. If zerofill_factor > 1 and return_cropped = False,
                 shape [(slices), yres*zerofill_factor, xres*zerofill_factor].
    """
    xres, yres = matrix_size

    if len(sinogram.shape) < 3:
        sinogram = sinogram[np.newaxis, :, :]
        n_slices = None
    else:
        n_slices = sinogram.shape[0]  # Value not used

    sinogram = np.transpose(sinogram, (1, 0, 2))  # Re-order to [views, slices, xres] as per code standard
    raw_data = np.fft.fftshift(np.fft.fft(np.fft.fftshift(sinogram, axes=-1), axis=-1), axes=-1)

    image = reconstruct_image(
        raw_data=raw_data,
        matrix_size=matrix_size,
        angles=None,
        zerofill_factor=zerofill_factor,
        do_corrections=force_kspace_corrections, quick_correct=quick_correct, ref_slice=ref_slice,
        gridding_matrix=gridding_matrix, M=M, L=L,
        return_complex=return_complex,
        return_cropped=return_cropped
    )

    if n_slices is None:
        image = np.squeeze(image)

    return image


def corrected_cartesian_data(raw_data, matrix_size, angles=None, golden_angle=False,
                             quick_correct=True, M=501, L=4, ref_slice=None):

    raw_data = do_kspace_corrections(raw_data, quick_correct=quick_correct, ref_slice=ref_slice)
    kgrid = grid_to_cartesian(raw_data, matrix_size,
                              angles=angles, golden_angle=golden_angle, M=M, L=L)  # Returns in [slices, xres, yres]

    return kgrid

