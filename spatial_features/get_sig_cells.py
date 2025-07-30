import numpy as np
import random
import warnings

def get_sig_cells(spike_train_this_epoch, hd, epoch_dur = 1, frame_rate = 30):
    num_shifts = 10000
    shift_min = 2*30
    shift_max = len(spike_train_this_epoch) - 20*30
    
    MRL_values = []


    current_data = spike_train_this_epoch
    current_angles = hd

    if np.nanmin(hd) < -np.pi or np.nanmax(hd) > np.pi:
        hd_rad = np.deg2rad(hd)
    else:
        hd_rad = hd
    occupancy_counts, _ = np.histogram(hd_rad, bins=72, range = [-np.pi, np.pi])
    occupancy_time = occupancy_counts / frame_rate 


    for shift_idx in range(num_shifts):
        # Get random shift value
        random_shift = random.randint(shift_min, shift_max)

        # Add or subtract the random_shift value from each element in the variable
        shifted_data = current_data + random_shift

        range_min = np.nanmin(current_data)
        range_max = np.nanmax(current_data)
        range_size = range_max - range_min + 1
        
        # Ensure shifted_data stays within the range [range_min, range_max]
        shifted_data = np.mod(shifted_data - range_min, range_size) + range_min

        # Calculate angles_degrees and MRL
        angles_radians= hd_rad[shifted_data]; 
        mask = ~np.isnan(angles_radians)
        angles_radians= angles_radians[mask]
        counts, bin_edges = np.histogram(angles_radians, bins=72,range = [-np.pi, np.pi] )
        bin_idx = np.digitize(angles_radians, bin_edges) - 1  # zero-based index for Python

        direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, np.nan, dtype=float), where=occupancy_time!=0)

        W = direction_firing_rate[bin_idx]

        MRL = resultant_vector_length(angles_radians, w=W)
        MRL_values.append( MRL)

    perc_95_val = np.percentile(MRL_values, 95)
    perc_99_val = np.percentile(MRL_values, 99)
    return perc_95_val, perc_99_val, MRL_values


def resultant_vector_length(alpha, w=None, d=None, axis=None,
                            axial_correction=1, ci=None, bootstrap_iter=None):
    """
    Copied from Pycircstat documentation
    Computes mean resultant vector length for circular data.

    This statistic is sometimes also called vector strength.

    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
               default None.
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
                          (number of samples if None)
    :return: mean resultant length

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()

    cmean = _complex_mean(alpha, w=w, axis=axis,
                          axial_correction=axial_correction)

    # obtain length
    r = np.abs(cmean)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    return r


def _complex_mean(alpha, w=None, axis=None, axial_correction=1):
    # Copied from picircstat documentation
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)

    assert w.shape == alpha.shape, "Dimensions of data " + str(alpha.shape) \
                                   + " and w " + \
        str(w.shape) + " do not match!"

    return ((w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) /
            np.sum(w, axis=axis))