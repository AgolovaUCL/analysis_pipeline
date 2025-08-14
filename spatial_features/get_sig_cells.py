import numpy as np
import random
import warnings

def get_sig_cells(spike_train_this_epoch, hd_rad,epoch_start_frame, epoch_end_frame,  occupancy_time, n_bins = 24, frame_rate = 25, num_shifts = 1000):
    """
    Shuffles the data num_shifts time and calculates the MRLs

    Input:
    spike_train_this_epoch: spike data for unit from one epoch
    hd_rad: hd array (for the whole trial and in radians!!!)
    epoch_start_frame: first frame of this epoch
    epoch_end_frame: last frame of this epoch
    occupancy_time: histogram of occupancy per hd bin for this epoch
    num_bins: number of bins for the histogram (default = 24, so 15 degree bins)
    frame_rate: frame rate of video (default = 25)
    num_shifts: number of times data is shuffled. Default is 1000 
    
    """
    # Setting shift values
    shift_min = 2*frame_rate # minimum shift: 2 second
    shift_max = np.int32(epoch_end_frame - epoch_start_frame) - 20*frame_rate # maximum shift: epoch length - 20 s
    
    if shift_min > shift_max:
        shift_max_temp = shift_min
        shift_min = shift_max
        shift_max = shift_max_temp
    if shift_min < 0:
        shift_min = 0

    MRL_values = []


    current_data = spike_train_this_epoch

    for shift_idx in range(num_shifts):
        # Get random shift value
        random_shift = random.randint(shift_min, shift_max)

        # Add or subtract the random_shift value from each element in the variable
        shifted_data = current_data + random_shift

        range_min = np.int32(epoch_start_frame)
        range_max = np.int32(epoch_end_frame)
        range_size = range_max - range_min + 1
        
        # Ensure shifted_data stays within the range [range_min, range_max]
        shifted_data = np.mod(shifted_data - range_min, range_size) + range_min

        # Calculate angles_degrees and MRL
        angles_radians= hd_rad[shifted_data]; 
        mask = ~np.isnan(angles_radians)
        angles_radians= angles_radians[mask]

        counts, bin_edges = np.histogram(angles_radians, bins=n_bins,range = [-np.pi, np.pi] )

        direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        MRL = resultant_vector_length(bin_centers, w = direction_firing_rate)
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