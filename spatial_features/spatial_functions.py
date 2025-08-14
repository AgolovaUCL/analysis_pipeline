import numpy as np
import astropy.convolution as cnv

def get_ratemaps(spikes, x, y, n: int, binsize = 15, stddev = 5):
    """
    Calculate the rate map for given spikes and positions.

    Args:
        spikes (array): spike train for unit
        x (array): x positions of animal
        y (array): y positions of animal
        n (int): kernel size for convolution
        binsize (int, optional): binning size of x and y data. Defaults to 15.
        stddev (int, optional): gaussian standard deviation. Defaults to 5.

    Returns:
        rmap: 2D array of rate map
        x_edges: edges of x bins
        y_edges: edges of y bins
    """
    x_bins = np.arange(np.nanmin(x), np.nanmax(x) + binsize, binsize)
    y_bins = np.arange(np.nanmin(y), np.nanmax(y)+ binsize, binsize)

    pos_binned, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    
    spikes = [np.int32(el) for el in spikes]
    
    spikes_x = x[spikes]
    spikes_y = y[spikes]
    spikes_binned, _, _ = np.histogram2d(spikes_x, spikes_y, bins=[x_bins, y_bins])
    
    g = cnv.Box2DKernel(n)
    g = cnv.Gaussian2DKernel(stddev, x_size=n, y_size=n)
    g = np.array(g)
    smoothed_spikes =cnv.convolve(spikes_binned, g)
    smoothed_pos = cnv.convolve(pos_binned, g)

    rmap = np.divide(
        smoothed_spikes,
        smoothed_pos,
        out=np.full_like(smoothed_spikes, np.nan),  
        where=smoothed_pos != 0              
    )
    
    return rmap, x_edges, y_edges


