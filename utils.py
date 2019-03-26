"""
Useful functions for sonar segmentation.

See also: teixeira2018multibeam (IROS 2018)
"""
import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import expon, rayleigh

from skimage.io import imread

"""
background pmf : (1-pi)*delta(x) + pi*expon.pdf(x,0,s)
object pmf :     rayleigh.pdf(x,0,s)

note: normalization missing from the above equations
"""

def compile(image_list, cfg_list, sonar, enhance=False):
    """
    Horizontally stack a set of images (512 rows by (Nx96) columns)
    """
    num_scans = len(image_list)
    data = imread(image_list[0], as_grey=True).astype(np.float64) # 0-255.0
    for i in range(1, num_scans):
        ping = imread(image_list[i], as_grey=True).astype(np.float64) # 0-255.0
        if enhance:
            sonar.load_config(cfg_list[i])
            ping = sonar.deconvolve(ping)
#             ping = sonar.removeTaper(ping) # TODO: enable removeTaper
        data = np.hstack((data, ping))

    data /= 255.0 # normalize to 0-1.0 range

    return data

def kld(sample, reference):
    """
    Compute KL divergence for two probability mass functions
    """
    p_pmf = sample
    q_pmf = reference
    q_pmf[np.where(q_pmf == 0)[0]] = 1 # we're dividing by q, so set all zero-values to 1
    r_pmf = p_pmf/q_pmf
    r_pmf[np.where(r_pmf == 0)[0]] = 1.0 # ln(1) = 0; ln(0)=nan

    return np.sum(p_pmf*np.log(r_pmf))

def background_pmf(x, pi_0, shape, levels=2**8):
    """
    Evaluate the background probability mass function

    x - where to evaluate the pmf
    pi_0 - the weight of the zero-bias pmf component
    shape - the shape of the exponential pmf component
    levels - the size of the discrete interval (default: 256)
    """
    prob = expon.pdf(x, loc=0, scale=shape) # background
    prob /= np.sum(expon.pdf(np.linspace(0, 1, levels), loc=0, scale=shape))
    prob *= pi_0
    # check if scalar or array
    if isinstance(prob, np.ndarray):
        prob[np.argwhere(x <= (1.0/levels))] += (1-pi_0)
    else:
        if x < (1.0/levels):
            prob += (1-pi_0)

    return prob

def object_pmf(x, shape, levels=2**8):
    """
    Evaluate the object probability mass function

    x - where to evaluate the pmf
    shape - the shape of the pmf
    levels - the size of the discrete interval (default: 256)
    """

    prob = rayleigh.pdf(x, 0, shape)
    prob /= np.sum(rayleigh.pdf(np.linspace(0, 1, levels), 0, shape))

    return prob

def mixture_pmf(x, pi_bg, pi_obj, s_bg, s_obj, levels=2**8):
    """
    Evaluate the mixture model probability mass function:

    (1-pi_bg-pi_obj)*delta(x) + pi_bg*exponential(x,s_bg) + pi_obj*rayleigh(x,s_obj)

    x - where to evaluate the pmf
    pi_bg - the weight of the exponential component (background pmf)
    pi_obj - the weight of the rayleigh component (object pmf)
    s_bg - the shape of the exponential component (background pmf)
    s_obj - the shape of the rayleigh component (object pmf)
    levels - the size of the discrete interval (default: 256)
    """
    # background
    prob_bg = expon.pdf(x, loc=0, scale=s_bg) # background
    prob_bg /= np.sum(expon.pdf(np.linspace(0, 1, levels), loc=0, scale=s_bg))

    # object
    prob_obj = rayleigh.pdf(x, 0, s_obj)
    prob_obj /= np.sum(rayleigh.pdf(np.linspace(0, 1, levels), loc=0, scale=s_obj))

    # print 'expon:', np.sum(prob_bg), 'rayleigh:', np.sum(p2)

    # mixture
    prob = pi_bg*prob_bg + pi_obj*prob_obj
    prob[x < (1.0/levels)] += (1 - pi_bg - pi_obj) # zero-bias

    return prob

def get_mixture_parameters(ping, levels=2**8):
    """
    Computes the mixture model parameters.

    Keyword arguments
    ping - the sonar image (0-1 range)

    Output
    (pi1, pi2, p1, s2)
    """
    bins = np.linspace(0, 1.0, levels+1)
    hist = np.histogram(ping.flatten(), bins)

    x_vals = hist[1][:-1].astype(np.float64)
    p_vals = hist[0][:].astype(np.float64)
    p_vals /= (0.0+np.sum(p_vals))

    # curve_fit(fcn, xdata, ydata, params)
    params, _ = curve_fit(mixture_pmf, x_vals, p_vals, p0=[0.3, 0.02, 0.02, 0.15])

    # TODO: check parameter sanity!

    mix = mixture_pmf(x_vals, params[0], params[1], params[2], params[3])

    k = kld(p_vals, mix)

    return (params, k)

def likelihood(x, pi1, pi2, s_1, s_2, levels=2**8):
    """

    """
    pi0 = (1 - pi1 - pi2)
    pi0 /= (1-pi2)

    num = rayleigh.pdf(x, loc=0, scale=s_2)
    num /= np.sum(rayleigh.pdf(np.linspace(0, 1.0, levels), loc=0, scale=s_2))

    den = expon.pdf(x, loc=0, scale=s_1)
    den /= np.sum(expon.pdf(np.linspace(0, 1.0, levels), loc=0, scale=s_1))
    den *= (1-pi0)
    den[x < 1.0/levels] += pi0

    return num/den


# def segment_np(x,pi1,pi2,s1,s2, p_fa=1e-3):
#     """
#     Neyman-Pearson segmentation.
#     """

#     return s


def segment_map(x, pi1, pi2, s_1, s_2):
    """
    MAP segmentation (binary local classifier)
    """
    eta = (1-pi2)/pi2

    s = likelihood(x, pi1, pi2, s_1, s_2)
    s[s < eta] = 0
    s[s >= eta] = 1.0

    return s

def segment_ping_map(ping):
    """
    MAP segmentation of a sonar scan.

    This function computes the mixture model for the ping and then uses it to compute the MAP
    segmentation
    """
    # 1) extract model parameters
    params, _ = get_mixture_parameters(ping)

    # 2) segment
    scan = segment_map(ping, params[0], params[1], params[2], params[3])

    return scan


# def segment_mrf(x,pi1, pi2, s1, s2):
#     """
#     MRF segmentation
#     """
#     s = np.zeros_like(x)
#     # NOT IMPLEMENTED
#     return s


def extract_max(ping, ping_binary, min_range, bin_length):
    """
    Extract the strongest return (per-beam) from a segmented image.
    """
    pping = np.copy(ping)
    pping[ping_binary <= 0] = 0
    intensities = np.amax(pping, axis=0)
    ranges = np.argmax(pping, axis=0)
    ranges = ranges*bin_length
    ranges[ranges <= 0] = -min_range
    ranges += min_range*(np.ones_like(ranges))

    return (ranges, intensities)


# def extract_first(x, b, min_range, bin_length):
#     """
#     Extract the first return (per-beam) from a segmented image.
#     UNIMPLEMENTED
#     """

#     ping = np.copy(x)
#     ping[b <= 0] = 0
#     intensities = np.amax(ping, axis=0)
#     ranges = np.argmax(ping, axis=0)
#     ranges = ranges*bin_length
#     ranges[ranges <= 0] = -min_range
#     ranges += min_range(np.ones_like(ranges))

#     return (ranges, intensities)


def remove_percentile(ping, percentile=99.0):
    """
    Set all pixels below the specified quantile to 0.
    """
    ping2 = np.copy(ping)
    p_th = np.percentile(ping[:], percentile)
    ping2[ping2 < p_th] = 0.0
    return ping2
