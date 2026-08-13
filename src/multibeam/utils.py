"""
Useful functions for sonar segmentation.

See also: teixeira2018multibeam (IROS 2018)
"""

import numpy as np

from scipy.optimize import curve_fit, minimize
from scipy.stats import entropy, expon, rayleigh
from skimage.io import imread


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

def mixture_pmf(x, pi_bg, pi_obj, s_bg, s_obj, levels=2**8):
    """Evaluate the mixture model probability mass function:

    (1-pi_bg-pi_obj)*delta(x) + pi_bg*exponential(x,s_bg) + pi_obj*rayleigh(x,s_obj)

    x - where to evaluate the pmf
    pi_bg - the weight of the exponential component (background pmf)
    pi_obj - the weight of the rayleigh component (object pmf)
    s_bg - the shape of the exponential component (background pmf)
    s_obj - the shape of the rayleigh component (object pmf)
    levels - the size of the discrete interval (default: 256)
    """

    x0 = np.linspace(0, 1, levels)

    # Background:  zero-biased exponential
    prob_bg = expon.pdf(x, loc=0, scale=s_bg) # background
    prob_bg /= np.sum(expon.pdf(x0, loc=0, scale=s_bg))

    # Object: Rayleigh
    prob_obj = rayleigh.pdf(x, 0, s_obj)
    prob_obj /= np.sum(rayleigh.pdf(x0, loc=0, scale=s_obj))

    # Mixture model
    prob = pi_bg*prob_bg + pi_obj*prob_obj
    # zero-bias
    prob[x < (1.0/levels)] += (1 - pi_bg - pi_obj) # zero-bias

    return prob


def obj_fcn(theta, y, p_y):
    p_mix = mixture_pmf(y, theta[0],theta[1],theta[2],theta[3])
    return entropy(p_y, p_mix)

def get_mixture(y, theta0=[0.3, 0.01, 0.03, 0.2], levels=2**8):
    """Estimate mixture model parameters through least squares.

    :param y:
    :param theta0:
    :param levels:
    """

    theta = np.copy(theta0)
    samples = np.copy(y.flatten())

    bins = np.linspace(0, 1.0, levels+1)

    hist = np.histogram(samples, bins)
    yv = hist[1][:-1].astype(np.float64)
    p_emp = hist[0][:].astype(np.float64)
    p_emp /= (np.sum(p_emp))

    result = minimize(obj_fcn, theta0, (yv, p_emp))
    theta = result.x

    p_mix = mixture_pmf(yv, theta[0], theta[1], theta[2], theta[3])

    div = entropy(p_emp, p_mix)

    return p_emp, p_mix, theta, div

def get_mixture_parameters(ping, p0=[0.32, 0.01, 0.03, 0.2],  levels=2**8):
    """Compute the mixture model parameters (DEPRECATED).

    Keyword arguments
    ping - the sonar image (0-1 range)

    Output
    (pi1, pi2, p1, s2)
    """
    bins = np.linspace(0, 1.0, levels+1)
    hist = np.histogram(ping.flatten(), bins)

    x_vals = hist[1][:-1].astype(np.float64)
    p_emp = hist[0][:].astype(np.float64)
    p_emp /= np.sum(p_emp)

    # curve_fit(fcn, xdata, ydata, params)
    # params, _ = curve_fit(mixture_pmf, x_vals, p_emp, p0=[0.3, 0.02, 0.02, 0.15])
    params, _ = curve_fit(mixture_pmf, x_vals, p_emp, p0=p0, bounds=([0.2, 0.0, 0.01, 1e-2 ], [ 0.5, 0.1, 0.05, 1e0 ]))

    # TODO: check parameter sanity!

    mix = mixture_pmf(x_vals, params[0], params[1], params[2], params[3])

    k = entropy(p_emp, mix)

    return (params, k)


# def mixture_pmf(x, pi_bg, pi_obj, s_bg, s_obj, levels=2**8):
#     """
#     Evaluate the mixture model probability mass function:

#     (1-pi_bg-pi_obj)*delta(x) + pi_bg*exponential(x,s_bg) + pi_obj*rayleigh(x,s_obj)

#     x - where to evaluate the pmf
#     pi_bg - the weight of the exponential component (background pmf)
#     pi_obj - the weight of the rayleigh component (object pmf)
#     s_bg - the shape of the exponential component (background pmf)
#     s_obj - the shape of the rayleigh component (object pmf)
#     levels - the size of the discrete interval (default: 256)
#     """
#     # background
#     prob_bg = expon.pdf(x, loc=0, scale=s_bg) # background
#     prob_bg /= np.sum(expon.pdf(np.linspace(0, 1, levels), loc=0, scale=s_bg))

#     # object
#     prob_obj = rayleigh.pdf(x, 0, s_obj)
#     prob_obj /= np.sum(rayleigh.pdf(np.linspace(0, 1, levels), loc=0, scale=s_obj))

#     # print 'expon:', np.sum(prob_bg), 'rayleigh:', np.sum(p2)

#     # mixture
#     prob = pi_bg*prob_bg + pi_obj*prob_obj
#     prob[x < (1.0/levels)] += (1 - pi_bg - pi_obj) # zero-bias

#     return prob

# def get_mixture_parameters(ping, p0=[0.3, 0.020, 0.034, 0.15],  levels=2**8):
#     """
#     Computes the mixture model parameters.

#     Keyword arguments
#     ping - the sonar image (0-1 range)

#     Output
#     (pi1, pi2, p1, s2)
#     """
#     bins = np.linspace(0, 1.0, levels+1)
#     hist = np.histogram(ping.flatten(), bins)

#     x_vals = hist[1][:-1].astype(np.float64)
#     p_vals = hist[0][:].astype(np.float64)
#     p_vals /= (0.0+np.sum(p_vals))

#     # curve_fit(fcn, xdata, ydata, params)
#     # params, _ = curve_fit(mixture_pmf, x_vals, p_vals, p0=[0.3, 0.02, 0.02, 0.15])
#     params, _ = curve_fit(mixture_pmf, x_vals, p_vals, p0=p0, bounds=([0.2, 0.0, 0.01, 1e-2 ],[ 0.9, 0.1, 0.1, 1e0 ]))

#     # TODO: check parameter sanity!

#     mix = mixture_pmf(x_vals, params[0], params[1], params[2], params[3])

#     k = kld(p_vals, mix)

#     return (params, k)


def likelihood(x, pi1, pi2, s_1, s_2, levels=2**8):
    """
    evaluate likelihood 

    TODO: replace w/ calls to object_pmf and background_pmf
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



def segment_map(ping, pi1, pi2, s_1, s_2):
    """Run MAP segmentation (binary local classifier).

    :param ping:
    :param pi1:
    :param pi2:
    :param s_1:
    :param s_2:
    """
    eta = (1 - pi2) / pi2

    s = likelihood(ping, pi1, pi2, s_1, s_2)
    s[s < eta] = 0
    s[s >= eta] = 1.0

    return s

def segment_ping_threshold(ping, threshold: float=0.5):
    """Perform fixed-threshold segmentation.

    :param ping:
    :param threshold:
    """
    ping_seg = np.copy(ping)
    ping_seg[ping_seg < threshold] = 0
    ping_seg[ping_seg >= threshold] = 1

    return ping_seg

def segment_ping_map(ping) -> tuple:
    """Perform MAP segmentation of a sonar scan.

    :param ping:

    This function computes the mixture model for the ping and then uses it to compute the MAP
    segmentation
    """
    # 1) extract model parameters
    # params, _ = get_mixture_parameters(ping)
    _, _, theta, _ = get_mixture(ping)

    # 2) segment
    scan = segment_map(ping, theta[0], theta[1], theta[2], theta[3])

    return (scan, theta)

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
    ping2[ping2 < p_th] = 0
    return ping2
