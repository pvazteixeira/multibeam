"""
Useful functions for sonar segmentation.

See also: teixeira2018multibeam (IROS 2018)
"""
import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import entropy, expon, rayleigh

from skimage.io import imread

from __future__ import division # integer division now yields floating-point numbers



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

def background_pmf(x, pi_0, shape, levels=2**8):
    """
    evaluate the background probability mass function

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

def exp_pmf(x, shape, levels=2**8):
    """
    evaluate the background probability mass function

    x - where to evaluate the pmf
    shape - the shape of the exponential pmf component
    levels - the size of the discrete interval (default: 256)
    """
    prob = expon.pdf(x, loc=0, scale=shape) # background
    prob /= np.sum(expon.pdf(np.linspace(0, 1, levels), loc=0, scale=shape))

    return prob


def ray_pmf(x, shape, levels=2**8):
    """
    Evaluate the object probability mass function

    x - where to evaluate the pmf
    shape - the shape of the pmf
    levels - the size of the discrete interval (default: 256)
    """

    prob = rayleigh.pdf(x, 0, shape)
    prob /= np.sum(rayleigh.pdf(np.linspace(0, 1, levels), 0, shape))

    return prob

def object_pmf(x, shape, levels=2**8):
    """
    Evaluate the object probability mass function

    x - where to evaluate the pmf
    shape - the shape of the pmf
    levels - the size of the discrete interval (default: 256)
    """


    return ray_pmf(x, shape, levels)


def rice_pmf(x, dist, shape, levels=2**8):
    """
    Evaluate the object probability mass function

    x - where to evaluate the pmf
    shape - the shape of the pmf
    levels - the size of the discrete interval (default: 256)
    """

    prob = rice.pdf(x, dist, 0, shape)
    prob /= np.sum(rice.pdf(np.linspace(0, 1.0, levels), dist, 0, shape))

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

    x0 = np.linspace(0, 1, levels)

    # Background:  zero-biased exponential
    prob_bg = expon.pdf(x, loc=0, scale=s_bg) # background
    prob_bg /= np.sum(expon.pdf(x0, loc=0, scale=s_bg))

    # Object: Rayleigh
    prob_obj = rayleigh.pdf(x, 0.0, s_obj)
    prob_obj /= np.sum(rayleigh.pdf(x0, loc=0, scale=s_obj))

    # Mixture model
    prob = pi_bg*prob_bg + pi_obj*prob_obj
    # zero-bias
    prob[x < (1.0/levels)] += (1 - pi_bg - pi_obj) # zero-bias

    return prob


# def mixture_pmf2(x, pi_bg, pi_obj, levels=2**8):
#     """
#     Evaluate the mixture model probability mass function with pre-assigned weights:

#     p(x) = (1-pi_bg-pi_obj)*delta(x) +
#            pi_bg*exponential(x,loc=0, scale=0.03) +
#            pi_obj*rice(x,0.2, loc=0.0, scale=0.01)

#     x - where to evaluate the pmf
#     pi_bg - the weight of the exponential component (background pmf)
#     pi_obj - the weight of the rayleigh component (object pmf)
#     levels - the size of the discrete interval (default: 256)
#     """

#     return mixture_pmf(x, pi_bg, pi_obj, 0.04, 0.3, 0.01, levels=levels)

def get_mixture_parameters(ping, p0=[0.3, 0.020, 0.3, 0.01],  levels=2**8):
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
    # params, _ = curve_fit(mixture_pmf, x_vals, p_vals, p0=[0.3, 0.02, 0.02, 0.15])
    params, _ = curve_fit(mixture_pmf, x_vals, p_vals, p0=p0)#, bounds=([0.2, 0.0, 0.01, 1e-2 ],[ 0.9, 0.1, 0.1, 1e0 ]))

    # TODO: check parameter sanity!

    mix = mixture_pmf(x_vals, params[0], params[1], params[2], params[3])

    k = entropy(p_vals, mix)

    return (params, k)

def get_mixture_weights(ping, w0=[0.3,0.01],levels=2**8):
    """
    Compute the weights for the mixture model, using pre-assigned weights.

    """
    bins = np.linspace(0, 1.0, levels+1)
    hist = np.histogram(ping.flatten(), bins)

    x_vals = hist[1][:-1].astype(np.float64)
    p_vals = hist[0][:].astype(np.float64)
    p_vals /= (0.0+np.sum(p_vals))

    weights, _ = curve_fit(mixture_pmf2, x_vals, p_vals, p0=w0)

    mix = mixture_pmf2(x_vals, weights[0], weights[1], levels=levels)

    k = entropy(p_vals, mix) # compute KL divergence

    return (weights, k)


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


def compute_roc(pi1, pi2, s1, s2, levels=2**8):
    """
    """
    k = np.linspace(0, 1.0, levels)
    b_pmf = background_pmf(k, pi1, s1, levels)
    p_fa = 1.0 - np.cumsum(b_pmf)

    o_pmf = object_pmf(k, s2, levels)
    p_d = 1.0 - np.cumsum(o_pmf)

    auc = np.trapz(p_d, p_fa)

    return(p_fa, p_d, auc)


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



def segment_map(x, pi1, pi2, s_1, s_2):
    """
    MAP segmentation (binary local classifier)
    """
    eta = (1-pi2)/pi2

    s = likelihood(x, pi1, pi2, s_1, s_2)
    s[s < eta] = 0
    s[s >= eta] = 1.0

    return s

def segment_ping_threshold(ping, threshold=0.5):
    """
    Fixed-threshold segmentation.
    """
    ping_seg = np.copy(ping)
    ping_seg[ping_seg < threshold] = 0.0
    ping_seg[ping_seg >= threshold] = 1.0

    return ping_seg

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

    return (scan, params)


# def segment_ping_mrf(ping,pi1, pi2, s1, s2):
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

def em(y, theta0, epsilon=1e-2, max_iter = 100, levels=2**8):
    """
    estimate mixture model parameters via the EM algorithm
    """
    theta = np.copy(theta0)
    samples = np.copy(y.flatten())

    samples_nz = np.delete(samples, np.argwhere(samples == 0)) # remove zero-bias
    n_nz = len(samples_nz.flatten())

    bins = np.linspace(0, 1.0, levels+1)
    k = np.linspace(0, 1.0, levels)

    hist_nz = np.histogram(samples_nz, bins)
    yv_nz = hist_nz[1][:-1].astype(np.float64)
    p_nz = hist_nz[0][:].astype(np.float64)
    p_nz /= np.sum(p_nz)

    for i in range(0, max_iter):
        # expectation: compute sample weights
        w_exp = exp_pmf(samples_nz, theta[2])
        w_ray = ray_pmf(samples_nz, theta[3])

        w_norm = w_exp + w_ray
        w_exp = np.divide(w_exp, w_norm)
        w_ray = np.divide(w_ray, w_norm)

        # maximization: update component parameters
        # exponential - MLE
        theta[2] = np.sum(np.multiply(w_exp, samples_nz))/np.sum(w_exp)
        # rayleigh - unbiased MLE
        theta[3] = np.sqrt(np.sum(np.multiply(w_ray, np.power(samples_nz, 2)))/(2*np.sum(w_ray)))

        # maximization: update component weights
        theta[0] = np.sum(w_exp)/n_nz
        theta[1] = 1-theta[0] # theta[1] = np.sum(w_ray)/N
        p_mix = theta[0]*exp_pmf(yv_nz, theta[2]) + theta[1]*ray_pmf(yv_nz, theta[3])
        div = entropy(p_nz[1:], p_mix[1:]) # ignore zero-bias in kld computation

        if div < epsilon:
            break

    # compute zero-bias weight
    hist = np.histogram(samples, bins)
    yv = hist[1][:-1].astype(np.float64)
    p_emp = hist[0][:].astype(np.float64)
    p_emp /= (np.sum(p_emp))

    scale = (1-p_emp[0])/(1-p_mix[0])

    theta[0] = scale*theta[0]
    theta[1] = scale*theta[1]
    p_mix = theta[0]*exp_pmf(yv, theta[2]) + theta[1]*ray_pmf(yv, theta[3])
    p_mix[yv < 1/levels] += (1-theta[0]-theta[1])

    div = entropy(p_emp, p_mix)

    return p_emp, p_mix, theta, div
