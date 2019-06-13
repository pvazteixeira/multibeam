"""
Useful functions for sonar segmentation.

See also: teixeira2018multibeam (IROS 2018)
"""
from __future__ import division # integer division now yields floating-point numbers

import numpy as np

from scipy.optimize import curve_fit, minimize
from scipy.signal import fftconvolve
from scipy.stats import entropy, expon, norm, rayleigh, rice

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

def background_pmf(x, pi_0, shape, levels=2**8):
    """
    evaluate the background probability mass function

    x - where to evaluate the pmf
    pi_0 - the weight of the zero-bias pmf component
    shape - the shape of the exponential pmf component
    levels - the size of the discrete interval (default: 256)
    """
    # prob = expon.pdf(x, loc=0, scale=shape) # background
    # prob /= np.sum(expon.pdf(np.linspace(0, 1, levels), loc=0, scale=shape))
    # prob *= pi_0
    # # check if scalar or array
    # if isinstance(prob, np.ndarray):
    #     prob[np.argwhere(x <= (1.0/levels))] += (1-pi_0)
    # else:
    #     if x < (1.0/levels):
    #         prob += (1-pi_0)
    prob = pi_0*exp_pmf(x, shape)
    prob[x < 1.0/levels] += (1-pi_0)

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

def norm_pmf(x, loc, scale, levels=2**8):
    """
    Evaluate the object probability mass function
    """

    prob = norm.pdf(x, loc, scale)
    prob /= np.sum(norm.pdf(np.linspace(0, 1, levels), loc, scale))

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

def obj_fcn(theta, y, p_y):
    p_mix = mixture_pmf(y, theta[0],theta[1],theta[2],theta[3])
    return entropy(p_y, p_mix)

def get_mixture(y, theta0=[0.3, 0.01, 0.03, 0.2], levels=2**8):
    """
    Estimate mixture model parameters through least squares.
    """

    theta = np.copy(theta0)
    samples = np.copy(y.flatten())

    bins = np.linspace(0, 1.0, levels+1)
    # k = np.linspace(0, 1.0, levels)

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
    """
    Computes the mixture model parameters (DEPRECATED).

    Keyword arguments
    ping - the sonar image (0-1 range)

    Output
    (pi1, pi2, p1, s2)
    """
    bins = np.linspace(0, 1.0, levels+1)
    hist = np.histogram(ping.flatten(), bins)

    x_vals = hist[1][:-1].astype(np.float64)
    p_emp = hist[0][:].astype(np.float64)
    p_emp /= (0.0+np.sum(p_emp))

    # curve_fit(fcn, xdata, ydata, params)
    # params, _ = curve_fit(mixture_pmf, x_vals, p_emp, p0=[0.3, 0.02, 0.02, 0.15])
    params, _ = curve_fit(mixture_pmf, x_vals, p_emp, p0=p0, bounds=([0.2, 0.0, 0.01, 1e-2 ], [ 0.5, 0.1, 0.05, 1e0 ]))

    # TODO: check parameter sanity!

    mix = mixture_pmf(x_vals, params[0], params[1], params[2], params[3])

    k = entropy(p_emp, mix)

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
    p_fa = 1.0 - np.cumsum(b_pmf) # = Sf0(y)

    o_pmf = object_pmf(k, s2, levels)
    p_d = 1.0 - np.cumsum(o_pmf)  # = Sf1(y)

    # flip order and add 1.0 to the end so it is easier to plot
    p_fa = np.append(p_fa[::-1], 1.0)
    p_d = np.append(p_d[::-1], 1.0)
    auc = np.trapz(p_d, p_fa)

    return(p_fa, p_d, auc)


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
    # params, _ = get_mixture_parameters(ping)
    _, _, theta, _ = get_mixture(ping)

    # 2) segment
    scan = segment_map(ping, theta[0], theta[1], theta[2], theta[3])

    return (scan, theta)


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
    Assumes an exponential+rayleigh mixture with a zero-bias.

    DEPRECATED/UNTESTED
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


"""
Sparse segmentation methods
"""

def detect(ping, threshold=0.3):
    """
    Detect occupied beams (fixed threshold on energy).
    """
    return (np.sum(np.power(ping, 2), axis=0) > threshold)

def annotate(ping, occupancy):
    """
    Annotate occupied beams in green, empty beams in red.
    """
    ping_rgb = np.dstack((ping, ping, ping))
    for i in range(0, len(occupancy)):
        ch = 0 # which channel to modify
        if occupancy[i]:
            ch = 1
        # max out the channel
        ping_rgb[:, i, ch] /= np.amax(ping_rgb[:, i, ch] )
    return ping_rgb

def get_template(dr=9.0/512, l=-10.0):
    """
    Get template function for the pseudo match filter
    """
    r = np.arange(0, 1, dr)
    pulse = np.exp(l*r)
    return pulse


def correlate(ping, pulse):
    """
    Compute radial correlation in image (matched filter)
    """
    pulse.shape = (len(pulse), 1)
    # q = correlate(ping, pulse, mode='full')
    # scipy's fftconvolve is much faster than correlate
    q_ping = fftconvolve(ping, pulse[::-1], mode='full') # 2ms
    q_ping = np.copy(q_ping[(len(pulse)-1):, :])
    return q_ping

def segment_smap(ping, pulse, threshold=1.0):
    """
    Scan segmentation via per-beam matched filter.

    Empty beams will have idx = 0
    """
    q_ping = correlate(ping, pulse)
    q_ping[q_ping < threshold] = 0
    idx = np.argmax(q_ping, axis=0)
    return idx


# def smrf_obj(idx, q_ping, l=-1, bw=1.0):
#     """
#     Objective function for sparse mrf computation

#     r: range estimate
#     q_ping: correlation image
#     l: exponential factor
#     bw: binary factor weight

#     TODO: handle non-contiguous vectors, or assume that is handled outside
#     """
#     # TODO: convert r to index
#     idx = (r)
#     u =  q_ping[idx, 0:len(r)] # unary cost: correlation at the given range
#     b =  np.sum(np.exp(l*np.abs(np.diff(idx))))# binary cost:
#     return u + bw*b

# def segment_smrf(ping, pulse, threshold=1.0):
#     q_ping = correlate(ping, pulse)
#     q_ping[q_ping < threshold] = 0

#     idx = np.argmax(q_ping, axis=0)

#     result = minimize(smrf_obj, (q_ping))

#     return result


def compute_transition_energy(ping, r, l=-0.10):
    """
    Compute transition energy matrix for the current label assignment.
    TODO: vectorize
    """
    bins, beams = ping.shape
    T = np.zeros((bins, beams))
    j = np.arange(bins)
#     ones = np.ones(bins)
    for i in range(beams):
        if i > 0 and i < beams-1:
            if (r[i] > 0) or (r[i-1] > 0 and r[i+1] > 0):
                T[:, i] = 0

                if r[i-1] > 0:
                    T[:, i] = np.exp(l*np.abs(r[i-1]-j))
                if r[i+1] > 0:
                    T[:, i] += np.exp(l*np.abs(r[i+1]-j))
        elif i == 0:
            if r[0] > 0:
                if r[1] > 0:
                    T[:, i] = np.exp(l*np.abs(r[1]-j))
                else:
                    T[:, 0] = 0
#                 for j in range(bins):
#                     T[j,0] = (np.exp(l*abs(j-r[1])), 0)[r[1]==0]
        else:
            if r[i] > 0:
                if r[i-1] > 0:
                    T[:, i] = np.exp(l*np.abs(r[i-1]-j))
                else:
                    T[:, i] = 0.0
    # this fixes single-beam gaps (double-nested for-loops)
#     for i in range(beams):
#         if i > 0 and i < beams-1:
#             if (r[i]>0) or (r[i-1]>0 and r[i+1]>0):
#                 for j in range(bins):
#                     T[j,i] = (np.exp(l*abs(j-r[i-1])), 0)[r[i-1]==0] + (np.exp(l*abs(j-r[i+1])),0)[r[i+1]==0]
#         elif i==0:
#             if r[0]>0:
#                 for j in range(bins):
#                     T[j,0] = (np.exp(l*abs(j-r[1])), 0)[r[1]==0]
#         else:
#             if r[i]>0:
#                 for j in range(bins):
#                      T[j,i] = (np.exp(l*abs(j-r[i-1])),0)[r[i-1]==0]
# this works, but there are still gaps
#         if r[i]>0:
#             # valid range measurement
#             if i > 0 and i < beams-1:
#                 for j in range(bins):
#                     T[j,i] = (np.exp(l*abs(j-r[i-1])), 0)[r[i-1]==0] + (np.exp(l*abs(j-r[i+1])),0)[r[i+1]==0]
#             elif i==0:
#                 for j in range(bins):
#                     T[j,0] = (np.exp(l*abs(j-r[1])), 0)[r[1]==0]
#             else:
#                 for j in range(bins):
#                     T[j,i] = (np.exp(l*abs(j-r[i-1])),0)[r[i-1]==0]
#         elif r[i+1] > 0 and r[i-1]>0:
#             for j in range(bins):
#                     T[j,i] = (np.exp(l*abs(j-r[i-1])), 0)[r[i-1]==0] + (np.exp(l*abs(j-r[i+1])),0)[r[i+1]==0]
    return T

def segment_smrf(ping, pulse, threshold=.40, iterations=10):
    """
    MRF segmentation through iterative maximization
    """

    Q = correlate(ping, pulse)
    Q[Q < threshold] = 0

    x0 = np.argmax(Q, axis=0) # local MAP solution as initialization
    x = np.copy(x0)

    for i in range(iterations):
        T = compute_transition_energy(ping, x, (-0.10*(i+1))/iterations)
        E = Q + T
        x = np.argmax(E, axis=0)
        # todo: check for convergence

    return x0, x, E
