import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import expon, rayleigh

"""
background pmf : (1-pi)*delta(x) + pi*expon.pdf(x,0,s)
object pmf :     rayleigh.pdf(x,0,s)

note: normalization missing from the above equations
"""

def kld(sample, reference):
    p = sample
    q = reference
    q[np.where(q==0)[0]] = 1 # we're dividing by q, so set all zero-values to 1
    r = p/q
    r[np.where(r==0)[0]] = 1 # ln(1) = 0; ln(0)=nan

    return np.sum(p*np.log(r)) 

def background_pmf(x, pi, s, L=2**8):
    """
    background_pmf
    """
    p = expon.pdf(x,loc=0,scale=s) # background
    p /= np.sum(expon.pdf(np.linspace(0,1,L),loc=0,scale=s))
    p*=pi
    # check if scalar or array
    if isinstance(p,np.ndarray):
        p[np.argwhere(x<=(1.0/L))] += (1-pi)
    else:
        if x < (1.0/L):
            p+=(1-pi)

    return p

def object_pmf(x, s, L=2**8):
    """
    object_pmf
    """

    p = rayleigh.pdf(x,0,s)
    p /= np.sum(rayleigh.pdf(np.linspace(0,1,L),0,s))

    return p

def mixture_pmf(x, pi1, pi2, s1, s2, L=2**8):
    """
    (1-pi1-pi2)*delta(x) + pi1*exponential(x,s1) + pi2*rayleigh(x,s2)
    """
    # background
    p1 = expon.pdf(x,loc=0,scale=s1) # background
    p1 /= np.sum(expon.pdf(np.linspace(0,1,L),loc=0,scale=s1))

    # object
    p2 = rayleigh.pdf(x,0,s2)
    p2 /= np.sum(rayleigh.pdf(np.linspace(0,1,L),0,s2))

    # print 'expon:', np.sum(p1), 'rayleigh:', np.sum(p2)

    # mixture
    p = pi1*p1 + pi2*p2
    p[x<(1.0/L)] += (1 - pi1 - pi2) # zero-bias

    return p

def getMixtureParameters(ping,L=2**8):
    """ Computes the mixture model parameters.

    Keyword arguments
    ping - the sonar image (0-1 range)

    Output
    (pi1, pi2, p1, s2)

    """
    bins = np.linspace(0, 1.0, L+1 )
    hi = np.histogram(ping.flatten(),bins)

    x = hi[1][:-1].astype(np.float64)
    h = hi[0][:].astype(np.float64)
    h /=(0.0+np.sum(h))

    # curve_fit(fcn, xdata, ydata, params)
    # p, v = curve_fit(mixture_pmf, x, h, p0=[0.3, 0.02, 0.02, 0.15], bounds=([0,0,0.0,0],[0.5,0.5,1.0,1.0]))
    p, v = curve_fit(mixture_pmf, x, h, p0=[0.3, 0.02, 0.02, 0.15])

    mix = mixture_pmf(x, p[0],p[1],p[2],p[3])

    k = kld(h, mix)

    return (p, k)

def likelihood(x, pi1, pi2, s1, s2, L=2**8):
    pi0 = ( 1 - pi1 - pi2 )
    pi0 /= (1-pi2)

    num = rayleigh.pdf(x,loc=0,scale=s2)
    num /= np.sum(rayleigh.pdf(np.linspace(0,1.0,L),loc=0,scale=s2))

    den = expon.pdf(x,loc=0,scale=s1)
    den /= np.sum(expon.pdf(np.linspace(0,1.0,L), loc=0, scale=s1))
    den *= (1-pi0)
    den[x<1.0/L]+=pi0

    l = num/den

    return l


def segment_np(x,pi1,pi2,s1,s2, p_fa):
    """

    """

    return s


def segment_map(x,pi1,pi2,s1,s2):
    """
    MAP segmentation (binary local classifier)
    """
    eta = (1-pi2)/pi2

    s = likelihood(x, pi1,pi2,s1,s2)
    s[s<eta] = 0
    s[s>=eta] = 1.0

    return s

def segment_mrf(x,pi1, pi2, s1, s2):
    """
    MRF segmentation
    """
    s = np.zeros_like(x)
    # NOT IMPLEMENTED 
    return s


def extract_max(ping, ping_binary, min_range, bin_length):
    """
    Extract the strongest return (per-beam) from a segmented image.
    """
    pping = np.copy(ping)
    pping[ping_binary<=0] = 0;
    intensities = np.amax(pping, axis=0)
    ranges = np.argmax(pping, axis=0)
    ranges = ranges*bin_length
    ranges[ranges<=0] = -min_range;
    ranges += min_range*(np.ones_like(ranges))

    return (ranges, intensities)


def extract_first(x, b, min_range, bin_length):
    """
    Extract the first return (per-beam) from a segmented image.
    UNIMPLEMENTED
    """

    ping = np.copy(x)
    ping[b<=0] = 0;
    intensities = np.amax(ping, axis=0)
    ranges = np.argmax(ping, axis=0)
    ranges = ranges*bin_length
    ranges[ranges<=0] = -min_range;
    ranges += min_range(np.ones_like(ranges))

    return (ranges, intensities)
