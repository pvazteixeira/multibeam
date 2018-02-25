import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import expon, rayleigh

"""
background pmf : (1-pi)*delta(x) + pi*expon.pdf(x,0,s)
object pmf :     rayleigh.pdf(x,0,s)

note: normalization missing from the above equations
"""

def background_pmf(x, pi, s):
    p0 = np.zeros_like(x) # zero-bias
    p0[0]=1

    p1 = expon.pdf(x,0,s) # background
    p1/=np.sum(p1)        # normalize

    p = (1-pi)*p0 + pi*p1 # mixture
    p/=np.sum(p)          # normalize (unnecessary)
    return p

def object_pmf(x, s):
    p = rayleigh.pdf(x,0,s)
    p/=np.sum(p)
    return p

def mixture_pmf(x, pi1, pi2, s1, s2):
    """
    (1-pi1-pi2)*delta(x) + pi1*exponential(x,s1) + pi2*rayleigh(x,s2)
    """
    p0 = np.zeros_like(x) # zero-bias
    p0[0]=1

    p1 = expon.pdf(x,0,s1) # background
    p1/=np.sum(p1)        # normalize

    p2 = rayleigh.pdf(x, 0, s2)
    p2/=np.sum(p2)

    p = (1-pi1-pi2)*p0 + pi1*p1 + pi2*p2
    p/=np.sum(p)

    return p

def likelihood(x, pi1, pi2, s1, s2):
    pi0 = ( 1 - pi1 - pi2 )
    pi0 /= (1-pi2)

    num = rayleigh.pdf(x,0,s2)
    den = (1-pi0)*expon.pdf(x,0,s1)
    den[x==0]+=pi0

    l = num/den

    return l


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
    return s

def getMixtureParameters(ping):
    """ Computes the mixture model parameters.

    Keyword arguments
    ping - the sonar image (0-1 range)

    Output
    (pi1, pi2, p1, s2)

    """
    bins = np.linspace(0, 1.0, 256 )
    hi = np.histogram(ping.flatten(),bins)

    x = hi[1][:-1].astype(np.float64)
    h = hi[0][:].astype(np.float64)
    h /=(0.0+np.sum(h))

    p, v = curve_fit(mixture_pmf, x, h, p0=[0.3, 0.02, 0.02, 0.15],bounds=([0,0,0.0,0],[0.5,0.5,1.0,1.0]))

    return p
