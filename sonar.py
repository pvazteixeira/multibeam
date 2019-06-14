# -*- coding: utf-8 -*-
"""
sonar.py

This module conains a set of useful routines to handle multibeam profiling sonar data processing.

See also: [teixeira2018multibeam].
"""

import json
import logging
import numpy as np

from matplotlib.image import imsave

# from skimage.io import imread, imsave

import cv2

# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Sonar(object):
    # pylint: disable=too-few-public-methods
    # pylint: disable=too-many-instance-attributes
    """
    A class to handle pre-processing of multibeam sonar data.

    Pings are assumed to be R rows by B columns, corresponding to the data in
    polar coordinates, where the first row is closest to the sonar, and azimuth
    increases with the number of columns.
    Sonars are assumed to have a FOV smaller than 180 degrees.

    Attributes:
        min_range: Minimum range [m].
        max_range: Maximum range [m].
        num_beams: Number of sonar beams.
        num_bins:  Number of range bins/samples.
        fov:       Field of view angle (in-plane) [rad].
    """

    def __init__(self):
        """
        Initializes a sonar object with reasonable values.
        """
        self.min_range = 1.0
        self.max_range = 10.0
        self.fov = np.deg2rad(90.0)
        self.num_beams = 128

        self.num_bins = 512
        self.psf = np.ones((1, 1))
        self.taper = np.ones((self.num_beams))

        self.noise = 0.01
        self.rx_gain = 0.0 # currently unused, but important to record

        # assume linear mapping between beam and azimuth angle
        self.azimuths = np.linspace(-self.fov/2.0, self.fov/2.0, self.num_beams)
        self.k_b2a = [self.fov/(self.num_beams+0.0), -self.fov/2.0]
        self.p_beam2azi = np.poly1d(self.k_b2a)
        self.k_a2b = [self.num_beams/self.fov, self.num_beams/2.0]
        self.p_azi2beam = np.poly1d(self.k_a2b)

        # look-up table used to speed up conversion from polar to cartesian, e.g.:
        # cart_img[row_cart,col_cart] = polar_img[row_polar, cart_polar]
        self.row_cart = []
        self.col_cart = []
        self.row_polar = []
        self.col_polar = []

        self.__compute_lookup__(0.01)

    def print_config(self):
        """Print the sonar configuration."""
        print 'Range:', self.min_range, '-', self.max_range
        print 'FOV:', self.fov
        print 'Beams:', self.num_beams
        print 'Bins:', self.num_bins
        print '1/SNR:', self.noise
        print 'Rx gain:', self.rx_gain

    def load_config(self, cfg_file='sonar.json'):
        """Load the sonar configuration from file."""
        with open(cfg_file) as sonar_config_file:
            cfg = json.load(sonar_config_file)

            # lazy update: we should only update look-up table if the config has changed!
            update_lut = False

            properties = ['min_range', 'max_range', 'fov', 'num_beams', 'num_bins', 'noise', 'rx_gain']

            for p in properties:
                if p in cfg:
                    if cfg[p] != getattr(self, p):
                        update_lut = True
                        setattr(self, p, cfg[p])
                else:
                    logging.warning('property %s not found', p)

            if 'psf' in cfg:
                #if len(cfg['psf']) == self.num_beams:
                if cfg['psf'] != 1:
                    self.psf = np.array(cfg['psf'])
                    self.psf.shape = (1, self.num_beams)

            # DEPRECATED
            if 'taper' in cfg:
                if cfg['taper'] != 1:
                    self.taper = np.array(cfg['taper'])

            if 'azimuths' in cfg:
                azimuths = np.array(cfg['azimuths'])
                if np.any(azimuths != self.azimuths):
                    update_lut = True
                    self.__update_azimuths__(azimuths)
            else:
                # update the current mapping with the new parameters
                self.k_b2a = [self.fov/(self.num_beams+0.0), -self.fov/2.0]
                self.p_beam2azi = np.poly1d(self.k_b2a)
                self.k_a2b = [self.num_beams/self.fov, self.num_beams/2.0]
                self.p_azi2beam = np.poly1d(self.k_a2b)

            if update_lut:
                # delta_r is the smallest cartesian length of a pixel in the polar image
                delta_r = (self.max_range-self.min_range)/(self.num_bins+0.0)
                delta_r = min(delta_r, self.min_range*self.fov/(self.num_beams))

                self.__compute_lookup__(delta_r)

    def save_config(self, cfg_file='sonar.json'):
        """
        Save the ping/sonar parameters to a JSON file.
        """
        cfg = {}
        cfg['max_range'] = self.max_range
        cfg['min_range'] = self.min_range
        cfg['fov'] = self.fov
        cfg['num_beams'] = self.num_beams
        cfg['num_bins'] = self.num_bins
        cfg['noise'] = self.noise
        cfg['rx_gain'] = self.rx_gain

        cfg['psf'] = np.squeeze(self.psf).tolist()
        cfg['azimuths'] = self.azimuths.tolist()
        # cfg['taper'] = self.taper.tolist()

        with open(cfg_file, 'w') as fp:
            json.dump(cfg, fp, sort_keys=True, indent=2)

    def to_json(self, ping):
        ping_dict = {}
        ping_dict['max_range'] = self.max_range
        ping_dict['min_range'] = self.min_range
        ping_dict['fov'] = self.fov
        ping_dict['num_beams'] = self.num_beams
        ping_dict['num_bins'] = self.num_bins
        ping_dict['noise'] = self.noise
        ping_dict['rx_gain'] = self.rx_gain

        ping_dict['psf'] = np.squeeze(self.psf).tolist()
        ping_dict['azimuths'] = self.azimuths.tolist()

        ping_dict['beams'] = {}
        for i in range(0, self.num_beams):
            ping_dict['beams'][str(i)] = np.squeeze(ping[:, i]).tolist()

        return ping_dict


    def to_csv_polar(self, filename, ping):
        """
        Export the ping as a csv file.
        """
        data = np.zeros((self.num_beams*self.num_bins, 3))
        for beam in range(0, self.num_beams):
            for rbin in range(0, self.num_bins):
                row_idx = beam*self.num_bins + rbin
                data[row_idx, 0] = self.range(rbin)
                data[row_idx, 1] = self.azimuth(beam)
                data[row_idx, 2] = ping[rbin, beam]
        np.savetxt(filename, data, delimiter=',', newline='\n')

    def to_csv_cart(self, filename, ping):
        """
        Export the ping as a csv file.
        """
        data = np.zeros((self.num_beams*self.num_bins, 3))
        for beam in range(0, self.num_beams):
            for rbin in range(0, self.num_bins):
                row_idx = beam*self.num_bins + rbin
                r = self.range(rbin)
                a = self.azimuth(beam)
                x = r*np.cos(a)
                y = r*np.sin(a)
                data[row_idx, 0] = x
                data[row_idx, 1] = y
                data[row_idx, 2] = ping[rbin, beam]
        np.savetxt(filename, data, delimiter=',', newline='\n')

    def range(self, rbin):
        """
        Returns the range (in meters) corresponding to the specified bin number
        """
        return self.min_range + rbin*((self.max_range - self.min_range)/self.num_bins)


    def range_to_bin(self, r):
        """Return the range bin index corresponding to the specified range"""
        dr = (self.max_range - self.min_range)/self.num_bins
        return int( (r-self.min_range)/dr)


    def azimuth(self, beam):
        """
        Returns the azimuth angle (in radians) corresponding to the specified beam number.
        """
        return self.p_beam2azi(beam+0.0)


    def beam(self, azimuth):
        """Return the beam number corresponding to the specified azimuth angle (in radians)."""
        return (np.round(self.p_azi2beam(azimuth))).astype(int)


    def __update_azimuths__(self, azimuths):
        """Update the interpolating functions that compute the mapping between azimuth and beam from a table."""
        assert len(azimuths) == self.num_beams
        self.azimuths = azimuths
        # update FOV
        self.fov = np.amax(azimuths) - np.amin(azimuths)
        # update maps
        self.k_b2a = np.polyfit(np.arange(0, self.num_beams)+0.0, azimuths, 5)
        self.p_beam2azi = np.poly1d(self.k_b2a)
        self.k_a2b = np.polyfit(azimuths, np.arange(0, self.num_beams)+0.0, 5)
        self.p_azi2beam = np.poly1d(self.k_a2b)
        # TODO: update angular gain table


    def __compute_lookup__(self, resolution=0.01):
        """
        Compute lookup table used in polar to cartesian conversion
        """
        # This function computes a look-up table of pairs (i,j), (k,l) to enable
        # fast conversion from polar to cartesian via vectorization of
        # ping_cart[i, j] = ping_polar[k, l]
        #
        # To do this, it computes two pairs of images
        # - Cartesian image indices (same shape as Cartesian image)
        #   - row_cart: contains the row index of the Cartesian image
        #   - col_cart: contains the column index of the Cartesian image
        # - polar image indices (same shape as polar image)
        #   - row_polar: contains the row index of the polar image
        #   - col_polar: contains the column index of the polar image
        #
        # To compute these images, we first compute the spatial coordinates of each
        # pixel and then transform it to the index using the sonar properties.

        bin_length = (self.max_range - self.min_range)/(self.num_bins + 0.0)
        # beamwidth = (self.fov)/(self.num_beams+0.0)
        assert(self.azimuths[0]<self.azimuths[-1])

        y0 = self.max_range*np.sin(self.azimuths[0])
        y1 = self.max_range*np.sin(self.azimuths[-1])
        self.width = int(np.around((y1-y0)/resolution))
        yres = (y1-y0)/(self.width+0.0) # resolution on y-axis, in m/px

        x0 = self.min_range*min(np.cos(self.azimuths[0]), np.cos(self.azimuths[-1]))
        x1 = self.max_range
        self.height = int(np.around((x1-x0)/resolution))
        xres = (x1-x0)/(self.height+0.0) # resolution on x-axis, in m/px

        logging.debug("Resolution: req=%f, x=%f, y=%f", resolution, xres, yres)

        row_cart = np.arange(0, self.height)
        row_cart.shape = (self.height, 1)
        row_cart = np.tile(row_cart, (1, self.width))
        x = x0 + xres*row_cart

        col_cart = np.arange(0, self.width)
        col_cart.shape = (1, self.width)
        col_cart = np.tile(col_cart, (self.height, 1))
        y = y0 + yres*col_cart

        # convert to range, azi
        (mag, angle) = cv2.cartToPolar(x.flatten(), y.flatten())

        # convert from cv's [0,2pi] range to [-pi,pi]
        angle[angle > np.pi] -= 2*np.pi
        angle[angle < -np.pi] += 2*np.pi

        # ensure that min and max angle are not out of bounds
        # NOTE: due to the use of the polynomial approximation to map between angle and beam, we run the risk of obtaining valid beam numbers for angles that are outside the sonar's field of view, as the polynomial approximation will be invalid in such regions. Therefore, we must get thes angles to just outside the FOV, where the approximation, despite not being valid, will not produce valid beam indices

        angle[angle < self.azimuths[0]] = self.azimuths[0]-0.1
        angle[angle > self.azimuths[-1]] = self.azimuths[-1]+0.1

        # reshape to the output image size
        mag.shape = (self.height, self.width)
        angle.shape = (self.height, self.width)
        # imsave('mag.png', mag)
        # imsave('angle.png', angle)

        # convert to beam index
        col_polar = self.p_azi2beam(angle).astype(int)
        col_polar.shape = (self.height, self.width)

        # convert to bin index
        row_polar = np.copy(mag) - self.min_range
        # row_polar = self.range2bin(mag)
        row_polar = np.around(row_polar/bin_length).astype(int)

        # DEBUG
        # imsave('col_polar_pre.png', col_polar)
        # imsave('row_polar_pre.png', row_polar)
        # imsave('col_cart_pre.png', col_cart)
        # imsave('row_cart_pre.png', row_cart)

        # map all points outside the FOV to 0,0
        # CONSIDER DELETING THESE ELEMENTS AND JUST PRE-ALLOCATING the output array
        # ...or maybe the performance hit from deleting elements offsets the gain from
        # reducing the number of look-ups?

        self.row_polar = np.copy(row_polar).astype(int)
        self.col_polar = np.copy(col_polar).astype(int)
        self.row_cart = np.copy(row_cart).astype(int)
        self.col_cart = np.copy(col_cart).astype(int)

        self.col_polar[row_polar < 0] = 0
        self.row_polar[col_polar < 0] = 0
        self.col_polar[col_polar < 0] = 0
        self.row_polar[row_polar < 0] = 0

        self.col_polar[row_polar >= self.num_bins] = 0
        self.row_polar[col_polar >= self.num_beams] = 0
        self.col_polar[col_polar >= self.num_beams] = 0
        self.row_polar[row_polar >= self.num_bins] = 0

        # DEBUG
        # imsave('col_polar.png', self.col_polar)
        # imsave('row_polar.png', self.row_polar)
        # imsave('col_cart.png', self.col_cart)
        # imsave('row_cart.png', self.row_cart)

    def reset_window(self, min_range, max_range, resolution=0.01):
        """Reset the sonar window and recompute lookup table."""
        self.min_range = min_range
        self.max_range = max_range
        # update lookup table
        self.__compute_lookup__(resolution)

        # currently, width is being ignored!
        # should actually take resolution [m/px] as argument
    def to_cart(self, ping, background=0.0):
        """Convert sonar scan from polar to Cartesian

        Keyword arguments:
        ping - the sonar scan, in polar representation
        width - the desired with of the Cartesian representation (default: 320)

        Note: some conversion performance values as a function of resolution:
        Sample results for 96 beams * 512 bins, 28.8deg FOV, 2.25-11.25m
        resolution | conversion time
        0.01 m/px    9ms
        0.02 m/px    2ms
        0.03 m/px    1ms
        """
        pingc = np.copy(ping)
        image = np.zeros((self.height, self.width))
        pingc[0, 0] = background
        image[self.row_cart.flatten(), self.col_cart.flatten()] = pingc[self.row_polar.flatten(), self.col_polar.flatten()]

        return image

    def deconvolve(self, ping):
        """
        Remove impulse response function from ping
        (derived from opencv's deconvolution sample)
        """
        assert ping.shape == (self.num_bins, self.num_beams)
        # convert to float, single channel

        ping = ping.astype(np.float64)

        # compute input ping's DFT
        img_f = cv2.dft(ping, flags=cv2.DFT_COMPLEX_OUTPUT)
        psf = self.psf
        psf /= psf.sum()
        psf_padded = np.zeros_like(ping)
        kh, kw = psf.shape
        psf_padded[:kh, :kw] = psf

        # compute (padded) psf's DFT
        psf_f = cv2.dft(psf_padded, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)

        psf_f_2 = (psf_f**2).sum(-1)
        ipsf_f = psf_f / (psf_f_2 + self.noise)[..., np.newaxis]

        result_f = cv2.mulSpectrums(img_f, ipsf_f, 0)
        result = cv2.idft(result_f, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        result = np.roll(result, -kh//2, 0)
        result = np.roll(result, -kw//2, 1)

        # result = result - result.min()
        # a) rescale to match original
        # result = (np.max(ping)/np.max(result))*result
        # b) normalize
        #result = (1.0/np.max(result))*result
        # c) clip to 0-1 range
        result[result < 0] = 0
        result = (np.max(ping)/np.max(result))*result
        # result[result>1.0] = 1.0
        return result.astype(ping.dtype)

    def tvg(self, r, k1, k2, k3):
        """
        Compute time-varying gain
        """
        return k1*np.log10(r) + k2*r + k3 + 0.0

    def remove_attenuation(self, ping, k=None):
        """
        remove attenuation effects

        Note: must 
        """
        if k is None:
            k = [0.25, 0.01375]

        rng = np.linspace(self.min_range, self.max_range, self.num_bins)
        gain = self.tvg(rng, k[0], k[1], 0.0)
        gain[gain<1] = 1
        gain /= gain[0]
        gain = np.tile(gain, (ping.shape[1], 1))
        ping_2 = np.multiply(gain.transpose(), ping)
        # normalize image
        ping_2 *= (np.amax(ping)/np.amax(ping_2))

        return ping_2

    def remove_taper(self, ping, k_taper=None, normalize=False):
        """
        remove beam pattern taper effects
        """
        if k_taper is None:
            k_taper = [3000, 0, -50, 0, 4, 0, 1]
        p_ka = np.poly1d(k_taper)
        gain = p_ka(self.azimuths)
        gain = np.tile(gain, (ping.shape[0], 1))
        ping_2 = np.multiply(gain, ping)
        if normalize:
            ping_2 *= np.amax(ping)/np.amax(ping_2)

        return ping_2


    # def removeTaper(self, ping):
    #     """
    #     Remove taper effects from a scan.
    #     """
    #     # TODO: revise!
    #     taper = np.tile(self.taper, (ping.shape[0], 1))
    #     ping2 = ping.astype(np.float64)
    #     ping2 /= taper

    #     ping2 = ping2 - ping2.min()
    #     ping2 = (np.max(ping)/np.max(ping2))*ping2
    #     #ping2*=((ping.max()+0.0)/ping2.max())
    #     # if (ping2.max()>1.0 ):
    #     #   # rescale if needed
    #     #   ping2*=(1.0/ping2.max())
    #     # ping2[ping2<0]=0
    #     # ping2[ping2>1.0] = 1.0

    #     return ping2

    def preprocess(self, ping_raw, renormalize=False):
        """
        Pre-process a ping. This entails removing beam-pattern effects, angular taper, and
        attenuation.
        """
        ping = np.copy(ping_raw)

        ping = self.remove_attenuation(ping) # maybe broken?
        ping = self.remove_taper(ping, normalize=renormalize)
        ping = self.deconvolve(ping)

        return ping


###################
## revise below! ##
###################


'''

def removeRange(self, ping):
    # this function captures absorption and geometrical spreading
    def attFcn(r,a,b,c):
      return a*np.exp(-b*r)/(r**c)

    bin_length = (self.max_range - self.min_range)/(self.num_bins+0.0)
    r = self.min_range + bin_length*np.arange(0,self.num_bins)

    # attenuation parameters (learned from data)
    attenuation = attFcn(r, 2.0, -0.1, 1.15) 
    attenuation.shape = (len(attenuation),1)
    att = np.tile(attenuation, (1,ping.shape[1]))
    ping2 = ping.astype(np.float64)
    ping2/=att
    # rescale to original
    ping2 = ping2 - ping2.min()
    ping2 = (np.max(ping)/np.max(ping2))*ping2
    return ping2

def segment(self, ping, threshold):
    """ Segments the image using a fixed threshold

    Keyword arguments:
    ping - 
   threshold - the segmenting threshold (0-1 range)

    Note:
    Return image type is the same as the input's.
    """

    ping_binary = ping.astype(np.float64)
    ping_binary[ping_binary<threshold]=0
    ping_binary[ping_binary>0]=1.0

    return ping_binary.astype(ping.dtype)

def getReturns(self, segmented_ping):
    """ Computes the location of the first return along the beam for all the beams

    Keyword arguments:
    segmented_ping - the segmented ping/scan (e.g. as provided by segment())
    Output:
    returns - a 2D vector containing the (x,y) positions of the return for each of the sonar beams;
              beams for which no return was found will have (-1,-1) as the return position.
    """
    # get the bin location for each beam
    # convert (bin, beam ) to (r,theta)
    # convert (r, theta) to (x,y)
    positions = np.zeros((2,np.num_beams))

    # (x,y) = cv2.polarToCart()
    return positions
'''
