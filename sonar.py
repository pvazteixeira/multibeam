# -*- coding: utf-8 -*-
"""
sonar.py

This module conains a set of useful routines to handle multibeam profiling sonar data processing.

See also: [teixeira2018multibeam].
"""

import json
import logging
import numpy as np

import cv2

from skimage.io import imread, imsave

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

        self.noise = 0.1
        self.rx_gain = 0.0 # currently unused, but important to record

        # assume linear mapping between beam and azimuth angle
        self.azimuths = []
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

            properties = ['min_range', 'max_range', 'fov', 'num_beams', 'num_bins', 'noise', 'rx_gain']

            for p in properties:
                if p in cfg:
                    setattr(self, p, cfg[p])
                else:
                    logging.warning('property %s not found', p)

            if 'psf' in cfg:
                #if len(cfg['psf']) == self.num_beams:
                if cfg['psf'] != 1:
                    self.psf = np.array(cfg['psf'])
                    self.psf.shape = (1, self.num_beams)

            if 'taper' in cfg:
                if cfg['taper'] != 1:
                    self.taper = np.array(cfg['taper'])

            if 'azimuths' in cfg:
                azimuths = np.array(cfg['azimuths'])
                self.__update_azimuths__(azimuths)
            else:
                # we update the LINEAR mapping defined at initialization with the new parameters
                self.k_b2a = [self.fov/(self.num_beams+0.0), -self.fov/2.0]
                self.p_beam2azi = np.poly1d(self.k_b2a)
                self.k_a2b = [self.num_beams/self.fov, self.num_beams/2.0]
                self.p_azi2beam = np.poly1d(self.k_a2b)

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
        cfg['taper'] = self.taper.tolist()

        with open(cfg_file, 'w') as fp:
            json.dump(cfg, fp, sort_keys=True, indent=2)

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

    def azimuth(self, beam):
        """
        Returns the azimuth angle (in radians) corresponding to the specified beam number.
        """
        return self.p_beam2azi(beam+0.0)

    def beam(self, azimuth):
        """Return the beam number corresponding to the specified azimuth angle (in radians)."""
        return int(np.round(self.p_azi2beam(azimuth)))

    def __update_azimuths__(self, azimuths):
        """Update the interpolating functions that compute the mapping between azimuth and beam from a table."""
        assert len(azimuths) == self.num_beams
        self.azimuths = azimuths
        # update FOV
        self.fov = np.amax(azimuths) - np.amin(azimuths)
        # update maps
        self.k_b2a = np.polyfit(np.arange(0, self.num_beams), azimuths, 5)
        self.p_beam2azi = np.poly1d(self.k_b2a)
        self.k_a2b = np.polyfit(azimuths, np.arange(0, self.num_beams)+0.0, 5)
        self.p_azi2beam = np.poly1d(self.k_a2b)

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

        #y0 = -self.max_range*np.sin(self.fov/2.0)
        y1 = self.max_range*np.sin(self.fov/2.0)
        width = np.around(2*y1/resolution)
        yres = 2*y1/width  # resolution on y-axis, in m/px
        self.width = int(width)

        x0 = self.min_range*np.cos(self.fov/2.0)
        x1 = self.max_range
        height = np.around((x1-x0)/resolution)
        xres = (x1-x0)/height+0.0
        self.height = int(height)

        logging.debug("Resolution: req=%f, x=%f, y=%f", resolution, xres, yres)

        row_cart = np.arange(0, height)
        row_cart.shape = (height, 1)
        row_cart = np.tile(row_cart, (1, width))
        x = x0 + xres*row_cart

        col_cart = np.arange(0, width)
        col_cart.shape = (1, width)
        col_cart = np.tile(col_cart, (height, 1))
        y = -y1 + yres*col_cart

        # convert to range, azi
        (mag, angle) = cv2.cartToPolar(x.flatten(), y.flatten())

        # convert from cv's [0,2pi] range to [-pi,pi]
        angle[angle > np.pi] -= 2*np.pi
        angle[angle < -np.pi] += 2*np.pi

        mag.shape = (height, width)
        angle.shape = (height, width)

        col_polar = self.p_azi2beam(angle)
        col_polar.shape = (height, width)

        row_polar = mag
        row_polar -= self.min_range
        row_polar = np.around(row_polar/bin_length)

        # cpoob = np.sum(col_polar < 0) + np.sum(col_polar >= self.num_beams)
        # rpoob = np.sum(row_polar < 0) + np.sum(row_polar >= self.num_bins)
        # logging.debug("Polar: %d out of %d elements out of bounds", cpoob+rpoob, width*height)

        # ccoob = np.sum(col_cart < 0) + np.sum(col_cart >= height)
        # rcoob = np.sum(row_cart < 0) + np.sum(row_cart >= width)
        # logging.debug("Cartesian: %d out of %d elements out of bounds", ccoob+rcoob, width*height)
        # col_polar+=self.fov/2.0
        # col_polar = np.around(col_polar/beamwidth)

        # map all points outside the FOV to 0,0
        col_polar[row_polar < 0] = 0
        row_polar[col_polar < 0] = 0
        col_polar[col_polar < 0] = 0
        row_polar[row_polar < 0] = 0

        col_polar[row_polar >= self.num_bins] = 0
        row_polar[col_polar >= self.num_beams] = 0
        col_polar[col_polar >= self.num_beams] = 0
        row_polar[row_polar >= self.num_bins] = 0

        # col_polar[row_polar>=self.num_bins] = 0
        # row_polar[col_polar>=self.num_beams] = 0
        # row_polar[row_polar>=self.num_bins] = 0
        # col_polar[col_polar>=self.num_beams] = 0

        self.row_polar = row_polar.astype(int)
        self.col_polar = col_polar.astype(int)
        self.row_cart = row_cart.astype(int)
        self.col_cart = col_cart.astype(int)

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
        image = np.zeros((self.height, self.width))
        ping[0, 0] = background
        image[self.row_cart.flatten(), self.col_cart.flatten()] = ping[self.row_polar.flatten(), self.col_polar.flatten()]

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

    def removeTaper(self, ping):
        """
        Remove taper effects from a scan.
        """
        # TODO: revise!
        taper = np.tile(self.taper, (ping.shape[0], 1))
        ping2 = ping.astype(np.float64)
        ping2 /= taper

        ping2 = ping2 - ping2.min()
        ping2 = (np.max(ping)/np.max(ping2))*ping2
        #ping2*=((ping.max()+0.0)/ping2.max())
        # if (ping2.max()>1.0 ):
        #   # rescale if needed
        #   ping2*=(1.0/ping2.max())
        # ping2[ping2<0]=0
        # ping2[ping2>1.0] = 1.0

        return ping2

    def preprocess(self, ping):
        """
        Pre-process a ping. This entails removing beam-pattern effects, angular taper, and
        attenuation.
        """
        ping2 = np.copy(ping)
        # deconvolve
        # remove taper
        # remote attenuation

        return ping2


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
