import cv2 
import numpy as np
import json
 
class Sonar:

  def __init__(self):
    # some sane defaults
    self.min_range = 1.0
    self.max_range = 10.0
    self.fov = np.deg2rad(90.0)
    self.num_beams = 128
    self.num_bins = 512
    self.psf = np.ones((1,1)) 
    self.taper = np.ones((self.num_beams))

    self.noise = 0.01

    # look-up table used to speed up conversion from polar to cartesian
    self.row_cart =[]
    self.col_cart =[]
    self.row_polar =[]
    self.col_polar = []
    self.__computeLookUp__(0.02) 

  def loadConfig(self, cfg_file='sonar.json'):
    with open(cfg_file) as sonar_config_file:
      cfg = json.load(sonar_config_file)
    self.min_range = cfg['min_range']
    self.max_range = cfg['max_range']
    self.fov = cfg['fov']
    self.num_beams = cfg['num_beams']
    self.num_bins = cfg['num_bins']
    self.psf = np.hstack(cfg['psf']) 
    self.psf.shape = (1,96)
    self.taper = np.array(cfg['taper'])
    self.noise = cfg['noise']
    self.__computeLookUp__(0.02)

  def saveConfig(self, cfg_file='sonar.json'):
    cfg={}
    cfg['min_range'] = self.min_range
    cfg['max_range'] = self.max_range
    cfg['fov'] = self.fov
    cfg['num_beams'] = self.num_beams
    cfg['num_bins'] = self.num_bins
    cfg['psf'] = self.psf
    cfg['taper'] = self.taper
    cfg['noise'] = self.noise
    with open(cfg_file, 'w') as fp:
      json.dump(cfg, fp, sort_keys=True, indent=2)

  def __computeLookUp__(self, resolution):
    """Compute lookup table used in polar to cartesian conversion

   """
    bin_length = (self.max_range - self.min_range)/(self.num_bins + 0.0)
    beamwidth  = (self.fov)/(self.num_beams+0.0)

    y0 = self.max_range*np.sin(self.fov/2.0) 
    width = np.around(2*y0/resolution)
    yres = 2*y0/width
    self.width = int(width)
    
    x0 = self.min_range*np.cos(self.fov/2.0)
    x1 = self.max_range
    height = np.around((x1-x0)/resolution)
    xres = (x1-x0)/height
    self.height = int(height)
    
    print 'Resolution'
    print 'Desired:',resolution,', x:',xres,', y:',yres

    row_cart = np.arange(0,height)
    row_cart.shape = (height,1)
    row_cart = np.tile(row_cart, (1,width))
    x = x0 + xres*row_cart
    
    col_cart = np.arange(0,width)
    col_cart.shape = (1,width)
    col_cart = np.tile(col_cart, (height,1))
    y = -y0 + yres*col_cart 
    
    (mag,angle) = cv2.cartToPolar(x.flatten(),y.flatten())

    angle[angle>np.pi]-=2*np.pi # convert from cv's [0,2pi] range to [-pi,pi]

    mag.shape = (height, width)
    angle.shape = (height, width)
   
    row_polar = mag
    col_polar = angle

    row_polar-=self.min_range
    row_polar = np.around(row_polar/bin_length)

    col_polar+=self.fov/2.0
    col_polar = np.around(col_polar/beamwidth)

    col_polar[row_polar<0]=0
    row_polar[col_polar<0]=0
    col_polar[col_polar<0]=0
    row_polar[row_polar<0]=0

    col_polar[row_polar>=self.num_bins] = 0
    row_polar[col_polar>=self.num_beams] = 0
    row_polar[row_polar>=self.num_bins] = 0
    col_polar[col_polar>=self.num_beams] = 0
    
    self.row_polar = row_polar.astype(int)
    self.col_polar = col_polar.astype(int)
    self.row_cart = row_cart.astype(int)
    self.col_cart = col_cart.astype(int)

  def resetWindow(self, min_range, max_range, resolution=0.02):
    self.min_range = min_range
    self.max_range = max_range
    # update lookup table
    self.__computeLookUp__(resolution)

  # currently, width is being ignored!
  # should actually take resolution [m/px] as argument
  def toCart(self, ping, width=320, bg=0.0):
    """Convert sonar scan from polar to Cartesian

    Keyword arguments:
    ping - the sonar scan, in polar representation
    width - the desired with of the Cartesian representation (default: 320)
    Note: some conversion performance values as a function of resolution: 
      resolution | conversion time
      0.01 m/px    9ms
      0.02 m/px    2ms
      0.03 m/px    1ms
   """
    image = np.zeros((self.height, self.width)) 
    ping[0,0] = bg
    image[self.row_cart.flatten(), self.col_cart.flatten()] = ping[self.row_polar.flatten(), self.col_polar.flatten()]

    return image

  def deconvolve(self, ping):
    """
    remove impulse response function from ping
    (derived from opencv's deconvolution sample)
    """
    assert ping.shape == (self.num_bins, self.num_beams)
    # convert to float, single channel
    
    ping = ping.astype(np.float64)
    
    # compute input ping's DFT
    img_f = cv2.dft(ping, flags = cv2.DFT_COMPLEX_OUTPUT) 
    psf = self.psf
    psf/=psf.sum()
    psf_padded = np.zeros_like(ping)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf

    # compute (padded) psf's DFT
    psf_f = cv2.dft(psf_padded, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)

    psf_f_2 = (psf_f**2).sum(-1)
    ipsf_f = psf_f / (psf_f_2 + self.noise)[..., np.newaxis]

    result_f = cv2.mulSpectrums(img_f, ipsf_f, 0)
    result = cv2.idft(result_f, flags = cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    result = np.roll(result, -kh//2, 0)
    result = np.roll(result, -kw//2, 1)

    # rescale to match original
    result = (np.max(ping)/np.max(result))*result
    result[result<0]=0

    return result

  def removeTaper(self,ping):
    taper = np.tile(self.taper, (ping.shape[0],1))
    ping2 = ping.astype(np.float64)
    ping2/=taper
    ping2*=((ping.max()+0.0)/ping2.max())
    return ping2

  def removeRange(self, ping):
    # this function captures absorption and geometrical spreading
    def attFcn(r,a,b,c):
      return a*np.exp(-b*r)/(r**c)

    bin_length = (self.max_range - self.min_range)/(self.num_bins+0.0)
    r = self.min_range + bin_length*np.arange(0,self.num_bins)

    attenuation = attFcn(r, 2.0, -0.1, 1.15) 
    
    attenuation.shape = (len(attenuation),1)
    att = np.tile(attenuation, (1,ping.shape[1]))
    ping2 = ping.astype(np.float64)
    ping2/=att
    # rescale to original
    ping2*=((ping.max()+0.0)/ping2.max())
    return ping2

  def segment(self, ping, threshold):
    """ Segments the image using a fixed threshold
    
    Keyword arguments:
    ping 
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
