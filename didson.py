import sonar
import numpy as np
import cv2

class Didson(Sonar):

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
   
        row_polar = mag   # bin
        col_polar = angle # beam

        # distortion removal formula

        row_polar-=self.min_range
        row_polar = np.around(row_polar/bin_length)

        #col_polar+=self.fov/2.0
        
        # default conversion
        # col_polar = np.around(col_polar/beamwidth)
        # didson-specific converstion (compensates for angular distortion)
        a = np.array([0.0030, -0.0055, 2.6829, 48.04]) # angular distortion model coefficients
        col_polar = np.rad2deg(col_polar)
        col_polar = np.round(a[0]*np.power(col_polar,3) + a[1]*np.power(col_polar,2) + a[2]*col_polar+(1+a[3])+np.ones_like(col_polar)); 
 
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


