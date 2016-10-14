from multibeam import Sonar
import numpy as np
import cv2
import matplotlib.pyplot as plt
# examples for:
"""
Short example on the use of the multibeam module 
"""

# instantiate a sonar object
didson = Sonar();

# set its parameters
didson.fov = np.deg2rad(28.8)
didson.num_beams = 96
didson.num_bins = 512

# update the sonar window parameters
didson.resetWindow(2.25, 11.75) 

# load image
ping = cv2.imread('data/DIDSON/tuna_can.png',cv2.CV_LOAD_IMAGE_GRAYSCALE)

# show original image
plt.figure()
plt.subplot(1,3,1)
plt.imshow(ping)

# convert original image to Cartesian
ping_cart = didson.toCart(ping) 
plt.subplot(1,3,2)
plt.imshow(ping_cart)

# define sonar's psf (sonar-specific)
beam_pattern_indices = np.array([0,8,16,24,32,40,48,56,64,72,80,88])
beam_pattern =  np.zeros((1,96))
beam_pattern[0, beam_pattern_indices] = [24,24,24,27,32,40,70,40,32,27,24,24]
psf = (1.0/np.sum(beam_pattern))*beam_pattern
didson.psf = psf

# deconvolve
ping_deconv = didson.deconvolve(ping)

# convert enhanced image to Cartesian
ping_deconv_cart = didson.toCart(ping_deconv)

# show enhanced image
plt.subplot(1,3,3)
plt.imshow(ping_deconv_cart)

# display results
plt.show()

