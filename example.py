from multibeam import Sonar
import numpy as np
import cv2
import matplotlib.pyplot as plt
# examples for:
# smc didson
# teledyne/blueview mb-2250


didson = Sonar();

# initialize beam pattern
beam_pattern_indices = np.array([0,8,16,24,32,40,48,56,64,72,80,88])
beam_pattern =  np.zeros((1,96))
beam_pattern[0, beam_pattern_indices] = [24,24,24,27,32,40,70,40,32,27,24,24]
# define PSF
psf = (1.0/np.sum(beam_pattern))*beam_pattern

didson.psf = psf
didson.min_range = 2.25
didson.max_range = 11.75 

didson.fov = np.deg2rad(28.8)
didson.num_beams = 96
didson.num_bins = 512
didson.__computeLookUp__(0.01)

# load image
img = cv2.imread('data/DIDSON/tuna_can.png',cv2.CV_LOAD_IMAGE_GRAYSCALE)

# show original image
plt.figure()
plt.subplot(1,3,1)
plt.imshow(img)

# convert original image to Cartesian
img_cart = didson.toCart(img) 
plt.subplot(1,3,2)
plt.imshow(img_cart)

print didson.psf

# deconvolve
img_deconv = didson.deconvolve(img)

# convert enhanced image to Cartesian
img_deconv_cart = didson.toCart(img_deconv)

# show enhanced image
plt.subplot(1,3,3)
plt.imshow(img_deconv_cart)
plt.show()

