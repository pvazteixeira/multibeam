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
didson.loadConfig('data/DIDSON/didson.json')

# load image
ping = cv2.imread('data/DIDSON/tuna_can.png',cv2.CV_LOAD_IMAGE_GRAYSCALE)

# show original image
plt.figure()
plt.subplot(1,4,1)
plt.imshow(ping)

# convert original image to Cartesian
ping_cart = didson.toCart(ping) 
plt.subplot(1,4,2)
plt.imshow(ping_cart)

# define sonar's psf (sonar-specific)
# beam_pattern_indices = np.array([0,8,16,24,32,40,48,56,64,72,80,88])
# beam_pattern =  np.zeros((1,96))
# beam_pattern[0, beam_pattern_indices] = [24,24,24,27,32,40,70,40,32,27,24,24]
# psf = (1.0/np.sum(beam_pattern))*beam_pattern
# didson.psf = psf

# deconvolve
ping_deconv = didson.deconvolve(ping)

# convert enhanced image to Cartesian
ping_deconv_cart = didson.toCart(ping_deconv)

# show enhanced image
plt.subplot(1,4,3)
plt.imshow(ping_deconv_cart)

print np.amax(ping_deconv_cart)
# segment ping
binary_ping = didson.segment(ping_deconv,100.0)
print np.amin(binary_ping), np.amax(binary_ping)
binary_ping_cart = didson.toCart(binary_ping)
plt.subplot(1,4,4)
plt.imshow(binary_ping_cart)

# display results
plt.show()


cv2.imwrite('ping.png',ping_deconv_cart)
