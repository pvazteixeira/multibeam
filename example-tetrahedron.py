from multibeam import Sonar
import numpy as np
import cv2
import matplotlib.pyplot as plt
# examples for:
"""
Short example on the use of the multibeam module 
"""

# instantiate a sonar object
tetrahedron = Sonar();
tetrahedron.loadConfig('data/tetrahedron/tetrahedron.json')

# load image
ping = cv2.imread('data/tetrahedron/chirp_7-9.png',cv2.CV_LOAD_IMAGE_GRAYSCALE)

# show original image
plt.figure()
plt.subplot(1,4,1)
plt.imshow(ping)

# convert original image to Cartesian
ping_cart = tetrahedron.toCart(ping) 
plt.subplot(1,4,2)
plt.imshow(ping_cart)

# deconvolve
ping_deconv = tetrahedron.deconvolve(ping)

# # convert enhanced image to Cartesian
ping_deconv_cart = tetrahedron.toCart(ping_deconv)

# # show enhanced image
plt.subplot(1,4,3)
plt.imshow(ping_deconv_cart)

# # segment ping
# binary_ping = tetrahedron.segment(ping_deconv,100.0)
# binary_ping_cart = tetrahedron.toCart(binary_ping)
# plt.subplot(1,4,4)
# plt.imshow(binary_ping_cart)

# display results
plt.show()

# save results
cv2.imwrite('ping.png',ping_deconv_cart)
