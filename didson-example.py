from sonar import Sonar
from didson import Didson
import numpy as np
import cv2
import matplotlib.pyplot as plt
# examples for:
"""
Short example on the use of the multibeam module 
"""

# instantiate a sonar object
didson = Didson();
didson.loadConfig('data/DIDSON/didson.json')

# load image
ping = cv2.imread('data/DIDSON/tuna_can.png',cv2.CV_LOAD_IMAGE_GRAYSCALE)

# show original image
plt.figure()
plt.subplot(1,4,1)
plt.imshow(ping)
plt.title('Original - polar')

# convert original image to Cartesian
ping_cart = didson.toCart(ping) 
plt.subplot(1,4,2)
plt.imshow(ping_cart)
plt.title('Original - Cartesian')

# deconvolve
ping = ping.astype(np.float64)
ping/=255.0
ping_deconv = didson.deconvolve(ping)
ping_deconv*=255.0
ping_deconv = ping_deconv.astype(np.uint8)

# convert enhanced image to Cartesian
ping_deconv_cart = didson.toCart(ping_deconv)

# show enhanced image
plt.subplot(1,4,3)
plt.imshow(ping_deconv_cart)
plt.title('Enhanced - Cartesian')


# segment ping
binary_ping = didson.segment(ping_deconv,100.0)
binary_ping_cart = didson.toCart(binary_ping)
plt.subplot(1,4,4)
plt.imshow(binary_ping_cart)
plt.title('Segmented - Cartesian')


# display results
plt.show()

# save results
cv2.imwrite('ping.png',ping_deconv_cart)
