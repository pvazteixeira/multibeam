#!/usr/bin/env python

"""
Small example on the use of the multibeam module with LCM.

This script subscribes to lcm messages of the type hauv_multibeam_t on the
channel MULTIBEAM_PING. Each received ping is processed using the multibeam.py
functionality, and republished as a message of the original type
(hauv_multibeam_t) on the MULTIBEAM_PING_ENHANCED channel.
"""

from multibeam import Sonar
from didson import Didson
import numpy as np
import cv2
import lcm
from hauv import multibeam_ping_t

__author__     = "Pedro Vaz Teixeira"
__copyright__  = ""
__credits__    = ["Pedro Vaz Teixeira"]
__license__    = ""
__version__    = "1.0.0"
__maintainer__ = "Pedro Vaz Teixeira"
__email__      = "pvt@mit.edu"
__status__     = "Development"


def pingHandler(channel, data):

    global lcm_node, didson 
    msg = multibeam_ping_t.decode(data)

    ping = np.copy(np.asarray(msg.image, dtype=np.int16))
    ping.shape = (msg.height, msg.width)

    # convert to [0,1] range 
    ping = ping.astype(np.float64)
    ping+=32768.0
    ping/=65535.0

    # deconvolve
    ping_deconv = didson.deconvolve(ping)

    # map back to original range
    ping_deconv*=65535.0 
    ping_deconv-=32768.0

    ping_deconv=ping_deconv.astype(np.int16)
    ping_deconv.shape=(msg.height*msg.width)

    # create outgoing message
    msg_out = msg 

    msg_out.image = np.copy(np.asarray(ping_deconv, dtype=np.int16))

    # publish
    lcm_node.publish("MULTIBEAM_PING_ENHANCED", msg_out.encode()) 

    img = didson.toCart(ping)

    # cv2.imshow('ping',img)
    # cv2.waitKey(1)

if __name__ == '__main__':    

    global lcm_node, didson
    # instantiate a sonar object
    didson = Didson();
    didson.loadConfig('data/DIDSON/didson.json')

    lcm_node = lcm.LCM()
    ping_subscription = lcm_node.subscribe("MULTIBEAM_PING",pingHandler)

    # cv2.namedWindow('ping',cv2.WINDOW_NORMAL)
    
    try:
        while True:
            lcm_node.handle()
    except KeyboardInterrupt:
        pass

