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
from bot_core import planar_lidar_t

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
    # TODO: replace hard-coded conversions with "astype"
    ping = ping.astype(np.float64)
    ping+=32768.0
    ping/=65535.0

    # deconvolve
    ping_deconv = didson.deconvolve(ping)

    # map back to original range
    # ping_deconv*=65535.0 
    # ping_deconv-=32768.0
    # ping_deconv=ping_deconv.astype(np.int16)
    # ping_deconv.shape=(msg.height*msg.width)

    # classify
    # ideally just a call to getReturns
    ping_binary = ping_deconv;
    ping_binary[ping_binary<0.35] = 0
    # pings are (512,96)
    intensities = np.amax(ping_binary,axis=0)
    ranges = np.argmax(ping_binary, axis=0)
    dr = (didson.max_range - didson.min_range)/(didson.num_bins + 0.0)
    ranges = ranges*dr;
    ranges[ranges<=0] = -didson.min_range # no return here
    ranges += didson.min_range*np.ones(msg.width)

    # create outgoing message
    msg_out = planar_lidar_t() 
    msg_out.utime = msg.time
    msg_out.nranges = didson.num_beams
    msg_out.ranges = ranges #np.zeros(didson.num_beams)

    msg_out.nintensities = didson.num_beams
    msg_out.intensities = intensities #np.zeros(didson.num_beams)

    msg_out.rad0 = didson.fov/2.0 
    msg_out.radstep = -didson.fov/(didson.num_beams + 0.0)

    # publish
    lcm_node.publish("MULTIBEAM_RETURNS", msg_out.encode()) 

    # img = didson.toCart(ping)

    # cv2.imshow('ping',img)
    # cv2.waitKey(1)

if __name__ == '__main__':    

    global lcm_node, didson
    # instantiate a sonar object
    didson = Didson();
    didson.loadConfig('data/DIDSON/didson.json')

    lcm_node = lcm.LCM()
    ping_subscription = lcm_node.subscribe("MULTIBEAM_PING",pingHandler)

    # this breaks with the conda-installed version of opencv
    # cv2.namedWindow('ping',cv2.WINDOW_NORMAL)
    
    try:
        while True:
            lcm_node.handle()
    except KeyboardInterrupt:
        pass

