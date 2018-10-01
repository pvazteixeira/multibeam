# multibeam

A simple Python module to handle multibeam sonar images.

Features:
 * polar to Cartesian conversion 
 * support for non-linear mappings between beam number and bearing angle

Planned features:
 * background and object intensity distribution estimation
 * image segmentation 
 ** fixed threshold
 ** MAP
 ** MRF
 * attenuation compensation

## dependencies

* [numpy]() - general-purpose array manipulation
* [opencv]() - image processing
* [scipy]() -
* [lcm]() (optional) - required by `lcm-example.py` (the `hauv_multibeam_ping_t` lcm type will
  also be necessary - its python binding must be in the python path)
