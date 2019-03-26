# multibeam

![Example scan (left to right): original, pre-processed, MAP classification](https://raw.githubusercontent.com/pvazteixeira/multibeam/feature/mrf/images/scans.png)

A simple Python module to handle multibeam sonar images.

Features:
 * polar to Cartesian conversion 
 * support for non-linear mappings between beam number and bearing angle
 * background and object intensity distribution estimation
 * MAP segmentation

## dependencies

* [numpy]() - general-purpose array manipulation
* [opencv]() - image processing
* [scipy]() -
