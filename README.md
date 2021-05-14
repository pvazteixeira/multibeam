# multibeam

![Example scan (left to right): original, pre-processed, MAP classification](https://raw.githubusercontent.com/pvazteixeira/multibeam/feature/mrf/images/scans.png)

A simple Python module to handle pre-processing and segmentation of multibeam sonar images.

Features:
 * polar to Cartesian conversion 
 * support for non-linear mappings between beam number and bearing angle
 * background and object intensity distribution estimation (via mixture model)
 * MAP and MRF segmentation

## dependencies

The main dependencies for this module are:

* [numpy](https://numpy.org) - general-purpose array manipulation
* [opencv](https://opencv.org) - image processing
* [scipy](https://www.scipy.org)

The `environment.yml` file contains the [conda](http://conda.io) environment specification for this package. 


##  references

The principles behind this package are detailed in 

```
@inproceedings{teixeira2018multibeam,
  title =        {Multibeam Data Processing for Underwater Mapping},
  pages =        {1877-1884},
  url =          {https://doi.org/10.1109/iros.2018.8594128},
  ISSN =         {2153-0866},
  doi =          {10.1109/iros.2018.8594128},
  author =       {Pedro V. Teixeira and Michael Kaess and Franz S. Hover and John J. Leonard},
  title =        {Multibeam Data Processing for Underwater Mapping},
  booktitle =    {{IEEE}/{RSJ} Int. Conf. Intelligent Robots and Systems ({IROS})},
  year =         2018,
  month =        Oct,
}
```
