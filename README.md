# multibeam

A Python module to handle multibeam images.

## dependencies

* json
* numpy
* opencv 

## usage

The file `example.py` shows how to use the multibeam module for operations like conversion from polar to Cartesian. 

The code below creates a sonar object named `didson` and configures it using the `didson.json` configuration file:
```python
# instantiate a sonar object
didson = Sonar();
didson.loadConfig('data/DIDSON/didson.json')
```
The contents of the `didson.json` detail sonar configuration, such as window parameters:

```json
"fov": 0.50265482457436694, 
"max_range": 11.75, 
"min_range": 2.25, 
"num_beams": 96, 
"num_bins": 512, 
"noise": 0.01, 
"psf": [ [1.0 ]]
```

The last two parameters, `noise` and `psf` are used to deconvolve the sonar's beam pattern from incoming scans.

## notes

Image columns are assumed to correspond to individual sonar beams (and, consequently, image rows correspond to constant-range arcs). In other words, pixel i,j of the original sonar scan (in polar coordinates) is assumed to correspond to the beam i's jth range bin. The first column corresponds to the beam aimed at -FOV/2.

## references
