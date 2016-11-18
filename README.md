# multibeam

A Python module to handle multibeam images.

## dependencies

Required:
* json - used to store sonar configurations
* numpy - general-purpose array manipulation
* opencv - image processing  

Optional:
* lcm - required by `lcm-example.py` (the `hauv_multibeam_ping_t` lcm type will
  also be necessary - its python binding must be in the python path)
