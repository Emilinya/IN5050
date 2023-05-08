# Testing
Fast encoding can be compared to reference encoding by running `make test`. This defaults to 10 frames, but that can be changed with the `COMP_FRAMES` variable.
Comparison requires reference codec, which can be downloaded at https://bitbucket.org/mpg_code/in5050-codec63/src. Comparison also uses the image `foreman.yuv`,
which can be downloaded from https://bitbucket.org/mpg_code/in5050-mjpeg.

# Profiling
The fast encoding can be profiled by running `make profile_fast`. The reference codec can be profiled by running `make profile_reference`, but this requires
adding `-pg` to the c flags in the makefile of the reference and fast codec.
