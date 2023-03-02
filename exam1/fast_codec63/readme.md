# Home Exam 1

## The makefile
Running `make all` or just `make` compiles the encoder, decoder and predictor. Running `make clean` deletes build files 

## Non-SIMD changes
We have done some modifications to the codec that do not change the speed of the program. We moved `sad_block_8x8` from `dsp.c` to `me.c`, and
renamed `dsp` to `cosine_transform` and `me` to `motion_estimate`. We also moved dct-related functions from `common.c` to `cosine_transform.c`, renamed `common` to `utils`, and added two new functions to `utils.c`. One of those was `errcheck_fopen`, which calls `fopen`, checks for errors, and if an error occurs, it prints an error message and exits the program. The other function is `free_yuv`, which just frees a `yuv_t` pointer. We also made a new function, `cl_utils.c`, which defines some functions that handle parsing command line inputs. Using the new utility functions, we simplified `c63enc.c`. We also added a new command line input to `c63enc.c`, `-p`,
which tells the program to encode multiple times in a row while timing each run, so it can calculate the mean runtime, and the standard deviation. 
