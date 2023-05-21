# Home exam 3

## Build instructions
The `run.sh` script will compile and launch the encoder on both nodes. `c63enc`
will be run on the PC, and `c63server` will be launched on the tegra. To specify
the cluster to run on, specify the `--tegra` parameter. The x86 node is always in5050-2016-10,
as the other machine did not seem to work. To pass arguments to `c63enc`, use `--args "arg1 arg2"`.

Example usage: \
`./run.sh --tegra tegra-1 --args "/mnt/sdcard/foreman.yuv -o output -w 352 -h 288"`


## Makefiles
Most of the interesting makefile rules are in `Makefile.include`. This file is
included from the tegra and x86 specific makefiles in `tegra-build` and
`x86-build`. There are make-rules that create symlinks from the `x86-build` and
`tegra-build` directories to the "real" source files. This prevents the `.o`
files from colliding if we're building on an NFS mount.

