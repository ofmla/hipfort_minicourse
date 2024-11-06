# hipfort_minicourse
Tutorial -- Minicourse on hipfort: a FORTRAN interface library for accessing GPU Kernels.

This course is strongly based on the excellent work from “Accelerated computing with Fortran and Hipfort” written by Dr. Toby Potter of Pelagos Consulting and Education and Dr. Joe Schoonover from Fluid Numerics, and some of the content—such as code, text, and figures—has been adapted from the original materials available at https://github.com/pelagos-consulting/Hipfort_Course under the the Creative Commons "Attribution-ShareAlike 4.0 International" license.

## Build instructions

To build the examples in this repository you need:

* A 2008 Fortran standard compliant Fortran compiler
* [ROCm](https://rocm.docs.amd.com)
* [HIPFort](https://github.com/ROCm/hipfort)
* GNU make

A `def file` is provided to help you build a custom container for running the example code. Since the systems available for course development only have NVIDIA GPUs, our `def file` retrieves a specific NVIDIA image/OS from Docker Hub using the Docker protocol and includes build instructions for setting up ROCm and HIPFort. While this setup has been tested to run on NVIDIA GPUs, please be aware that the container definition file may not be optimal for all use cases, particularly on systems with AMD hardware.

> As part of ongoing improvements, refining this container definition is on the to-do list

### Build the apptainer image

You can build the apptainer image, with the following command

```shell
$ apptainer build --warn-unused-build-args container_hipfort_minicourse.sif rocky8_cuda_hip.def
```

… then wait for apptainer to build an image for our minicourse apps (it might tight a take).

> Make sure apptainer is installed on your machine.

## Run example

After logging into the cluster, use the `salloc` command to allocate one node with GPUs, and then `ssh` to the allocated node for processing:
```shell
$ salloc -N 1 -A <account> -p <group of nodes that have GPUs>
$ ssh username@allocated_node
$ cd path_to_folder_with_apptainer_image
```
### Run the container

To run all the codes from the apptainer image use (note the use of the [--nv option](https://apptainer.org/docs/user/1.0/gpu.html#nvidia-gpus-cuda-standard))

```shell
$ apptainer shell --nv container_hipfort_minicourse.sif
<container_hipfort_minicourse.sif>[username@allocated_node hipfort_minicourse]$
```
Enter to the `example` folder and use `make` to compile the program, i.e., for the `veacadd` subfolder 
```shell
<container_hipfort_minicourse.sif>[username@allocated_node hipfort_minicourse]$ cd example/vecadd
<container_hipfort_minicourse.sif>[username@allocated_node hipfort_minicourse]$ FC=/usr/local/hipfort/bin/hipfc make
<container_hipfort_minicourse.sif>[username@allocated_node hipfort_minicourse]$ ./main
-- Running test 'vecadd' (Fortran 2008 interfaces)- device: Tesla V100-SXM2-32GB -  PASSED!
```
Alternatively you can use [CMake](https://cmake.org/) for building the `laplacian` and `tensoradd` examples
```shell
<container_hipfort_minicourse.sif>[username@allocated_node hipfort_minicourse]$ export HIP_PLATFORM=nvidia
<container_hipfort_minicourse.sif>[username@allocated_node hipfort_minicourse]$ export HIPFORT_ROOT=/usr/local/hipfort/
<container_hipfort_minicourse.sif>[username@allocated_node hipfort_minicourse]$ FC=/usr/local/hipfort/bin/hipfc cmake -B _build -DCMAKE_INSTALL_MESSAGE=LAZY -DCMAKE_VERBOSE_MAKEFILE=OFF -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_INSTALL_PREFIX=${PWD} -DCMAKE_BUILD_TYPE=Release -Dhip_DIR=/opt/rocm/lib/cmake/hip
<container_hipfort_minicourse.sif>[username@allocated_node hipfort_minicourse]$ cmake --build _build
<container_hipfort_minicourse.sif>[username@allocated_node hipfort_minicourse]$ cmake --install _build
```
## License

These materials are for a course intended to provide a briew introduction to hipfort. The course is aimed at a scientific audience. Comments, corrections, and additions are welcome.

All code in this repository is licensed under the MIT License, except for the `tensoradd` example, which is provided under the Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0) International license.

Unless otherwise specified, other (non-code) content is available under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png

The full text of these licenses is provided in the [LICENSE](https://github.com/ofmla/hipfort_minicourse/blob/main/LICENSE) file.
