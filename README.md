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
apptainer build --warn-unused-build-args container_hipfort_minicourse.sif rocky8_cuda_hip.def
```

… then wait for apptainer to build an image for our apps (it might tight a take).

> Make sure apptainer is installed on your machine.

### Run the container

To run all the codes from the appatiner image use

```shell
apptainer shell --nv container_hipfort_minicourse.sif
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
