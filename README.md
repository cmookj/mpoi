# MPOI: Multi-Processing Object Interface

This library helps writing a program which uses OpenCL for parallel computation.

## How to Build

This repository depends on Bazel.

Build MPOI library:
```shell
bazel build --compilation_mode=opt --cxxopt=-std=c++17 //core:mpoi
```

## Example Programs

Note: the example programs require C++20 because it uses `std::format`.

### Example 1

A silly example only to show how to use the MPOI library.

```shell
bazel build --compilation_mode=opt --cxxopt=-std=c++20 //examples:ex1
bazel-bin/examples/ex1
```

The example program compares the computational performance between CPU and GPU.
The output of the program on M4 Pro:
```shell
================================ S U M M A R Y =================================
  Parallel (msec)      Serial (msec)        Ratio (P/S)          Difference
--------------------------------------------------------------------------------
        127                 441              0.28798187         1.403664e-14
         74                 442              0.16742082         1.403664e-14
         57                 440              0.12954545         1.403664e-14
         59                 473              0.12473573         1.403664e-14
         63                 455              0.13846155         1.403664e-14
         63                 441              0.14285715         1.403664e-14
         59                 441              0.13378684         1.403664e-14
         57                 438              0.13013698         1.403664e-14
         59                 442              0.13348417         1.403664e-14
         57                 440              0.12954545         1.403664e-14
--------------------------------------------------------------------------------
        67.5               445.3             0.1517956
```

### Example 2

2D image processing example which shows a $9\times9$ Gaussian blur (convolution).

```shell
bazel build --compilation_mode=opt --cxxopt=-std=c++20 //examples:ex2
bazel-bin/examples/ex2
```

Output of the program on M4 Pro:
```shell
Running time for serial computation = 24 msec
1 OpenCL platform(s) found.
Platform # 0
1 GPU device(s) found for platform 0x7fff0000
Device vendor ID: 16940800
Device has 20 compute units.
Local item size = 128
Running time for parallel computation = 3 msec
```
