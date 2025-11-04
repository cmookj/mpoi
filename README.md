# MPOI: Multi-Processing Object Interface

This library helps writing a program which uses OpenCL for parallel computation.

## How to Build

This repository depends on Bazel.

Build MPOI library:
```shell
bazel build --compilation_mode=opt --cxxopt=-std=c++17 //core:mpoi
```

Build example program:
```shell
bazel build --compilation_mode=opt --cxxopt=-std=c++20 //examples:ex1
```

Note: the example program requires C++20 because it uses `std::format`.

## Example Program

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
