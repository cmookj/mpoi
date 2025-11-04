load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "mpoi",
    srcs = ["mpoi.cc"],
    hdrs = ["mpoi.h"],
    visibility = ["//visibility:public"],
    linkopts = ["-framework", "OpenCL"],
    copts = select({
        "@bazel_tools//src/conditions:windows": ["/std:c++17"],
        "//conditions:default": ["-framework", "OpenCL", "-std=c++17"],
    }),
)
