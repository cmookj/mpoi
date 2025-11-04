load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_binary(
    name = "ex1",
    srcs = ["ex1.cc"],
    deps = [
        "//core:mpoi",
    ],
    copts = select({
        "@bazel_tools//src/conditions:windows": ["/std:c++20"],
        "//conditions:default": ["-std=c++20"],
    }),
    linkopts = ["-framework", "OpenCL"],
)
