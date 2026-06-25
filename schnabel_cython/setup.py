"""
setup.py — Build the schnabel_ransac Cython extension.

Usage:
    python setup.py build_ext --inplace

This compiles:
    1. bridge.cpp              (our thin bridge)
    2. All Schnabel .cpp files (except main.cpp)
    3. schnabel_ransac.pyx     (Cython → C++)

Into a single shared library: schnabel_ransac.so (or .pyd on Windows)
"""

import os
import glob
import platform

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# ------- Locate Schnabel C++ source -------
HERE = os.path.dirname(os.path.abspath(__file__))
SCHNABEL_DIR = os.path.join(
    HERE, "..", "Efficient-RANSAC-for-Point-Cloud-Shape-Detection"
)
SCHNABEL_DIR = os.path.abspath(SCHNABEL_DIR)

# Collect all .cpp files — EXCLUDE main.cpp (has its own main())
schnabel_sources = [
    f for f in sorted(glob.glob(os.path.join(SCHNABEL_DIR, "*.cpp")))
    if os.path.basename(f) != "main.cpp"
]
misclib_sources = sorted(
    glob.glob(os.path.join(SCHNABEL_DIR, "MiscLib", "*.cpp"))
)

# ------- Compiler flags -------
# Use -std=c++14 for compatibility with Schnabel's legacy code.
# -w suppresses all warnings from the 2009-era C++ code.
if platform.system() == "Windows":
    # Microsoft Visual C++ (MSVC) flags
    # We can likely use /O2 optimization safely on Windows x64.
    compile_args = ["/std:c++14", "/O2", "/w"]
else:
    # macOS/Linux (Clang/GCC) flags
    # We use -O0 because -O1/-O2 causes a strict-aliasing/alignment segfault in KdTree on Apple Silicon (arm64).
    compile_args = ["-g", "-std=c++14", "-O0", "-w", "-Wno-c++11-narrowing", "-fdelayed-template-parsing", "-fms-extensions"]
link_args = []
macros = [
    ("POINTSWITHINDEX", "1"),  # enables Point::index for tracking original indices
]

# macOS: Apple Clang does NOT support -fopenmp natively.
# If libomp is installed (brew install libomp), uncomment below:
# if platform.system() == "Darwin":
#     compile_args += ["-Xpreprocessor", "-fopenmp"]
#     link_args += ["-lomp"]
#     macros.append(("DOPARALLEL", "1"))
# else:
#     compile_args.append("-fopenmp")
#     link_args.append("-fopenmp")
#     macros.append(("DOPARALLEL", "1"))

# ------- Extension definition -------
ext = Extension(
    name="schnabel_ransac",
    sources=[
        "schnabel_ransac.pyx",
        "bridge.cpp",
    ] + schnabel_sources + misclib_sources,
    include_dirs=[
        SCHNABEL_DIR,                                   # for <PointCloud.h>, etc.
        os.path.join(SCHNABEL_DIR, "MiscLib"),          # for bare MiscLib includes
        HERE,                                           # for "bridge.h"
        np.get_include(),                               # for numpy C headers
    ],
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    define_macros=macros,
)

# ------- Setup -------
setup(
    name="schnabel_ransac",
    version="0.1.0",
    description="Cython wrapper for Schnabel's Efficient RANSAC (point-cloud shape detection)",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",       # Python 3 semantics
            "boundscheck": False,        # no bounds checks for speed
            "wraparound": False,         # no negative index wrap-around
        },
    ),
)
