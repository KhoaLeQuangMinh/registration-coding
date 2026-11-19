"""
Build Self-P2IR's two KPConv C++ helpers (grid subsampling, radius neighbours) for this machine.

Their setup.py needs numpy.distutils (removed from recent NumPy/setuptools), and the wrapper glue
uses NumPy 1 C-API spellings. We copy cpp_wrappers to a build folder, add a small compatibility
shim to the two wrapper.cpp files (no change to the neighbour or subsampling algorithms), build
there, and copy the .so files next to the original sources so `import cpp_wrappers...` works.

Usage: python build_cpp_wrappers.py <Self-P2IR checkout> <build folder>
"""
import glob
import os
import shutil
import sys

import numpy as np
from setuptools import Extension, setup

SHIM = """
// --- NumPy 1/2 compatibility shim (added for this build only) ---
#ifndef NPY_IN_ARRAY
#define NPY_IN_ARRAY NPY_ARRAY_IN_ARRAY
#endif
#define PyArray_NDIM(o) PyArray_NDIM((PyArrayObject*)(o))
#define PyArray_DIM(o, i) PyArray_DIM((PyArrayObject*)(o), i)
#define PyArray_DATA(o) PyArray_DATA((PyArrayObject*)(o))
// -----------------------------------------------------------------
"""

MODULES = {
    "cpp_subsampling": ("grid_subsampling", ["grid_subsampling/grid_subsampling.cpp", "wrapper.cpp"]),
    "cpp_neighbors": ("radius_neighbors", ["neighbors/neighbors.cpp", "wrapper.cpp"]),
}


def main():
    repo, build = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    src = os.path.join(repo, "cpp_wrappers")
    dst = os.path.join(build, "cpp_wrappers")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.so"))

    for folder, (name, sources) in MODULES.items():
        base = os.path.join(dst, folder)
        wrapper = os.path.join(base, "wrapper.cpp")
        text = open(wrapper).read()
        anchor = "#include <numpy/arrayobject.h>"
        assert anchor in text, f"{wrapper}: numpy include not found"
        open(wrapper, "w").write(text.replace(anchor, anchor + SHIM, 1))

        os.chdir(base)
        setup(name=name, script_args=["build_ext", "--inplace"],
              ext_modules=[Extension(name, sources=["../cpp_utils/cloud/cloud.cpp"] + sources,
                                     include_dirs=[np.get_include()], extra_compile_args=["-std=c++11"])])
        for so in glob.glob(os.path.join(base, f"{name}*.so")):
            shutil.copy2(so, os.path.join(src, folder))
            print("built", os.path.join(src, folder, os.path.basename(so)))


if __name__ == "__main__":
    main()
