from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "monad_core",
        [
            "src/python/bindings.cpp",
            "src/monad/engine.cpp",
        ],
        include_dirs=["src"],
        cxx_std=17,
        # On Windows, /O2 is default for Release.
    ),
]

setup(
    name="monad_core",
    version="0.0.1",
    author="MonadAI",
    description="Monad Core Binding",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
