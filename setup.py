from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

# Eigen path (use local copy or system install)
# Check multiple possible locations for Eigen
eigen_paths = [
    os.path.join(os.path.dirname(__file__), "vendor", "eigen"),  # Local vendor
    os.path.expanduser("~/Desktop/Projects/Monad/build_phase3/_deps/eigen-src"),  # From Monad project
    os.path.expanduser("~/Desktop/Projects/statelix/vendor/eigen"),  # From statelix
]

eigen_include = None
for p in eigen_paths:
    if os.path.exists(os.path.join(p, "Eigen")):
        eigen_include = p
        break

if eigen_include is None:
    print("WARNING: Eigen not found. SSJ features will not work.")
    print("Please install Eigen or create vendor/eigen with Eigen headers.")
    eigen_include = "vendor/eigen"  # Placeholder

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "monad_core",
        [
            "src/bindings.cpp",
            "src/monad/engine.cpp",
        ],
        include_dirs=["src", eigen_include],
        cxx_std=17,
        extra_compile_args=["/bigobj"] if os.name == 'nt' else [],
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
