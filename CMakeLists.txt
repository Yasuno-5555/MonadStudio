cmake_minimum_required(VERSION 3.14)
project(monad_core)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization
if(MSVC)
    add_compile_options(/O2)
else()
    add_compile_options(-O3 -march=native -fopenmp)
endif()

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Define the module "monad_core"
# Now usage: src/python/bindings.cpp + src/monad/engine.cpp
pybind11_add_module(monad_core 
    src/python/bindings.cpp
    src/monad/engine.cpp
)

# Set library output to root for easy import
set_target_properties(monad_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
