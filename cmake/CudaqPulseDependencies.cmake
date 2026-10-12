# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard(DIRECTORY)

# Build-time dependency discovery for the GPU runtime, Python, and nanobind.
# MLIR/LLVM is resolved in the top-level CMakeLists.txt.

# The GPU runtime is an explicit choice, never auto-detected: OFF configures
# the compiler-only package, ON makes every GPU dependency REQUIRED.
option(CUDAQ_PULSE_ENABLE_GPU
  "Build the cuDensityMat GPU runtime (requires the CUDA toolkit and cuDensityMat)"
  OFF)

if(CUDAQ_PULSE_ENABLE_GPU)
  find_package(CUDAToolkit REQUIRED)
  find_package(cuDensityMat REQUIRED)
  message(STATUS
    "cudaq-pulse GPU runtime enabled with cuDensityMat ${cuDensityMat_VERSION} "
    "and CUDA ${CUDAToolkit_VERSION}")
  add_subdirectory(core/runtime)
endif()

# nanobind resolves its Python dependency through FindPython, whereas MLIR and
# the top-level project use FindPython3. Seed the former from the latter so both
# modules agree on a single interpreter.
if(NOT Python_EXECUTABLE)
  set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
endif()
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind ships its own CMake package inside the Python wheel. Ask the
# interpreter where it is, while still letting -Dnanobind_DIR override.
if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE _cudaq_pulse_nanobind_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _cudaq_pulse_nanobind_error
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _cudaq_pulse_nanobind_status)
  if(NOT _cudaq_pulse_nanobind_status EQUAL 0)
    message(FATAL_ERROR
      "nanobind was not found by ${Python3_EXECUTABLE}: "
      "${_cudaq_pulse_nanobind_error}\n"
      "Install it with `pip install \"nanobind>=2.12\"`, or point at an "
      "existing CMake package with -Dnanobind_DIR=<dir>.")
  endif()
  set(nanobind_DIR "${_cudaq_pulse_nanobind_dir}")
endif()
find_package(nanobind CONFIG REQUIRED)
