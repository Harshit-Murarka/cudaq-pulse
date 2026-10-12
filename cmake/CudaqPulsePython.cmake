# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard(DIRECTORY)

include(CudaqPulseStagePython)

# The build tree mirrors the installed package: one PYTHONPATH entry exposes
# the staged Python sources, the native extension, and the GPU runtime.
file(GLOB_RECURSE _cudaq_pulse_python_sources
  RELATIVE "${CUDAQ_PULSE_FRONTEND_DIR}"
  "${CUDAQ_PULSE_FRONTEND_DIR}/*.py"
  "${CUDAQ_PULSE_FRONTEND_DIR}/*.pyi")
list(APPEND _cudaq_pulse_python_sources py.typed)

cudaq_pulse_stage_python_sources(CudaqPulsePythonStaging
  ROOT_DIR "${CUDAQ_PULSE_FRONTEND_DIR}"
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_pulse"
  SOURCES ${_cudaq_pulse_python_sources})
add_dependencies(_cudaq_pulse_native CudaqPulsePythonStaging)

set(CUDAQ_PULSE_PYTHONPATH "${CMAKE_BINARY_DIR}/python")

if(SKBUILD)
  set(CUDAQ_PULSE_PACKAGE_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
else()
  set(CUDAQ_PULSE_PACKAGE_VERSION "0.0.0+cmake")
endif()

set(_cudaq_pulse_build_targets
  cudaq-pulse-opt
  _cudaq_pulse_native
  _cudaq_pulse_native_stub
  CudaqPulsePythonStaging)

if(CUDAQ_PULSE_ENABLE_GPU)
  list(APPEND _cudaq_pulse_build_targets cudm_runtime)
  set(CUDAQ_PULSE_GPU_RUNTIME_ENABLED_PY True)
  set(CUDAQ_PULSE_GPU_RUNTIME_LIBRARY_PY
    "\"lib/${CMAKE_SHARED_LIBRARY_PREFIX}cudm-runtime${CMAKE_SHARED_LIBRARY_SUFFIX}\"")
  set(CUDAQ_PULSE_CUDENSITYMAT_VERSION_PY "\"${cuDensityMat_VERSION}\"")
  set(CUDAQ_PULSE_CUDA_VERSION_PY "\"${CUDAToolkit_VERSION}\"")
else()
  set(CUDAQ_PULSE_GPU_RUNTIME_ENABLED_PY False)
  set(CUDAQ_PULSE_GPU_RUNTIME_LIBRARY_PY None)
  set(CUDAQ_PULSE_CUDENSITYMAT_VERSION_PY None)
  set(CUDAQ_PULSE_CUDA_VERSION_PY None)
endif()

configure_file(
  "${CUDAQ_PULSE_SOURCE_DIR}/cmake/_build_config.py.in"
  "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_build_config.py"
  @ONLY)

add_custom_target(pulse DEPENDS ${_cudaq_pulse_build_targets})

install(DIRECTORY "${CUDAQ_PULSE_FRONTEND_DIR}"
  DESTINATION .
  COMPONENT CudaqPulse
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)
install(FILES "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_build_config.py"
  DESTINATION cudaq_pulse
  COMPONENT CudaqPulse)
