# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudm_runtime SHARED
  cudm_runtime.cpp
)

# Use build/install-interface generator expressions: the raw source path is only
# valid while building from this source tree, whereas consumers of an installed
# cudaq-pulse resolve the header from the install prefix.
target_include_directories(cudm_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudm_runtime PRIVATE
  cuDensityMat::cuDensityMat
  CUDA::cudart
  ${CMAKE_DL_LIBS})

get_target_property(_cudensitymat_location
  cuDensityMat::cuDensityMat IMPORTED_LOCATION)
get_filename_component(_cudensitymat_library_dir
  "${_cudensitymat_location}" DIRECTORY)

# The installed library lives at site-packages/cudaq_pulse/lib and resolves
# cuDensityMat and the CUDA runtime from the wheels named by the [gpu] extra.
set_target_properties(cudm_runtime PROPERTIES
  OUTPUT_NAME "cudm-runtime"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_pulse/lib"
  BUILD_RPATH "${_cudensitymat_library_dir};${CUDAToolkit_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN/../../cuquantum/lib;$ORIGIN/../../nvidia/cuda_runtime/lib"
)

install(TARGETS cudm_runtime
  LIBRARY DESTINATION cudaq_pulse/lib
  COMPONENT CudaqPulse
)
