# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

nanobind_add_module(
  _cudaq_pulse_native
  NB_STATIC
  bindings.cpp
)

target_link_libraries(_cudaq_pulse_native PRIVATE
  CudaqPulsePulse
  CudaqPulsePulseTransforms
  CudaqPulseQOp
  CudaqPulseCuDensityMat
  CudaqPulsePulseToQOp
  CudaqPulseQOpToCuDensityMat
  CudaqPulseCuDensityMatToLLVM
  MLIRExecutionEngine
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRParser
  MLIRSCFDialect
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)

set_target_properties(_cudaq_pulse_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_native"
)

set(_native_dir "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_native")
nanobind_add_stub(
  _cudaq_pulse_native_stub
  MODULE _cudaq_pulse_native
  OUTPUT "${_native_dir}/_cudaq_pulse_native.pyi"
  PYTHON_PATH "${_native_dir}"
  DEPENDS _cudaq_pulse_native
)

install(
  TARGETS _cudaq_pulse_native
  LIBRARY DESTINATION cudaq_pulse/_native
  COMPONENT CudaqPulse
)
install(
  FILES "${_native_dir}/_cudaq_pulse_native.pyi"
  DESTINATION cudaq_pulse/_native
  COMPONENT CudaqPulse
)
