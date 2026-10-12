#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Runs once per manylinux_2_28 container before cibuildwheel builds any wheel.
# Installs exactly the build-time GPU dependencies the [gpu] extra pins at run
# time, and unpacks the checksum-verified MLIR toolchain. Any failure aborts
# the wheel build.

set -euo pipefail

CUDA_VERSION="12-9"
CUDENSITYMAT_VERSION="0.7.0"

case "$(uname -m)" in
  x86_64) cuda_repo_arch="x86_64" ;;
  aarch64) cuda_repo_arch="sbsa" ;;
  *) echo "error: unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

# CUDA toolkit pieces needed to compile and link against the CUDA runtime.
dnf config-manager --add-repo \
  "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${cuda_repo_arch}/cuda-rhel8.repo"
dnf install -y "cuda-nvcc-${CUDA_VERSION}" "cuda-cudart-devel-${CUDA_VERSION}"

# cuDensityMat headers and library, from the same wheel the [gpu] extra installs.
python3.12 -m pip install --no-deps --target /opt/cudensitymat \
  "cudensitymat-cu12==${CUDENSITYMAT_VERSION}"
test -f /opt/cudensitymat/cuquantum/include/cudensitymat.h
test -f /opt/cudensitymat/cuquantum/lib/libcudensitymat.so.0

# The workflow downloads the toolchain archive into .cache/mlir/downloads
# before cibuildwheel starts; this verifies its checksum and unpacks it.
python3.12 tools/fetch-mlir-toolchain.py
