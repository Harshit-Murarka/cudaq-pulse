# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Single gate for every GPU execution path.

Whether the GPU runtime exists is recorded by CMake in ``_build_config``; it
is never discovered by searching the filesystem. ``require_gpu_runtime()``
either returns a loaded, device-backed runtime or raises one of three
distinct errors, so "built without GPU support", "runtime failed to load",
and "no usable device" are never confused.

The CUDA libraries the runtime depends on are installed by the ``[gpu]``
extra as separate wheels whose directories are not all on cuDensityMat's
RUNPATH, so they are loaded from those wheels first, in dependency order.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import _build_config

__all__ = [
    "GpuDeviceUnavailableError",
    "GpuRuntime",
    "GpuRuntimeError",
    "GpuRuntimeLoadError",
    "GpuSupportNotBuiltError",
    "require_gpu_runtime",
]

_OVERRIDE_ENV = "CUDM_RUNTIME_LIB"

# Wheel-relative paths, ordered so that each library's DT_NEEDED entries are
# already resident when it loads.
_CUDA_WHEEL_LIBRARIES = (
    "nvidia/nvjitlink/lib/libnvJitLink.so.12",
    "nvidia/cuda_runtime/lib/libcudart.so.12",
    "nvidia/cublas/lib/libcublasLt.so.12",
    "nvidia/cublas/lib/libcublas.so.12",
    "nvidia/cusparse/lib/libcusparse.so.12",
    "nvidia/cusolver/lib/libcusolver.so.11",
    "nvidia/curand/lib/libcurand.so.10",
    "cutensor/lib/libcutensor.so.2",
    "cuquantum/lib/libcutensornet.so.2",
    "cuquantum/lib/libcudensitymat.so.0",
)


class GpuRuntimeError(RuntimeError):
    """Base class for every reason GPU execution cannot proceed."""


class GpuSupportNotBuiltError(GpuRuntimeError):
    """This cudaq-pulse build does not contain the GPU runtime."""


class GpuRuntimeLoadError(GpuRuntimeError):
    """The GPU runtime is part of this build but could not be loaded."""


class GpuDeviceUnavailableError(GpuRuntimeError):
    """The GPU runtime loaded, but no usable NVIDIA device or driver exists."""


@dataclass(frozen=True)
class GpuRuntime:
    """A loaded cuDensityMat runtime with at least one usable device."""

    library: ctypes.CDLL
    path: Path
    device_count: int


_lock = threading.Lock()
_runtime: GpuRuntime | None = None


def require_gpu_runtime() -> GpuRuntime:
    """Return the loaded GPU runtime, or raise a specific ``GpuRuntimeError``."""
    global _runtime
    with _lock:
        if _runtime is None:
            path = _runtime_library_path()
            not_in_wheels = _preload_cuda_wheels() if sys.platform.startswith("linux") else []
            library = _load(path, not_in_wheels)
            _runtime = GpuRuntime(
                library=library, path=path, device_count=_device_count(library, path)
            )
        return _runtime


def _runtime_library_path() -> Path:
    override = os.environ.get(_OVERRIDE_ENV)
    if override:
        path = Path(override)
        if not path.is_file():
            raise GpuRuntimeLoadError(
                f"{_OVERRIDE_ENV} is set to {override!r}, which is not a file."
            )
        return path
    if not _build_config.GPU_RUNTIME_ENABLED:
        raise GpuSupportNotBuiltError(
            "This cudaq-pulse build does not include GPU support, so it can "
            "compile pulse programs but not execute them. Install a wheel "
            'with GPU support via `pip install "cudaq-pulse[gpu]"` on Linux, '
            "or rebuild from source with -DCUDAQ_PULSE_ENABLE_GPU=ON."
        )
    assert _build_config.GPU_RUNTIME_LIBRARY is not None
    path = Path(__file__).resolve().parent.parent / _build_config.GPU_RUNTIME_LIBRARY
    if not path.is_file():
        raise GpuRuntimeLoadError(
            f"cudaq-pulse was built with GPU support, but its runtime library "
            f"is missing from {path}. The installation is incomplete; "
            f"reinstall cudaq-pulse."
        )
    return path


def _locate_cuda_wheel_libraries(
    package_roots: Callable[[str], Sequence[Path]],
) -> tuple[list[Path], list[str]]:
    """Split ``_CUDA_WHEEL_LIBRARIES`` into files found under the installed
    packages (in load order) and the names of those that are not installed."""
    found: list[Path] = []
    missing: list[str] = []
    for relative in _CUDA_WHEEL_LIBRARIES:
        top, _, rest = relative.partition("/")
        candidates = [root / rest for root in package_roots(top)]
        match = next((c for c in candidates if c.is_file()), None)
        if match is None:
            missing.append(Path(relative).name)
        else:
            found.append(match)
    return found, missing


def _installed_package_roots(top: str) -> list[Path]:
    spec = importlib.util.find_spec(top)
    if spec is None or spec.submodule_search_locations is None:
        return []
    return [Path(location) for location in spec.submodule_search_locations]


def _preload_cuda_wheels() -> list[str]:
    found, missing = _locate_cuda_wheel_libraries(_installed_package_roots)
    for library in found:
        try:
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            raise GpuRuntimeLoadError(
                f"cudaq-pulse could not load the CUDA library {library}: "
                f'{error}. Reinstall with `pip install "cudaq-pulse[gpu]"`.'
            ) from error
    return missing


def _load(path: Path, not_in_wheels: Sequence[str]) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        detail = ""
        if not_in_wheels:
            detail = (
                f" These CUDA libraries are not installed as Python "
                f"packages, so they had to come from the system library "
                f"path: {', '.join(not_in_wheels)}."
            )
        raise GpuRuntimeLoadError(
            f"cudaq-pulse could not load its GPU runtime {path}: {error}.{detail} "
            f"Install the GPU dependencies with "
            f'`pip install "cudaq-pulse[gpu]"`.'
        ) from error


def _device_count(library: ctypes.CDLL, path: Path) -> int:
    get_count = library.cudaGetDeviceCount
    get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_count.restype = ctypes.c_int
    error_string = library.cudaGetErrorString
    error_string.argtypes = [ctypes.c_int]
    error_string.restype = ctypes.c_char_p

    count = ctypes.c_int(0)
    status = get_count(ctypes.byref(count))
    if status != 0:
        raw = error_string(status)
        reason = raw.decode("utf-8", errors="replace") if raw else "unknown"
        raise GpuDeviceUnavailableError(
            f"The cudaq-pulse GPU runtime ({path}) loaded, but CUDA reports no "
            f"usable device: cudaGetDeviceCount returned error {status} "
            f"({reason}). Check the NVIDIA driver and CUDA_VISIBLE_DEVICES."
        )
    if count.value == 0:
        raise GpuDeviceUnavailableError(
            f"The cudaq-pulse GPU runtime ({path}) loaded, but CUDA reports "
            f"zero devices. Check the NVIDIA driver and CUDA_VISIBLE_DEVICES."
        )
    return count.value
