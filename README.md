# CUDA-Q pulse

CUDA-Q pulse (`cudaq-pulse`) is a pulse-level quantum programming research
package built on MLIR. It provides a Python kernel DSL, pulse and operator
dialects, compiler passes, and an experimental GPU execution path on NVIDIA
cuDensityMat.

> [!WARNING]
> CUDA-Q pulse is **research-preview software**. It is not production software
> and is not a product-supported CUDA-Q feature. Its Python APIs, MLIR
> dialects, runtime interfaces, build options, numerical behavior, and file
> layout may change incompatibly or be removed without notice. No stability,
> compatibility, performance, or production-readiness guarantee is provided.

`cudaq-pulse` does not depend on CUDA-Q. The wheel contains its own statically
linked MLIR, so `pip` is the only installer you need.

## Install

```bash
python -m venv .venv && source .venv/bin/activate

# Compiler only (Linux and macOS): compile kernels and inspect the MLIR.
pip install cudaq-pulse

# Compiler plus GPU simulation (Linux x86_64 and aarch64): also installs
# cuDensityMat and the CUDA 12 runtime libraries from PyPI.
pip install "cudaq-pulse[gpu]"
```

GPU execution additionally needs an NVIDIA GPU (compute capability 7.0 or
newer) and a driver that supports CUDA 12.9. No CUDA toolkit, cuQuantum SDK,
or container is required.

Without a usable GPU, everything except execution still works. Execution fails
with a specific error rather than degrading:

| Situation | Error raised by `pulse.evolve` / `CompiledKernel.run` |
| --- | --- |
| Wheel without GPU support (macOS, or a compiler-only source build) | `GpuSupportNotBuiltError` |
| GPU support built in, but `[gpu]` libraries missing or unloadable | `GpuRuntimeLoadError` |
| Libraries load, but no device or driver | `GpuDeviceUnavailableError` |

All three derive from `cudaq_pulse.runtime.gpu.GpuRuntimeError`.

## Quick example

```python
import cudaq_pulse as pulse


@pulse.kernel
def rabi_oscillation(qubit):
    drive_line, tone = get_drive_line(qubit)
    drive(drive_line, gaussian(64, 0.5, 16.0), tone)


compiled_kernel = pulse.compile(
    rabi_oscillation,
    [pulse.qudit_ref()],
    qubit_freq_hz={0: 5.0e9},
)
print(compiled_kernel.mlir)
```

The compiler traces the Python kernel, builds Pulse dialect IR, applies the
selected transformations, and emits scheduled MLIR. The wheel also installs
`cudaq-pulse-opt`, the MLIR driver for the pulse dialects:

```bash
cudaq-pulse-opt --pulse-to-qop kernel.mlir
```

The [user guide](docs/index.rst) covers the kernel model, operations,
compilation, passes, and GPU execution. [`examples/`](examples) has runnable
programs.

## Run the test suite against your installation

The tests live in this repository, not in the wheel. They import whichever
`cudaq_pulse` is installed, so they check exactly what you installed:

```bash
git clone https://github.com/NVIDIA/cudaq-pulse.git
cd cudaq-pulse
pip install --group test        # pytest, hypothesis, lit (pip 25.1+)

pytest tests -m "not gpu"       # any installation
pytest tests                    # GPU installations: also runs the GPU suite
```

Selecting the GPU tests on an installation that cannot run them stops pytest
immediately with the reason, instead of skipping them.

## Build from source

See [CONTRIBUTING.md](CONTRIBUTING.md). In short, `pip install .` from a
clone builds a wheel locally after `python tools/fetch-mlir-toolchain.py` has
downloaded the pinned, checksum-verified MLIR toolchain. Nothing else is
built from source.

## Current scope and limitations

- The compiler, dialects, lowering experiments, and CPU unit tests are the
  primary research surface.
- GPU evolution models two-level transmons in per-qubit rotating frames.
  Target T1/T2 data, XX couplings, residual ZZ terms, and calibrated
  drive-amplitude scaling are supported. Transmon anharmonicity remains target
  metadata because a faithful leakage model requires three or more levels.
- GPU evolution supports the `rk1`, `rk2`, `rk4`, `magnus`, and
  `crank_nicolson` integrators. Readout/acquisition, observable evaluation,
  neutral-atom and multilevel models, unspecialized parameters, arbitrary
  Python waveform callbacks, and waveform algebra are rejected explicitly by
  the execution lowering.
- GPU numerical results are research regression coverage, with no production
  compatibility or accuracy commitment.

## Repository contents

- `core/frontend/` — Python DSL, compiler pipeline, passes, targets, and
  runtime
- `core/mlir/` — Pulse, `QOp`, and CuDensityMat dialects, transformations,
  bindings, and `cudaq-pulse-opt`
- `core/runtime/` — cuDensityMat runtime library loaded by JIT-compiled code
- `cmake/` — dependency, Python staging, testing, and documentation modules
- `tools/` — MLIR toolchain build/fetch scripts and CI helpers
- `test/` — lit and `FileCheck` compiler regression tests
- `tests/` — pytest unit, workload, and GPU tests
- `examples/` — pulse programming examples
- `docs/` — user, API, and architecture documentation
- `benchmarks/` — compiler and simulation benchmarks

## License

[Apache License 2.0](LICENSE).
