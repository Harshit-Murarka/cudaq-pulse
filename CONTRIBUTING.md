# Contributing to cudaq-pulse

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it provides Python, CMake, Ninja, and
  every Python tool below)
- A C++17 compiler: Clang or GCC on Linux, Xcode command-line tools on macOS
- For GPU work only: Linux, an NVIDIA GPU and driver, and the CUDA 12.9 toolkit

You do not need CUDA-Q, an LLVM build, or Docker.

## Set up

```bash
git clone https://github.com/NVIDIA/cudaq-pulse.git
cd cudaq-pulse

uv sync                                  # .venv with build, test, lint, docs tools
python3 tools/fetch-mlir-toolchain.py    # pinned MLIR 22.1.8 into .cache/mlir/
```

`uv sync` installs tooling only (`[tool.uv] package = false`); the package is
built by CMake, below. The fetch script downloads the toolchain archive for
your platform, verifies its SHA-256 against `tools/mlir-toolchain.lock.json`,
and unpacks it. It refuses archives it cannot verify, and CMake refuses to
configure without the toolchain.

## Build and test

```bash
source .venv/bin/activate

cmake -S . -B build/dev -G Ninja \
  -DCUDAQ_PULSE_BUILD_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/dev --target check-pulse
```

`check-pulse` builds everything and runs the lit/FileCheck tests in `test/`
and the non-GPU pytest suite in `tests/`. The staged package lives in
`build/dev/python`; to use it directly:

```bash
export PYTHONPATH="$PWD/build/dev/python"
export PATH="$PWD/build/dev/bin:$PATH"     # cudaq-pulse-opt
python examples/01_single_qubit_rabi.py
```

Python sources are staged as symlinks, so edits to `core/frontend/` take
effect without rebuilding; C++ changes need `cmake --build build/dev`.

## Lint and type-check

```bash
pre-commit install                       # once; runs the hooks on each commit
pre-commit run --all-files

MYPYPATH=build/dev/python mypy -p cudaq_pulse
```

The hooks run the ruff linter and formatter, clang-format, whitespace and
YAML/TOML checks, and a license-header check
(`tools/check-license-headers.py`). New source files need the NVIDIA
Apache-2.0 header used by their neighbors.

mypy runs in strict mode against the staged package, because it needs the
type stub that the build generates for the native extension.

## Benchmarks

```bash
PYTHONPATH=build/dev/python python benchmarks/compiler/compile_time.py \
  --baseline benchmarks/baselines/compile_time.json
```

CI fails if any workload's op count differs from the baseline. If a change
alters op counts on purpose, regenerate it with `--write-baseline` and
explain the change in the pull request. Timing is only gated when you pass
`--gate FACTOR`; it is too noisy on shared runners to gate in CI.

## GPU builds

GPU support is off by default and, when requested, mandatory: configuration
fails rather than silently producing a compiler-only build.

```bash
uv pip install -r pyproject.toml --extra gpu
CUDENSITYMAT_ROOT="$(python -c 'import cuquantum; print(cuquantum.__path__[0])')"

cmake -S . -B build/gpu -G Ninja \
  -DCUDAQ_PULSE_BUILD_TESTS=ON \
  -DCUDAQ_PULSE_ENABLE_GPU=ON \
  -DCUDENSITYMAT_ROOT="$CUDENSITYMAT_ROOT" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/gpu --target check-pulse
cmake --build build/gpu --target check-pulse-gpu
```

The first command installs the `[gpu]` extra's dependencies (not the package
itself) into `.venv`: the same libraries `pip install "cudaq-pulse[gpu]"`
gives users, including the cuDensityMat headers the build compiles against.
`check-pulse-gpu` runs the cuDensityMat linkage smoke test and the numerical
GPU suite, and fails if no device is usable.

## Build a wheel

```bash
pip install .            # or: uv build --wheel
```

This uses the same fetched toolchain. Wheels are compiler-only unless
`CUDAQ_PULSE_ENABLE_GPU=ON` is set in the environment (as the release build
does on Linux; see `[tool.cibuildwheel]` in `pyproject.toml`).

## The MLIR toolchain

`tools/build-mlir-toolchain.sh` builds the slim, static LLVM/MLIR 22.1.8
distribution that cudaq-pulse links against. The `mlir-toolchain` workflow
runs it for every platform; after a change, run that workflow with
`publish` enabled and commit the printed checksums to
`tools/mlir-toolchain.lock.json`.

## Continuous integration

- `ci.yml`: ruff and pre-commit; build, lit, pytest, mypy, and the benchmark
  baseline on Linux x86_64, Linux aarch64, and macOS. A manually dispatched
  run with `gpu` enabled also builds with GPU support on the runner named by
  the `GPU_RUNNER` repository variable.
- `wheels.yml`: cibuildwheel for CPython 3.10–3.14 on manylinux_2_28 (x86_64,
  aarch64, with the GPU runtime) and macOS arm64 (compiler-only), plus an
  sdist. Tags matching `v*` publish to PyPI.
- `docs.yml`: Sphinx with warnings as errors; deploys `main` to GitHub Pages.
