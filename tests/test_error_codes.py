# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A CompilationError carries its code, its fix, a close name and the kernel
line that raised it, as fields, beside the message it always had."""

from __future__ import annotations

import ast
import inspect
import pickle
import re
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import ErrorCode
from cudaq_pulse.controller import ControllerLimitError
from cudaq_pulse.kernel.ir_builder import SymbolicValueError

from tests._handles import Q

_DOCS = Path(__file__).resolve().parents[1] / "docs" / "api" / "errors.rst"


def test_the_codes_are_the_documented_vocabulary():
    assert [c.value for c in ErrorCode] == [
        "unknown_name",
        "unknown_attribute",
        "unassigned_name",
        "wrong_argument",
        "unsupported_construct",
        "runtime_value",
        "symbolic_parameter",
        "invalid_duration",
        "invalid_waveform",
        "calibration",
        "limit_exceeded",
        "kernel_raised",
        "python_error",
        "control_system",
        "own_schedule",
        "timing",
        "controller_limit",
        "other",
    ]
    for code in ErrorCode:
        assert code.name == code.value.upper()
        assert code == code.value


@pytest.mark.skipif(not _DOCS.is_file(), reason="needs the source tree's docs/")
def test_the_errors_page_documents_every_code():
    documented = set(re.findall(r"``(\w+)``", _DOCS.read_text()))
    assert {c.value for c in ErrorCode} <= documented


def test_every_code_the_package_raises_is_an_error_code():
    sources = Path(pulse.__file__).resolve().parent
    given = set()
    for path in sources.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "code" and isinstance(keyword.value, ast.Constant):
                        given.add(keyword.value.value)
    assert given and given <= {c.value for c in ErrorCode}


def test_importing_cudaq_pulse_does_not_make_the_enum():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, cudaq_pulse; codes = 'cudaq_pulse._error_codes'; "
            "print(codes in sys.modules, cudaq_pulse.ErrorCode.OTHER.value, codes in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out == ["False", "other", "True"]


def _raised(kernel, *args):
    with pytest.raises(pulse.CompilationError) as info:
        pulse.compile(kernel).lower(*args)
    return info.value


def _line_of(kernel, text):
    lines, first = inspect.getsourcelines(kernel.__wrapped__)
    return first + next(i for i, line in enumerate(lines) if text in line)


def test_a_misspelled_op_names_the_op_it_means_and_the_line():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drve(d, square(8, 0.1))  # noqa: F821

    error = _raised(k, Q[0])
    assert str(error) == "unknown pulse op: drve; did you mean drive()?"
    assert error.code is ErrorCode.UNKNOWN_NAME
    assert error.did_you_mean == "drive"
    assert error.fix == "did you mean drive()?"
    where = error.location
    assert where is not None
    assert (where.file, where.line, where.path) == (__file__, _line_of(k, "drve("), "k")
    assert where.source == "drve(d, square(8, 0.1))  # noqa: F821"
    assert (where.message, where.fix) == ("unknown pulse op: drve", "did you mean drive()?")
    assert where.value == "drve"
    repaired = where.source.replace(where.value, error.did_you_mean, 1)
    assert repaired == "drive(d, square(8, 0.1))  # noqa: F821"


def test_a_misspelled_line_method_names_the_method_it_means():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, d.get_calibraton(pulse.gates.X))

    error = _raised(k, Q[0])
    assert str(error) == (
        "a drive line has no attribute 'get_calibraton'; "
        "its attributes are tone, tones and get_calibration"
    )
    assert error.code is ErrorCode.UNKNOWN_ATTRIBUTE
    assert error.did_you_mean == "get_calibration"
    assert error.location.line == _line_of(k, "get_calibraton")
    assert error.location.value == "get_calibraton"


@pytest.mark.parametrize(
    ("body", "code", "fix"),
    [
        (
            "drive(q, square(8, 0.1))",
            ErrorCode.WRONG_ARGUMENT,
            "pass a line of the qudit, such as get_drive_line(q)",
        ),
        (
            "shift_phase(get_drive_line(q), 1.0)",
            ErrorCode.WRONG_ARGUMENT,
            "pass the line's tone: shift_phase(line.tone, ...)",
        ),
        (
            "m = readout(get_readout_line(q), square(8, 0.1))\n    if m == 1:\n        pass",
            ErrorCode.RUNTIME_VALUE,
            "branch on it directly with `if result:` or `if not result:`",
        ),
        (
            "n = 0\n    while n < 2:\n        n += 1",
            ErrorCode.UNSUPPORTED_CONSTRUCT,
            "use a for loop",
        ),
        (
            "pulse.drive(get_drive_line(q), square(8, 0.1))",
            ErrorCode.UNKNOWN_ATTRIBUTE,
            "write drive(...) or cudaq_pulse.ops.drive(...)",
        ),
        ("wait(get_drive_line(q), 2.5)", ErrorCode.INVALID_DURATION, ""),
    ],
)
def test_each_error_has_its_code_and_its_fix_apart_from_the_message(tmp_path, body, code, fix):
    path = tmp_path / "kernels.py"
    path.write_text(f"import cudaq_pulse as pulse\n\n@pulse.kernel\ndef k(q):\n    {body}\n")
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    error = _raised(namespace["k"], Q[0])
    assert error.code is code
    assert error.fix == fix
    if fix:
        assert str(error).endswith(f"; {fix}")
        assert error.location.message + "; " + fix == str(error)
    assert error.location.file == str(path)
    assert error.location.line >= 5
    assert error.location.source == path.read_text().splitlines()[error.location.line - 1].strip()


def test_errors_of_a_kind_have_its_code_by_default():
    assert pulse.CompilationError("x").code is ErrorCode.OTHER
    assert pulse.TimingError("x").code is ErrorCode.TIMING
    assert SymbolicValueError("x").code is ErrorCode.SYMBOLIC_PARAMETER
    assert ControllerLimitError(["x"]).code is ErrorCode.CONTROLLER_LIMIT
    error = pulse.CompilationError("x")
    assert (error.fix, error.did_you_mean, error.location) == ("", None, None)


def test_the_code_fix_and_location_survive_pickling():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drve(d, square(8, 0.1))  # noqa: F821

    error = _raised(k, Q[0])
    copy = pickle.loads(pickle.dumps(error))
    assert str(copy) == str(error)
    assert (copy.code, copy.fix, copy.did_you_mean) == (
        ErrorCode.UNKNOWN_NAME,
        "did you mean drive()?",
        "drive",
    )
    assert copy.location == error.location


def test_a_message_given_with_a_fix_reads_as_one():
    error = pulse.CompilationError("what is wrong", fix="what to write", did_you_mean="x")
    assert str(error) == "what is wrong; what to write"
    assert error.args == ("what is wrong; what to write",)
    assert (error.fix, error.did_you_mean) == ("what to write", "x")
