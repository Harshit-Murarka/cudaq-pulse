# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ErrorCode``, the kind of a ``CompilationError``. Imported when first
used: by ``cudaq_pulse.ErrorCode``, or by reading an error's ``code``.
"""

from enum import StrEnum

__all__ = ["ErrorCode"]


class ErrorCode(StrEnum):
    """What kind of ``CompilationError`` an error is: its ``code``.

    The codes are stable, so a program, a test or a tool can act on one
    without reading the message. Each member is a string; the plain string
    (``"unknown_name"``) compares equal to it.
    """

    #: A name the kernel calls or reads that nothing in its scope defines,
    #: such as a misspelled operation; ``did_you_mean`` names a close one.
    UNKNOWN_NAME = "unknown_name"
    #: An attribute the value does not have, such as a misspelled method of
    #: a line; ``did_you_mean`` names a close one.
    UNKNOWN_ATTRIBUTE = "unknown_attribute"
    #: A name read before it is assigned on the path the trace took.
    UNASSIGNED_NAME = "unassigned_name"
    #: An operation given an argument of the wrong kind, such as a qudit
    #: where it takes a line, or the wrong number of arguments.
    WRONG_ARGUMENT = "wrong_argument"
    #: Python the kernel language does not trace: a ``while`` loop, ``try``,
    #: a generator, a class definition.
    UNSUPPORTED_CONSTRUCT = "unsupported_construct"
    #: A value known only when the program runs, such as a measurement
    #: result, used where tracing needs its value: compared, converted,
    #: iterated or hashed; or a branch on one that does what a branch taken
    #: at run time cannot.
    RUNTIME_VALUE = "runtime_value"
    #: A ``float`` parameter whose value the program needs while it
    #: compiles (``SymbolicValueError``).
    SYMBOLIC_PARAMETER = "symbolic_parameter"
    #: A duration or time that is not a whole number of clock cycles, a
    #: float where a duration takes an int or a time, or arithmetic that
    #: mixes times with numbers of cycles.
    INVALID_DURATION = "invalid_duration"
    #: A waveform that cannot be made: a ``custom()`` envelope that returns
    #: what is not a number, waveform arithmetic on different durations,
    #: samples made for another rate.
    INVALID_WAVEFORM = "invalid_waveform"
    #: A calibration the kernel plays that cannot play, or that its trace
    #: cannot read.
    CALIBRATION = "calibration"
    #: A program larger than a limit of the compiler: more operations than
    #: the controller holds, a loop unrolled past its limit, kernel calls
    #: nested too deep.
    LIMIT_EXCEEDED = "limit_exceeded"
    #: An ``assert`` that failed, or a ``raise``, in the kernel.
    KERNEL_RAISED = "kernel_raised"
    #: Python code of the kernel that failed as it would outside one: a
    #: subscript, an unpacking or a helper call that raised; the message
    #: holds Python's error.
    PYTHON_ERROR = "python_error"
    #: A program the control system cannot play as written: a link it does
    #: not have, a qudit the quantum system does not define, a call across
    #: its controllers.
    CONTROL_SYSTEM = "control_system"
    #: A kernel called with its own ``schedule=`` that the program cannot
    #: schedule on its own: under ``rcp`` or ``alap_rcp``, which schedule a
    #: whole program, or with a ``max_duration_cycles`` that
    #: ``flatten_schedules=True`` cannot hold. ``fix`` says whether to
    #: compile with ``flatten_schedules=True`` or without it.
    OWN_SCHEDULE = "own_schedule"
    #: A program that misses a timing it declares (``TimingError``).
    TIMING = "timing"
    #: A program over a limit of its controller (``ControllerLimitError``).
    CONTROLLER_LIMIT = "controller_limit"
    #: Any other compilation error; the message says what is wrong.
    OTHER = "other"
