# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""cudaq-pulse compiler passes (experimental pass-authoring API).

Provides verification, scheduling, canonicalization, and optimization passes
that operate on the lightweight ``Program``/``Op`` IR, plus the
``ProgramBuilder`` used to construct programs directly.

Each pass is a plain ``Program -> Program`` (or ``Program -> (events, metrics)``
for schedulers) function, so you can compose the built-ins or write your own
transform and apply it, then emit MLIR with ``program_to_pulse_mlir``. For the
common "compile a kernel end-to-end" path, use ``cudaq_pulse.compile()``.

This surface is experimental and may change without notice.
"""

from ._builder import ProgramBuilder
from .canonicalize import run_canonicalize
from .fusion import run_fusion
from .ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    _mk,
    clone_program,
    duration_of,
    is_loop_or_barrier,
)
from .loop_passes import run_licm, run_loop_strength_reduction
from .pulse_to_operator import OperatorProgram, run_pulse_to_operator
from .scheduling import (
    MachineModel,
    ScheduledEvent,
    ScheduleMetrics,
    schedule_alap,
    schedule_asap,
    schedule_rcp,
)
from .to_pulse_mlir import program_to_pulse_mlir
from .verify import (
    check_drive_exclusivity,
    check_linearity,
    check_loop_structure,
    check_monotone_time,
    check_waveform_validity,
    verify,
)
from .virtual_z import run_virtual_z

__all__ = [
    "Op",
    "OpKind",
    "Program",
    "Value",
    "ValueType",
    "_mk",
    "clone_program",
    "duration_of",
    "is_loop_or_barrier",
    "ScheduledEvent",
    "ScheduleMetrics",
    "MachineModel",
    "ProgramBuilder",
    "OperatorProgram",
    "verify",
    "check_linearity",
    "check_monotone_time",
    "check_drive_exclusivity",
    "check_loop_structure",
    "check_waveform_validity",
    "schedule_asap",
    "schedule_alap",
    "schedule_rcp",
    "run_canonicalize",
    "run_virtual_z",
    "run_fusion",
    "run_licm",
    "run_loop_strength_reduction",
    "run_pulse_to_operator",
    "program_to_pulse_mlir",
]
