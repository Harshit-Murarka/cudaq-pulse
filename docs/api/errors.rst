..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0


Errors
======

Every error CUDA-Q Pulse raises on purpose derives from ``PulseError``, so
``except cudaq_pulse.PulseError`` catches them all. Each also has the
built-in base of its kind of error, so ``except ValueError`` or ``except
RuntimeError`` catches it as it would the built-in one: ``InvalidValueError``
is a ``ValueError``, ``NotFoundError`` and ``UnknownQuantumSystemError`` are
``KeyError``\ s, ``IndexOutOfRangeError`` is an ``IndexError``, and ``CompilationError`` and ``NoProcessQuantumSystemError`` are
``RuntimeError``\ s.

.. autoexception:: cudaq_pulse.PulseError
   :members: problems, location

An error that finds several problems at once, such as
``QuantumSystemDefinitionError``, lists each as a ``Problem`` in ``problems``:

.. autoclass:: cudaq_pulse.Problem

Arguments and names
-------------------

.. autoexception:: cudaq_pulse.InvalidValueError

.. autoexception:: cudaq_pulse.InvalidTypeError

.. autoexception:: cudaq_pulse.NotFoundError

.. autoexception:: cudaq_pulse.IndexOutOfRangeError

.. autoexception:: cudaq_pulse.UnsupportedError

Compiling
---------

.. autoexception:: cudaq_pulse.CompilationError
   :members: code

.. autoexception:: cudaq_pulse.TimingError

.. autoexception:: cudaq_pulse.compiler.UnsupportedInterpreterError

.. _error-codes:

Error codes
~~~~~~~~~~~

Every ``CompilationError`` has a ``code``, an ``ErrorCode``, so a program
or a tool can act on the kind of error without reading the message. The
codes are stable: a code keeps its meaning, and a new kind of error gets a
new code. ``fix`` holds the part of the message that says what to write
instead (empty when the message gives none), ``did_you_mean`` the name to
write in place of a misspelled one, and ``location``, for an error raised
while the kernel is traced, a ``Problem`` with the kernel's ``file``,
``line`` and ``source`` (the text of that line), as ``path`` the name of
the function and, with ``did_you_mean``, as ``value`` the misspelled name.
The message is the same as without them. For a kernel ``k`` that
misspells an operation, and its qudit ``q``:

.. code-block:: python
   :class: fragment

   try:
       pulse.compile(k).lower(q)
   except pulse.CompilationError as error:
       if error.code == pulse.ErrorCode.UNKNOWN_NAME and error.did_you_mean:
           where = error.location
           fixed = where.source.replace(where.value, error.did_you_mean, 1)
           print(f"{where.file}:{where.line}: write {fixed!r}")

=========================== ============================================================
Code                        What it means
=========================== ============================================================
``unknown_name``            A name the kernel calls or reads that nothing in its scope
                            defines; ``did_you_mean`` names a close one.
``unknown_attribute``       An attribute the value does not have; ``did_you_mean``
                            names a close one.
``unassigned_name``         A name read before it is assigned on the path traced.
``wrong_argument``          An argument of the wrong kind, or the wrong number of them.
``unsupported_construct``   Python the kernel language does not trace, such as a
                            ``while`` loop or ``try``.
``runtime_value``           A value known only when the program runs, such as a
                            measurement result, used where tracing needs its value.
``symbolic_parameter``      A ``float`` parameter whose value compiling needs.
``invalid_duration``        A duration that is not a whole number of cycles, or that
                            mixes times with numbers of cycles.
``invalid_waveform``        A waveform that cannot be made.
``calibration``             A calibration that cannot play, or that the trace cannot
                            read.
``limit_exceeded``          A program larger than a limit of the compiler.
``kernel_raised``           An ``assert`` that failed, or a ``raise``, in the kernel.
``python_error``            Python code of the kernel that failed as it would outside
                            one; the message holds Python's error.
``control_system``          A program the control system cannot play as written.
``own_schedule``            A kernel called with its own ``schedule=`` that the program
                            cannot schedule on its own, such as under ``rcp``; ``fix``
                            says whether to compile with ``flatten_schedules=True``.
``timing``                  A missed deadline or length limit (``TimingError``).
``controller_limit``        A limit of the controller exceeded
                            (``ControllerLimitError``).
``other``                   Any other compilation error.
=========================== ============================================================

.. autoclass:: cudaq_pulse.ErrorCode
   :members:
   :undoc-members:

Running
-------

.. autoexception:: cudaq_pulse.ExecutionError

.. autoexception:: cudaq_pulse.InternalError

Quantum systems
---------------------------

.. autoexception:: cudaq_pulse.QuantumSystemDefinitionError

.. autoexception:: cudaq_pulse.NoQuantumSystemError

.. autoexception:: cudaq_pulse.NoProcessQuantumSystemError

.. autoexception:: cudaq_pulse.QuantumSystemMismatchError

.. autoexception:: cudaq_pulse.quantum_systems.UnknownQuantumSystemError

GPU simulation
--------------

``pulse.simulate(..., backend="gpu")`` raises one of the three subclasses of
``GpuRuntimeError`` (``cudaq_pulse.GpuRuntimeError``), an ``ExecutionError``,
when GPU simulation cannot run; see :ref:`gpu-errors`.

.. autoexception:: cudaq_pulse.runtime.GpuRuntimeError

.. autoexception:: cudaq_pulse.runtime.GpuSupportNotBuiltError

.. autoexception:: cudaq_pulse.runtime.GpuRuntimeLoadError

.. autoexception:: cudaq_pulse.runtime.GpuDeviceUnavailableError

Controller backends
-------------------

Compiling for a controller whose limits a program exceeds raises
``ControllerLimitError``; finding a backend raises a ``BackendError``. See
:doc:`controller`.

.. autoexception:: cudaq_pulse.controller.ControllerLimitError

.. autoexception:: cudaq_pulse.backends.BackendError

.. autoexception:: cudaq_pulse.backends.BackendNotFoundError

.. autoexception:: cudaq_pulse.backends.IncompatibleBackendError

Warnings
--------

.. autoexception:: cudaq_pulse.PulseWarning

.. autoexception:: cudaq_pulse.UpgradeNotice

.. autoexception:: cudaq_pulse.UnitWarning

See :ref:`unit warnings <unit-warnings>` for what ``unit_warnings=True`` checks.
