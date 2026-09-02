"""Automatic exploration-bound discovery for error budgeting."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from deltakit_circuit._circuit import Circuit

from deltakit_explorer.analysis.error_budget._memory import (
    MemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.analysis.error_budget._parameters import SamplingParameters


class BoundsDiscoveryError(RuntimeError):
    """Raised when usable error-budget exploration bounds cannot be found."""


@dataclass(frozen=True)
class BoundsSearchResult:
    """Diagnostics and bounds returned by automatic bounds discovery."""

    bounds: tuple[tuple[float, float], ...]
    stop_reasons: tuple[str, ...]
    endpoint_logical_error_estimates: tuple[tuple[float, float], ...]
    sensitivity_snrs: tuple[float, ...]
    shots_used: tuple[int, ...]
    insensitive: tuple[bool, ...]
    failed_parameters: tuple[int, ...]


def find_error_budget_bounds(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_per_distance: Mapping[int, Sequence[int]],
    *,
    sampling_parameters: SamplingParameters | None = None,
    parameter_indices: Sequence[int] | None = None,
    initial_relative_width: float = 0.1,
    sensitivity_threshold: float = 3.0,
    logical_error_rate_min: float | None = None,
    logical_error_rate_max: float = 0.4,
    max_iterations: int = 12,
    max_relative_width: float = 16.0,
    seed: int | None = None,
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
) -> BoundsSearchResult:
    """Find feasible axis-aligned exploration bounds around ``p / 2``.

    This is the public API skeleton. The search and sampling implementation will be
    added incrementally behind this stable interface.
    """
    del (
        noise_model,
        noise_parameters,
        num_rounds_per_distance,
        sampling_parameters,
        parameter_indices,
        initial_relative_width,
        sensitivity_threshold,
        logical_error_rate_min,
        logical_error_rate_max,
        max_iterations,
        max_relative_width,
        seed,
        memory_generator,
    )
    raise NotImplementedError("automatic error-budget bounds are not implemented yet")
