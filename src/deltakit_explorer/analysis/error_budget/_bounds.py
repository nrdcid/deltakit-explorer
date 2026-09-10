# (c) Copyright Riverlane 2020-2026. All rights reserved.
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
    gradient_evaluation_point: npt.NDArray[np.floating] | Sequence[float] | None = None,
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
    """Return initial axis-aligned bounds around the gradient evaluation point.

    ``gradient_evaluation_point`` gives the parameter values at which the gradient
    will be evaluated and must have the same shape as ``noise_parameters``.
    It defaults to ``noise_parameters / 2``. Pass ``noise_parameters`` to center
    the bounds at ``p``, or a custom vector for any other evaluation point.
    ``initial_relative_width`` is relative to the chosen evaluation point.

    Search and sampling are not implemented yet; diagnostics are empty.
    """
    parameters = np.asarray(noise_parameters)
    centers = (
        parameters / 2
        if gradient_evaluation_point is None
        else np.asarray(gradient_evaluation_point)
    )
    if centers.shape != parameters.shape:
        msg = "gradient_evaluation_point must have the same shape as noise_parameters"
        raise ValueError(msg)
    return BoundsSearchResult(
        bounds=tuple(
            (
                float(center * (1 - initial_relative_width)),
                float(center * (1 + initial_relative_width)),
            )
            for center in centers
        ),
        stop_reasons=(),
        endpoint_logical_error_estimates=(),
        sensitivity_snrs=(),
        shots_used=(),
        insensitive=(),
        failed_parameters=(),
    )
