# (c) Copyright Riverlane 2020-2026. All rights reserved.
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import numpy.typing as npt
from deltakit_circuit._circuit import Circuit

from deltakit_explorer.analysis.error_budget._discretisation import (
    DiscretisationStrategy,
)
from deltakit_explorer.analysis.error_budget._memory import (
    MemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.analysis.error_budget._parameters import (
    BoundSearchParameters,
    FittingParameters,
    SamplingParameters,
    _resolve_gradient_point,
)


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


# Sampling-related arguments are reserved for the simulation adapter implementation.
def find_error_budget_bounds(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],  # noqa: ARG001
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],  # noqa: ARG001
    *,
    search_parameters: BoundSearchParameters,
    gradient_evaluation_scale: float = 0.5,
    shots_per_trial: int = 10_000,
    fitting_parameters: FittingParameters = FittingParameters(),
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator  # noqa: ARG001
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
    enable_correlations: bool = False,  # noqa: ARG001
    seed: int | None = None,  # noqa: ARG001
) -> BoundsSearchResult:
    """Validate discovery inputs and return initial, unvalidated search intervals.

    Search and sampling are not implemented yet. Returned intervals have not been
    screened for feasibility or sensitivity; diagnostics remain empty.

    Args:
        noise_model: Callable adding noise to a circuit using the supplied vector.
        noise_parameters: Finite, nonempty one-dimensional calibration vector.
        num_rounds_by_distances: Memory-experiment round counts for each distance.
        search_parameters: Required search configuration with one domain per
            calibration parameter. The scaled center must lie inside each domain.
        gradient_evaluation_scale: Finite positive scalar multiplying the entire
            calibration vector. Defaults to 0.5, selecting half calibration.
        shots_per_trial: Positive number of shots per distance/round circuit at
            each probed vector, not a total divided across the circuits. Reserved
            for sampling; no shots are taken by this initial implementation.
        fitting_parameters: Production fit configuration. Used to enforce positive
            intervals for logarithmic discretisation; degree and point count are
            preserved and no fit design is generated here.
        sampling_parameters: Execution settings for discovery. Only batch_size
            and max_workers are used; max_shots and early-stopping settings are
            ignored in favour of shots_per_trial and fixed-shot sampling.
        memory_generator: Callable generating noiseless memory circuits, or a
            distance-to-rounds mapping of precomputed circuits.
        enable_correlations: Correlated PyMatching setting for the future sampler.
        seed: Optional seed for the future sampler.

    Returns:
        Initial intervals with empty diagnostics. These are not discovered bounds.

    Raises:
        ValueError: If calibration, scale, shot/execution settings, domains, or
            initial intervals are invalid or incompatible with the fit strategy.
    """
    centers = _resolve_gradient_point(noise_parameters, gradient_evaluation_scale)
    search_parameters.validate_parameter_count(len(centers))
    for name, value in (
        ("shots_per_trial", shots_per_trial),
        ("batch_size", sampling_parameters.batch_size),
        ("max_workers", sampling_parameters.max_workers),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            msg = f"{name} must be a positive integer."
            raise ValueError(msg)

    domains = np.asarray(search_parameters.parameter_domains)
    if np.any((centers <= domains[:, 0]) | (centers >= domains[:, 1])):
        msg = "The evaluation point must lie strictly inside every parameter domain."
        raise ValueError(msg)
    if np.any(centers == 0):
        msg = "A zero evaluation coordinate needs explicit bounds; relative width is undefined."
        raise ValueError(msg)

    logarithmic = (
        fitting_parameters.discretisation_strategy == DiscretisationStrategy.LOGARITHMIC
    )
    if logarithmic and np.any(centers <= 0):
        msg = "A logarithmic fit requires strictly positive evaluation coordinates."
        raise ValueError(msg)
    with np.errstate(over="ignore", under="ignore"):
        half_widths = search_parameters.initial_relative_half_width * np.abs(centers)
        lower = np.maximum(domains[:, 0], centers - half_widths)
        upper = np.minimum(domains[:, 1], centers + half_widths)
    if logarithmic:
        # Move halfway toward zero rather than inventing a tiny physical scale.
        lower = np.where(lower <= 0, centers / 2, lower)
    if np.any((lower >= centers) | (upper <= centers)) or (
        logarithmic and np.any(lower <= 0)
    ):
        msg = "Initial intervals cannot strictly contain the evaluation point; supply explicit bounds."
        raise ValueError(msg)

    return BoundsSearchResult(
        bounds=tuple(
            (float(lo), float(hi)) for lo, hi in zip(lower, upper, strict=True)
        ),
        stop_reasons=(),
        endpoint_logical_error_estimates=(),
        sensitivity_snrs=(),
        shots_used=(),
        insensitive=(),
        failed_parameters=(),
    )
