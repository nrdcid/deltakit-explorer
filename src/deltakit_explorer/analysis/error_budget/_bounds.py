# (c) Copyright Riverlane 2020-2026. All rights reserved.
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from numbers import Integral
from time import perf_counter

import numpy as np
import numpy.typing as npt
import pandas as pd
from deltakit_circuit._circuit import Circuit

from deltakit_explorer.analysis._binomial_fit import ConfidenceInterval
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


class BoundSearchStatus(Enum):
    """Termination status of one parameter's bound search."""

    NOT_EVALUATED = auto()
    CONVERGED = auto()
    INSUFFICIENT_COUNTS = auto()
    LOW_SNR = auto()
    SATURATED = auto()
    DOMAIN_LIMIT = auto()
    TRIAL_LIMIT = auto()
    INVALID_ESTIMATE = auto()
    NON_MONOTONE = auto()


@dataclass(frozen=True)
class CircuitPilotObservation:
    """Raw counts and logical-error probability for one pilot circuit.

    Attributes:
        distance: Code distance.
        num_rounds: Number of memory-experiment rounds.
        fails: Actual observed logical failures.
        shots: Actual completed shots, including unsuccessful probes.
        lep_interval: Binomial LEP estimate and bounds, if available.
    """

    distance: int
    num_rounds: int
    fails: int
    shots: int
    lep_interval: ConfidenceInterval | None = None


@dataclass(frozen=True)
class LambdaPilotObservation:
    """Observation at one exact noise vector, including failed probes.

    Attributes:
        point_id: Stable identifier within this discovery run.
        noise_parameters: Full noise vector, preserving distinct nearby points.
        circuits: Per-circuit counts and LEP intervals.
        lambda_estimate: Valid Lambda estimate, or None when unavailable/invalid.
        lambda_stddev: Standard deviation of Lambda, if valid.
        inverse_lambda_estimate: Valid inverse-Lambda estimate, if available.
        inverse_lambda_stddev: Standard deviation of inverse Lambda, if valid.
        warnings: Estimator warnings retained without discarding raw evidence.
        invalidity_reasons: Reasons this point cannot support a valid estimate.
    """

    point_id: int
    noise_parameters: tuple[float, ...]
    circuits: tuple[CircuitPilotObservation, ...]
    lambda_estimate: float | None = None
    lambda_stddev: float | None = None
    inverse_lambda_estimate: float | None = None
    inverse_lambda_stddev: float | None = None
    warnings: tuple[str, ...] = ()
    invalidity_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParameterSearchDiagnostic:
    """Outcome and evidence for one axis search.

    Attributes:
        parameter_index: Coordinate being varied.
        status: Search termination status.
        candidate_interval: Last proposed interval, even when unresolved.
        trial_count: Actual noncentral trials performed for this parameter.
        endpoint_snr: Endpoint inverse-Lambda signal-to-noise ratio, if available.
        min_failures: Minimum observed failure count across endpoint circuits.
        max_lep: Maximum observed endpoint logical-error probability.
        reason: Explanation of the outcome or missing evidence.
    """

    parameter_index: int
    status: BoundSearchStatus
    candidate_interval: tuple[float, float] | None = None
    trial_count: int = 0
    endpoint_snr: float | None = None
    min_failures: int | None = None
    max_lep: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class BoundSearchTimings:
    """Measured discovery durations in seconds; never runtime deadlines.

    Attributes:
        construction_seconds: Circuit and decoder construction time.
        sampling_seconds: Combined sampling and decoding time.
        post_processing_seconds: Statistical post-processing time.
        total_seconds: Total elapsed time, including setup and orchestration.
    """

    construction_seconds: float = 0.0
    sampling_seconds: float = 0.0
    post_processing_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class BoundsSearchResult:
    """Validated bounds and retained evidence from a possibly incomplete search.

    Attributes:
        bounds: One validated interval per coordinate; None for unresolved axes.
            Unvalidated candidate intervals belong only in diagnostics.
        evaluation_point: Full gradient evaluation vector.
        gradient_evaluation_scale: Multiplier used to resolve the evaluation point.
        diagnostics: Per-parameter search outcomes in coordinate order.
        pilot_data: Raw sampler report rows, including failed probes.
        observations: One observation per unique sampled noise vector. Repeated
            cached lookups do not add observations or contribute to totals.
        phase_timings: Measured phase and total discovery durations.

    Raises:
        ValueError: If bounds do not match the evaluation vector or a resolved
            interval is non-finite or does not strictly contain its coordinate.
    """

    bounds: tuple[tuple[float, float] | None, ...]
    evaluation_point: tuple[float, ...]
    gradient_evaluation_scale: float
    diagnostics: tuple[ParameterSearchDiagnostic, ...]
    pilot_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    observations: tuple[LambdaPilotObservation, ...] = ()
    phase_timings: BoundSearchTimings = field(default_factory=BoundSearchTimings)

    def __post_init__(self) -> None:
        if not self.bounds or len(self.bounds) != len(self.evaluation_point):
            msg = "bounds must contain one entry per nonempty evaluation coordinate."
            raise ValueError(msg)
        for bound, center in zip(self.bounds, self.evaluation_point, strict=True):
            if bound is not None and (
                len(bound) != 2
                or not np.all(np.isfinite(bound))
                or not bound[0] < center < bound[1]
            ):
                msg = "Resolved bounds must be finite and strictly contain the evaluation point."
                raise ValueError(msg)

    @property
    def success(self) -> bool:
        """Whether every parameter has a validated interval."""
        return all(bound is not None for bound in self.bounds)

    @property
    def total_trials(self) -> int:
        """Actual unique sampled vectors, including the center and failed probes."""
        return len(self.observations)

    @property
    def total_shots(self) -> int:
        """Actual shots across all pilot circuits, including failed probes."""
        return sum(
            circuit.shots for point in self.observations for circuit in point.circuits
        )


class BoundsDiscoveryError(RuntimeError):
    """Discovery failure retaining its partial result for inspection.

    Attributes:
        result: Partial discovery result, when one was produced.
    """

    def __init__(self, message: str, result: BoundsSearchResult | None = None) -> None:
        """Create an error with the discovery evidence available at failure.

        Args:
            message: Explanation of the failure.
            result: Partial result, if discovery produced one.
        """
        super().__init__(message)
        self.result = result


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
    """Validate discovery inputs and return an unresolved search result.

    Search and sampling are not implemented yet. Initial candidate intervals are
    retained in diagnostics; all bounds are unresolved until sampling validates them.

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
        A partial result with unresolved bounds and unsampled candidate diagnostics.

    Raises:
        ValueError: If calibration, scale, shot/execution settings, domains, or
            initial intervals are invalid or incompatible with the fit strategy.
    """
    started = perf_counter()
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

    diagnostics = tuple(
        ParameterSearchDiagnostic(
            parameter_index=i,
            status=BoundSearchStatus.NOT_EVALUATED,
            candidate_interval=(float(lo), float(hi)),
            reason="Statistical search and pilot sampling are not implemented yet.",
        )
        for i, (lo, hi) in enumerate(zip(lower, upper, strict=True))
    )
    return BoundsSearchResult(
        bounds=(None,) * len(centers),
        evaluation_point=tuple(map(float, centers)),
        gradient_evaluation_scale=float(gradient_evaluation_scale),
        diagnostics=diagnostics,
        phase_timings=BoundSearchTimings(total_seconds=perf_counter() - started),
    )
