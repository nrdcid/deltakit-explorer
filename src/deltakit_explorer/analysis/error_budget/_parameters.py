# (c) Copyright Riverlane 2020-2026. All rights reserved.
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import numpy.typing as npt

from deltakit_explorer.analysis.error_budget._discretisation import (
    DiscretisationStrategy,
)


@dataclass(frozen=True)
class FittingParameters:
    num_points_per_parameters: int = 10
    """Number of different values to try for each noise parameter.

    Corresponds to the number of points that will be used to fit a degree ``fitting_degree``
    polynomial. As such, should be greater than ``fitting_degree + 1``.
    """
    discretisation_strategy: DiscretisationStrategy = DiscretisationStrategy.LOGARITHMIC
    """A strategy to generate points that will be used to compute 1 / Λ on different
    values and fit a degree ``fitting_degree`` polynomial.

    Default to logarithmically spaced points.
    """
    fitting_degree: int = 3
    """Degree of polynomial that will be used to approximate 1 / Λ and to compute each
    of its derivatives.

    Should be lower than ``num_points_per_parameters - 1``. Higher values will incur
    higher standard deviation. Default to ``3``, which seems to be a good compromise
    between fit accuracy and resulting standard deviation.
    """

    def __post_init__(self) -> None:
        if self.num_points_per_parameters + 1 < self.fitting_degree + 2:
            msg = (
                f"Estimation of the standard deviation requires at least "
                f"fitting_degree + 2 = {self.fitting_degree + 2} discretisation points, "
                f"but only {self.num_points_per_parameters} + 1 are provided. Please "
                f"increase num_points_per_parameters to at least "
                f"{self.fitting_degree + 1}."
            )
            raise ValueError(msg)

    def get_discretisation(
        self, a: float, b: float, c: float
    ) -> npt.NDArray[np.floating]:
        return self.discretisation_strategy(
            a, b, c, self.num_points_per_parameters, self.fitting_degree
        )


@dataclass(frozen=True)
class SamplingParameters:
    max_shots: int = 10_000_000
    """Maximum number of shots per sampling task.

    A sampling task may stop with a lower number of samples if additional conditions are
    met, see ``lep_target_rse`` or ``lep_computation_min_fails`` for more details.
    """
    batch_size: int = 10_000
    """Number of sampling experiments that are submitted per batch."""
    lep_target_rse: float = 1e-4
    """Target relative standard error under which a sampling task is considered precise
    enough and can be stopped before ``max_shots`` sampling tasks have returned."""
    lep_computation_min_fails: int = 10
    """Minimum number of failures that should be witnessed before stopping a sampling task.

    A sampling task may stop with less failures, for example if ``max_shots`` shots have
    been performed."""
    max_workers: int = 1
    """Max number of parallel processes used by the function.

    Default to ``1`` which means fully sequential.
    """


@dataclass(frozen=True)
class BoundSearchParameters:
    """Configuration for automatic error-budget bound discovery.

    Attributes:
        parameter_domains: Finite, ordered limits for each noise parameter. Domains
            are copied to immutable tuples and need not be probability intervals.
        initial_relative_half_width: Positive initial half-width relative to the
            absolute evaluation coordinate.
        expansion_factor: Finite factor greater than one for interval expansion.
        sensitivity_z_score: Positive endpoint signal-to-noise threshold.
        min_logical_failures: Positive integer failure-count floor per experiment.
        max_lep: Logical error probability ceiling, strictly between zero and 0.5.
        max_trials_per_parameter: Positive integer cap on new noncentral probes
            for each parameter.

    Raises:
        ValueError: If domains are empty, malformed, non-finite, or unordered, or
            a configuration field is outside its allowed range.
    """

    parameter_domains: Sequence[tuple[float, float]]
    initial_relative_half_width: float = 0.25
    expansion_factor: float = 2.0
    sensitivity_z_score: float = 3.0
    min_logical_failures: int = 10
    max_lep: float = 0.45
    max_trials_per_param: int = 16

    def __post_init__(self) -> None:
        domain_error = (
            "parameter_domains must be a nonempty sequence of finite "
            "(lower, upper) pairs with lower < upper."
        )
        try:
            domains = np.asarray(self.parameter_domains, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(domain_error) from exc
        if (
            domains.ndim != 2
            or domains.shape[0] == 0
            or domains.shape[1] != 2
            or not np.all(np.isfinite(domains))
            or np.any(domains[:, 0] >= domains[:, 1])
        ):
            raise ValueError(domain_error)
        object.__setattr__(
            self,
            "parameter_domains",
            tuple((float(lower), float(upper)) for lower, upper in domains),
        )

        for name, minimum in (
            ("initial_relative_half_width", 0),
            ("expansion_factor", 1),
            ("sensitivity_z_score", 0),
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= minimum:
                msg = f"{name} must be finite and greater than {minimum}."
                raise ValueError(msg)
        if not np.isfinite(self.max_lep) or not 0 < self.max_lep < 0.5:
            msg = "max_lep must be finite and strictly between 0 and 0.5."
            raise ValueError(msg)
        for name in ("min_logical_failures", "max_trials_per_parameter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                msg = f"{name} must be a positive integer."
                raise ValueError(msg)

    def validate_parameter_count(self, num_parameters: int) -> None:
        """Check that every calibration parameter has a domain.

        Args:
            num_parameters: Number of entries in the calibration vector.

        Raises:
            ValueError: If the number of domains does not match the vector.
        """
        if len(self.parameter_domains) != num_parameters:
            msg = (
                f"parameter_domains must contain {num_parameters} domains to match "
                f"noise_parameters; got {len(self.parameter_domains)}."
            )
            raise ValueError(msg)
