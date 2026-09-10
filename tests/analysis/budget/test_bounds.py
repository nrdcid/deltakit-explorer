import numpy as np
import pytest

from deltakit_explorer.analysis.error_budget._bounds import find_error_budget_bounds
from tests.analysis.budget.test_generation import noise_model


def test_find_error_budget_bounds_returns_interval_containing_half_parameter():
    result = find_error_budget_bounds(
        noise_model=noise_model,
        noise_parameters=np.array([0.02]),
        num_rounds_per_distance={3: [3]},
        max_iterations=1,
        seed=0,
    )

    lower, upper = result.bounds[0]

    assert lower < 0.01 < upper


@pytest.mark.parametrize(
    "point",
    [None, np.array([0.02, 0.04]), np.array([0.01, 0.02]), [0.015, 0.03]],
)
def test_find_error_budget_bounds_centers_on_gradient_evaluation_point(point):
    parameters = np.array([0.02, 0.04])
    original_parameters = parameters.copy()
    expected_center = parameters / 2 if point is None else np.asarray(point).copy()

    result = find_error_budget_bounds(
        noise_model=noise_model,
        noise_parameters=parameters,
        num_rounds_per_distance={3: [3]},
        gradient_evaluation_point=point,
        initial_relative_width=0.2,
    )

    np.testing.assert_allclose(
        result.bounds,
        np.column_stack((expected_center * 0.8, expected_center * 1.2)),
    )
    np.testing.assert_array_equal(parameters, original_parameters)
    if point is not None:
        np.testing.assert_array_equal(point, expected_center)


@pytest.mark.parametrize("point", [0.01, [0.01], [[0.01, 0.02]]])
def test_find_error_budget_bounds_rejects_mismatched_evaluation_point(point):
    with pytest.raises(ValueError, match="same shape as noise_parameters"):
        find_error_budget_bounds(
            noise_model=noise_model,
            noise_parameters=[0.02, 0.04],
            num_rounds_per_distance={3: [3]},
            gradient_evaluation_point=point,
        )
