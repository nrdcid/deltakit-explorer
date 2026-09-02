import numpy as np

from deltakit_explorer.analysis.error_budget import find_error_budget_bounds


def fake_noise_model(circuit, parameters):
    return circuit


def test_find_error_budget_bounds_returns_interval_containing_half_parameter():
    result = find_error_budget_bounds(
        noise_model=fake_noise_model,
        noise_parameters=np.array([0.02]),
        num_rounds_per_distance={3: [3]},
        max_iterations=1,
        seed=0,
    )

    lower, upper = result.bounds[0]

    assert lower < 0.01 < upper
